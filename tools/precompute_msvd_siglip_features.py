#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Precompute SigLIP patch features for MSVD (or any) videos.

This replaces the SigLIP vision tower inside training: features are computed once
here and then read from disk by ``train.py`` via ``--video_feature_folder``,
which removes ~2.6 GB of VRAM and one vision-tower forward pass per sample.

Output format
-------------
One ``.npy`` per video named ``{video_id}.npy`` holding the SigLIP penultimate
hidden states for ``--num_frames`` uniformly sampled candidate frames:

    (num_frames, 729, 1152)  uint16-packed bfloat16      # 32 -> 53.7 MB

uint16 packing keeps the file at half the size of float32 while staying
bit-identical to what the tower produced (bfloat16 -> float32 is lossless).

Usage
-----
    # all MSVD videos referenced by datasets/train.json, on GPU 0
    PYTHONNOUSERSITE=1 python tools/precompute_msvd_siglip_features.py \
        --data_json datasets/train.json --video_root datasets \
        --out_dir datasets/MSVD_sft/siglip_feat32 --num_frames 32 --device cuda:0

    # shard across 4 GPUs
    for i in 0 1 2 3; do ... --shard $i --num_shards 4 --device cuda:$i & done
"""
import argparse
import json
import os
import sys
import time

# Everything lives locally on this cluster; without these, every
# from_pretrained() spends ~10 minutes on huggingface.co connect timeouts.
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import SiglipVisionModel, SiglipImageProcessor          # noqa: E402
from videollama2.mm_utils import process_video, save_video_features       # noqa: E402

DEFAULT_MODEL = 'google/siglip-so400m-patch14-384'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_json', default='datasets/train.json',
                   help='JSON list whose entries carry a "video" field '
                        '(relative to --video_root).')
    p.add_argument('--video_root', default='datasets',
                   help='Prefix prepended to the "video" field.')
    p.add_argument('--videos_dir', default=None,
                   help='Alternative: cache every video file in this directory.')
    p.add_argument('--out_dir', required=True, help='Where to write {video_id}.npy')
    p.add_argument('--num_frames', type=int, default=32,
                   help='Candidate frames to sample per video (keyframe indices '
                        'live in this space).')
    p.add_argument('--select_layer', type=int, default=-2,
                   help='hidden_states index (training uses mm_vision_select_layer=-2).')
    p.add_argument('--model_name', default=DEFAULT_MODEL)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float32'],
                   help='Storage dtype. bfloat16 is bit-exact vs the tower output.')
    p.add_argument('--frame_batch', type=int, default=32,
                   help='How many frames to push through the tower at once.')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num_shards', type=int, default=1)
    p.add_argument('--limit', type=int, default=0, help='Only do the first N videos.')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--aspect_ratio', default='pad', choices=['pad', 'square'])
    return p.parse_args()


def collect_videos(args):
    """Return a de-duplicated list of (video_id, absolute_video_path)."""
    seen, videos = set(), []
    if args.videos_dir:
        names = sorted(os.listdir(args.videos_dir))
        for n in names:
            vid = os.path.splitext(n)[0]
            if vid in seen:
                continue
            seen.add(vid)
            videos.append((vid, os.path.join(args.videos_dir, n)))
    else:
        with open(args.data_json, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        for e in entries:
            rel = e.get('video')
            if not rel:
                continue
            vid = os.path.basename(rel).split('.')[0]
            if vid in seen:
                continue
            seen.add(vid)
            videos.append((vid, os.path.join(args.video_root, rel)))
    return videos


@torch.no_grad()
def extract(model, processor, video_path, args, device, dtype):
    pixels = process_video(video_path, processor, aspect_ratio=args.aspect_ratio,
                           num_frames=args.num_frames)          # (T, 3, H, W) float32
    chunks = []
    for start in range(0, pixels.shape[0], args.frame_batch):
        batch = pixels[start:start + args.frame_batch].to(device)
        # Mirror training: bf16 weights + autocast, float32 pixel input.
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=(dtype == torch.bfloat16)):
            out = model(batch, output_hidden_states=True)
        chunks.append(out.hidden_states[args.select_layer].to(dtype).cpu())
    return torch.cat(chunks, dim=0)                             # (T, N, H)


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32

    videos = collect_videos(args)
    if args.limit:
        videos = videos[:args.limit]
    videos = videos[args.shard::args.num_shards]
    os.makedirs(args.out_dir, exist_ok=True)

    todo = []
    for vid, path in videos:
        out = os.path.join(args.out_dir, vid + '.npy')
        if os.path.exists(out) and not args.overwrite:
            continue
        if not os.path.exists(path):
            print('[skip] missing video %s' % path, flush=True)
            continue
        todo.append((vid, path, out))

    print('[feat] %d videos total, shard %d/%d -> %d to compute, out=%s'
          % (len(videos), args.shard, args.num_shards, len(todo), args.out_dir), flush=True)
    if not todo:
        print('[feat] nothing to do', flush=True)
        return

    t0 = time.time()
    model = SiglipVisionModel.from_pretrained(args.model_name,
                                              torch_dtype=dtype).to(device).eval()
    processor = SiglipImageProcessor.from_pretrained(args.model_name)
    n_par = sum(p.numel() for p in model.parameters())
    print('[feat] tower ready: %.1fM params, dtype=%s, device=%s (%.1fs)'
          % (n_par / 1e6, args.dtype, device, time.time() - t0), flush=True)

    done = skipped = failed = 0
    t0 = time.time()
    for i, (vid, path, out) in enumerate(todo):
        try:
            feats = extract(model, processor, path, args, device, dtype)
            save_video_features(out, feats)
            done += 1
            if done % 20 == 0 or done == 1:
                el = time.time() - t0
                print('[feat] %d/%d  %s  shape=%s  %.1f s/video  eta %.1f min'
                      % (done, len(todo), vid, tuple(feats.shape), el / done,
                         el / done * (len(todo) - done) / 60), flush=True)
        except Exception as ex:
            failed += 1
            print('[feat] FAILED %s (%s): %s: %s' % (vid, path, type(ex).__name__, ex),
                  flush=True)

    el = time.time() - t0
    print('[feat] done: %d written, %d failed, %d already present, %.1f min (%.2f s/video)'
          % (done, failed, skipped, el / 60, el / max(done, 1)), flush=True)


if __name__ == '__main__':
    main()
