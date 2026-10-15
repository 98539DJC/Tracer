#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate a caption for a single MSVD clip from a fine-tuned checkpoint.

Verifies that INFERENCE works with the precomputed-feature architecture
(no SigLIP, no wordweights) and that the trained projector -- including the
BCM output gate added in `TracerMambaConnector` -- is restored from
`non_lora_trainables.bin`.

Usage:
  python tools/eval_one_video.py \
      --ckpt output/MSVD/checkpoints/_smoke_v9/checkpoint-20 \
      --video_id eyhzdC936uk_15_27
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer
from videollama2 import mm_infer
from videollama2.mm_utils import load_video_features
from videollama2.model.videollama2_qwen2 import Videollama2Qwen2ForCausalLM
from videollama2.utils import disable_torch_init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='checkpoint dir with adapter + non_lora_trainables.bin')
    ap.add_argument('--video_id', required=True)
    ap.add_argument('--feature_dir', default='datasets/MSVD_sft/siglip_feat16')
    ap.add_argument('--model_path', default='DAMO-NLP-SG/VideoLLaMA2.1-7B-16F')
    ap.add_argument('--instruct', default='Use only one short sentence to describe the video.')
    a = ap.parse_args()

    disable_torch_init()

    config = AutoConfig.from_pretrained(a.model_path)
    config.mm_projector_type = 'route_mamba'
    config.mm_vision_tower = 'precomputed'          # no SigLIP at inference
    config.use_precomputed_features = True
    model = Videollama2Qwen2ForCausalLM.from_pretrained(
        a.model_path, config=config, torch_dtype=torch.bfloat16)
    model = model.to('cuda')

    sd = torch.load(os.path.join(a.ckpt, 'non_lora_trainables.bin'), map_location='cpu')
    sd = {(k[11:] if k.startswith('base_model.') else k): v for k, v in sd.items()}
    if any(k.startswith('model.model.') for k in sd):
        sd = {(k[6:] if k.startswith('model.') else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)

    gate = [k for k in sd if 'downsample_b' in k]
    wfus = [k for k in sd if k.endswith('W_fus.weight')]
    print('[eval] non_lora tensors      :', len(sd))
    print('[eval] BCM gate keys restored:', gate or '*** MISSING ***')
    print('[eval] W_fus keys restored   :', wfus or '*** MISSING ***')
    for k in gate + wfus:
        print('       %-60s max|.|=%.6g' % (k, float(sd[k].abs().max())))
    print('[eval] missing keys (<=5)    :', missing[:5], '... total', len(missing))

    model = PeftModel.from_pretrained(model, a.ckpt)
    model = model.merge_and_unload()
    model.eval()

    tok = AutoTokenizer.from_pretrained(a.model_path, use_fast=False)
    if tok.pad_token is None and tok.unk_token is not None:
        tok.pad_token = tok.unk_token

    feat_path = os.path.join(a.feature_dir, a.video_id + '.npy')
    feat = load_video_features(feat_path)
    print('[eval] features %s %s <- %s' % (tuple(feat.shape), feat.dtype, feat_path))

    with torch.no_grad():
        out = mm_infer(feat, a.instruct, model=model, tokenizer=tok,
                       do_sample=False, modal='video')
    print('[eval] video_id  :', a.video_id)
    print('[eval] prediction:', out)


if __name__ == '__main__':
    main()
