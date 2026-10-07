# -*- coding: utf-8 -*-
"""
生成 Tracer 复现所需的两个辅助数据文件（默认值版，先跑通用）：

1. wordweight : per-word weighted CE 的权重，默认全 1（等价普通 CE）
2. keyframe   : 关键帧索引，默认均匀取 16 帧（np.arange(16)）

真实算法（per-word 权重、TCVR 关键帧）论文正文未给细节，后续再补。

用法（在 ~/Tracer-main 目录下，已激活 tracer 环境）：
  python tools/make_msvd_aux.py --model_path DAMO-NLP-SG/VideoLLaMA2.1-7B-16F
"""
import argparse
import json
import os
from collections import OrderedDict

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", default="datasets/train.json")
    parser.add_argument("--model_path", default="DAMO-NLP-SG/VideoLLaMA2.1-7B-16F")
    parser.add_argument("--wordweight_dir",
                        default="datasets/MSVD_sft/MSVD_wordweight/Qwen2_wordweight_after")
    parser.add_argument("--keyframe_dir", default="datasets/MSVD_sft/tcvr_32_16")
    parser.add_argument("--num_frames", type=int, default=16)
    args = parser.parse_args()

    with open(args.train_json, "r", encoding="utf-8") as f:
        samples = json.load(f)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    os.makedirs(args.wordweight_dir, exist_ok=True)
    os.makedirs(args.keyframe_dir, exist_ok=True)

    # 按 video 分组，保持 id 排序；video_id/gt_id 的确定方式和 train.py 的 __getitem__ 一致
    groups = OrderedDict()
    for s in samples:
        groups.setdefault(s["video"], []).append(s)

    keyframe_index = np.arange(args.num_frames, dtype=np.int64)

    n = 0
    for video_path, entries in groups.items():
        video_id = video_path.split("/")[-1].split(".")[0]
        entries_sorted = sorted(entries, key=lambda x: x["id"])
        for gt_id, entry in enumerate(entries_sorted):
            caption = entry["conversations"][1]["value"]

            # wordweight：全 1，长度 = caption 的 token 数
            ids = tokenizer.encode(caption)
            wordweight = np.ones(len(ids), dtype=np.float32)

            ww_path = os.path.join(args.wordweight_dir, f"{video_id}_{gt_id}.npy")
            kf_path = os.path.join(args.keyframe_dir, f"{video_id}_{gt_id}.npy")
            np.save(ww_path, wordweight)
            np.save(kf_path, keyframe_index)
            n += 1

    print(f"done: {n} samples -> wordweight + keyframe")


if __name__ == "__main__":
    main()
