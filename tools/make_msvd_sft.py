# -*- coding: utf-8 -*-
"""
自制 MSVD 数据集脚本（复现 TRACER 用）。

输入（需要自行下载，见 README 步骤）：
  ann_dir/cap_train.json  cap_val.json  cap_test.json   (LAVIS 镜像的 MSVD 标准划分标注)

输出：
  datasets/MSVD_sft/make_sft/train.jsonl   每行 {"video_id": ..., "golden_captions": [...]}
  datasets/MSVD_sft/make_sft/val.jsonl     同上
  datasets/MSVD_sft/make_sft/test.jsonl    同上
  datasets/train.json                      VideoLLaMA2 SFT 格式训练集

用法：
  python tools/make_msvd_sft.py --ann_dir datasets/MSVD_sft/annotations \
      --video_dir datasets/MSVD_sft/videos --out_train datasets/train.json
"""
import argparse
import json
import os
from collections import OrderedDict

# 与 generate.py 评测时使用的指令保持一致
INSTRUCT = "Use only one short sentence to describe the video."


def load_annotations(ann_dir):
    splits = {}
    for split in ("train", "val", "test"):
        path = os.path.join(ann_dir, f"cap_{split}.json")
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        # 按视频分组，保持标注原始顺序
        groups = OrderedDict()
        for e in entries:
            vid = e["video"]
            groups.setdefault(vid, []).append(e["caption"].strip())
        splits[split] = groups
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_dir", default="datasets/MSVD_sft/annotations")
    parser.add_argument("--video_dir", default="datasets/MSVD_sft/videos")
    parser.add_argument("--out_train", default="datasets/train.json")
    parser.add_argument("--sft_dir", default="datasets/MSVD_sft/make_sft")
    args = parser.parse_args()

    splits = load_annotations(args.ann_dir)

    os.makedirs(args.sft_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_train), exist_ok=True)

    # 1) 评测用 jsonl：每行一个视频 + 全部参考字幕
    for split, groups in splits.items():
        out_path = os.path.join(args.sft_dir, f"{split}.jsonl")
        n_missing = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for vid, caps in groups.items():
                if not os.path.exists(os.path.join(args.video_dir, vid)):
                    n_missing += 1
                    continue
                f.write(json.dumps({
                    "video_id": os.path.splitext(vid)[0],
                    "golden_captions": caps,
                }, ensure_ascii=False) + "\n")
        print(f"{split}: {len(groups)} videos, missing video files: {n_missing} -> {out_path}")

    # 2) 训练用 train.json：VideoLLaMA2 SFT 格式，每个 (视频, 字幕) 对一条样本
    train_groups = splits["train"]
    train_samples = []
    idx = 0
    n_missing = 0
    for vid, caps in train_groups.items():
        if not os.path.exists(os.path.join(args.video_dir, vid)):
            n_missing += 1
            continue
        rel_path = os.path.relpath(os.path.join(args.video_dir, vid),
                                   os.path.dirname(args.out_train))
        for cap in caps:
            train_samples.append({
                "id": idx,
                "video": rel_path.replace(os.sep, "/"),
                "conversations": [
                    {"from": "human", "value": f"<video>\n{INSTRUCT}"},
                    {"from": "gpt", "value": cap},
                ],
            })
            idx += 1
    with open(args.out_train, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    print(f"train.json: {len(train_groups) - n_missing} videos, "
          f"{len(train_samples)} samples -> {args.out_train}")


if __name__ == "__main__":
    main()
