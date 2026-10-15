#!/bin/bash
# =============================================================================
# Find the largest per-device batch that fits in one GPU's VRAM, for the MSVD
# QLoRA recipe.  Run this after changing the model / sequence length, then put
# the result into PER_DEVICE_BS in finetune_qlora_MSVD_v9.sh.
#
#   bash tools/probe_batch.sh                 # 8,12,16,24 on GPU 2
#   GPU=3 BATCHES="12 14 16" bash tools/probe_batch.sh
#
# One GPU is used on purpose: with 2 GPUs + ZeRO-2 the per-GPU footprint is the
# same or slightly smaller, so a batch that fits here is safe for two.
# max_steps=1 -> train.py prints its [MEMSTAT] peak line when training returns.
# =============================================================================
set -o pipefail
export LC_ALL=C
cd "$(dirname "$0")/.." || exit 1
PY=/home/megaium/miniconda3/envs/tracer/bin/python
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU=${GPU:-2}
BATCHES=${BATCHES:-"8 12 16 24"}
CKPT=${CKPT:-True}
OUT=output/MSVD/checkpoints/_probe_batch
mkdir -p "$OUT"

echo "probing on GPU ${GPU} (gradient_checkpointing=${CKPT})"
printf '  %-8s %-6s %-10s %-14s %-14s %-12s %s\n' batch exit OOM peak_alloc reserved wall_s samples/s

for bs in $BATCHES; do
  CUDA_VISIBLE_DEVICES=$GPU timeout 1200 $PY -u -m torch.distributed.run \
    --nnodes 1 --nproc_per_node 1 --master_port 29771 videollama2/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 1e-5 --bits 4 \
    --deepspeed scripts/zero2.json --model_type videollama2_qwen2 \
    --model_path DAMO-NLP-SG/VideoLLaMA2.1-7B-16F --vision_tower precomputed \
    --use_precomputed_features True --video_feature_folder datasets/MSVD_sft/siglip_feat32 \
    --mm_projector_type route_mamba \
    --pretrain_mm_mlp_adapter DAMO-NLP-SG/VideoLLaMA2.1-7B-16F-Base/mm_projector.bin \
    --data_path datasets/train.json --data_folder datasets/ --mm_vision_select_layer -2 \
    --image_aspect_ratio pad --num_frames 16 --bf16 True --tf32 True --fp16 False \
    --num_train_epochs 1 --per_device_train_batch_size $bs --gradient_accumulation_steps 1 \
    --max_steps 1 --save_strategy no --learning_rate 1e-5 --lr_scheduler_type cosine \
    --logging_steps 1 --model_max_length 2048 --gradient_checkpointing $CKPT \
    --dataloader_num_workers 4 --report_to none \
    --output_dir $OUT --run_name _probe > /tmp/pb_$bs.log 2>&1
  rc=$?
  peak=$(grep -ao "max_alloc=[0-9.]*GB" /tmp/pb_$bs.log | tail -1 | cut -d= -f2)
  resv=$(grep -ao "reserved=[0-9.]*GB"  /tmp/pb_$bs.log | tail -1 | cut -d= -f2)
  wall=$(grep -ao "'train_runtime': [0-9.]*" /tmp/pb_$bs.log | tail -1 | cut -d' ' -f2)
  sps=$(grep -ao "'train_samples_per_second': [0-9.]*" /tmp/pb_$bs.log | tail -1 | cut -d' ' -f2)
  printf '  %-8s %-6s %-10s %-14s %-14s %-12s %s\n' "$bs" "$rc" \
     "$(grep -ac 'out of memory' /tmp/pb_$bs.log)" "${peak:-n/a}" "${resv:-n/a}" \
     "${wall:-n/a}" "${sps:-n/a}"
  rm -f /tmp/pb_$bs.log
done
rm -rf "$OUT"
echo
echo "The card has 97.9 GB; pick the largest batch whose reserved stays under ~94 GB."
