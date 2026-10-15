#!/bin/bash
# 等 GPU 2,3 空闲 → 评估 checkpoint-100 → 算 CIDEr，全程写 log
# 主 log：eval_ckpt100.log；分块推理 log：eval_chunk0.log / eval_chunk1.log

MAINLOG="eval_ckpt200.log"
CHUNK0="eval_chunk0.log"
CHUNK1="eval_chunk1.log"

export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1   # keep ~/.local (broken flash_attn) out of sys.path
export HF_HUB_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# ---------- 1. 等 GPU 2,3 空闲 ----------
echo "[$(date '+%m-%d %H:%M:%S')] 等待 GPU 0,1 空闲（占用<3GB）..." | tee -a "$MAINLOG"
while true; do
    busy=0
    for g in 0 1; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)
        [ -z "$used" ] && { echo "GPU $g 查询失败，检查编号" | tee -a "$MAINLOG"; exit 1; }
        [ "$used" -gt 3000 ] && busy=1
    done
    [ "$busy" -eq 0 ] && break
    sleep 60
done
echo "[$(date '+%m-%d %H:%M:%S')] GPU 空闲，开始评估 checkpoint-200" | tee -a "$MAINLOG"

# ---------- 2. 2 卡并行推理 ----------
CUDA_VISIBLE_DEVICES=0 /home/megaium/miniconda3/envs/tracer/bin/python -u gen_chunk.py 0 2 > "$CHUNK0" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 /home/megaium/miniconda3/envs/tracer/bin/python -u gen_chunk.py 1 2 > "$CHUNK1" 2>&1 &
P1=$!

wait $P0; R0=$?
wait $P1; R1=$?

echo "[$(date '+%m-%d %H:%M:%S')] 推理完成 chunk0退出码=$R0 chunk1退出码=$R1" | tee -a "$MAINLOG"
grep 'chunk.*done' "$CHUNK0" "$CHUNK1" | tee -a "$MAINLOG"

# ---------- 3. 算 CIDEr ----------
echo "[$(date '+%m-%d %H:%M:%S')] 开始算 CIDEr" | tee -a "$MAINLOG"
/home/megaium/miniconda3/envs/tracer/bin/python merge_eval.py 2>&1 | tee -a "$MAINLOG"

echo "[$(date '+%m-%d %H:%M:%S')] 全部完成，结果见 $MAINLOG" | tee -a "$MAINLOG"
