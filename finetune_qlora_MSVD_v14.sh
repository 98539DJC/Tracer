#!/bin/bash
# =============================================================================
# MSVD fine-tuning -- QLoRA 4-bit, Tracer `route_mamba` projector,
# precomputed SigLIP features (no video decoding, no vision tower).
#
#   GPUs          : 2,3                       (see CUDA_VISIBLE_DEVICES below)
#   logs          : every run is teed into output/MSVD/logs/<run>_<timestamp>.log
#                   with an env/config header, per-step losses, and a final
#                   peak-memory summary; <run>.latest.log points at the newest.
#   VRAM          : PER_DEVICE_BS was MEASURED on this machine (97.9 GB cards):
#
#                     per-device bs   peak alloc   reserved   fits
#                          8            62.3 GB     63.8 GB    yes
#                         12            87.8 GB     89.3 GB    yes   <- default (~90%)
#                         16             >94 GB        OOM     no
#
#                   memory is dominated by the TracerMambaConnector activations
#                   (~3.7 GB/sample), NOT by the LLM/logits (~0.7 GB/sample).
#                   Re-measure with tools/probe_batch.sh after model changes.
#
# usage:
#     bash finetune_qlora_MSVD_v9.sh
#     PER_DEVICE_BS=8 GRAD_ACC=8 bash finetune_qlora_MSVD_v9.sh    # global 128
#     MAX_STEPS=3 bash finetune_qlora_MSVD_v9.sh                   # smoke test
# =============================================================================
set -o pipefail
export LC_ALL=C
cd /mnt/lzz/Tracer-main || exit 1

# ------------------------------- hardware -----------------------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NPROC_PER_NODE=2

# ------------------------------- batching ----------------------------------
# per-device batch that fills the card (measured, see tools/probe_batch.sh)
PER_DEVICE_BS=${PER_DEVICE_BS:-12}
GRAD_ACC=${GRAD_ACC:-5}                       # 12 x 5 x 2 = 120 samples / step
GRAD_CKPT=${GRAD_CKPT:-True}
EFFECTIVE_GLOBAL=$(( PER_DEVICE_BS * GRAD_ACC * NPROC_PER_NODE ))
MAX_STEPS=${MAX_STEPS:-}                      # empty = run the full epoch
SAVE_STEPS=${SAVE_STEPS:-50}

# ------------------------------- paths -------------------------------------
RUN_NAME=${RUN_NAME:-my_run_qlora_v14}
DATA_DIR=datasets
FEATURE_DIR=${DATA_DIR}/MSVD_sft/siglip_feat32
MODEL_PATH=DAMO-NLP-SG/VideoLLaMA2.1-7B-16F
PROJECTOR_PATH=DAMO-NLP-SG/VideoLLaMA2.1-7B-16F-Base/mm_projector.bin
OUTP_DIR=output/MSVD/checkpoints/${RUN_NAME}
LOG_DIR=output/MSVD/logs
mkdir -p "$OUTP_DIR" "$LOG_DIR"
LOG="${LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
ln -sfn "$(basename "$LOG")" "${LOG_DIR}/${RUN_NAME}.latest.log"

export PYTHONNOUSERSITE=1            # keep ~/.local's broken flash_attn out
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
PY=/home/megaium/miniconda3/envs/tracer/bin/python

# ------------------------------- pre-flight --------------------------------
{
echo "=============================================================================="
echo " MSVD fine-tuning            start: $(date '+%F %T')"
echo "=============================================================================="
echo " host          : $(hostname)"
echo " run name      : ${RUN_NAME}"
echo " log           : ${LOG}"
echo " git           : $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo " GPUs          : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  nproc=${NPROC_PER_NODE}"
echo " batch         : per_device=${PER_DEVICE_BS} x grad_accum=${GRAD_ACC} x ${NPROC_PER_NODE} = ${EFFECTIVE_GLOBAL}"
echo " grad ckpt     : ${GRAD_CKPT}      max_steps: ${MAX_STEPS:-<full epoch>}"
echo " model (LLM)   : ${MODEL_PATH}"
echo "                 -> $(readlink -f ${MODEL_PATH} 2>/dev/null)"
echo " projector     : ${PROJECTOR_PATH}"
echo "                 -> $(readlink -f ${PROJECTOR_PATH} 2>/dev/null)"
echo " feature cache : ${FEATURE_DIR}"
echo " loss          : L_cap = sum(w*CE)  (wordweighted, chunked CE)"
echo "                 + 0.1*L_cor + 0.7*L_cyc + 0.3*L_div  "
echo "------------------------------------------------------------------------------"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
echo "------------------------------------------------------------------------------"
} | tee -a "$LOG"

for f in "${MODEL_PATH}/config.json" "${PROJECTOR_PATH}" "/home/megaium/Tracer-main/datasets/train.json"; do
  [ -e "$f" ] || { echo "ERROR: missing $f" | tee -a "$LOG"; exit 1; }
done

# every video referenced by train.json must have a cached feature file
$PY - "$FEATURE_DIR" "/home/megaium/Tracer-main/datasets/train.json" <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, os, sys
feat_dir, train_json = sys.argv[1], sys.argv[2]
rows = json.load(open(train_json, encoding='utf-8'))
vids = {os.path.basename(r['video']).rsplit('.', 1)[0] for r in rows if 'video' in r}
missing = [v for v in vids if not os.path.exists(os.path.join(feat_dir, v + '.npy'))]
print('[preflight] train rows   : %d' % len(rows))
if missing:
    print('[preflight] %d clip(s) without a cached feature file, e.g. %s'
          % (len(missing), missing[:3]))
    print('[preflight] cached-feature mode needs them all -> abort '
          '(tools/precompute_msvd_siglip_features.py)')
    sys.exit(1)
print('[preflight] feature cache complete for every clip in train.json  OK')
PYEOF
[ "${PIPESTATUS[0]}" = "0" ] || { echo "pre-flight FAILED, aborting (see log above)" | tee -a "$LOG"; exit 1; }

# ------------------------------- train -------------------------------------
EXTRA=""
[ -n "${MAX_STEPS}" ] && EXTRA="--max_steps ${MAX_STEPS}"

echo "[$(date '+%T')] launching training ..." | tee -a "$LOG"
$PY -u -m torch.distributed.run --nnodes 1 --nproc_per_node ${NPROC_PER_NODE} \
    --master_addr=127.0.0.1 --master_port=29669 --node_rank=0 \
    videollama2/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 1e-4 --bits 4 \
    --deepspeed scripts/zero2.json \
    --model_type videollama2_qwen2 \
    --model_path ${MODEL_PATH} \
    --vision_tower precomputed \
    --use_precomputed_features True \
    --video_feature_folder ${FEATURE_DIR} \
    --mm_projector_type route_mamba \
    --pretrain_mm_mlp_adapter ${PROJECTOR_PATH} \
    --data_path /home/megaium/Tracer-main/datasets/train.json \
    --data_folder ${DATA_DIR}/ \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio pad \
    --num_frames 16 \
    --bf16 True --tf32 True --fp16 False \
    --output_dir ${OUTP_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${PER_DEVICE_BS} \
    --per_device_eval_batch_size ${PER_DEVICE_BS} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --gradient_checkpointing ${GRAD_CKPT} \
    --save_strategy steps --save_steps ${SAVE_STEPS} --save_total_limit 99 \
    --learning_rate 1e-5 --weight_decay 0. --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 --logging_first_step True --log_level info \
    --model_max_length 2048 \
    --dataloader_num_workers 16 --dataloader_pin_memory True \
    --use_wordweight_loss True --wordweight_normalize sum \
    --gamma_1 0.1 --gamma_2 0.7 --gamma_3 0.3 \
    --report_to tensorboard \
    --run_name ${RUN_NAME} ${EXTRA} 2>&1 | tee -a "$LOG"
TRAIN_RC=${PIPESTATUS[0]}

# ------------------------------- summary -----------------------------------
{
echo "------------------------------------------------------------------------------"
echo " finished      : $(date '+%F %T')   exit=${TRAIN_RC}"
echo " log           : ${LOG}"
grep -a "MEMSTAT\] allocated" "$LOG" | tail -1
grep -ao "'train_runtime': [0-9.]*"            "$LOG" | tail -1
grep -ao "'train_loss': [0-9.]*"               "$LOG" | tail -1
grep -ao "'train_samples_per_second': [0-9.]*" "$LOG" | tail -1
echo "=============================================================================="
} | tee -a "$LOG"

exit ${TRAIN_RC}
