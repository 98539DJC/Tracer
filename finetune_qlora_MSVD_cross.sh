#!/bin/bash
# =============================================================================
# 方案 v2：Tracer + Vim Cross-Mamba + RFM —— A1–A5 训练脚本
#
#   A1: ST_FUSION_MODE=conv ST_SCAN_MODE=off   ST_ALIGN_MODE=nte_center
#   A2: ST_FUSION_MODE=rfm  ST_SCAN_MODE=off   ST_ALIGN_MODE=nte_center
#   A3: ST_FUSION_MODE=rfm  ST_SCAN_MODE=self  ST_ALIGN_MODE=nte_center
#   A4: ST_FUSION_MODE=rfm  ST_SCAN_MODE=cross ST_ALIGN_MODE=nte_center
#   A5: ST_FUSION_MODE=rfm  ST_SCAN_MODE=cross ST_ALIGN_MODE=slot
#
#   共同起点 B0@250（时间投影器 + LoRA），冻结一切旧参数；
#   只训新增：Vim(sp_norm/sp_mamba) + Cross-Mamba(scan_*) + 融合(fuse)。
#   新参数 LR = mm_projector_lr = 1e-4；A 组只用加权 caption loss 反向。
#   首轮短跑 200 步，每 50 步存 ckpt，验证集选点。
#
#   usage:
#     ST_FUSION_MODE=rfm ST_SCAN_MODE=cross ST_ALIGN_MODE=nte_center \
#       RUN_NAME=a4_smoke MAX_STEPS=2 bash finetune_qlora_MSVD_cross.sh
# =============================================================================
set -o pipefail
export LC_ALL=C
cd "$(dirname "$0")" || exit 1

# ------------------------------- hardware -----------------------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}

# ------------------------------- batching ----------------------------------
# 方案 §13.2 / 用户确认：首轮 A 组统一 2 卡 × bs3 × acc10 = 60。
PER_DEVICE_BS=${PER_DEVICE_BS:-3}
GRAD_ACC=${GRAD_ACC:-10}
GRAD_CKPT=${GRAD_CKPT:-True}
# ------------------------------- mode ---------------------------------------
# JOINT=1: 从头时间+空间+融合+LoRA 联合训练（Base 投影器初始化，812 步）
# JOINT=0: 方案 §6.1 冻结时间分支，仅训新模块（B0@250 初始化，A 组）
JOINT=${JOINT:-0}
MAX_STEPS=${MAX_STEPS:-$([ "$JOINT" = "1" ] && echo 812 || echo 200)}
SAVE_STEPS=${SAVE_STEPS:-50}
SEED=${SEED:-42}

# ------------------------------- lr -----------------------------------------
# 默认 = B0 原配方；减半对照用 LORA_LR=5e-6 PROJ_LR=5e-5（NTE/BCM 跟随投影器 LR）
LORA_LR=${LORA_LR:-1e-5}
PROJ_LR=${PROJ_LR:-1e-4}

# ------------------------------- v2 switches --------------------------------
ST_FUSION_MODE=${ST_FUSION_MODE:-rfm}
ST_SCAN_MODE=${ST_SCAN_MODE:-cross}
ST_ALIGN_MODE=${ST_ALIGN_MODE:-nte_center}
ST_FUSION_SCALE=${ST_FUSION_SCALE:-0.1}
ST_RANK_WEIGHT=${ST_RANK_WEIGHT:-0}
ST_FRAME_CHUNK=${ST_FRAME_CHUNK:-2}
if [ "$JOINT" = "1" ]; then
  # 时间分支参与训练：恢复 B0/ C-Joint 的辅助损失
  export ST_AUX_LOSS=${ST_AUX_LOSS:-1}
  export NTE_BCM_LR_MULT=${NTE_BCM_LR_MULT:-1.0}
else
  export ST_AUX_LOSS=${ST_AUX_LOSS:-0}
fi
export ST_DEBUG=${ST_DEBUG:-1}
export ST_DEBUG_COORD=${ST_DEBUG_COORD:-1}   # 首步打印真实 NTE 坐标/ATD/采样中心
export ST_RFM_CKPT=${ST_RFM_CKPT:-0}

# ------------------------------- paths --------------------------------------
B0_DIR=${B0_DIR:-output/MSVD/checkpoints/tracer_b0_lr1e4_vim1protocol_s42}
B0_CKPT=${B0_CKPT:-250}
BASE_PROJ=DAMO-NLP-SG/VideoLLaMA2.1-7B-16F-Base/mm_projector.bin
if [ "$JOINT" = "1" ]; then
  PROJECTOR_INIT=${PROJECTOR_INIT:-${BASE_PROJ}}
  LORA_INIT=""
  TRAINABLE_SCOPE=${TRAINABLE_SCOPE:-default}
  RUN_NAME=${RUN_NAME:-st_joint_${ST_FUSION_MODE}_${ST_SCAN_MODE}_${ST_ALIGN_MODE}_s${SEED}}
else
  PROJECTOR_INIT=${PROJECTOR_INIT:-${B0_DIR}/checkpoint-${B0_CKPT}/non_lora_trainables.bin}
  LORA_INIT=${LORA_INIT:-${B0_DIR}/checkpoint-${B0_CKPT}/adapter_model.safetensors}
  TRAINABLE_SCOPE=${TRAINABLE_SCOPE:-st}
  RUN_NAME=${RUN_NAME:-st_${ST_FUSION_MODE}_${ST_SCAN_MODE}_${ST_ALIGN_MODE}_s${SEED}}
fi

MM_PROJECTOR_TYPE=tracer_vim_cross

DATA_DIR=datasets
FEATURE_DIR=${DATA_DIR}/MSVD_sft/siglip_feat32
MODEL_PATH=DAMO-NLP-SG/VideoLLaMA2.1-7B-16F
OUTP_DIR=output/MSVD/checkpoints/${RUN_NAME}
LOG_DIR=output/MSVD/logs
mkdir -p "$OUTP_DIR" "$LOG_DIR"
LOG="${LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
ln -sfn "$(basename "$LOG")" "${LOG_DIR}/${RUN_NAME}.latest.log"

export PYTHONNOUSERSITE=1
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
PY=${PY:-/home/megaium/miniconda3/envs/tracer/bin/python}

# ------------------------------- pre-flight --------------------------------
{
echo "=============================================================================="
echo " v2 $([ "$JOINT" = "1" ] && echo JOINT-from-scratch || echo A-group-frozen-B0)  start: $(date '+%F %T')"
echo "=============================================================================="
echo " run name      : ${RUN_NAME}"
echo " log           : ${LOG}"
echo " GPUs          : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  nproc=${NPROC_PER_NODE}"
echo " batch         : per_device=${PER_DEVICE_BS} x grad_accum=${GRAD_ACC} x ${NPROC_PER_NODE} = $((PER_DEVICE_BS*GRAD_ACC*NPROC_PER_NODE))"
echo " projector     : ${MM_PROJECTOR_TYPE}   trainable_scope: ${TRAINABLE_SCOPE}"
echo " mode          : JOINT=${JOINT}  (1=从头联合训练, 0=冻结 B0 时间分支)"
echo " st switches   : fusion=${ST_FUSION_MODE} scan=${ST_SCAN_MODE} align=${ST_ALIGN_MODE}"
echo "                 gamma=${ST_FUSION_SCALE} rank=${ST_RANK_WEIGHT} frame_chunk=${ST_FRAME_CHUNK}"
echo " aux loss      : ST_AUX_LOSS=${ST_AUX_LOSS} (0 = 仅 caption 反向)"
echo " max_steps     : ${MAX_STEPS}   save=${SAVE_STEPS}   seed=${SEED}"
echo " lr            : LoRA=${LORA_LR}  projector=${PROJ_LR}  NTE/BCM x${NTE_BCM_LR_MULT:-1.0}"
echo " init projector: ${PROJECTOR_INIT} -> $(readlink -f ${PROJECTOR_INIT} 2>/dev/null)"
echo " init LoRA     : ${LORA_INIT:-<fresh>}"
echo " feature cache : ${FEATURE_DIR}"
echo "------------------------------------------------------------------------------"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
echo "------------------------------------------------------------------------------"
} | tee -a "$LOG"

for f in "${MODEL_PATH}/config.json" "${PROJECTOR_INIT}" "${DATA_DIR}/train.json"; do
  [ -e "$f" ] || { echo "ERROR: missing $f" | tee -a "$LOG"; exit 1; }
done
if [ -n "${LORA_INIT}" ] && [ ! -e "${LORA_INIT}" ]; then
  echo "ERROR: missing ${LORA_INIT}" | tee -a "$LOG"; exit 1
fi

# ------------------------------- train -------------------------------------
EXTRA=""
[ -n "${MAX_STEPS}" ] && EXTRA="--max_steps ${MAX_STEPS}"
LORA_ARG=""
[ -n "${LORA_INIT}" ] && LORA_ARG="--lora_init_path ${LORA_INIT}"

echo "[$(date '+%T')] launching v2 (${MM_PROJECTOR_TYPE}, JOINT=${JOINT}, seed=${SEED}) ..." | tee -a "$LOG"
$PY -u -m torch.distributed.run --nnodes 1 --nproc_per_node ${NPROC_PER_NODE} \
    --master_addr=127.0.0.1 --master_port=29681 --node_rank=0 \
    videollama2/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr ${PROJ_LR} --bits 4 \
    --deepspeed scripts/zero2.json \
    --model_type videollama2_qwen2 \
    --model_path ${MODEL_PATH} \
    --vision_tower precomputed \
    --use_precomputed_features True \
    --video_feature_folder ${FEATURE_DIR} \
    --mm_projector_type ${MM_PROJECTOR_TYPE} \
    --pretrain_mm_mlp_adapter ${PROJECTOR_INIT} \
    --data_path ${DATA_DIR}/train.json \
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
    --learning_rate ${LORA_LR} --weight_decay 0. --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 --logging_first_step True --log_level info \
    --model_max_length 4096 \
    --seed ${SEED} \
    --dataloader_num_workers 16 --dataloader_pin_memory True \
    --use_wordweight_loss True --wordweight_normalize sum \
    --gamma_1 0.1 --gamma_2 0.7 --gamma_3 0.3 \
    --trainable_scope ${TRAINABLE_SCOPE} \
    ${LORA_ARG} \
    --st_fusion_mode ${ST_FUSION_MODE} \
    --st_scan_mode ${ST_SCAN_MODE} \
    --st_align_mode ${ST_ALIGN_MODE} \
    --st_fusion_scale ${ST_FUSION_SCALE} \
    --st_rank_weight ${ST_RANK_WEIGHT} \
    --st_frame_chunk ${ST_FRAME_CHUNK} \
    --report_to tensorboard \
    --run_name ${RUN_NAME} ${EXTRA} 2>&1 | tee -a "$LOG"
TRAIN_RC=${PIPESTATUS[0]}

# ------------------------------- summary -----------------------------------
{
echo "------------------------------------------------------------------------------"
echo " finished      : $(date '+%F %T')   exit=${TRAIN_RC}"
echo " log           : ${LOG}"
grep -a "\[ST\]" "$LOG" | tail -1
grep -a "\[SCOPE\]" "$LOG" | tail -1
grep -a "\[PROJ\] loaded" "$LOG" | tail -1
grep -a "\[CREATE_OPTIM\] req_grad" "$LOG" | tail -1
grep -a "MEMSTAT\] allocated" "$LOG" | tail -1
grep -ao "'train_runtime': [0-9.]*" "$LOG" | tail -1
grep -ao "'train_loss': [0-9.]*" "$LOG" | tail -1
echo "=============================================================================="
} | tee -a "$LOG"

exit ${TRAIN_RC}
