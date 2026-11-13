#!/bin/bash
# =============================================================================
# 方案 v2 完整结构（A4）—— 从头时间+空间+融合+LoRA 联合训练
#
#   与 A 组（冻结 B0 时间分支）的区别：
#     - 初始化：Base mm_projector.bin（不加载 B0@250）
#     - 可训练：时间分支 NTE/BCM/原投影器 + Vim + Cross-Mamba + RFM + LoRA
#     - 损失：caption + 原 NTE 辅助损失（cor/cyc/div），ST_AUX_LOSS=1
#     - 812 步（1 epoch），2 卡 × bs3 × acc10
#
#   可调（默认 = 原 A4 全 LR 配方）：
#     RUN / MAX_STEPS / SAVE_STEPS / LORA_LR / PROJ_LR
#     减半 LR 探峰：LORA_LR=5e-6 PROJ_LR=5e-5 RUN=st_a4_joint_lrhalf_s42 SAVE_STEPS=25
#     EVAL_CKPTS  默认覆盖密集早期点（峰值在第一个测试点时应加密）
#     EVAL_SPLIT  默认 test（与 A4/B0-lrhalf 已有对照表同口径）
#
#   usage:
#     bash run_st_cross_joint.sh
#     LORA_LR=5e-6 PROJ_LR=5e-5 RUN=st_a4_joint_lrhalf_s42 SAVE_STEPS=25 \
#       EVAL_CKPTS="25 50 75 100 125 150 200 250 300" bash run_st_cross_joint.sh
# =============================================================================
set -u
cd "$(dirname "$0")" || exit 1

RUN=${RUN:-st_a4_joint_s42}
SMOKE=${RUN}_smoke
GPUS=${GPUS:-2,3}
MAX_STEPS=${MAX_STEPS:-812}
SAVE_STEPS=${SAVE_STEPS:-50}
EVAL_SPLIT=${EVAL_SPLIT:-test}
EVAL_CKPTS=${EVAL_CKPTS:-"25 50 75 100 125 150 200 250 300"}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=output/MSVD/logs/run_st_cross_joint_${STAMP}.log
log() { echo "[st-joint] $(date '+%F %T') $*" | tee -a "$LOG"; }

# 可选 LR 覆盖（未设置则用 finetune_qlora_MSVD_cross.sh 的默认 1e-5 / 1e-4）
LR_ARGS=()
[ -n "${LORA_LR:-}" ] && LR_ARGS+=("LORA_LR=${LORA_LR}")
[ -n "${PROJ_LR:-}" ] && LR_ARGS+=("PROJ_LR=${PROJ_LR}")

log "LR override: ${LR_ARGS[*]:-<default 1e-5 / 1e-4>}"

log "STEP 1/3: smoke (${SMOKE}, 2 steps)"
CUDA_VISIBLE_DEVICES=${GPUS} JOINT=1 RUN_NAME=${SMOKE} MAX_STEPS=2 \
  ST_FUSION_MODE=rfm ST_SCAN_MODE=cross ST_ALIGN_MODE=nte_center \
  env "${LR_ARGS[@]}" bash finetune_qlora_MSVD_cross.sh
rc=$?
log "smoke exit=${rc}"
[ "$rc" != "0" ] && { log "smoke failed — abort"; exit 1; }

log "STEP 2/3: full joint training (${RUN}, ${MAX_STEPS} steps, save/${SAVE_STEPS})"
CUDA_VISIBLE_DEVICES=${GPUS} JOINT=1 RUN_NAME=${RUN} MAX_STEPS=${MAX_STEPS} SAVE_STEPS=${SAVE_STEPS} \
  ST_FUSION_MODE=rfm ST_SCAN_MODE=cross ST_ALIGN_MODE=nte_center \
  env "${LR_ARGS[@]}" bash finetune_qlora_MSVD_cross.sh
rc=$?
log "full exit=${rc}"
[ "$rc" != "0" ] && { log "full run failed"; exit "$rc"; }

log "STEP 3/3: ${EVAL_SPLIT} sweep (加密早期点)"
EVAL_GPUS=${GPUS} EVAL_SPLIT=${EVAL_SPLIT} \
  EVAL_CKPTS="${EVAL_CKPTS}" \
  bash eval_all_checkpoints.sh "${RUN}" "${GPUS}"
log "${EVAL_SPLIT} sweep exit=$?  summary=output/MSVD/${RUN}_all_ckpt_summary_${EVAL_SPLIT}.tsv"
log "ALL DONE"
