#!/bin/bash
# =============================================================================
# 方案 v2 首轮驱动：A1–A5 依次短跑（默认 200 步）+ valid 全 ckpt 扫描
#
#   ARMS="a1 a2" bash run_st_cross_round1.sh          # 只跑指定组
#   FULL=1 bash run_st_cross_round1.sh                # 完整 812 步
#   SMOKE_FIRST=0 bash run_st_cross_round1.sh         # 跳过每组 2 步冒烟
#
#   A1 conv/off/nte_center   A2 rfm/off/nte_center   A3 rfm/self/nte_center
#   A4 rfm/cross/nte_center  A5 rfm/cross/slot
#
#   每组训完自动 EVAL_SPLIT=valid 扫 ckpt（按方案：验证集选点，锁定后才 test）。
# =============================================================================
set -u
cd "$(dirname "$0")" || exit 1

ARMS=${ARMS:-"a1 a2 a3 a4 a5"}
SMOKE_FIRST=${SMOKE_FIRST:-1}
FULL=${FULL:-0}
MAX_STEPS=${MAX_STEPS:-200}
GPUS=${GPUS:-2,3}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=output/MSVD/logs/run_st_cross_round1_${STAMP}.log
log() { echo "[st-driver] $(date '+%F %T') $*" | tee -a "$LOG"; }

[ "$FULL" = "1" ] && MAX_STEPS=812

arm_cfg() {
  case "$1" in
    a1)  echo "conv off  nte_center" ;;
    a1s) echo "conv off  slot" ;;        # 用户补充：A1 的坐标对照（只切 align）
    a2)  echo "rfm  off  nte_center" ;;
    a3)  echo "rfm  self nte_center" ;;
    a4)  echo "rfm  cross nte_center" ;;
    a5)  echo "rfm  cross slot" ;;
    *)   return 1 ;;
  esac
}

log "arms=${ARMS}  full=${FULL}  max_steps=${MAX_STEPS}  gpus=${GPUS}"

for arm in $ARMS; do
  cfg=$(arm_cfg "$arm") || { log "unknown arm: $arm"; exit 1; }
  read -r fm sm am <<< "$cfg"
  run=st_${arm}_s42

  if [ "$SMOKE_FIRST" = "1" ]; then
    log "smoke ${arm} (2 steps) ..."
    CUDA_VISIBLE_DEVICES=${GPUS} RUN_NAME=${run}_smoke MAX_STEPS=2 \
      ST_FUSION_MODE=$fm ST_SCAN_MODE=$sm ST_ALIGN_MODE=$am \
      bash finetune_qlora_MSVD_cross.sh
    rc=$?
    log "smoke ${arm} exit=${rc}"
    [ "$rc" != "0" ] && { log "smoke failed — stop"; exit 1; }
  fi

  log "train ${arm}: run=${run} fusion=${fm} scan=${sm} align=${am} steps=${MAX_STEPS}"
  CUDA_VISIBLE_DEVICES=${GPUS} RUN_NAME=${run} MAX_STEPS=${MAX_STEPS} \
    ST_FUSION_MODE=$fm ST_SCAN_MODE=$sm ST_ALIGN_MODE=$am \
    bash finetune_qlora_MSVD_cross.sh
  rc=$?
  log "train ${arm} exit=${rc}"
  [ "$rc" != "0" ] && log "train ${arm} FAILED — continuing to val sweep anyway"

  log "val sweep ${arm} ..."
  if [ "${MAX_STEPS}" = "812" ]; then
    _cks="50 100 150 200 300 400 500 600 700 800 812"
  else
    _cks="50 100 150 200"
  fi
  EVAL_GPUS=${GPUS} EVAL_SPLIT=valid EVAL_CKPTS="${_cks}" \
    bash eval_all_checkpoints.sh "${run}" "${GPUS}"
  log "val sweep ${arm} exit=$?  summary=output/MSVD/${run}_all_ckpt_summary_valid.tsv"
done

log "ALL DONE"
