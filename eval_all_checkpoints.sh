#!/bin/bash
# ============================================================================
# 评测某个 run 的所有 checkpoint（50..800 每 50 步 + 最终模型 813），
# 输出训练曲线 tsv（CIDEr/全指标 vs step），用于判断收敛/过拟合/最佳步数。
#
#   usage: bash eval_all_checkpoints.sh <RUN_NAME> [EVAL_GPUS=2,3]
#
#   - 串行评测（避免与训练争抢 GPU），逐个 checkpoint 推理 + 官方评分
#   - 汇总写入 output/MSVD/<RUN>_all_ckpt_summary.tsv
#   - 完整日志 output/MSVD/logs/eval_all_<RUN>_<ts>.log
# ============================================================================
set -u
cd "$(dirname "$0")" || exit 1

RUN=${1:?usage: eval_all_checkpoints.sh <RUN_NAME> [EVAL_GPUS]}
EVAL_GPUS=${2:-2,3}
export EVAL_GPUS
EVAL_SPLIT=${EVAL_SPLIT:-test}
export EVAL_SPLIT

CKPT_DIR=output/MSVD/checkpoints/${RUN}
SUMMARY=output/MSVD/${RUN}_all_ckpt_summary_${EVAL_SPLIT}.tsv
LOG=output/MSVD/logs/eval_all_${RUN}_${EVAL_SPLIT}_$(date +%Y%m%d_%H%M%S).log

if [ -n "${EVAL_CKPTS:-}" ]; then
  # 显式指定要评测的 checkpoint 列表（空格分隔），用于中途终止训练的场景
  ALL_CKPTS="${EVAL_CKPTS}"
else
  # 中间 checkpoint（checkpoint-* 目录，升序）
  CKPTS=$(ls -d ${CKPT_DIR}/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n)
  # 最终模型 step：从 run 根目录 trainer_state.json 读 global_step（B0=812），回退 813
  FINAL=$(python3 -c "import json;print(json.load(open('${CKPT_DIR}/trainer_state.json'))['global_step'])" 2>/dev/null || echo 813)
  ALL_CKPTS="${CKPTS} ${FINAL}"
fi

printf 'ckpt\tBleu_1\tBleu_2\tBleu_3\tBleu_4\tMETEOR\tROUGE_L\tCIDEr\tSPICE\n' > "$SUMMARY"

{
  echo "======================================================================"
  echo " all-checkpoint eval : ${RUN}   start $(date '+%F %T')"
  echo " checkpoints          : ${ALL_CKPTS}"
  echo " summary              : ${SUMMARY}"
  echo "======================================================================"

  for ckpt in ${ALL_CKPTS}; do
    echo "----------------------------------------------------------------------"
    echo " eval ${RUN} @ ${ckpt}  start $(date '+%F %T')"
    echo "----------------------------------------------------------------------"
    EVAL_GPUS=${EVAL_GPUS} bash eval_msvd.sh "${RUN}" "${ckpt}"
    RC=$?
    echo " eval ${RUN} @ ${ckpt}  exit=${RC}  $(date '+%F %T')"

    SCORE=output/MSVD/score/${RUN}-${ckpt}_${EVAL_SPLIT}.json
    if [ -f "$SCORE" ]; then
      python3 - "$SCORE" "$ckpt" "$SUMMARY" <<'PY'
import json, sys
score, ckpt, summary = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(score, encoding='utf-8'))
keys = ['Bleu_1','Bleu_2','Bleu_3','Bleu_4','METEOR','ROUGE_L','CIDEr','SPICE']
row = [ckpt] + [f"{d.get(k,'NA')}" for k in keys]
with open(summary, 'a') as f:
    f.write('\t'.join(row) + '\n')
print('ROW:', '\t'.join(row))
PY
    else
      echo "[WARN] score file missing: $SCORE"
    fi
  done

  echo "======================================================================"
  echo " all-checkpoint eval DONE  $(date '+%F %T')"
  echo "======================================================================"
  echo "--- 训练曲线（CIDEr ×100 vs step）---"
  awk -F'\t' 'NR==1{print "ckpt\tCIDEr(x100)"} NR>1{printf "%s\t%.2f\n", $1, $8*100}' "$SUMMARY"
} 2>&1 | tee -a "$LOG"
