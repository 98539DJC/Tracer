#!/bin/bash
# ============================================================================
#  MSVD evaluation -- one entry point:  checkpoint -> captions -> metrics
#
#  Usage:
#     bash eval_msvd.sh <RUN_NAME> <CKPT> [N_CHUNKS]
#  e.g.
#     bash eval_msvd.sh my_run_qlora_v16 400          # 推理 + 评估（默认 2 卡）
#     INFER=0 bash eval_msvd.sh my_run_qlora_v16 400  # 只评估已有预测（不加载模型）
#
#  Environment:
#     EVAL_GPUS     GPU list for the sharded inference, default "0,1"
#     N_CHUNKS      number of shards (default = number of GPUs listed)
#     INFER         1 = run inference first (default), 0 = score existing results
#     EVAL_SPLIT    default "test"
#     EVAL_OUT_DIR  default "output/MSVD"
#     EVAL_MODEL_DIR / EVAL_DATASET / OFFICIAL_EVAL_ROOT : passed through to
#                   gen_chunk.py / merge_eval.py unchanged
#
#  Metrics are produced by the official pycocoevalcap port in
#  evaluation/pycocoevalcap/eval.py: Bleu_1..4, METEOR, ROUGE_L, CIDEr, SPICE.
#  This script only drives it -- there is no second place where a score is
#  computed, and nothing here adjusts a number.
# ============================================================================
set -u
export LC_ALL=C
cd "$(dirname "$0")" || exit 1
PY=${PY:-/home/megaium/miniconda3/envs/tracer/bin/python}
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

RUN_NAME=${1:-}
CKPT=${2:-}
if [ -z "$RUN_NAME" ] || [ -z "$CKPT" ]; then
  echo "usage: bash eval_msvd.sh <RUN_NAME> <CKPT> [N_CHUNKS]" >&2
  exit 2
fi
EVAL_GPUS=${EVAL_GPUS:-0,1}
N_CHUNKS=${3:-$(echo "$EVAL_GPUS" | tr ',' '\n' | wc -l)}
INFER=${INFER:-1}
EVAL_SPLIT=${EVAL_SPLIT:-test}
EVAL_OUT_DIR=${EVAL_OUT_DIR:-output/MSVD}
export EVAL_SPLIT EVAL_OUT_DIR EVAL_NUM_CHUNKS=$N_CHUNKS

LOG_DIR="$EVAL_OUT_DIR/logs"
mkdir -p "$LOG_DIR" "$EVAL_OUT_DIR/result" "$EVAL_OUT_DIR/score"
MAIN="$LOG_DIR/eval_${RUN_NAME}-${CKPT}_$(date +%Y%m%d_%H%M%S).log"

echo "==============================================================================" | tee -a "$MAIN"
echo " MSVD evaluation" | tee -a "$MAIN"
echo "   run name : $RUN_NAME" | tee -a "$MAIN"
echo "   ckpt     : $CKPT" | tee -a "$MAIN"
echo "   gpus     : $EVAL_GPUS   shards: $N_CHUNKS   split: $EVAL_SPLIT   infer: $INFER" | tee -a "$MAIN"
echo "   output   : $EVAL_OUT_DIR" | tee -a "$MAIN"
echo "==============================================================================" | tee -a "$MAIN"

CKPT_DIR="$EVAL_OUT_DIR/checkpoints/${RUN_NAME}/checkpoint-${CKPT}"
if [ "$INFER" != "0" ]; then
  for f in non_lora_trainables.bin adapter_model.safetensors; do
    if [ ! -f "$CKPT_DIR/$f" ]; then
      echo "ERROR: $CKPT_DIR/$f not found" | tee -a "$MAIN"; exit 1
    fi
  done
  echo "checkpoint found: $CKPT_DIR" | tee -a "$MAIN"

  # ---- sharded inference, one shard per GPU -------------------------------
  echo "--- inference ---" | tee -a "$MAIN"
  i=0
  for gpu in $(echo "$EVAL_GPUS" | tr ',' ' '); do
    CUDA_VISIBLE_DEVICES=$gpu EVAL_FOLDER=$RUN_NAME EVAL_CKPT=$CKPT \
        $PY -u gen_chunk.py $i $N_CHUNKS > "$LOG_DIR/gen_${RUN_NAME}-${CKPT}_chunk${i}.log" 2>&1 &
    i=$(( i + 1 ))
  done
  FAIL=0
  wait || FAIL=1
  for k in $(seq 0 $(( N_CHUNKS - 1 ))); do
    n=$(grep -ac '^\[chunk' "$LOG_DIR/gen_${RUN_NAME}-${CKPT}_chunk${k}.log" 2>/dev/null)
    printf '    chunk %-2s captions: %s\n' "$k" "$n" | tee -a "$MAIN"
    [ "$n" -gt 0 ] || FAIL=1
  done
  if [ "$FAIL" != "0" ]; then
    echo "ERROR: inference failed, see $LOG_DIR/gen_${RUN_NAME}-${CKPT}_chunk*.log" | tee -a "$MAIN"
    exit 1
  fi
else
  echo "INFER=0 -> skipping inference, scoring existing results" | tee -a "$MAIN"
fi

# ---- merge shards + compute the official metrics --------------------------
echo "--- scoring (official pycocoevalcap) ---" | tee -a "$MAIN"
EVAL_FOLDER=$RUN_NAME EVAL_CKPT=$CKPT $PY merge_eval.py 2>&1 | tee -a "$MAIN"
RC=${PIPESTATUS[0]}
[ "$RC" != "0" ] && { echo "ERROR: merge_eval.py exited $RC" | tee -a "$MAIN"; exit 1; }

SCORE="$EVAL_OUT_DIR/score/${RUN_NAME}-${CKPT}_${EVAL_SPLIT}.json"
if [ ! -f "$SCORE" ]; then
  echo "ERROR: expected score file $SCORE was not written" | tee -a "$MAIN"; exit 1
fi

# ---- summary --------------------------------------------------------------
echo | tee -a "$MAIN"
echo "==============================================================================" | tee -a "$MAIN"
echo " MSVD $EVAL_SPLIT results : ${RUN_NAME} @ ${CKPT}" | tee -a "$MAIN"
echo "==============================================================================" | tee -a "$MAIN"
$PY - "$SCORE" <<'PYEOF' 2>&1 | tee -a "$MAIN"
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
order = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
w = max(len(k) for k in order) + 2
for k in order:
    if k in d:
        print('  %-*s %.4f   (x100: %.2f)' % (w, k, d[k], d[k] * 100))
extra = [k for k in d if k not in order]
for k in extra:
    print('  %-*s %s' % (w, k, d[k]))
print()
print('  reference (paper, MSVD test):  Bleu_4 0.8250  METEOR 0.5590  ROUGE_L 0.9070  CIDEr 1.8770')
PYEOF
echo | tee -a "$MAIN"
echo " score file : $SCORE" | tee -a "$MAIN"
echo " full log   : $MAIN" | tee -a "$MAIN"
echo "==============================================================================" | tee -a "$MAIN"
