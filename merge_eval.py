import os
import json
import sys
sys.path.append('./')
from evaluation.pycocoevalcap.eval import COCOEvalCap

dataset = os.environ.get('EVAL_DATASET', 'MSVD')
split = os.environ.get('EVAL_SPLIT', 'test')
folder = os.environ.get('EVAL_FOLDER', 'my_run_qlora_v7')
ckpt_id = int(os.environ.get('EVAL_CKPT', '200'))
num_chunks = int(os.environ.get('EVAL_NUM_CHUNKS', '1'))
RESULT_ROOT = os.environ.get('EVAL_OUT_DIR', f'output/{dataset}')

merged = []
for i in range(num_chunks):
    p = f'{RESULT_ROOT}/result/{folder}-{ckpt_id}_{split}_chunk{i}.json'
    merged.extend(json.load(open(p, encoding='utf-8')))

result_path = f'{RESULT_ROOT}/result/{folder}-{ckpt_id}_{split}.json'
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(merged, f, indent=4, ensure_ascii=False)

eval_path = f'{RESULT_ROOT}/score/{folder}-{ckpt_id}_{split}.json'
cocoEval = COCOEvalCap(result_path)
cocoEval.evaluate()

for metric, score in cocoEval.eval.items():
    print('%s: %.4f' % (metric, score))

os.makedirs(os.path.dirname(eval_path), exist_ok=True)
with open(eval_path, 'w', encoding='utf-8') as f:
    json.dump(cocoEval.eval, f, indent=4, ensure_ascii=False)
print('score written to %s' % eval_path)

with open(eval_path, 'w') as f:
    json.dump(cocoEval.eval, f, indent=4)
print(f"merged {len(merged)} items; scores saved to {eval_path}")
