import os
import json
import sys
sys.path.append('./')
from evaluation.pycocoevalcap.eval import COCOEvalCap

dataset = 'MSVD'
split = 'test'
folder = 'my_run_qlora_v7'
ckpt_id = 200
num_chunks = 1

merged = []
for i in range(num_chunks):
    p = f'output/{dataset}/result/{folder}-{ckpt_id}_{split}_chunk{i}.json'
    merged.extend(json.load(open(p, encoding='utf-8')))

result_path = f'output/{dataset}/result/{folder}-{ckpt_id}_{split}.json'
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(merged, f, indent=4, ensure_ascii=False)

eval_path = f'output/{dataset}/score/{folder}-{ckpt_id}_{split}.json'
cocoEval = COCOEvalCap(result_path)
cocoEval.evaluate()
for metric, score in cocoEval.eval.items():
    print('%s: %.3f' % (metric, score))

os.makedirs(os.path.dirname(eval_path), exist_ok=True)
with open(eval_path, 'w') as f:
    json.dump(cocoEval.eval, f, indent=4)
print(f"merged {len(merged)} items; scores saved to {eval_path}")
