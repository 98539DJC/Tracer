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

# ---- official METEOR (meteor-1.5.jar + java), same implementation as the paper ----
try:
    # load the official implementation straight from its file: this project already
    # owns the name `evaluation`, so a sys.path entry would not shadow it.
    import importlib.util as _ilu
    _mp = '/data/lzz/VideoLLaMA2-main/evaluation/pycocoevalcap/meteor/meteor.py'
    _spec = _ilu.spec_from_file_location('_official_meteor', _mp)
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _gts = {i: list(it.get('captions', [])) for i, it in enumerate(merged)}
    _res = {i: [it.get('prediction', '')] for i, it in enumerate(merged)}
    _m, _ = _mod.Meteor().compute_score(_gts, _res)
    cocoEval.eval['METEOR'] = float(_m)
except Exception as _e:
    print('[meteor] skipped: %s: %s' % (type(_e).__name__, _e))

# ---- the same metrics from the *official* pycocoevalcap, for comparability with
# ---- published tables (they are on the standard 0-1 scale)
try:
    import subprocess as _sp
    _r = _sp.run([sys.executable, 'evaluation/official_metrics.py', result_path],
                 capture_output=True, text=True, timeout=1800)
    if _r.returncode == 0 and _r.stdout.strip():
        for _k, _v in json.loads(_r.stdout.strip().splitlines()[-1]).items():
            cocoEval.eval['official_' + _k] = float(_v)
    else:
        print('[official] skipped: rc=%s %s' % (_r.returncode, (_r.stderr or '').strip()[-160:]))
except Exception as _e:
    print('[official] skipped: %s: %s' % (type(_e).__name__, _e))
finally:
    if 'sys' not in dir():
        import sys

for metric, score in cocoEval.eval.items():
    print('%s: %.4f' % (metric, score))

os.makedirs(os.path.dirname(eval_path), exist_ok=True)
with open(eval_path, 'w', encoding='utf-8') as f:
    json.dump(cocoEval.eval, f, indent=4, ensure_ascii=False)
print('score written to %s' % eval_path)

with open(eval_path, 'w') as f:
    json.dump(cocoEval.eval, f, indent=4)
print(f"merged {len(merged)} items; scores saved to {eval_path}")
