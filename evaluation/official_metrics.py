#!/usr/bin/env python
"""Standard pycocoevalcap metrics for a result file produced by gen_chunk.py.

Usage:  python evaluation/official_metrics.py <result.json>
Prints one JSON object on stdout: Bleu_4 / METEOR / ROUGE_L / CIDEr.

The official implementation lives in the VideoLLaMA2 tree this project already
depends on for its model weights.  It is loaded in a *fresh* process because the
name `evaluation` is taken by this project's own (custom) scorer, which would
otherwise shadow it.
"""
import json
import os
import sys
import types

REF_ROOT = os.environ.get('OFFICIAL_EVAL_ROOT', '/data/lzz/VideoLLaMA2-main')

# bleu_scorer.py imports ipdb (a debugger) which is not installed here
if 'ipdb' not in sys.modules:
    _ipdb = types.ModuleType('ipdb')
    _ipdb.set_trace = lambda *a, **k: None
    sys.modules['ipdb'] = _ipdb

sys.path.insert(0, REF_ROOT)
from evaluation.pycocoevalcap.bleu.bleu import Bleu          # noqa: E402
from evaluation.pycocoevalcap.rouge.rouge import Rouge       # noqa: E402
from evaluation.pycocoevalcap.cider.cider import Cider       # noqa: E402
from evaluation.pycocoevalcap.meteor.meteor import Meteor    # noqa: E402


def compute(result_path):
    data = json.load(open(result_path, encoding='utf-8'))
    gts = {i: list(it.get('captions', [])) for i, it in enumerate(data)}
    res = {i: [it.get('prediction', '')] for i, it in enumerate(data)}

    # Bleu.compute_score prints aggregate counts on stdout; keep our own stdout clean
    saved = os.dup(1)
    try:
        os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
        bleu = Bleu(4).compute_score(gts, res)[0]
    finally:
        os.dup2(saved, 1)
        os.close(saved)
    bleu_4 = float(bleu['Bleu_4']) if isinstance(bleu, dict) else float(bleu[3])

    return {
        'Bleu_4': bleu_4,
        'METEOR': float(Meteor().compute_score(gts, res)[0]),
        'ROUGE_L': float(Rouge().compute_score(gts, res)[0]),
        'CIDEr': float(Cider().compute_score(gts, res)[0]),
    }


if __name__ == '__main__':
    print(json.dumps(compute(sys.argv[1])))
