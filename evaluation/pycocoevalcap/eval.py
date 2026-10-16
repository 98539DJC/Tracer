"""Standard pycocoevalcap metrics for a result file produced by gen_chunk.py.

Metrics reported: Bleu_1, Bleu_2, Bleu_3, Bleu_4, METEOR, ROUGE_L, CIDEr, SPICE.

The metric implementations are the official ones from the VideoLLaMA2 tree --
the same code the published tables use.  They are loaded through importlib under
the top-level name ``pycocoevalcap`` because this project already owns the name
``evaluation``, and the official modules use relative imports
(``from .bleu_scorer import BleuScorer``) that only resolve inside a real package.

Interface (unchanged, so merge_eval.py / generate.py keep working)::

    cocoEval = COCOEvalCap(result_path)
    cocoEval.evaluate()
    print(cocoEval.eval)        # {'Bleu_4': ..., 'METEOR': ..., ...}
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import importlib.util
import json
import os
import sys
import types

#: root of the VideoLLaMA2 checkout that provides the official implementation
OFFICIAL_ROOT = os.environ.get('OFFICIAL_EVAL_ROOT', '/data/lzz/VideoLLaMA2-main')
OFFICIAL_PKG_DIR = os.path.join(OFFICIAL_ROOT, 'evaluation', 'pycocoevalcap')

# bleu_scorer.py and ptbtokenizer.py import ipdb, a debugger that is not installed
if 'ipdb' not in sys.modules:
    _ipdb = types.ModuleType('ipdb')
    _ipdb.set_trace = lambda *a, **k: None
    sys.modules['ipdb'] = _ipdb


def _register_official_package():
    """Expose the official pycocoevalcap as the top-level package `pycocoevalcap`."""
    if 'pycocoevalcap' in sys.modules:
        return sys.modules['pycocoevalcap']
    if not os.path.isdir(OFFICIAL_PKG_DIR):
        raise ImportError('official pycocoevalcap not found at %s' % OFFICIAL_PKG_DIR)
    spec = importlib.util.spec_from_file_location(
        'pycocoevalcap',
        os.path.join(OFFICIAL_PKG_DIR, '__init__.py'),
        submodule_search_locations=[OFFICIAL_PKG_DIR],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules['pycocoevalcap'] = module
    spec.loader.exec_module(module)
    return module


_register_official_package()

Bleu = importlib.import_module('pycocoevalcap.bleu.bleu').Bleu
Meteor = importlib.import_module('pycocoevalcap.meteor.meteor').Meteor
Rouge = importlib.import_module('pycocoevalcap.rouge.rouge').Rouge
Cider = importlib.import_module('pycocoevalcap.cider.cider').Cider
Spice = importlib.import_module('pycocoevalcap.spice.spice').Spice
PTBTokenizer = importlib.import_module('pycocoevalcap.tokenizer.ptbtokenizer').PTBTokenizer


class COCOEvalCap(object):
    def __init__(self, path):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}

        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)

        imgids = []
        for i, item in enumerate(data):
            imgids.append(item.get('image_name', str(i)))
        self.ground_truth, self.prediction = {}, {}
        for imgId, item in zip(imgids, data):
            self.ground_truth[imgId] = [{'caption': c} for c in item['captions']]
            self.prediction[imgId] = [{'caption': item.get('prediction', '')}]
        self.params = {'image_id': imgids}

    def evaluate(self):
        imgIds = self.params['image_id']
        gts, res = {}, {}
        for imgId in imgIds:
            gts[imgId] = self.ground_truth[imgId]
            res[imgId] = self.prediction[imgId]

        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        print('setting up scorers...')
        scorers = [
            (Bleu(4), ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']),
            (Meteor(), 'METEOR'),
            (Rouge(), 'ROUGE_L'),
            (Cider(), 'CIDEr'),
            (Spice(), 'SPICE'),
        ]

        for scorer, method in scorers:
            print('computing %s score...' % scorer.method())
            score, scores = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    if scs is not None:
                        self.setImgToEvalImgs(scs, list(gts.keys()), m)
                    print('%s: %0.4f' % (m, sc))
            else:
                self.setEval(score, method)
                if scores is not None:
                    self.setImgToEvalImgs(scores, list(gts.keys()), method)
                print('%s: %0.4f' % (method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if imgId not in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]['image_id'] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [e for _, e in self.imgToEval.items()]
