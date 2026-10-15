import json
import math
import re
from collections import defaultdict

NGRAM = 4
SIGMA = 6.0


def tokenize(s):
    s = (s or '').lower().strip()
    s = re.sub(r"[^a-z0-9']+", ' ', s)
    return [t for t in s.split() if t]


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _counts(tokens, n):
    c = defaultdict(int)
    for k in range(1, n + 1):
        for g in _ngrams(tokens, k):
            c[g] += 1
    return c


def _lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _geom(precs):
    if any(p <= 0 for p in precs):
        return 0.0
    return math.exp(sum(math.log(p) for p in precs) / len(precs))


def compute_bleu(all_refs, all_hyps, n=NGRAM):
    clipped = [0] * n
    total = [0] * n
    hyp_len = 0
    ref_len = 0
    for refs, hyp in zip(all_refs, all_hyps):
        hyp_len += len(hyp)
        if refs:
            ref_len += min((abs(len(r) - len(hyp)), len(r)) for r in refs)[1]
        hc = _counts(hyp, n)
        rc = defaultdict(int)
        for r in refs:
            for g, c in _counts(r, n).items():
                rc[g] = max(rc[g], c)
        for g, c in hc.items():
            total[len(g) - 1] += c
            clipped[len(g) - 1] += min(c, rc[g])
    bp = 1.0
    if hyp_len < ref_len:
        bp = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0
    scores = {}
    precs = []
    for k in range(n):
        p = clipped[k] / total[k] if total[k] > 0 else 0.0
        precs.append(p)
        scores['Bleu_%d' % (k + 1)] = bp * _geom(precs)
    return scores


def compute_rouge(all_refs, all_hyps):
    vals = []
    for refs, hyp in zip(all_refs, all_hyps):
        best = 0.0
        for r in refs:
            l = _lcs(hyp, r)
            if len(hyp) and len(r):
                p = l / len(hyp)
                rec = l / len(r)
                f = 2 * p * rec / (p + rec) if (p + rec) else 0.0
                best = max(best, f)
        vals.append(best)
    return sum(vals) / len(vals) if vals else 0.0


def compute_cider(all_refs, all_hyps, n=NGRAM, sigma=SIGMA):
    N = len(all_hyps)
    df = defaultdict(int)
    for refs in all_refs:
        seen = set()
        for r in refs:
            for g in _counts(r, n):
                seen.add(g)
        for g in seen:
            df[g] += 1
    idf = {g: math.log(N / c) for g, c in df.items()}

    cider_n = [0.0] * n
    ciderd_n = [0.0] * n
    for refs, hyp in zip(all_refs, all_hyps):
        for k in range(1, n + 1):
            hc = defaultdict(int)
            for g in _ngrams(hyp, k):
                hc[g] += 1
            htotal = sum(hc.values())
            hvec = {}
            for g, c in hc.items():
                if g in idf:
                    hvec[g] = (c / htotal) * idf[g]
            hnorm = math.sqrt(sum(v * v for v in hvec.values()))
            if hnorm == 0:
                continue
            sims = []
            simsd = []
            for r in refs:
                rc = defaultdict(int)
                for g in _ngrams(r, k):
                    rc[g] += 1
                rtotal = sum(rc.values())
                rvec = {}
                for g, c in rc.items():
                    if g in idf:
                        rvec[g] = (c / rtotal) * idf[g]
                rnorm = math.sqrt(sum(v * v for v in rvec.values()))
                if rnorm == 0:
                    sims.append(0.0)
                    simsd.append(0.0)
                    continue
                dot = sum(hvec[g] * rvec.get(g, 0.0) for g in hvec)
                cos = dot / (hnorm * rnorm)
                sims.append(cos)
                diff = len(hyp) - len(r)
                penalty = math.exp(-(diff * diff) / (2.0 * sigma * sigma))
                simsd.append(cos * penalty)
            if sims:
                cider_n[k - 1] += sum(sims) / len(sims)
                ciderd_n[k - 1] += sum(simsd) / len(simsd)

    for k in range(n):
        cider_n[k] = cider_n[k] / N * 10.0
        ciderd_n[k] = ciderd_n[k] / N * 10.0
    return sum(cider_n) / n, sum(ciderd_n) / n


class COCOEvalCap:
    def __init__(self, result_file_path):
        self.result_file_path = result_file_path
        self.eval = {}

    def evaluate(self):
        with open(self.result_file_path, encoding='utf-8') as f:
            data = json.load(f)
        all_hyps = []
        all_refs = []
        for item in data:
            pred = item.get('prediction', '') or ''
            caps = item.get('captions', [])
            if isinstance(caps, str):
                caps = [caps]
            all_hyps.append(tokenize(pred))
            all_refs.append([tokenize(c) for c in caps])
        self.eval.update(compute_bleu(all_refs, all_hyps))
        self.eval['ROUGE_L'] = compute_rouge(all_refs, all_hyps)
        cider, ciderd = compute_cider(all_refs, all_hyps)
        self.eval['CIDEr'] = cider
        self.eval['CIDEr-D'] = ciderd
