import sys, json, argparse

# 用 VideoLLaMA2 官方的 pycocoevalcap 包（标准 coco-caption 实现）
sys.path.insert(0, '/data/lzz/VideoLLaMA2-main/evaluation/pycocoevalcap')
from cider.cider_scorer import CiderScorer

parser = argparse.ArgumentParser()
parser.add_argument('result_json', help='merged result json, e.g. output/MSVD/result/my_run_qlora-200_test.json')
parser.add_argument('--sigma', type=float, default=6.0, help='Gaussian penalty sigma; 6.0=Cider-D, 1e9=plain CIDEr')
args = parser.parse_args()

data = json.load(open(args.result_json, encoding='utf-8'))

scorer = CiderScorer(n=4, sigma=args.sigma)
for item in data:
    # item['captions'] = 参考字幕列表, item['prediction'] = 模型预测
    scorer += (item['prediction'] or '', item['captions'])

score, scores = scorer.compute_score()
print('CIDEr (VideoLLaMA2 pkg, sigma=%s): %.4f' % (args.sigma, score))
