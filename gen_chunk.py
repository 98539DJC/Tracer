import os
import sys
import json
import numpy as np
import torch
sys.path.append('./')
from functools import partial
from peft import PeftModel
from videollama2 import mm_infer
from videollama2.utils import disable_torch_init
from videollama2.mm_utils import process_image, process_video
from transformers import AutoTokenizer, AutoConfig
from videollama2.model.videollama2_qwen2 import Videollama2Qwen2ForCausalLM

disable_torch_init()

dataset = 'MSVD'
split = os.environ.get('EVAL_SPLIT', 'test')
folder = os.environ.get('EVAL_FOLDER', 'my_run_qlora_v7')
ckpt_id = int(os.environ.get('EVAL_CKPT', '200'))
chunk_idx = int(sys.argv[1])
num_chunks = int(sys.argv[2])

video_dir = 'datasets/MSVD_sft/videos/'
video_ext = '.avi'
valid_jsonl_path = f'datasets/MSVD_sft/make_sft/{split}.jsonl'
# EVAL_MODEL_DIR: use an explicit checkpoint dir instead of the in-project layout.
# EVAL_OUT_DIR  : write result/ and score/ somewhere other than output/<dataset>/.
RESULT_ROOT = os.environ.get('EVAL_OUT_DIR', f'output/{dataset}')
lora_model_path = (os.environ.get('EVAL_MODEL_DIR')
                   or f'output/{dataset}/checkpoints/{folder}/checkpoint-{ckpt_id}')
output_path = f'{RESULT_ROOT}/result/{folder}-{ckpt_id}_{split}_chunk{chunk_idx}.json'

modal = 'video'
GEN_KWARGS = {}
for _k, _c in (('num_beams', int), ('length_penalty', float),
                ('repetition_penalty', float), ('no_repeat_ngram_size', int)):
    _v = os.environ.get('EVAL_' + _k.upper())
    if _v not in (None, ''):
        GEN_KWARGS[_k] = _c(_v)

NUM_CAND = int(os.environ.get('EVAL_NUM_CANDIDATES', '1'))
TEMP = float(os.environ.get('EVAL_TEMPERATURE', '0.0'))
TOPP = float(os.environ.get('EVAL_TOP_P', '0.9'))

instruct = os.environ.get('EVAL_INSTRUCT',
                              'Use only one short sentence to describe the video.')
model_path = 'DAMO-NLP-SG/VideoLLaMA2.1-7B-16F'

config = AutoConfig.from_pretrained(model_path)
# Vim 单块空间分支：从 checkpoint 的 config.json 读 projector 类型，保证推理时构建
# 与训练一致的 connector（route_mamba vs vim_spatial）。否则 Vim-S checkpoint 会被
# 错误地当 route_mamba 加载（空间参数 s1_s/sp_mamba 被 strict=False 忽略）→ 评测作废。
_ckpt_cfg_path = os.path.join(lora_model_path, 'config.json')
if os.path.exists(_ckpt_cfg_path):
    _ckpt_cfg = json.load(open(_ckpt_cfg_path, encoding='utf-8'))
    config.mm_projector_type = _ckpt_cfg.get('mm_projector_type', 'route_mamba')
    # 方案 v2：tracer_vim_cross 的 fusion/scan/align 开关必须与训练一致，
    # 否则默认 st_fusion_mode='off' 会静默退化成 B0。
    for _k in ('st_fusion_mode', 'st_scan_mode', 'st_align_mode',
               'st_fusion_scale', 'st_rank_weight', 'st_frame_chunk'):
        if _k in _ckpt_cfg:
            setattr(config, _k, _ckpt_cfg[_k])
else:
    config.mm_projector_type = 'route_mamba'

# ---- cached SigLIP features: never build or load the vision tower ----
# 主口径 feat32（+ uniform 16/32）：旧 B0 在此口径下 final=187.79≈论文 187.70、peak 195；
# 而 GitHub 仓库默认的 feat16 实测整体低 ~7 分（旧 B0 final 180.80）。故默认 feat32。
# 如需完全照仓库 feat16 原样评测：EVAL_FEATURE_DIR=datasets/MSVD_sft/siglip_feat16。
FEATURE_DIR = os.environ.get('EVAL_FEATURE_DIR', 'datasets/MSVD_sft/siglip_feat32')
USE_FEATURE_CACHE = os.environ.get('USE_FEATURE_CACHE', '1') == '1'
if USE_FEATURE_CACHE:
    config.mm_vision_tower = 'precomputed'
    config.use_precomputed_features = True
model = Videollama2Qwen2ForCausalLM.from_pretrained(
    model_path,
    config=config, torch_dtype=torch.bfloat16
)
model = model.to('cuda')

non_lora_trainables = torch.load(
    os.path.join(lora_model_path, 'non_lora_trainables.bin'),
    map_location='cpu'
)
non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
if any(k.startswith('model.model.') for k in non_lora_trainables):
    non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
model.load_state_dict(non_lora_trainables, strict=False)

print('Loading LoRA weights...')
model = PeftModel.from_pretrained(model, lora_model_path)
print('Merging LoRA weights...')
model = model.merge_and_unload()
print('Model loaded.')

if USE_FEATURE_CACHE:
    from videollama2.mm_utils import load_video_features
    print(f'[FEAT] reading cached SigLIP features from {FEATURE_DIR} '
          f'(SigLIP is not loaded)', flush=True)

    def load_input(video_path):
        vid = os.path.basename(video_path).rsplit('.', 1)[0]
        feat_path = os.path.join(FEATURE_DIR, vid + '.npy')
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f'feature cache miss: {feat_path}')
        # 与 GitHub 权威版（62ad82c）一致：默认 feat16，直接读全部 16 帧、不采样。
        # 仅当显式把 EVAL_FEATURE_DIR 指到 feat32 cache 时，才按训练 uniform 规则挑 16/32
        # （供可选的 feat32 对照实验；主口径是 feat16）。
        if 'feat32' in FEATURE_DIR:
            _idx_env = os.environ.get('EVAL_FRAME_INDICES')
            if _idx_env:
                idx = np.array([int(x) for x in _idx_env.split(',')], dtype=np.int64)
            else:
                idx = np.linspace(0, 31, 16).round().astype(np.int64)
            return load_video_features(feat_path, idx)
        return load_video_features(feat_path)
else:
    vision_tower = model.get_vision_tower()
    _image_processor = vision_tower.image_processor
    processor = {
        'image': partial(process_image, processor=_image_processor, aspect_ratio=None),
        'video': partial(process_video, processor=_image_processor, aspect_ratio='pad', num_frames=16),
    }

    def load_input(video_path):
        return processor[modal](video_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
if tokenizer.pad_token is None and tokenizer.unk_token is not None:
    tokenizer.pad_token = tokenizer.unk_token

lines = open(valid_jsonl_path, encoding='utf8').readlines()
chunk_lines = lines[chunk_idx::num_chunks]

output_list = []
for line in chunk_lines:
    data = json.loads(line)
    video_id = data["video_id"]
    video_path = video_dir + video_id + video_ext
    try:
        if NUM_CAND > 1:
            cands = []
            for _ci in range(NUM_CAND):
                try:
                    _c = mm_infer(
                        load_input(video_path), instruct,
                        model=model, tokenizer=tokenizer, do_sample=True,
                        temperature=TEMP, top_p=TOPP, modal=modal,
                        **GEN_KWARGS
                    )
                    cands.append(_c.lower().rstrip('.'))
                except Exception as _e:
                    print(f"Error {video_id} cand{_ci}: {_e}")
            output = cands[0] if cands else ""
        else:
            output = mm_infer(
                load_input(video_path), instruct,
                model=model, tokenizer=tokenizer, do_sample=False, modal=modal,
                **GEN_KWARGS
            )
            output = output.lower().rstrip('.')
            cands = [output]
    except Exception as e:
        print(f"Error {video_id}: {e}")
        output = ""
    output_dict = {
        "split": split,
        "image_name": video_path,
        "captions": data["golden_captions"],
        "prediction": output,
        "candidates": cands,
    }
    print(f"[chunk{chunk_idx}] {video_id}: {output}")
    output_list.append(output_dict)

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output_list, f, indent=4, ensure_ascii=False)
print(f"chunk{chunk_idx} done, {len(output_list)} items -> {output_path}")
