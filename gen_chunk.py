import os
import sys
import json
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
split = 'test'
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
config.mm_projector_type = 'route_mamba'

# ---- cached SigLIP features: never build or load the vision tower ----
FEATURE_DIR = os.environ.get('MSVD_FEATURE_DIR', 'datasets/MSVD_sft/siglip_feat16')
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
