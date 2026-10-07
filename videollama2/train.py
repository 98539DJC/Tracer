# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import re
import os
import gc
import copy
import json
import random
import pathlib
import traceback
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

# torch-related packages
# NOTE: torch must be imported before transformers. Otherwise, `Segmentation fault (core dumped)` will occur.
import torch
import numpy as np
from torch.utils.data import Dataset

import transformers
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

import sys
sys.path.append('./')
from videollama2.model import *
from accelerate.utils.deepspeed import DeepSpeedEngineWrapper
def _ds_backward(self, loss, **kw):
    self.engine.backward(loss, **kw)
    flag = self.engine.inside_no_sync_ctxt
    self.engine.inside_no_sync_ctxt = False
    try:
        self.engine.step()
    finally:
        self.engine.inside_no_sync_ctxt = flag
DeepSpeedEngineWrapper.backward = _ds_backward
from videollama2.constants import NUM_FRAMES, IGNORE_INDEX, MODAL_INDEX_MAP
from videollama2.mm_utils import tokenizer_multimodal_token, process_video, process_image
from videollama2.videollama2_trainer import (VideoLLaMA2Trainer,
    get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, 
    find_all_linear_names, safe_save_model_for_hf_trainer
)

from videollama2.mm_utils import expand2square
from PIL import Image
import os

# NOTE: fast tokenizer warning issue: https://github.com/huggingface/transformers/issues/5486   
os.environ["TOKENIZERS_PARALLELISM"] = "true"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    :param seed: An integer value to be used as the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class ModelArguments:
    # LLM Arguments
    model_type: Optional[str] = field(default="videollama2", metadata={"help": "Model type selected in the list: " + ", ".join(VLLMs.keys())})
    model_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.5")
    version: Optional[str] = field(default="v1", metadata={"help": "Version of the conversation template."})
    freeze_backbone: bool = field(default=False, metadata={"help": "Whether to freeze the LLM backbone."})
    # Connector Arguments
    mm_projector_type: Optional[str] = field(default='linear')
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    # Vision tower Arguments
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    # Path Arguments
    data_path: List[str] = field(default=None, metadata={"help": "Path to the training data."})
    # image_folder: Optional[str] = field(default=None)
    # video_folder: Optional[str] = field(default=None)
    data_folder: Optional[str] = field(default=None)
    # Loading Arguments
    is_multimodal: bool = False
    lazy_preprocess: bool = False
    num_frames: Optional[int] = field(default=None)
    # Preprocess Arguments
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    mm_projector_lr: Optional[float] = None
    freeze_mm_mlp_adapter: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    # Training Data Arguments 
    group_by_modality_length: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # Lora or Quant Arguments
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    modal_token: str = None,
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    conversations = []
    input_ids = []
    targets = []
    for source in sources:
        # 1. apply chat template for input conversation
        assert len(source) == 2
        assert modal_token in source[0]['value']
        message = [
            {'role': 'user', 'content': modal_token},
            {'role': 'assistant', 'content': source[1]['value']}
        ]
        conversation = " ".join([sentence['value'] for sentence in source])

        input_id = tokenizer_multimodal_token(conversation, tokenizer, modal_token, return_tensors='pt')
        target = copy.deepcopy(input_id)
        target[input_id == MODAL_INDEX_MAP[modal_token]] = IGNORE_INDEX

        input_ids.append(input_id)
        targets.append(target)

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    modal_token: str = None,
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    # Apply prompt templates
    conversations = []
    input_ids = []
    targets = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != "user":
            # Skip the first one if it is not from human
            source = source[1:]

        message = [{'role': roles[sentence['from']], 'content': sentence['value']} for sentence in source]
        conversation = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        input_ids.append(tokenizer_multimodal_token(conversation, tokenizer, modal_token, return_tensors='pt'))
        targets.append(copy.deepcopy(input_ids[-1]))

        assert len(source) % 2 == 0, f"Invalid conversation length {len(source)}."

        cur = 0
        message = []
        for idx, sentence in enumerate(source):
            if idx % 2 == 1:
                tmp_message = [
                    {'role': roles[source[idx-1]['from']], 'content': source[idx-1]['value']}, 
                    {'role': roles[sentence['from']], 'content': sentence['value']}
                ]

                instruction = tokenizer.apply_chat_template(message + tmp_message[:1], tokenize=False, add_generation_prompt=True)
                conversation = tokenizer.apply_chat_template(message + tmp_message, tokenize=False, add_generation_prompt=False)

                instruction_len = len(tokenizer_multimodal_token(instruction, tokenizer, modal_token, return_tensors='pt'))
                conversation_len = len(tokenizer_multimodal_token(conversation, tokenizer, modal_token, return_tensors='pt'))

                targets[-1][cur:instruction_len] = IGNORE_INDEX

                cur = conversation_len
                message += tmp_message

    return dict(input_ids=input_ids, labels=targets)


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments,
    modal_token: str = None,
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    assert modal_token in MODAL_INDEX_MAP, f"Unsupported modal token {modal_token}."

    for source in sources:
        for sentence in source:
            if modal_token in sentence['value']:
                sentence['value'] = sentence['value'].replace(modal_token, '').strip()
                sentence['value'] = modal_token + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
            replace_token = modal_token
            # TODO: fix this for multimedia, e.g., <video>, <audio>, etc.
            sentence["value"] = sentence["value"].replace(modal_token, replace_token)

    return sources


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = []
        for dp in data_path:
            _datas = json.load(open(dp, "r"))
            list_data_dict.extend(_datas)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        # 预建 video -> 排序条目 映射，避免 __getitem__ 里 O(N) 线性扫描
        _video_entries = {}
        for _entry in list_data_dict:
            if 'video' in _entry:
                _video_entries.setdefault(_entry['video'], []).append(_entry)
        self._video_to_entries = {
            _v: sorted(_es, key=lambda x: x['id'])
            for _v, _es in _video_entries.items()
        }

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 576 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        image_processor = self.data_args.image_processor
        video_processor = self.data_args.video_processor

        num_frames = NUM_FRAMES if self.data_args.num_frames is None else self.data_args.num_frames

        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.data_folder
            image_file = os.path.join(image_folder, image_file)

            try:
                image = process_image(image_file, image_processor, aspect_ratio=self.data_args.image_aspect_ratio)
            except:
                traceback.print_exc()
                backup_idx = random.randint(0, len(self.list_data_dict) - 1)
                print(f"Encounted error when reading image {image_file}, use {backup_idx}-th example instead!!!")
                return self.__getitem__(backup_idx)

            # place <image> tag to question head.
            modal_token = "<image>"
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args, modal_token)
        elif 'video' in sources[0]:
            video_file = self.list_data_dict[i]['video']
            video_folder = self.data_args.data_folder
            video_file = os.path.join(video_folder, video_file)

            try:
                video = process_video(video_file, video_processor, aspect_ratio=self.data_args.image_aspect_ratio, num_frames=num_frames)
            except Exception as e:
                traceback.print_exc()
                backup_idx = random.randint(0, len(self.list_data_dict) - 1)
                print(f"Encounted error when reading video {video_file}, use {backup_idx}-th example instead!!!")
                return self.__getitem__(backup_idx)

            # place <video> tag to question head.
            modal_token = "<video>"
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args, modal_token)
        else:
            modal_token = None
            sources = copy.deepcopy([e["conversations"] for e in sources])

        if self.data_args.is_pretraining:
            data_dict = preprocess_plain(sources, self.tokenizer, modal_token=modal_token)
        else:
            data_dict = preprocess(sources, self.tokenizer, modal_token=modal_token)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif 'video' in self.list_data_dict[i]:
            # data_dict['video'] = video
            # MSVD 命名规则不同,确定video_id和gt_id
            current_entry = self.list_data_dict[i]
            video_path = current_entry['video']
            video_id = video_path.split('/')[-1].split(".")[0]
            # 筛选出相同video的所有条目（预建缓存，O(1)）
            sorted_entries = self._video_to_entries[video_path]
            # 找到当前条目在排序后的列表中的索引
            gt_id = sorted_entries.index(current_entry)               
            
            # lzz MSRVTT确定video_id和gt_id
            # video_id = i//20 #每个视频20个GT
            # gt_id = i%20 
            
            # 根据video_id和gt_id加载wordweight,不取BOS和EOS 
            wordweight_path = os.path.join('datasets/MSVD_sft/MSVD_wordweight/Qwen2_wordweight_after', f'{video_id}_{gt_id}.npy')
            wordweights = np.load(wordweight_path)
            wordweights = torch.from_numpy(wordweights).float()  # 转换为torch张量
            data_dict['wordweights'] = wordweights
            
            #  MSRVTT&MSVD 取keyframe这里设置均匀采样32帧但是根据keyframe_index取16帧
            keyframe_path = os.path.join('datasets/MSVD_sft/tcvr_32_16', f'{video_id}_{gt_id}.npy')
            keyframe_index = np.load(keyframe_path)
            keyframe_indices = torch.from_numpy(keyframe_index).long()
            selected_video = video[keyframe_indices]
            data_dict['video'] = selected_video   
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            data_dict['image'] = torch.zeros(3, self.data_args.image_size, self.data_args.image_size)
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, wordweights = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels", "wordweights"))
        # input_ids, labels = tuple([instance[key] for instance in instances]
        #                     for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            wordweights=wordweights,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # work for 'images' argument in `prepare_inputs_labels_for_multimodal` of LlavaMetaForCausalLM in llava_arch.py
        batch['images'] = []
        for instance in instances:
            for modal_token in MODAL_INDEX_MAP.keys():
                modal_token = modal_token.lower()
                # MODAL_TOKEN shape like: <image>, <video>, ...
                modal_name = re.findall(f'[<](.*)[>]', modal_token)
                assert len(modal_name) == 1
                modal_name = modal_name[0]
                if modal_name in instance:
                    batch['images'].append((instance[modal_name], modal_name))

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank
    # 设置随机种子确保实验可重复性
    set_seed(42)
    
    # 解析命令行参数到三个数据类: 模型参数、数据参数、训练参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank   # 分布式训练中的本地进程编号
    # 根据训练精度设置计算数据类型(f16/bf32/fp32)
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # 初始化量化配置参数
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:    # 4位或8位量化配置
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            # 创建量化配置对象
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,   # 4bit量化开关
                load_in_8bit=training_args.bits == 8,   # 8bit量化开关
                llm_int8_skip_modules=["mm_projector"], # 跳过量化的模块
                llm_int8_threshold=6.0,                 # 激活值阈值
                llm_int8_has_fp16_weight=False,         # 不使用fp16权重
                bnb_4bit_compute_dtype=compute_dtype,   # 4bit计算类型
                bnb_4bit_use_double_quant=training_args.double_quant,   # 二次量化
                bnb_4bit_quant_type=training_args.quant_type, # {'fp4', 'nf4'}  # 量化类型(nf4/fp4)
                bnb_4bit_quant_storage=compute_dtype,   # 量化存储类型
            )
        ))
        
    # 从预训练加载模型配置
    config = VLLMConfigs[model_args.model_type].from_pretrained(model_args.model_path, trust_remote_code=True)
    config._attn_implementation = attn_implementation   # 设置注意力实现方式(flash_attention_2)

    model = VLLMs[model_args.model_type].from_pretrained(
        model_args.model_path,
        config=config,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),   # 设置模型精度
        do_sample=True,     # 生成时使用采样
        **bnb_model_from_pretrained_args
    )
    # 因为vision_tower也会参与qlora,先删除后面再初始化
    del model.model.vision_tower
    del model.model.mm_projector
    model.config.use_cache = False      # 训练时禁用缓存以获得更优显存使用

    # 冻结主干网络参数(当需要微调部分模块时使用)
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    # 准备低精度训练模型(4/8bit量化训练)
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    # 梯度检查点设置(用时间换显存)
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
             # 通过hook强制embedding层输出需要梯度
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # LoRA适配器配置
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,     # LoRA秩
            lora_alpha=training_args.lora_alpha,    # 缩放因子
            target_modules=find_all_linear_names(model),    # 需要添加LoRA的模块
            lora_dropout=training_args.lora_dropout,        # dropout率
            bias=training_args.lora_bias,           # 偏置项处理方式
            task_type="CAUSAL_LM",
        )
        # 转换模型精度以适应LoRA训练
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)# 应用LoRA到Qwen2

    # 加载tokenizer并配置
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_path,
        model_max_length=training_args.model_max_length,    # 最大长度限制
        padding_side="right",   # 右侧填充
        use_fast=True,          # 使用快速tokenizer
    )
    # 处理pad_token缺失情况
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    # 多模态处理部分
    if model_args.vision_tower is not None:
        # initialize vision encoder + multi-modal projector
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
        
        # 配置vision_tower参数
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        # 配置数据参数
        data_args.image_size = vision_tower.image_size

        data_args.image_processor = vision_tower.image_processor
        data_args.video_processor = vision_tower.video_processor if hasattr(vision_tower, "video_processor") else vision_tower.image_processor

        # 启用多模态标志
        data_args.is_multimodal = True

        # 更新模型配置
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        # 配置多模态adapter微调
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:  # 仅微调mm_projector
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        # 设置训练模式(预训练/微调)
        if model_args.tune_mm_mlp_adapter:
            data_args.is_pretraining = True
        else:
            data_args.is_pretraining = False

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        
        # 冻结多模态adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

         # 量化模式下调整mm_projector
        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        # 配置学习率和视频帧数
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.num_frames = NUM_FRAMES if data_args.num_frames is None else data_args.num_frames
        # vision_tower is not trainable in VideoLLaMA2
        model.get_model().vision_tower.requires_grad_(False)   # 冻结vision_tower
        model.get_model().mm_projector.requires_grad_(True)    # 解冻mm_projector

    # 量化模型参数精度调整
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):   # LoRA层精度调整
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:      # 归一化层保持fp32
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:     # 输出层精度调整
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # print("Current model:", model)
    # 创建监督学习数据模块
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # 初始化训练器
    trainer = VideoLLaMA2Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    # ---- 诊断：hook 投影器 W_sim 梯度 ----
    if training_args.local_rank in [-1, 0]:
        def _grad_hook(name):
            def _h(grad):
                if grad is None:
                    print(f'[GRAD] {name}: None', flush=True)
                else:
                    print(f'[GRAD] {name}: {grad.detach().float().norm().item():.3e}', flush=True)
            return _h
        for _n, _p in model.named_parameters():
            if 'mm_projector' in _n and 'W_sim' in _n and _p.requires_grad:
                _p.register_hook(_grad_hook(_n))
        print('[HOOK] registered W_sim grad hooks', flush=True)
    # 检查是否存在ckpt并恢复训练
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    if training_args.local_rank in [-1, 0]:
        for _i, _pg in enumerate(trainer.optimizer.param_groups):
            _numel = sum(p.numel() for p in _pg['params'])
            _shapes = [tuple(p.shape) for p in _pg['params'][:2]]
            print(f'[OPTIM] group {_i}: lr={_pg["lr"]:.3e} n={len(_pg["params"])} numel={_numel} shapes={_shapes}', flush=True)
        for _n, _p in model.named_parameters():
            if ('mm_projector' in _n and 'W_sim' in _n) or 'lora_B' in _n:
                print(f'[WEIGHT] {_n}: max={_p.detach().float().abs().max().item():.3e}', flush=True)
    trainer.save_state()    # 保存训练状态

    model.config.use_cache = True   # 推理时启用缓存

    # 模型保存逻辑
    if training_args.lora_enable:
        # 分离保存LoRA参数和基础模型参数
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        # 全参数保存
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    # 使用Flash Attention 2实现启动训练
    train("flash_attention_2")