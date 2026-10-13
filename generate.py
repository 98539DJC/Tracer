import os
import re
import sys
import json
import torch
sys.path.append('./')
from functools import partial
from evaluation.pycocoevalcap.eval import COCOEvalCap
from peft import PeftModel
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
from videollama2.mm_utils import process_image, process_video
from transformers import AutoTokenizer, AutoConfig
from videollama2.model.videollama2_qwen2 import Videollama2Qwen2ForCausalLM

class VideoCaptionEvaluator:
    def __init__(self):
        self.dataset_config = {
            'MSVD': {
                'video_ext': '.avi',
                'video_dir': 'datasets/MSVD_sft/videos/',
                'sft_dir': 'datasets/MSVD_sft/make_sft/'
            },
            'MSRVTT': {
                'video_ext': '.mp4', 
                'video_dir': 'datasets/MSRVTT_sft/videos/',
                'sft_dir': 'datasets/MSRVTT_sft/make_sft/'
            },
              'VATEX': {
                'video_ext': '.mp4', 
                'video_dir': 'datasets/VATEX_sft/videos/',
                'sft_dir': 'datasets/VATEX_sft/make_sft/'
            }
        }
        
    def get_paths(self, dataset, split, ckpt_id, folder):
        """根据数据集和分割生成所有相关路径"""
        config = self.dataset_config[dataset]
        
        paths = {
            'valid_jsonl_path': f"{config['sft_dir']}{split}.jsonl",
            'video_dir': config['video_dir'],
            'video_ext': config['video_ext'],
            'lora_model_path': f'output/{dataset}/checkpoints/{folder}/checkpoint-{ckpt_id}',
            'output_path': f'output/{dataset}/result/{folder}-{ckpt_id}_{split}.json',
            # 'output_path': f'output/{dataset}/result/mamba_baseline+bidirectional-{ckpt_id}_{split}.json',
            'eval_output_path': f'output/{dataset}/score/{folder}-{ckpt_id}_{split}.json'
        }
        return paths
    
    def inference(self, dataset, split, ckpt_id, folder):
        """生成caption结果"""
        disable_torch_init()
        
        # 获取路径配置
        paths = self.get_paths(dataset, split, ckpt_id, folder)
        
        # 模型配置
        modal = 'video'
        instruct = 'Use only one short sentence to describe the video.'
        model_path = 'DAMO-NLP-SG/VideoLLaMA2.1-7B-16F'
        
        # 加载模型
        config = AutoConfig.from_pretrained(model_path)
        # 更换mm_projector
        config.mm_projector_type = 'route_mamba'
        model = Videollama2Qwen2ForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True,
            config=config, device_map='auto', torch_dtype=torch.float16
        )

        # 加载非lora参数
        non_lora_trainables = torch.load(
            os.path.join(paths['lora_model_path'], 'non_lora_trainables.bin'), 
            map_location='cpu'
        )
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)
        
        # 加载和合并lora权重
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, paths['lora_model_path'])
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
             
        # 处理器和tokenizer
        vision_tower = model.get_vision_tower()
        processor = vision_tower.image_processor
        processor = {
            'image': partial(process_image, processor=processor, aspect_ratio=None),
            'video': partial(process_video, processor=processor, aspect_ratio=None, num_frames=16),
        }
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        if tokenizer.pad_token is None and tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token

        # 生成结果
        output_list = []
        with open(paths['valid_jsonl_path'], "r", encoding="utf8") as f:
            for line in f:
                data = json.loads(line)
                video_id = data["video_id"]
                video_path = paths['video_dir'] + video_id + paths['video_ext']
                
                try:
                    output = mm_infer(
                        processor[modal](video_path), instruct, 
                        model=model, tokenizer=tokenizer, do_sample=False, modal=modal
                    ) 
                    output = output.lower().rstrip('.')
                except Exception as e:
                    print(f"Error processing {video_id}: {e}")
                    output = ""
                
                output_dict = {
                    "split": split,
                    "image_name": video_path,
                    "captions": data["golden_captions"],
                    "prediction": output,
                }
                print(f"{video_id}: {output}")
                output_list.append(output_dict)

        # 保存结果
        os.makedirs(os.path.dirname(paths['output_path']), exist_ok=True)
        with open(paths['output_path'], 'w', encoding='utf-8') as outfile:
            json.dump(output_list, outfile, indent=4, ensure_ascii=False)
        
        # 清理资源
        del model
        torch.cuda.empty_cache()
        
        return paths['output_path']
    
    def evaluate(self, result_file_path, eval_file_path):
        """评估生成结果"""
        cocoEval = COCOEvalCap(result_file_path)
        cocoEval.evaluate()
                
        for metric, score in cocoEval.eval.items():
            print('%s: %.3f' % (metric, score))
        
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        with open(eval_file_path, 'w') as fp:
            json.dump(cocoEval.eval, fp, indent=4)
        return cocoEval.eval
    
    def get_checkpoint_numbers(self, dataset, folder):
        """获取checkpoint序号列表"""
        folder_path = f"output/{dataset}/checkpoints/{folder}"
        if not os.path.exists(folder_path):
            print(f"警告：文件夹 {folder_path} 不存在")
            return []
            
        checkpoint_numbers = sorted([
            int(re.match(r'checkpoint-(\d+)', filename).group(1))
            for filename in os.listdir(folder_path)
            if re.match(r'checkpoint-(\d+)', filename)
        ])
        return checkpoint_numbers


def main():
    # 在这里修改数据集和分割
    dataset = 'MSVD'  # 可选: 'MSVD', 'MSRVTT'
    split = 'test'    # 可选: 'valid', 'test'
    folder = 'my_run_qlora'
    
    # 初始化评估器
    evaluator = VideoCaptionEvaluator()
    
    # 获取checkpoint序号
    checkpoint_numbers = [350]
    print(f"找到的checkpoint序号: {checkpoint_numbers}")
    
    if not checkpoint_numbers:
        print("没有找到可用的checkpoint文件")
        return
    
    # 存储所有评估结果
    all_eval_results = {}
    
    # 遍历checkpoint（这里可以根据需要调整范围）
    for ckpt_id in checkpoint_numbers:
        print(f"\n{'='*50}")
        print(f"处理 checkpoint-{ckpt_id}, 数据集: {dataset}, 分割: {split}")
        print(f"{'='*50}")
        
        try:
            # 1. 生成结果
            result_path = evaluator.inference(dataset, split, ckpt_id, folder)
            
            # 2. 评估结果
            paths = evaluator.get_paths(dataset, split, ckpt_id, folder)
            eval_results = evaluator.evaluate(result_path, paths['eval_output_path'])
            
            # 3. 存储结果
            all_eval_results[ckpt_id] = eval_results
            
        except Exception as e:
            print(f"处理 checkpoint-{ckpt_id} 时出错: {e}")
            continue
    

if __name__ == "__main__":
    main()