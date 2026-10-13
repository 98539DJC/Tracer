#!/bin/bash

# Environment Variables
# 只用 2、3 号卡，不影响其他卡上的实验（仅此脚本内生效）
export CUDA_VISIBLE_DEVICES=2,3
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WORLD_SIZE=1
NPROC_PER_NODE=2
MASTER_ADDR="127.0.0.1"
# 冷门端口，避免和其他人的分布式训练冲突
MASTER_PORT=29617
RANK=0

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# Training Arguments
GLOBAL_BATCH_SIZE=128
LOCAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]

# Log Arguments
export TRANSFORMERS_OFFLINE=1
RUN_NAME=my_run_qlora_v3
DATA_DIR=datasets
OUTP_DIR=output/MSVD/checkpoints/my_run_qlora_v3
mkdir -p ${OUTP_DIR}

/home/megaium/miniconda3/envs/tracer/bin/python -u -m torch.distributed.run --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $RANK \
    videollama2/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 1e-5 --bits 4 \
    --deepspeed scripts/zero2.json \
    --model_type videollama2_qwen2 \
    --model_path DAMO-NLP-SG/VideoLLaMA2.1-7B-16F \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_projector_type route_mamba \
    --pretrain_mm_mlp_adapter DAMO-NLP-SG/VideoLLaMA2.1-7B-16F-Base/mm_projector.bin \
    --data_path   ${DATA_DIR}/train.json \
    --data_folder ${DATA_DIR}/ \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio pad \
    --num_frames 16 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ${OUTP_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size $LOCAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 99 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    --run_name $RUN_NAME 2>&1 | tee ${OUTP_DIR}/train.log
