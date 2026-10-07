#!/bin/bash

# 小实验：只把投影器 LR 从 1e-5 提到 1e-3，跑 20 步看 CE loss 会不会降
export CUDA_VISIBLE_DEVICES=2,3
WORLD_SIZE=1
NPROC_PER_NODE=2
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29618
RANK=0

LOCAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
export TRANSFORMERS_OFFLINE=1

RUN_NAME=exp_tracer_fix
DATA_DIR=datasets
OUTP_DIR=output/MSVD/checkpoints/exp_tracer_fix
mkdir -p ${OUTP_DIR}

/home/megaium/miniconda3/envs/tracer/bin/python -u -m torch.distributed.run --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $RANK \
    videollama2/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 1e-3 --bits 4 \
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
    --max_steps 100 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size $LOCAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --save_strategy "no" \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --run_name $RUN_NAME 2>&1 | tee ${OUTP_DIR}/train.log
