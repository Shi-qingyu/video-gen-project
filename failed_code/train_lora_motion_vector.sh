#!/bin/bash

export MODEL_PATH="THUDM/CogVideoX-5b"
export DATASET_PATH="./data/car"
export OUTPUT_PATH="results/motion_vector_step_500_r_128-lr_1e-5_car_5b"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


#  --use_8bit_adam is necessary for CogVideoX-5B-I2V
# if you are not using wth 8 gus, change `accelerate_config_machine_single.yaml` num_processes as your gpu number
accelerate launch --config_file configs/accelerate_config_machine_single.yaml --multi_gpu \
  train_lora_motion_vector.py \
  --gradient_checkpointing \
  --use_8bit_adam  \
  --pretrained_model_name_or_path $MODEL_PATH \
  --enable_tiling \
  --enable_slicing \
  --instance_data_root $DATASET_PATH \
  --caption_column prompts.txt \
  --video_column videos.txt \
  --num_validation_videos 1 \
  --validation_epochs 500 \
  --seed 0 \
  --rank 128 \
  --mixed_precision bf16 \
  --output_dir $OUTPUT_PATH \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --skip_frames_start 0 \
  --skip_frames_end 0 \
  --train_batch_size 1 \
  --max_train_steps 500 \
  --checkpointing_steps 100 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95
