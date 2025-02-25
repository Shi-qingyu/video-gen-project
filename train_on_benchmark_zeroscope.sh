#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define the base directories
BASE_DATA_DIR="./MTBench_subset/MTBench_easy"
BASE_OUTPUT_DIR="checkpoints"

# Define the model path
MODEL_PATH="cerspense/zeroscope_v2_576w"

# Define CUDA allocation configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Iterate over each subdirectory in the base data directory
for DATASET_SUBDIR in "$BASE_DATA_DIR"/*/; do
    # Ensure it's a directory
    if [ -d "$DATASET_SUBDIR" ]; then
        # Extract the dataset name (e.g., "horsejump" from "./data/horsejump/")
        DATASET_NAME=$(basename "$DATASET_SUBDIR")
        
        echo "--------------------------------------------"
        echo "Processing dataset: $DATASET_NAME"
        echo "--------------------------------------------"

        # Set the DATASET_PATH for the current dataset
        DATASET_PATH="$BASE_DATA_DIR/$DATASET_NAME"

        # Define the OUTPUT_PATH based on the dataset name
        OUTPUT_PATH="$BASE_OUTPUT_DIR/lr_1e-5_skipconv1d_conv1d_kernel_3_mid_64_gas_4_mse_1.0_zeroscope_$DATASET_NAME"

        # Export the environment variables for the current iteration
        export DATASET_PATH
        export OUTPUT_PATH

        echo "MODEL_PATH: $MODEL_PATH"
        echo "DATASET_PATH: $DATASET_PATH"
        echo "OUTPUT_PATH: $OUTPUT_PATH"

        # Create the OUTPUT_PATH directory if it doesn't exist
        mkdir -p "$OUTPUT_PATH"

            # Execute the training command
            accelerate launch --config_file configs/accelerate_config_machine_single.yaml --main_process_port 8002 --multi_gpu \
            train_conv1d_zeroscope.py \
            --gradient_checkpointing \
            --use_8bit_adam  \
            --pretrained_model_name_or_path $MODEL_PATH \
            --enable_tiling \
            --enable_slicing \
            --rank 64 \
            --kernel_size 3 \
            --version skipconv1d \
            --module_type conv1d \
            --instance_data_root $DATASET_PATH \
            --caption_column prompts.txt \
            --video_column videos.txt \
            --seed 0 \
            --mixed_precision no \
            --output_dir $OUTPUT_PATH \
            --height 320 \
            --width 576 \
            --fps 8 \
            --max_num_frames 24 \
            --skip_frames_start 0 \
            --skip_frames_end 0 \
            --train_batch_size 1 \
            --max_train_steps 500 \
            --checkpointing_steps 100 \
            --resume_from_checkpoint "" \
            --gradient_accumulation_steps 4 \
            --learning_rate 1e-5 \
            --optimizer AdamW \
            --adam_beta1 0.9 \
            --adam_beta2 0.95 \
            --mse_weight 1.0 \

        echo "Training completed for dataset: $DATASET_NAME"
        echo "Output saved to: $OUTPUT_PATH"
        echo ""

    else
        echo "Skipping $DATASET_SUBDIR as it is not a directory."
    fi
done

echo "All datasets have been processed."
