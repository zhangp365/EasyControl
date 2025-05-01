export MODEL_DIR="black-forest-labs/FLUX.1-dev" # your flux path
export OUTPUT_DIR="./models/style_model"  # your save path
export CONFIG="./default_config.yaml"
export TRAIN_DATA="./examples/mine_craft.jsonl"  # your data jsonl file
export LOG_PATH="$OUTPUT_DIR/log"

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file $CONFIG train.py \
    --pretrained_model_name_or_path $MODEL_DIR \
    --cond_size=512 \
    --noise_size=1024 \
    --subject_column="None" \
    --spatial_column="source" \
    --target_column="target" \
    --caption_column="prompt" \
    --ranks 128 \
    --network_alphas 128 \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOG_PATH \
    --mixed_precision="bf16" \
    --train_data_dir=$TRAIN_DATA \
    --learning_rate=1e-4 \
    --train_batch_size=1 \
    --validation_prompt "a voxel art style, reminiscent of Minecraft, blocky and smiling robots. The robot is constructed from cubes." \
    --num_train_epochs=100 \
    --validation_steps=1000 \
    --checkpointing_steps=1000 \
    --spatial_test_images "./examples/style_data/5.png" \
    --subject_test_images None \
    --test_h 1024 \
    --test_w 1024 \
    --num_validation_images=2 \
    --resume_from_checkpoint="$OUTPUT_DIR/checkpoint-8000" \
    --pretrained_lora_path="$OUTPUT_DIR/checkpoint-8000/lora.safetensors" # 生成路径：style_model/checkpoint-1000/lora.safetensors
# 
