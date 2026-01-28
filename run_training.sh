#!/bin/bash
# ============================================================================
# Quick Start Script for Tau-Bench RL Training
# ============================================================================
# Usage:
#   ./run_training.sh [config_profile]
#
# Examples:
#   ./run_training.sh a100_40gb
#   ./run_training.sh h100_80gb
#   ./run_training.sh dev
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CONFIG_PROFILE=${1:-default}
CONFIG_FILE="training_configs.ini"

echo -e "${GREEN}======================================"
echo "Tau-Bench RL Training Quick Start"
echo -e "======================================${NC}"

# Check if dataset exists
if [ ! -f "dataset.jsonl" ]; then
    echo -e "${RED}Error: dataset.jsonl not found!${NC}"
    echo "Please generate dataset first using task_generator.py"
    exit 1
fi

# Check if environment exists
if [ ! -d "envs/retail" ]; then
    echo -e "${RED}Error: envs/retail not found!${NC}"
    echo "Please ensure retail environment data is available"
    exit 1
fi

# Create necessary directories
mkdir -p outputs logs

# Detect GPU type
GPU_TYPE="unknown"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    echo -e "${GREEN}Detected GPU: $GPU_NAME${NC}"
    
    if [[ $GPU_NAME == *"A100"* ]]; then
        GPU_TYPE="a100"
        if [[ $GPU_NAME == *"80GB"* ]]; then
            GPU_TYPE="a100_80gb"
        else
            GPU_TYPE="a100_40gb"
        fi
    elif [[ $GPU_NAME == *"H100"* ]]; then
        GPU_TYPE="h100_80gb"
    fi
    
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo -e "${GREEN}Number of GPUs: $NUM_GPUS${NC}"
else
    echo -e "${YELLOW}Warning: nvidia-smi not found. Cannot detect GPU.${NC}"
fi

# Suggest config if not specified
if [ "$CONFIG_PROFILE" = "default" ] && [ "$GPU_TYPE" != "unknown" ]; then
    if [ $NUM_GPUS -gt 1 ]; then
        CONFIG_PROFILE="multi_${GPU_TYPE}"
    else
        CONFIG_PROFILE="$GPU_TYPE"
    fi
    echo -e "${YELLOW}Auto-selected config profile: $CONFIG_PROFILE${NC}"
fi

echo -e "${GREEN}Using configuration profile: $CONFIG_PROFILE${NC}"

# Parse configuration from INI file
if [ -f "$CONFIG_FILE" ]; then
    echo -e "${GREEN}Loading configuration from $CONFIG_FILE...${NC}"
    
    # Extract values from INI file (simple parser)
    MODEL_NAME=$(grep -A 20 "\[$CONFIG_PROFILE\]" $CONFIG_FILE | grep "model_name" | head -1 | cut -d'=' -f2 | xargs)
    MAX_SEQ_LENGTH=$(grep -A 20 "\[$CONFIG_PROFILE\]" $CONFIG_FILE | grep "max_seq_length" | head -1 | cut -d'=' -f2 | xargs)
    LORA_RANK=$(grep -A 20 "\[$CONFIG_PROFILE\]" $CONFIG_FILE | grep "lora_rank" | head -1 | cut -d'=' -f2 | xargs)
    LOAD_4BIT=$(grep -A 20 "\[$CONFIG_PROFILE\]" $CONFIG_FILE | grep "load_in_4bit" | head -1 | cut -d'=' -f2 | xargs)
    BATCH_SIZE=$(grep -A 20 "\[$CONFIG_PROFILE\]" $CONFIG_FILE | grep "batch_size" | head -1 | cut -d'=' -f2 | xargs)
    GRAD_ACCUM=$(grep -A 20 "\[$CONFIG_PROFILE\]" $CONFIG_FILE | grep "gradient_accumulation_steps" | head -1 | cut -d'=' -f2 | xargs)
    LEARNING_RATE=$(grep -A 20 "\[$CONFIG_PROFILE\]" $CONFIG_FILE | grep "learning_rate" | head -1 | cut -d'=' -f2 | xargs)
    NUM_EPOCHS=$(grep -A 20 "\[$CONFIG_PROFILE\]" $CONFIG_FILE | grep "num_epochs" | head -1 | cut -d'=' -f2 | xargs)
    
    # Set defaults if not found
    MODEL_NAME=${MODEL_NAME:-unsloth/gpt-oss-20b}
    MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-2048}
    LORA_RANK=${LORA_RANK:-16}
    LOAD_4BIT=${LOAD_4BIT:-true}
    BATCH_SIZE=${BATCH_SIZE:-2}
    GRAD_ACCUM=${GRAD_ACCUM:-8}
    LEARNING_RATE=${LEARNING_RATE:-5e-5}
    NUM_EPOCHS=${NUM_EPOCHS:-3}
else
    echo -e "${YELLOW}Warning: Config file not found. Using defaults.${NC}"
    MODEL_NAME="unsloth/gpt-oss-20b"
    MAX_SEQ_LENGTH=2048
    LORA_RANK=16
    LOAD_4BIT="true"
    BATCH_SIZE=2
    GRAD_ACCUM=8
    LEARNING_RATE=5e-5
    NUM_EPOCHS=3
fi

# Calculate effective batch size
EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))

# Display configuration
echo -e "${GREEN}======================================"
echo "Training Configuration:"
echo "======================================"
echo "Model: $MODEL_NAME"
echo "Max Sequence Length: $MAX_SEQ_LENGTH"
echo "LoRA Rank: $LORA_RANK"
echo "4-bit Quantization: $LOAD_4BIT"
echo "Batch Size per GPU: $BATCH_SIZE"
echo "Gradient Accumulation: $GRAD_ACCUM"
echo "Effective Batch Size: $EFFECTIVE_BATCH"
echo "Learning Rate: $LEARNING_RATE"
echo "Number of Epochs: $NUM_EPOCHS"
echo -e "======================================${NC}"

# Ask for confirmation
read -p "Continue with training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Build command
TRAIN_CMD="python tau_bench_rl_trainer.py \
    --model_name $MODEL_NAME \
    --dataset_path dataset.jsonl \
    --envs_path envs/retail \
    --output_dir outputs/tau_bench_rl_$(date +%Y%m%d_%H%M%S) \
    --max_seq_length $MAX_SEQ_LENGTH \
    --lora_rank $LORA_RANK \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE"

# Add 4-bit flag if enabled
if [ "$LOAD_4BIT" = "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --load_in_4bit"
fi

# Multi-GPU training
if [ $NUM_GPUS -gt 1 ]; then
    echo -e "${GREEN}Starting multi-GPU training with $NUM_GPUS GPUs...${NC}"
    TRAIN_CMD="torchrun --nproc_per_node=$NUM_GPUS $TRAIN_CMD"
else
    echo -e "${GREEN}Starting single-GPU training...${NC}"
fi

# Run training
echo -e "${GREEN}======================================"
echo "Executing: $TRAIN_CMD"
echo -e "======================================${NC}"

eval $TRAIN_CMD

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}======================================"
    echo "Training completed successfully!"
    echo -e "======================================${NC}"
    
    # Offer to run evaluation
    read -p "Run evaluation? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}Running evaluation...${NC}"
        python tau_bench_rl_trainer.py \
            --dataset_path dataset.jsonl \
            --envs_path envs/retail \
            --eval_only \
            --eval_samples 50
    fi
else
    echo -e "${RED}======================================"
    echo "Training failed with exit code $TRAIN_EXIT_CODE"
    echo -e "======================================${NC}"
    exit $TRAIN_EXIT_CODE
fi

echo -e "${GREEN}======================================"
echo "All done! 🎉"
echo -e "======================================${NC}"
