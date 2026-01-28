#!/bin/bash
#SBATCH --job-name=tau_bench_rl_training
#SBATCH --output=logs/tau_bench_rl_%j.out
#SBATCH --error=logs/tau_bench_rl_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

# ============================================================================
# SLURM Job Script for Tau-Bench RL Training on DGX Systems
# ============================================================================
#
# Usage:
#   sbatch dgx_train.slurm
#
# For multi-GPU training, modify:
#   --gres=gpu:a100:4  (for 4 GPUs)
#   --nodes=2          (for multi-node)
#
# For H100:
#   --gres=gpu:h100:1
# ============================================================================

# Environment setup
echo "======================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "======================================"

# Create logs directory
mkdir -p logs

# Load modules (adjust for your DGX cluster)
module purge
module load cuda/12.1
module load python/3.10
module load gcc/11.2.0

# Activate virtual environment (or create if not exists)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install unsloth transformers trl torch torchvision torchaudio
pip install bitsandbytes accelerate datasets tensorboard

# Verify GPU availability
echo "======================================"
echo "GPU Information:"
nvidia-smi
echo "======================================"
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'Number of GPUs: {torch.cuda.device_count()}')"
echo "======================================"

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on allocated GPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

# Training configuration
MODEL_NAME="unsloth/gpt-oss-20b"
DATASET_PATH="dataset.jsonl"
ENVS_PATH="envs/retail"
OUTPUT_DIR="outputs/tau_bench_rl_${SLURM_JOB_ID}"
MAX_SEQ_LENGTH=2048
LORA_RANK=16
NUM_EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM_STEPS=8
LEARNING_RATE=5e-5

# Create output directory
mkdir -p $OUTPUT_DIR

# Single GPU training
if [ "$SLURM_GPUS" = "1" ] || [ -z "$SLURM_GPUS" ]; then
    echo "======================================"
    echo "Starting Single GPU Training"
    echo "======================================"
    
    python tau_bench_rl_trainer.py \
        --model_name $MODEL_NAME \
        --dataset_path $DATASET_PATH \
        --envs_path $ENVS_PATH \
        --output_dir $OUTPUT_DIR \
        --max_seq_length $MAX_SEQ_LENGTH \
        --lora_rank $LORA_RANK \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
        --learning_rate $LEARNING_RATE \
        --load_in_4bit

# Multi-GPU training
else
    echo "======================================"
    echo "Starting Multi-GPU Training"
    echo "Number of GPUs: $SLURM_GPUS"
    echo "======================================"
    
    torchrun \
        --nproc_per_node=$SLURM_GPUS \
        --nnodes=$SLURM_NNODES \
        --node_rank=$SLURM_NODEID \
        --master_addr=$(scontrol show hostname $SLURM_NODELIST | head -n 1) \
        --master_port=29500 \
        tau_bench_rl_trainer.py \
            --model_name $MODEL_NAME \
            --dataset_path $DATASET_PATH \
            --envs_path $ENVS_PATH \
            --output_dir $OUTPUT_DIR \
            --max_seq_length $MAX_SEQ_LENGTH \
            --lora_rank $LORA_RANK \
            --num_epochs $NUM_EPOCHS \
            --batch_size $BATCH_SIZE \
            --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
            --learning_rate $LEARNING_RATE \
            --load_in_4bit
fi

TRAIN_EXIT_CODE=$?

# Evaluation
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "======================================"
    echo "Training completed successfully!"
    echo "Starting evaluation..."
    echo "======================================"
    
    python tau_bench_rl_trainer.py \
        --model_name $OUTPUT_DIR/final_model \
        --dataset_path $DATASET_PATH \
        --envs_path $ENVS_PATH \
        --output_dir $OUTPUT_DIR \
        --eval_only \
        --eval_samples 50
    
    EVAL_EXIT_CODE=$?
    
    if [ $EVAL_EXIT_CODE -eq 0 ]; then
        echo "Evaluation completed successfully!"
    else
        echo "Evaluation failed with exit code $EVAL_EXIT_CODE"
    fi
else
    echo "======================================"
    echo "Training failed with exit code $TRAIN_EXIT_CODE"
    echo "======================================"
fi

# Summary
echo "======================================"
echo "Job completed at: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "======================================"

# Copy logs to output directory
cp logs/tau_bench_rl_${SLURM_JOB_ID}.out $OUTPUT_DIR/
cp logs/tau_bench_rl_${SLURM_JOB_ID}.err $OUTPUT_DIR/

# Deactivate virtual environment
deactivate

exit $TRAIN_EXIT_CODE
