#!/bin/bash
#SBATCH --job-name=reasoning_eval
#SBATCH --output=logs/reasoning_eval%A_%a.out
#SBATCH --error=logs/reasoning_eval%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4           
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=71:59:59

nvidia-smi

# --- Parameters ---
MODEL_NAME="Qwen/Qwen3-8B" # "Qwen/Qwen3-8B" or "Qwen/Qwen3-4B"
ATTN_TYPE="lim" # "lim":LessIsMore; "tidal":TidalDecode; "None": full attention
N_ATTEMPTS=32
MAX_GEN_LEN=10 # 32768
TEMPERATURE=0.6
SPARSE_START_LAYER=2
CORRECTION_LAYER=12 #8B:12; 4B:20;
ATTENTION_SINK=4
TASKS=("aime") #  "aime2025" "gpqa" "math"
LIM_RATIO=(0.75) # default 0.75, only matters When ATTN_TYPE is "lim"
TOP_K_VALUES=(5)

# --- Directories ---
SLURM_SUBMIT_DIR=$(pwd)
SLURM_JOB_ID=$(date +%s)
RESULTS_DIR="${SLURM_SUBMIT_DIR}/results/"
LOG_DIR_PYTHON="${SLURM_SUBMIT_DIR}/logs/"
mkdir -p "${RESULTS_DIR}"
mkdir -p "${LOG_DIR_PYTHON}"

# --- Main Execution Logic ---
SCRIPT_PATH="${SLURM_SUBMIT_DIR}/experiments/reasoning/eval_reasoning.py"

echo "Starting Reasoning evaluation runs..."
for TASK in "${TASKS[@]}"; do
  for SCALE in "${LIM_RATIO[@]}"; do
    for TOP_K in "${TOP_K_VALUES[@]}"; do
      echo "----------------------------------------------------------------------"
      RUN_LOG_FILE="${LOG_DIR_PYTHON}/eval_${TASK}_model-${MODEL_NAME//\//_}_attn-${ATTN_TYPE}_job-${SLURM_JOB_ID}_r${SCALE}_k${TOP_K}.out"

      python "${SCRIPT_PATH}" \
        --model_name "${MODEL_NAME}" \
        --attn_type "${ATTN_TYPE}" \
        --top_k "${TOP_K}" \
        --n "${N_ATTEMPTS}" \
        --max_gen_len "${MAX_GEN_LEN}" \
        --temperature "${TEMPERATURE}" \
        --correction_layer "${CORRECTION_LAYER}" \
        --sparse_layer_start "${SPARSE_START_LAYER}" \
        --attention_sink "${ATTENTION_SINK}" \
        --lim_ratio "${SCALE}" \
        --task "${TASK}" \
        > "${RUN_LOG_FILE}" 2>&1

    done
  done
done

echo "Waiting for all parallel runs to finish..."
wait
echo "Check SLURM output in: ${SLURM_SUBMIT_DIR}/logs/reasoning_eval${SLURM_JOB_ID}_*.out"
echo "Check Python script JSON results in: ${SLURM_SUBMIT_DIR}/results/"