#!/bin/bash

#SBATCH --job-name=
#SBATCH --time 04:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=
#SBATCH --output=./translate/logs/faltan_eu_%j.out
#SBATCH --error=./translate/logs/faltan_eu_%j.err

source ./.venv/bin/activate

export HF_HOME=$SCRATCH/cache

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export OMP_NUM_THREADS=16
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo CUDA_VISIBLE_DEVICES "${CUDA_VISIBLE_DEVICES}"

python ./translate/translate.py \
    --model_path  HiTZ/Latxa-Llama-3.1-70B-Instruct\
    --dataset_path <path_to_dataset> \
    --few_shot_folder feedback_collection \
    --lang eu \
    --tensor_parallel_size 4 \
