#!/bin/bash
#SBATCH --job-name=
#SBATCH --cpus-per-task=
#SBATCH --gres=
#SBATCH --mem=
#SBATCH --time=

#SBATCH --output=./evaluation/logs/%x_%j.out
#SBATCH --error=./evaluation/logs/%x_%j.err

source ./.venv/bin/activate

export HF_HOME=$SCRATCH/cache/

# To test Meta models on HF, uncomment and set your token. Not required for other baselines.
# export HF_TOKEN=<your_huggingface_token>

language_list="en,es,eu"

echo "Processing model: $MODEL, languages: $language_list"

python ./evaluation/gen_constrained.py \
    --model_path $MODEL_PATH$MODEL \
    --benchmark $BENCHMARK \
    --lang_list $language_list \
    --tensor_parallel_size 2 \
    --repetitions 3\
