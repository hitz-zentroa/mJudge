#!/bin/bash

### HUGGINGFACE BASELINES: ____________________________________________________________________###

model_path="HiTZ/"
model_list=(Latxa-Llama-3.1-8B-Instruct)

# model_path="HiTZ/"
# model_list=(Latxa-Llama-3.1-70B-Instruct)

# model_path="meta-llama/"
# model_list=(Llama-3.1-8B-Instruct)

# model_path="meta-llama/"
# model_list=(Llama-3.1-70B-Instruct)

# model_path="prometheus-eval/"
# model_list=(prometheus-7b-v2.0)

# model_path="prometheus-eval/"
# model_list=(prometheus-7b-v1.0)

# model_path="Unbabel/"
# model_list=(M-Prometheus-7B)

### LOCAL MODELS: _____________________________________________________________________________###

# model_path=$DATA/hf_checkpoints/
# model_list=($(ls -d "${model_path}"*/ | xargs -n 1 basename'))

### SLURM JOB SUBMISSION: _____________________________________________________________________###

# Select benchmark. Options are recon and flask.
bench=recon #flask

for model in "${model_list[@]}"; do
    sbatch --export=ALL,MODEL=$model,MODEL_PATH=$model_path,BENCHMARK=$bench ./evaluation/bench.slurm
done