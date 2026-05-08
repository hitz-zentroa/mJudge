#!/bin/bash

model_list=(Latxa-Llama-3.1-8B-Instruct) # Name of localy stored models in $WORK/BaseModels/. Said path can be changed in ./llama-recipes/recipes/quickstart/finetuning/finetune_models.slurm. We did not directly use the Hugging Face model hub for training due to the server not having internet access.

#8B models
#model_list=(Latxa-Llama-3.1-8B-Instruct Llama-3.1-8B-Instruct DeepSeek-R1-Distill-Llama-8B)

#70B models
#model_list=(Latxa-Llama-3.1-70B-Instruct Llama-3.1-70B-Instruct)

lang_list=(en)

#full list of language settings
#lang_list=(es eu en_es_eu io_es io_eu io_en_es_eu)
dataset=feedback_dataset

for model in "${model_list[@]}"; do
    for lang in "${lang_list[@]}"; do
        sbatch --export=ALL,MODEL=$model,LANG=$lang ./llama-recipes/recipes/quickstart/finetuning/finetune_models.slurm
    done
done