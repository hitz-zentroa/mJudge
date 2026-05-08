#!/bin/bash
#SBATCH --job-name=
#SBATCH --time=1-00:00:00
#SBATCH --nodes 4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --output=./llama-recipes/ft_logs/%x_%j.out
#SBATCH --error=./llama-recipes/ft_logs/%x_%j.err


# We used 32 nodes to finetune the 70B models

source ./.venv/bin/activate

export HF_HOME=$SCRATCH/cache/

model_path=$WORK/BaseModels/

cd llama-recipes/

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INFO

export TOKENIZERS_PARALLELISM=true
export NCCL_SOCKET_IFNAME=$(ip -o -4 route get 1.1.1.1 | awk '{print $5}')

MASTER_PORT=9327
MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

srun  torchrun --nnodes=$SLURM_NNODES --nproc_per_node 4 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $MAIN_PROCESS_IP:$MASTER_PORT\
    ./recipes/quickstart/finetuning/finetuning.py \
    --enable_fsdp \
    --fsdp_config.pure_bf16 \
    --model_name ${model_path}$MODEL \
    --num_epochs 3 \
    --lr 1e-5 \
    --context_length 8100 \
    --dataset feedback_dataset \
    --lang $LANG \
    --dist_checkpoint_root_folder $SCRATCH/model_checkpoints_4nodes \
    --dist_checkpoint_folder $LANG \
    --save_optimizer \
    --output_dir $SCRATCH/models/$LANG \
    --use_wandb \
    --save_model \
    --run_name 4-$MODEL-$LANG-1e-5\
    --use_fast_kernels \
    --batch_size_training 1 \
    --gradient_accumulation_steps 4