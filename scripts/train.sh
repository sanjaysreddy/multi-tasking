#!/bin/bash
#SBATCH -p gpu_a100_8
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 64G
#SBATCH -t 0-01:30
#SBATCH --job-name="codemix"
#SBATCH -o /scratch/aruna/codemix-outs/cf-xlm-large-%j.out
#SBATCH -e /scratch/aruna/codemix-outs/cf-xlm-large-%j.err
#SBATCH --mail-user=f20190083@hyderabad.bits-pilani.ac.in
#SBATCH --mail-type=NONE

# Load modules
spack unload 
spack load gcc@11.2.0
spack load cuda@11.7.1%gcc@11.2.0 arch=linux-rocky8-zen2
spack load python@3.9.13%gcc@11.2.0 arch=linux-rocky8-zen2


# Activate Environment
source ~/omkar/envs/codemix/bin/activate

# Run 
cd /home/aruna/omkar/multitask

srun ~/omkar/envs/codemix/bin/python main.py \
--run_name "xlm-base-cf-2-v1" \
--base_model xlm-roberta-base \
--epochs 100 \
--ner_lr 3e-4 \
--lid_lr 3e-5 \
--batch_size 32 \
--logger wandb \
--exp_path "/scratch/aruna/codemix-runs" \
--dataset_dir "/scratch/aruna/codemix-data/ner" \
--k 8