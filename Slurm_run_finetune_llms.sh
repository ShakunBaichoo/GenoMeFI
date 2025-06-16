#!/bin/bash
#SBATCH -J finetune_llm
#SBATCH -t 6:00:00
#SBATCH -p gpu
#SBATCH --account=bwanggroup_gpu 
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --cpus-per-task=10
#SBATCH -N 1
#SBATCH --output=logs/finetune_llm_%j.out
#SBATCH --error=logs/finetune_llm_%j.err
#SBATCH --mail-user=shakun.baichoo@gmail.com # please use your email
#SBATCH --mail-type=ALL
#SBATCH --export=ALL

echo "Job started on $(hostname) at $(date)"
cd $SLURM_SUBMIT_DIR

# Activate your conda environment
# Mine is as follows:
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dna_bert2

# Forward all arguments to Python script
python 2_finetune_llms.py "$@"

echo "Job ended at $(date)"
