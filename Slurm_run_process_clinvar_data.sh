#!/bin/bash
#SBATCH -J Process_ClinvarData
#SBATCH -t 6:00:00
#SBATCH -p gpu
#SBATCH --account=bwanggroup_gpu 
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --cpus-per-task=8
#SBATCH -N 1
#SBATCH --output=logs/Process_ClinvarData_%j.out
#SBATCH --error=logs/Process_ClinvarData_%j.err
#SBATCH --mail-user=shakun.baichoo@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --export=ALL

echo "Job started on $(hostname) at $(date)"
module load samtools
module load bedtools

# Activate your conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dna_bert2

# Forward all arguments to Python script
python 1_process_clinvar_data.py

echo "Job ended at $(date)"