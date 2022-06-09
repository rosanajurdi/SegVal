#!/bin/bash
#SBATCH --job-name=Brain-fold1
#SBATCH --partition=gpu-volta
#SBATCH --gres=gpu:1
#SBATCH --time=400:00:00
#SBATCH --mem=64G
#SBATCH --mail-user rosana.eljurdi@icm-institute.org
#SBATCH --cpus-per-task=10
#SBATCH --chdir=/network/lustre/iss02/aramis/users/rosana.eljurdi
#SBATCH --output=/network/lustre/iss02/aramis/users/rosana.eljurdi/SegVal_Project/Training/logs/output.o%J
#SBATCH --error=/network/lustre/iss02/aramis/users/rosana.eljurdi/SegVal_Project/Training/logs/error.e%J
conda activate Collab_Contour
python3 /network/lustre/iss02/aramis/users/rosana.eljurdi/SegVal_Project/Training/main_brain_1.py