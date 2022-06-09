#!/bin/bash
#SBATCH --job-name=Brain-inference
#SBATCH --partition=normal
##SBATCH --gres=gpu:1
#SBATCH --time=500:00:00
#SBATCH --mem=64G
#SBATCH --mail-user rosana.eljurdi@icm-institute.org
#SBATCH --cpus-per-task=20
#SBATCH --chdir=/network/lustre/iss02/aramis/users/rosana.eljurdi
#SBATCH --output=/network/lustre/iss02/aramis/users/rosana.eljurdi/SegVal_Project/Training/logs/output_brain.o%J
#SBATCH --error=/network/lustre/iss02/aramis/users/rosana.eljurdi/SegVal_Project/Training/logs/error_brain.e%J
conda activate Collab_Contour
python3 /network/lustre/iss02/aramis/users/rosana.eljurdi/SegVal_Project/Training/compute-metric.py

