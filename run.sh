#!/bin/bash
#SBATCH --nodes 2
#SBATCH --time 12:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --cpus-per-task=4
#SBATCH --job-name tunnel
#SBATCH --output=output/stdout/slurm-%J.txt
#SBATCH --error=output/stderr/slurm-%J.err

subjects=("R2488" "R2280" "R2490" "R2487")   # Add more subjects as needed

for subj in "${subjects[@]}"; do
    echo "Processing subject: $subj"
    srun --exclusive -n 1 python data_enhancement.py --subject "$subj" &
done
wait

