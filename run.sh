#!/bin/bash
#SBATCH --nodes 1
#SBATCH --time 3:00:00
#SBATCH --mem-per-cpu 32G
#SBATCH --cpus-per-task=2
#SBATCH --job-name tunnel
#SBATCH --output output/slurm-%J.txt


python test.py