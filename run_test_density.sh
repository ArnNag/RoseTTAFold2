#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 2
#SBATCH --mem=64g
#SBATCH -o log
#SBATCH --gres=gpu:a4000:1
#SBATCH -t 00:15:00

python -m network.predict -model weights/RF2_jan24.pt -inputs tests/a3m/8CZC_full.a3m
