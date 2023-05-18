#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -t 07-00:00:00
#SBATCH --mem=10g
#SBATCH -n 1

evaluationEstimator=$1
nObs=$2
w=$3

python 3_simulation1D3.py $evaluationEstimator $nObs $w