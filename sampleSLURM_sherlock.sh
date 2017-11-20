#!/bin/bash
#SBATCH --job-name=runmain
#SBATCH -o dump/jobid%Anumber%a.o
#SBATCH -e dump/jobid%Anumber%a.e
#SBATCH --time=6:00:00
#SBATCH --nodes=1
# how much memory --mem=200000
#SBATCH --mem=10000
# which queue --qos=bigmem --partition=bigmem
#SBATCH --qos=normal

python -u main.py > out_main

