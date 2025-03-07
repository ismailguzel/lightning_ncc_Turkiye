#!/bin/bash

#SBATCH --job-name=lightning
#SBATCH --output=./out/%x.%j.out # Note: %x == job-name
#SBATCH --error=./out/%x.%j.err # %j == job_id
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=0 # request the whole memory
#SBATCH --time=0-01:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=tra25_castiel2
#SBATCH --reservation=s_tra_castiel2

mkdir -p ./out

module load profile/deeplrn cineca-ai/4.3.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python3 toy-example.py

exit
