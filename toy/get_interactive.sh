#!/bin/bash

srun --partition=boost_usr_prod --account=tra25_castiel2 --reservation=s_tra_castiel2 --time=1:10:00 --nodes=1 --ntasks-per-node=2 --gpus-per-node=2 --cpus-per-task=8 --pty /bin/bash -i
module load profile/deeplrn cineca-ai/4.3.0