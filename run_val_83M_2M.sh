#!/bin/bash
#
#SBATCH --job-name=equiformerv2_val
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --time=30:00:00
#SBATCH --mem=16GB
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=songxia23@gmail.com

singularity exec --nv \
            --overlay /scratch/sx801/singularity-envs/ocp-py39-50G-10M.ext3:ro \
            /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
            bash -c "source /ext3/env.sh; \
            python $1 main.py \
            --config-yml configs/s2ef/2M/equiformer_v2/equiformer_v2_N@12_L@6_M@2.yml \
            --checkpoint ./checkpoints/downloaded/eq2_83M_2M.pt \
            --mode validate "
