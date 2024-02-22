#!/bin/bash
#
#SBATCH --job-name=write_out
#SBATCH --output=logs/write_out.out
#SBATCH --error=logs/write_out.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joanplepi@gmail.com
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128GB
#SBATCH --gres=gpu:0
#SBATCH --time=48:00:00

source /home/plepi/anaconda3/etc/profile.d/conda.sh
conda activate perception
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/plepi/anaconda3/lib/


python -m scripts.write_out \
    --output_base_path "data/"\
    --tasks "bbq" \
    --sets "test" \
    --num_fewshot 0 \
    --num_examples 10 \