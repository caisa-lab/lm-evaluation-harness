#!/bin/bash
#
#SBATCH --job-name=bbq_eval
#SBATCH --output=logs/bbq_eval2.out
#SBATCH --error=logs/bbq_eval2.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joanplepi@gmail.com
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

source /home/plepi/anaconda3/etc/profile.d/conda.sh
conda activate perception
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/plepi/anaconda3/lib/

python llm_src/eval_llama2.py \
--model_name "meta-llama/Llama-2-7b-chat-hf" \
--tasks "bbq" \
--write_out \
--output_base_path "data/llm_eval/write_info"
# --tasks="crows_pairs_english,crows_pairs_english_race_color,crows_pairs_english_socioeconomic,crows_pairs_english_gender,\
#             crows_pairs_english_age,crows_pairs_english_religion,crows_pairs_english_disability,crows_pairs_english_sexual_orientation,\
#             crows_pairs_english_nationality,crows_pairs_english_physical_appearance,crows_pairs_english_autre,crows_pairs_french,\
#             crows_pairs_french_race_color,crows_pairs_french_socioeconomic,crows_pairs_french_gender,crows_pairs_french_age,\
#             crows_pairs_french_religion,crows_pairs_french_disability,crows_pairs_french_sexual_orientation,crows_pairs_french_nationality,\
#             crows_pairs_french_physical_appearance,crows_pairs_french_autre"
