#!/bin/bash

# Define Paths ==================
script_dir='../seq2cells/scripts'

config_eval="./test_configs/config_anndata_eval.yml"

# Run Predictions & Evaluation ===
python ${script_dir}/training/eval_single_cell_model.py \
config_file=${config_eval} \
checkpoint_file=$1



