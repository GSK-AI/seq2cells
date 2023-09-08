#!/bin/bash

# Define Paths ==================
script_dir='../seq2cells/scripts/'

config_train="./test_configs/config_anndata_fine_tune.yml"

# Run Training ==================
python ${script_dir}/training/fine_tune_on_anndata.py \
config_file=${config_train}


