#!/bin/bash

# Define Paths ==================
script_dir='../seq2cells/scripts/'
test_dir='.'

config_vep="${test_dir}/test_configs/config_predict_variant_effect.yml"

# Run Prediction ==================
python ${script_dir}/inference/predict_variant_effect.py \
config_file=${config_vep} \
checkpoint_file=$1 \
fasta_file=$2

# Expected output =================
# the results should look like this with exact predictions varying depending
# on the model trained and checkpoint selected.
#
# cd vep_out
# ls
# filtered_variants.tsv matched_variants.tsv
# predictions.pt pred_ref.npy pred_var.npy
#
# head matched_variants.tsv (variants matched with sequence queries matched by
# gene ID)
#
#	var_id	var_chr	var_pos	var_ref	var_alt	var_linked_gene	var_linked_gene_strip	chr	seq_start	seq_end	seq_strand	patch_id	group_id	add_id	center	num_roi	stretch	strands_roi	positions_roi	bins_roi	group_id_stripped	_merge	var_tss_distance
#0	chr1_931513_T_C_b38	chr1	931513	T	C	ENSG00000187634.11	ENSG00000187634	chr1	825555	1022162	+	ENSG00000187634.13_0	ENSG00000187634.13	SAMD11	923923	1	0	['+']	923923	448	ENSG00000187634	both	-7590
#1	chr1_942951_C_T_b38	chr1	942951	C	T	ENSG00000187634.11	ENSG00000187634	chr1	825555	1022162	+	ENSG00000187634.13_0	ENSG00000187634.13	SAMD11	923923	1	0	['+']	923923	448	ENSG00000187634	both	-19028
#2	chr1_1050658_G_C_b38	chr1	1050658	G	C	ENSG00000188976.10	ENSG00000188976	chr1	860888	1057495	-	ENSG00000188976.11_0	ENSG00000188976.11	NOC2L	959256	1	0	['-']	959256	448	ENSG00000188976	both	-91402
#
# python
#>>> import numpy as np
#>>> a = np.load('pred_ref.npy')
#>>> a
#array([[0.54566896, 0.55636966, 0.64093524, ..., 0.6288615 , 0.5729825 ,
#        0.6204464 ],
#       [0.54566896, 0.55636966, 0.64093524, ..., 0.6288615 , 0.5729825 ,
#        0.6204464 ],
#       [0.6731915 , 0.45564386, 0.533731  , ..., 0.6135318 , 0.49362114,
#        0.7129273 ]], dtype=float32)
#>>> b = np.load('pred_var.npy')
#>>> b
#array([[0.5456814 , 0.5563059 , 0.6409684 , ..., 0.6288126 , 0.57296413,
#        0.6204777 ],
#       [0.54572475, 0.5564203 , 0.6409311 , ..., 0.6288786 , 0.573027  ,
#        0.62047726],
#       [0.6731911 , 0.4556419 , 0.53373235, ..., 0.6135369 , 0.49361694,
#        0.7129307 ]], dtype=float32)
#>>> a.shape
#(3, 2600)
#
# referring to 3 variants x 2600 cells
