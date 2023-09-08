#!/bin/bash

# Define Paths ==================
script_dir='../seq2cells/scripts'
output_dir='.'

cd ${output_dir}

# Inputs ==============================================================
# TSS (ROI) in bed format
tss='./resources/sequence_data/toy_tss.bed'
# Enformer sequences
enf_seqs='./resources/sequence_data/enformer_sequence.bed'
# single cell data
sc_anndata='./resources/single_cell_data/pbmc_toy_example_orig.h5ad'

# if pre-computing the sequence embeddings yourself a matching
# reference genome as fasta file (.fa)
# and index (.fa.fai file in same directory).
# Download hg38 from e.g. https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/
# wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
# index with e.g. samtools http://www.htslib.org/doc/samtools-faidx.html
# path to fasta file
#fasta_file='hg38.fa'
fasta_file=$1

# (Intermediate) Outputs =============================================
query_file='toy_query_tss_out.tsv'
enformer_out_name='toy_seq_embeddings_out'
set_ids='intersect_out'
sc_anndata_out='pbmc_toy_example_with_emb_out'

# ==================
echo "=== Running creating query from regions of interest ==="
python ${script_dir}/preprocessing/create_seq_window_queries.py \
    --in ${tss} \
    --ref_genome ${fasta_file} \
    --out ${query_file} \
    --chromosome_col 1\
    --position_col 3\
    --position_base 0 \
    --strand_col 6 \
    --group_id_col 7 \
    --additional_id_col 8 \
    --no-stitch

# ==================
# Note this step is computationally expensive and requires
# downloading the full Enformer weights (done automatically)
# you may want to skip this step and use the supplied
# precomputed embeddings 'resources/embeddings/toy_seq_embeddings.h5'
embeddings_file='./resources/embeddings/toy_seq_embeddings.h5'
# else uncomment embedding run (below) and set new embeddings_file name
#echo "=== Running computing Enformer embeddings ==="
#embeddings_file="${enformer_out_name}.h5"
#python ${script_dir}/preprocessing/calc_embeddings_and_targets.py \
#--in_query ${query_file} \
#--ref_genome ${fasta_file} \
#--out_name ${enformer_out_name} \
#--position_base 1 \
#--add_bins 0 \
#--store_h5 \
#--targets '4675:5312'  # for all Enformer CAGE-seq targets

# ==================
echo "=== Running intersection with Enformer to mark\
train/test/valid ==="
python ${script_dir}/preprocessing/intersect_queries_with_enformer_regions.py \
--query ${query_file} \
--enf_seqs ${enf_seqs} \
--out_name ${set_ids} \
--strip

# ==================
echo "=== Running adding sequence embeddings to single cell data ==="
python ${script_dir}/preprocessing/add_embeddings_to_anndata.py \
  --query ${query_file} \
  --anndata ${sc_anndata} \
  --emb ${embeddings_file} \
  --valid_ids "${set_ids}_valid.txt" \
  --test_ids "${set_ids}_test.txt" \
  --out_name "${sc_anndata_out}" \
  --strip

# Your directory should look like this afterwards =========
# ls
# seq2cells  hg38.fa  hg38.fa.fai  intersect_out_test.txt
# intersect_out_train.txt  intersect_out_valid.txt  pbmc_toy_example_with_emb_out.h5ad  toy_query_tss_out.tsv
# toy_seq_embeddings_out.h5 (- only if run Enformer yourself)
#
# head *tsv *txt
#==> toy_query_tss_out.tsv <==
#chr     seq_start       seq_end seq_strand      patch_id        group_id        add_id  center  num_roi stretch strands_roi     positions_roi   bins_roi
#chr1    11556510        11753117        +       ENSG00000132879.14_0    ENSG00000132879.14      FBXO44  11654878        1       0       ['+']   11654878        448
#chr1    26274383        26470990        -       ENSG00000176083.18_0    ENSG00000176083.18      ZNF683  26372751        1       0       ['-']   26372751        448
#chr1    33083676        33280283        -       ENSG00000116525.14_0    ENSG00000116525.14      TRIM62  33182044        1       0       ['-']   33182044        448
#chr1    88966841        89163448        -       ENSG00000117228.11_0    ENSG00000117228.11      GBP1    89065209        1       0       ['-']   89065209        448
#chr1    149969409       150166016       +       ENSG00000136631.16_0    ENSG00000136631.16      VPS45   150067777       1       0       ['+']   150067777       448
#chr1    173770013       173966620       +       ENSG00000185278.16_0    ENSG00000185278.16      ZBTB37  173868381       1       0       ['+']   173868381       448
#chr1    203763232       203959839       +       ENSG00000182004.13_0    ENSG00000182004.13      SNRPE   203861600       1       0       ['+']   203861600       448
#chr10   70790198        70986805        -       ENSG00000166228.9_0     ENSG00000166228.9       PCBD1   70888566        1       0       ['-']   70888566        448
#chr10   102297338       102493945       +       ENSG00000077150.21_0    ENSG00000077150.21      NFKB2   102395706       1       0       ['+']   102395706       448
#
#==> intersect_out_test.txt <==
#ENSG00000073008
#ENSG00000100603
#ENSG00000105732
#ENSG00000119630
#ENSG00000130669
#
#==> intersect_out_train.txt <==
#ENSG00000006652
#ENSG00000058056
#ENSG00000066827
#ENSG00000071127
#ENSG00000077150
#ENSG00000101773
#ENSG00000102524
#ENSG00000102543
#ENSG00000102796
#ENSG00000108370
#
#==> intersect_out_valid.txt <==
#ENSG00000136240
#ENSG00000162971
#ENSG00000168393

