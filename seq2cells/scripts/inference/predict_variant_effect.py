"""
Predict the effect of sequence variants gene expression
=========================================
Copyright 2023 GlaxoSmithKline Research & Development Limited. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=========================================

Notes
-----
This script runs variant effect predictions on single cell or celltype
specific models. It combines a model trunk (e.g. Enformer trunk) to
process a one hot encoded DNA sequence to a sequence embedding and a head
trained against bulk or single cells to predict gene expression from the
embeddings.
"""
import logging
import os
from typing import Literal, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from enformer_pytorch import Enformer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from seq2cells.data.enformer_tss_based import Variant2EnformerDataset
from seq2cells.models.embedding2target import (
    Embedding2Target,
    EnformerTrunk,
    Variant2Embedding2Target,
)
from seq2cells.utils.predict_utils import BatchWiseWriter

# set logger
logger = logging.getLogger(__name__)


# helper functions =============================
def load_variant_data(
    variant_file: str,
    variant_file_format: Literal["tsv"],
    strip_gene: bool,
    gene_colname: str,
    stripped_gene_colname: str,
) -> pd.DataFrame:
    """Load sequence variants from file

    Parameter
    ---------
    variant_file: str
        Path to variant file to read.
    variant_file_format: Literal['tsv']
        File format of the variant file to adapt reading in.
    strip_gene: bool
        If to strip a gene id column of the TSS(isoform) identifier.
        Everything from and including the .
    gene_colname: str
        Name of the gene id column to use.
    stripped_gene_colname: str
        Name to give the stripped gene id column.

    Return
    ------
    var_df: pd.DataFrame
        A six or column dataframe listing the id, chr, position,
         reference_allele and variant_allele for each variant.

    Notes
    -----
    Optionally, strip the variant gene id of the TSS identified (e.g. .1234).
    """
    if variant_file_format == "tsv":
        var_df = pd.read_csv(
            variant_file,
            sep="\t",
            header=None,
            names=[
                "var_id",
                "var_chr",
                "var_pos",
                "var_ref",
                "var_alt",
                "var_linked_gene",
            ],
        )
    else:
        # no other formats supported yet
        pass

    if strip_gene:
        # strip tss identifier per gene
        var_df[stripped_gene_colname] = var_df[gene_colname].str.replace(
            r"\.\w+", "", regex=True
        )

    return var_df


def load_sequence_window_query(
    seq_query_file: str, strip_gene: bool, gene_colname: str, stripped_gene_colname: str
) -> pd.DataFrame:
    """Load sequence windows query from file

    Parameter
    ---------
    seq_query_file: str
        Expects a tab separate file listing:
        chr, seq_start, seq_end, seq_strand, patch_id, group_id,
        add_id, center, num_roi, stretch, strands_roi,
        positions_roi, bins_roi
    strip_gene: bool
        If to strip a gene id column of the TSS(isoform) identifier.
        Everything from and including the .
    gene_colname: str
        Name of the gene id column to use.
    stripped_gene_colname: str
        Name to give the stripped gene id column.

    Return
    ------
    seq_query_df: pd.DataFrame
        13 or 14 column dataframe with the same information as
        the query file expects as input + optionally an additional
        column for the stripped gene id.

    Notes
    -----
    Optionally, strip the gene id of the TSS identified (e.g. .1234).
    """
    seq_query_df = pd.read_csv(seq_query_file, sep="\t")

    # strip the tss identifier per gene id
    if strip_gene:
        seq_query_df[stripped_gene_colname] = seq_query_df[gene_colname].str.replace(
            r"\.\w+", "", regex=True
        )

    return seq_query_df


def combine_variants_with_seq_windows(
    var_df: pd.DataFrame,
    seq_query_df: pd.DataFrame,
    merge_var_on: str,
    merge_seq_on: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Link and merge variants with sequence windows

    Parameter
    ---------
    var_df: pd.DataFrame
        Variant data frame. Expects a column whos name matches
        merge_var_on. WIll usually have more columns for example:
        var_id, var_chr, var_pos, var_ref, var_alt, var_linked_gene
        var_linked_gene_strip
    seq_query_df: pd.DataFrame
        Sequence window dataframe. Expects a column whos name matches
        merge_seq_on. WIll usually have more columns for example:
        chr, seq_start, seq_end, seq_strand, patch_id, group_id,
        add_id, center, num_roi, stretch, strands_roi,
        positions_roi, bins_roi, stripped_gene_id
    merge_var_on: str
        Column name in var_df to use for joining the data frames.
    merge_seq_on: str
        Column name in seq_query_df to use for joining the data frames.

    Return
    ------
    var_query_df: pd.DataFrame
        Merged variant and sequence window data frame with at least,
        but usually more than, the following columns:
            '_merge' - indicating how that row entry was merged
        Example:
        var_id, var_chr, var_pos, var_ref, var_alt,
        var_linked_gene, var_linked_gene_strip,
        chr, seq_start, seq_end, seq_strand,
        patch_id, group_id, add_id, center, num_roi, stretch,
        strands_roi,  positions_roi,  bins_roi, group_id_stripped, _merge

    unmatched: pd.DataFrame
        Same as var_query_df but all the antries that do not have a "both"
        entry in the _merge column.
    """
    var_query_df = var_df.merge(
        seq_query_df,
        left_on=merge_var_on,
        right_on=merge_seq_on,
        how="left",
        indicator=True,
    )

    # split unmatched from matched
    unmatched = var_query_df.loc[var_query_df["_merge"] != "both", :]
    logger.info(
        f"Could not match {unmatched.shape[0]} variants to sequence " f"windows!"
    )

    # continue with rest
    var_query_df = var_query_df.loc[var_query_df["_merge"] == "both", :]

    # cast integer columns back to integer
    # pandas behavior in case of unmatched records
    # https://github.com/pandas-dev/pandas/issues/9958
    for column in [
        "seq_start",
        "seq_end",
        "center",
        "num_roi",
        "stretch",
        "positions_roi",
        "bins_roi",
    ]:
        var_query_df[column] = var_query_df[column].astype("Int64")

    return var_query_df, unmatched


def filter_out_of_reach_variants(
    df: pd.DataFrame,
    oor_threshold: int,
    var_pos_column: str,
    roi_pos_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter out of reach variants from dataframe

    Parameter
    ---------
    df: pd.DataFrame
        Matched variant seq window data frame.
        Expects a columns whos names match var_pos_column and
        roi_pos_column. Will usually have more columns for example:
        var_id, var_chr, var_pos, var_ref, var_alt,
        var_linked_gene, var_linked_gene_strip,
        chr, seq_start, seq_end, seq_strand,
        patch_id, group_id, add_id, center, num_roi, stretch,
        strands_roi,  positions_roi,  bins_roi, group_id_stripped, _merge
    oor_threshold: int
        Distance threshold at or below which a variant is considered to be
        within reach of a ROI (e.g. TSS)
    var_pos_column: str
        Name of the column that contains the variant genomic position
    roi_pos_column: str
        Name of the column that contains the ROI (e.g. TSS) genomic position

    Return
    ------
    df: pd.DataFrame
        Matched variant and sequence window data frame filtered for variants
        whose genomic position is below a threshold of distance from the TSS
        of the matched gene. Example:
        var_id, var_chr, var_pos, var_ref, var_alt,
        var_linked_gene, var_linked_gene_strip,
        chr, seq_start, seq_end, seq_strand,
        patch_id, group_id, add_id, center, num_roi, stretch,
        strands_roi,  positions_roi,  bins_roi, group_id_stripped, _merge

    oor_variants: pd.DataFrame
        Same as var_query_df but all the entries whose variant is out of
        reach of the ROI (e.g. TSS) are int this data frame.
    """
    oor_variants = df.loc[
        abs(df[var_pos_column] - df[roi_pos_column]) >= oor_threshold,
        :,
    ]

    # keep within reach variants
    df = df.loc[
        abs(df[var_pos_column] - df[roi_pos_column]) < oor_threshold,
        :,
    ]

    return df, oor_variants


def filter_indels(
    df: pd.DataFrame,
    ref_base_column: str,
    var_base_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter indels from a dataframe

    Parameter
    ---------
    df: pd.DataFrame
        Matched variant seq window data frame.
        Expects a columns whos names match ref_base_column and
        var_base_column. Will usually have more columns for example:
        var_id, var_chr, var_pos, var_ref, var_alt,
        var_linked_gene, var_linked_gene_strip,
        chr, seq_start, seq_end, seq_strand,
        patch_id, group_id, add_id, center, num_roi, stretch,
        strands_roi,  positions_roi,  bins_roi, group_id_stripped, _merge
    ref_base_column: str
        Name of the column that contains the reference base.
    var_base_column: str
        Name of the column that contains reh variant base.

    Return
    ------
    df: pd.DataFrame
        Matched variant and sequence window data frame filtered for variants
        that are not single base pair exchanges. Example:
        var_id, var_chr, var_pos, var_ref, var_alt,
        var_linked_gene, var_linked_gene_strip,
        chr, seq_start, seq_end, seq_strand,
        patch_id, group_id, add_id, center, num_roi, stretch,
        strands_roi,  positions_roi,  bins_roi, group_id_stripped, _merge

    bp_variants: pd.DataFrame
        Same as var_query_df but all the entries which are not single base
        pair exchanges are in this dataframe.
    """
    # ensure deletions are not encoded as .
    df[ref_base_column] = df[ref_base_column].str.replace(".", "", regex=False)
    df[var_base_column] = df[var_base_column].str.replace(".", "", regex=False)

    # save non single base pair exchanges.
    bp_fil_variants = df.loc[
        (df[ref_base_column].str.len() != 1) | (df[var_base_column].str.len() != 1)
    ]

    # filter keep all others
    df = df.loc[
        (df[ref_base_column].str.len() == 1) & (df[var_base_column].str.len() == 1)
    ]

    return df, bp_fil_variants


# Run ==========================================
def main(config: dict):
    """Run predictions."""
    if config["debug"]:
        logging.basicConfig(level=logging.INFO)

    # create output directory if not present
    if not os.path.exists(config["saving"]["output_path"]):
        os.makedirs(config["saving"]["output_path"])

    # 1) Load and format data ===========
    # load variants
    logger.info("Loading variant data ...")

    assert config["data"]["variants"]["variant_file_format"] in ["tsv"], (
        f"The selected variant file format "
        f"{config['data']['variants']['variant_file_format']} "
        f"is currently not supported."
    )

    var_df = load_variant_data(
        config["data"]["variants"]["variant_file"],
        variant_file_format=config["data"]["variants"]["variant_file_format"],
        strip_gene=config["data"]["variants"]["strip_id"],
        gene_colname="var_linked_gene",
        stripped_gene_colname="var_linked_gene_strip",
    )

    # load the seq_window queries matching the roi/tss
    logger.info("Loading sequence window query data:")

    seq_query_df = load_sequence_window_query(
        config["data"]["sequence"]["seq_window_query"],
        strip_gene=True,
        gene_colname="group_id",
        stripped_gene_colname="group_id_stripped",
    )

    # combine variants and seq windows ===========
    logger.info("Combine variants and sequence windows ...")

    var_query_df, unmatched = combine_variants_with_seq_windows(
        var_df,
        seq_query_df,
        merge_var_on="var_linked_gene_strip",
        merge_seq_on="group_id_stripped",
    )

    # filter out matched but out-of-seq_context reach variants
    logger.info("Filtering variants that are out of reach of TSS ...")

    var_query_df, oor_variants = filter_out_of_reach_variants(
        var_query_df,
        oor_threshold=config["data"]["sequence"]["seq_context_length"] / 2,
        var_pos_column="var_pos",
        roi_pos_column="positions_roi",
    )

    # filter out indels (non single basepair variants)
    logger.info("Filtering InDels ...")

    var_query_df, bp_fil_variants = filter_indels(
        var_query_df, ref_base_column="var_ref", var_base_column="var_alt"
    )

    # combine and save filtered variants
    filtered_variants = pd.concat(
        [unmatched, oor_variants, bp_fil_variants],
        keys=["unmatched", "out_of_reach", "indel"],
    )

    if config["saving"]["save_filtered_variants"]:
        filtered_variants_file_path = os.path.join(
            config["saving"]["output_path"],
            config["saving"]["save_filtered_variants_file"],
        )
        logger.info(
            f"Writing filtered out variants to " f"{filtered_variants_file_path} ..."
        )
        filtered_variants.to_csv(filtered_variants_file_path, sep="\t")

    # reset index to only the matched variants
    var_query_df.reset_index(drop=True, inplace=True)

    # calculate variant distance to TSS
    var_query_df["var_tss_distance"] = (
        var_query_df["positions_roi"] - var_query_df["var_pos"]
    )

    # write var_query to disk
    if config["saving"]["save_matched_variants"]:
        matched_variant_file_path = os.path.join(
            config["saving"]["output_path"],
            config["saving"]["save_matched_variants_file"],
        )
        logger.info(f"Writing matched variants to {matched_variant_file_path} ...")
        var_query_df.to_csv(matched_variant_file_path, sep="\t")

    # build dataset
    variant_dataset = Variant2EnformerDataset(
        var_query=var_query_df,
        fasta_file=config["data"]["sequence"]["reference_genome"],
        context_length=config["data"]["sequence"]["seq_context_length"],
        var_pos_base=config["data"]["variants"]["var_pos_base"],
        seq_pos_base=config["data"]["sequence"]["seq_pos_base"],
        strict_mode=config["task"]["input"]["strict_mode"],
        strand_specific=config["task"]["input"]["strand_specific_mode"],
    )

    # build Dataloader
    batch_size = config["resource"]["pred_batch_size"]
    if batch_size is None:
        batch_size = len(variant_dataset)

    variant_dataloader = DataLoader(
        variant_dataset, batch_size=batch_size, shuffle=False
    )

    # 2) load and combine models ==============
    # load seq2emb trunk
    trunk = EnformerTrunk(
        Enformer.from_pretrained(config["model"]["trunk"]["model_trunk_chkpt_path"]),
    )

    # load emb2tar head
    head = Embedding2Target.load_from_checkpoint(
        config["model"]["head"]["model_head_chkpt_path"]
    )

    # Combine
    model = Variant2Embedding2Target(
        trunk, head, add_bin=config["task"]["merge_adjacent_bins"]
    )
    model.eval()

    # 3) Setup Trainer for predictions ========
    callbacks_to_use = []

    # currently blocking fro running in temp_write_pred mode
    # moving this to an individual PR
    assert not config["resource"][
        "temp_write_pred"
    ], "Predicting in temp_write_pred mode is currently not supported."
    if config["resource"]["temp_write_pred"]:
        # set up batch wise prediction for writing to disk
        logger.info("Writing predictions in batches to disk ...")
        pred_writer = BatchWiseWriter(
            output_dir=config["saving"]["output_path"], write_interval="batch"
        )
        callbacks_to_use.append(pred_writer)

    # enable profiling
    profiler = None
    if config["resource"]["profile"]:
        from pytorch_lightning.profilers import PyTorchProfiler

        profiler = PyTorchProfiler()

    trainer = pl.Trainer(
        accelerator=config["resource"]["device"],
        callbacks=callbacks_to_use,
        devices=1,
        strategy=None,
        profiler=profiler,
        logger=False,
        inference_mode=True,
    )

    # 4) Run predictions ======================
    if config["resource"]["temp_write_pred"]:
        # predict storing in memory
        trainer.predict(model, dataloaders=variant_dataloader, return_predictions=False)
    else:
        # predict storing in memory
        predictions = trainer.predict(
            model, dataloaders=variant_dataloader, return_predictions=True
        )

    # temp save predictions ===================
    torch.save(
        predictions, os.path.join(config["saving"]["output_path"], "predictions.pt")
    )

    # 5) Assemble and save predictions =========
    if config["resource"]["temp_write_pred"]:
        # assemble from temp stored pt files
        # passing as this mode is currently not supported
        pass
    else:
        # unpack predictions and store
        ref_assemble = np.zeros(
            [len(variant_dataset), len(predictions[0][0][0])], dtype=np.float32
        )
        var_assemble = np.zeros(
            [len(variant_dataset), len(predictions[0][0][0])], dtype=np.float32
        )

        for i in range(len(predictions)):
            ref_assemble[i, :] = predictions[i][0][0].numpy()
            var_assemble[i, :] = predictions[i][1][0].numpy()

        ref_pred_path = os.path.join(
            config["saving"]["output_path"], config["saving"]["save_ref_pred_file"]
        )
        var_pred_path = os.path.join(
            config["saving"]["output_path"], config["saving"]["save_var_pred_file"]
        )

        np.save(ref_pred_path, ref_assemble)
        np.save(var_pred_path, var_assemble)


if __name__ == "__main__":
    # load config ==============================
    # Overwrite CLI args
    cli_config = OmegaConf.from_cli()
    with open(cli_config.config_file, "r") as f:
        config = yaml.safe_load(f)
    config = OmegaConf.to_object(OmegaConf.merge(config, cli_config))

    # overwrite if cli supplied checkpoint / fasta location
    if 'checkpoint_file' in cli_config:
        config['model']['head']['model_head_chkpt_path'] = cli_config.checkpoint_file
    if 'fasta_file' in cli_config:
        config['data']['sequence']['reference_genome'] = cli_config.fasta_file

    # run ======================================
    main(config=config)
