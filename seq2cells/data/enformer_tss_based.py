"""
Implements gene/tss based embedding datasets.
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
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset

from seq2cells.utils.data_seq import (
    FastaIntervalPysam,
    identity,
    one_hot_reverse_complement,
    seq_indices_reverse_complement,
    str_to_one_hot,
    str_to_seq_indices,
)


@dataclass
class EnformerTssDataset(Dataset):
    """Enformer TSS based dataset. Emb2Tar dataset"""

    def __init__(
        self,
        targets_file: str,
        target_index: List[int],
        inputs_file: str,
        data_split: str,
        input_type: str,
        covered_indices: Union[np.ndarray, List[int]],
    ) -> None:
        """Initialise the dataset

        Parameters
        ----------
        targets_file: str
            Path to .h5 file holding the observed targets in hdf5 format.
            Expects the keys: ['targets_train', 'targets_test', 'targets_valid']
            to be present. Each holding a numpy array.
        target_index: List[int]
            List of integers indicating which targets are trained against.
        inputs_file: str
            Path to .h5 file holding the
        data_split: str ['train', 'valid', 'test']
            String indicating which set to load in the dataset.
            Data are stored under separate keys in the hdf5 files.
        input_type: str ['embeddings', 'pred_targets']
            Indicate what type of inputs to use. Stored are embeddings and
            predicted targets.
        covered_indices: Union[np.ndarray, List[int]]
            Indices to subset the pre-calculated embeddings (predicted targets).
            E.g. to indicate for which of them Enformer windows cover TSS.
        """
        super().__init__()

        self.data_split = data_split
        self.input_type = input_type
        self.covered_indices = covered_indices
        self.target_index = target_index

        # init targets
        # select targets matching the set
        self.targets = h5py.File(targets_file, "r")[f"targets_{self.data_split}"]
        # subset to targets/classes of interest
        self.targets = self.targets[:, self.target_index]

        # init inputs (embeddings or predicted targets)
        assert self.input_type in [
            "embeddings",
            "pred_targets",
        ], "No valid input_type selected! Must be 'embeddings' or 'pred_targets'"

        if self.input_type == "pred_targets":
            self.inputs = pd.read_hdf(inputs_file, "tar", "r")
        else:
            self.inputs = pd.read_hdf(inputs_file, "emb", "r")
        # subset to covered indices
        self.inputs = self.inputs.loc[self.covered_indices, :]

    def __len__(self):
        """Get length of the dataset"""
        return len(self.targets)

    def __getitem__(self, idx: int):
        """Get one input and target based on index"""
        x = Tensor(self.inputs.iloc[idx, :].values)
        y = Tensor(self.targets[idx, :])

        return x, y


@dataclass
class AnnDataEmbeddingTssDataset(Dataset):
    """AnnData Embedding based TSS dataset.

    Notes
    -----
    Creates a dataset using pre-computed embeddings as input.
    Requires stored in the .varm key 'seq_embedding' of the AnnData object.
    """

    def __init__(
        self,
        ann_in: Union[str, ad.AnnData],
        data_split: str,
        split_name: str,
        subset_genes_col: Optional[str] = None,
        use_layer: Optional[str] = None,
        backed_mode: Optional[bool] = False,
    ) -> None:
        """Initialise the dataset

        Parameters
        ----------
        ann_in: Union[str, ad.AnnData]
            Path to .h5ad AnnData file.
        data_split: str ['train', 'valid', 'test', 'all']
            String indicating which set to load in the dataset.
            Data are stored under separate keys in the hdf5 files.
        split_name: str
            Name of the observations column that indicated which
            gene belongs to train/test/valdiation set
        subset_genes_col: Optional[str] = None
            Name of observations columns (the column must be bool or 0/1) to
            further filter the genes.
        use_layer: Optional[str]=None
            Name of the layer of the anndata object to use as observed counts.
            Will use anndata.X if None supplied
        backed_mode: Load AnnData partially in backed mode. For large files.
        """
        super().__init__()

        self.data_split = data_split
        self.subset_genes_col = subset_genes_col
        if self.subset_genes_col == "None":
            self.subset_genes_col = None

        # load AnnData object from file if provided as path string
        if isinstance(ann_in, str):
            if backed_mode:
                # optionally in backed mode
                self.anndata = ad.read(ann_in, backed="r")
            else:
                self.anndata = ad.read(ann_in)
        else:
            self.anndata = ann_in

        # set layer of interest as default observed layer
        if use_layer and use_layer != "None" and use_layer != "X":
            assert (
                use_layer in self.anndata.layers
            ), "Specified layer is not in the provided adata object!"
            self.anndata.X = self.anndata.layers[use_layer]

        # check if to further subset the genes and if subset column is in obs
        if self.subset_genes_col is not None:
            assert (
                self.subset_genes_col in self.anndata.obs_keys()
            ), f"{self.subset_genes_col} is not a valid column in anndata.obs"

        # get indices matching the desired set
        if self.data_split != "all":
            if self.subset_genes_col is not None:
                use_idx = self.anndata.obs[
                    (self.anndata.obs[split_name] == self.data_split)
                    & (self.anndata.obs[subset_genes_col])
                ].index
            else:
                use_idx = self.anndata.obs[
                    self.anndata.obs[split_name] == self.data_split
                ].index
        else:
            if self.subset_genes_col is not None:
                use_idx = self.anndata.obs[self.anndata.obs[subset_genes_col]].index
            else:
                use_idx = self.anndata.obs.index

        # subset targets
        self.targets = self.anndata[use_idx, :]
        # init inputs from embeddings embedding - subset to set of interest
        self.inputs = self.anndata[use_idx, :].obsm["seq_embedding"]

    def __len__(self):
        """Get length of the dataset"""
        return self.targets.shape[0]

    def __getitem__(self, idx: int):
        """Get one input and target based on index

        Parameters
        ----------
        idx: int
            Index (of gene) to retrieve.

        Returns
        -------
        x: torch.Tensor
            A torch tensor of length [embedding_dim].
            E.g. for Enformer embeddings 3072.
        y: torch.Tensor
            A torch tensor of length [num_cells].
        """
        x = Tensor(self.inputs.iloc[idx, :].values)
        y = Tensor(self.targets.chunk_X(select=[idx])[0])

        return x, y


@dataclass
class GenomeInterval2EnformerTargetDataset(Dataset):
    """Genome interval dataset with Enformer targets.

    Notes
    -----
    Adapted from pytorch enformer implementation to use Enformer query windows and
    hold and load target data as well as the DNA sequence.
    Hold a genome interval data set and corresponding targets.
    On query retrieves the sequence as one-hot-encoded sequence.
    """

    def __init__(
        self,
        query: Union[pd.DataFrame, str],
        fasta_file: str,
        context_length: int,
        pos_base: int,
        targets_file: str,
        target_index: List[int],
        data_split: str,
        covered_indices: Union[np.ndarray, List[int]],
        shift_augs: Optional[List[int]] = None,
        rc_force: Optional[bool] = False,
        rc_aug: Optional[bool] = False,
        return_augs: Optional[bool] = False,
        return_seq_indices: Optional[bool] = False,
        filter_df_fn: Optional[Callable] = identity,
    ) -> None:
        """Initialise the dataset

        Parameters
        ----------
        query: Union[pd.DataFrame, str],
            Enformer window query file/dataframe indicating the sequence window of
            the ROI/TSS regions. Can be path to query file or a pandas data frame
            with the query loaded.
        fasta_file: str,
            Path to fasta file. Needs matching .fai index file.
        context_length: int,
            Context length of the DNA sequence windows in base pairs.
        pos_base: int,
            If the query is in 0 or 1 based format.
        targets_file: str,
            Path to targets .h5 file
        target_index: List[int],
            List of integer indices indicating which targets
            (columns / cell types /experiments)
            of the extracted Enformer targets to load.
        data_split: str ['train', 'valid', 'test']
            String indicating which set to load in the dataset.
            Data are stored under separate keys in the hdf5 files.
        covered_indices: list[int]
            Indices to subset the pre-calculated embeddings (predicted targets).
            E.g. to indicate for which of them Enformer windows cover TSS.
        shift_augs: Pptional[List[int]] = None,
            List of integers +- bps to randomly apply/choose from
            for shifting the DNA sequence window.
        rc_force: Optional[bool] = False,
            If to always apply revers complement augmentation
        rc_aug: Optional[bool] = False,
            If to randomly apply reverse complement augmentation.
        return_augs: Optional[bool] = False,
            If to return the applied augmentations alongside the
            DNA sequence (and the targets).
        return_seq_indices: Optional[bool] = False,
            If to retrun the sequnce as indices rather than one hot encoded.
        filter_df_fn: Optional[Callable] = identity,
            A function to filter the queries.
        """
        super().__init__()

        self.data_split = data_split
        self.target_index = target_index
        # init targets
        # select targets matching the set
        self.targets = h5py.File(targets_file, "r")[f"targets_{self.data_split}"]
        # subset to targets/classes of interest
        self.targets = self.targets[:, self.target_index]

        if isinstance(query, pd.DataFrame):
            df = query
        else:
            # read from file
            query_path = Path(query)
            assert query_path.exists(), "path to .bed file must exist"

            df = pd.read_csv(str(query_path), sep="\t")

        # filter dataframe based on custom function
        df = filter_df_fn(df)

        # ensure right base index is used
        assert pos_base in [0, 1], "pos_base must be 0 or 1!"
        # remove 1 from start coordinate to confirm with bed format
        if pos_base == 1:
            df.iloc[:, 1] -= 1

        # subset to covered indices
        df = df.iloc[covered_indices, :]

        self.df = df

        self.fasta = FastaIntervalPysam(
            fasta_file=fasta_file,
            context_length=context_length,
            return_seq_indices=return_seq_indices,
            shift_augs=shift_augs,
            rc_force=rc_force,
            rc_aug=rc_aug,
        )

        self.return_augs = return_augs

    def __len__(self):
        """Get the length of the interval dataset/frame"""
        return len(self.df)

    def __getitem__(self, ind):
        """Retrieve sequence and targets matching the interval by index"""
        interval = self.df.iloc[ind, :]

        seq = self.fasta(
            interval["chr"],
            interval["seq_start"],
            interval["seq_end"],
            interval["seq_strand"],
            return_augs=self.return_augs,
        )
        tar = Tensor(self.targets[ind, :])

        if self.return_augs:
            return seq[0], tar, (seq[1], seq[2], seq[3])
        else:
            return seq, tar


@dataclass
class Variant2EnformerDataset(Dataset):
    """Dataset linking genomic variants with Enformer seq windows.

    Notes
    -----
    Designed to hold individual genetic variants linked with enformer like
    sequence windows. Each variant may be linked to one or more sequence
    windows, e.g. linked to one gene (eQTL) or multiple candidate genes (
    e.g. GWAS) and thus either linked to a single enformer sequence
    window or multiple variant x seq_window pairs per variant.

    The dataset also holds the link to a corresponding fasta file so that
    retrieving a variant retrieves the DNA sequence.

    Note: var position and start stop of seq_window are stored 0-based
     (bed format) in the internal data frame.

    Assumes variants are encoded on the + strand.
    """

    def __init__(
        self,
        var_query: Union[pd.DataFrame, str],
        fasta_file: str,
        context_length: int,
        var_pos_base: int,
        seq_pos_base: int,
        strict_mode: bool,
        strand_specific: Optional[bool] = False,
        return_seq_indices: Optional[bool] = False,
        filter_df_fn: Optional[Callable] = identity,
    ) -> None:
        """Initialise the dataset

        Parameters
        ----------
        var_query: Union[pd.DataFrame, str],
            Variant matched with enformer window query file/dataframe
            indicating the variant id pos and ref alt and the sequence
            window matched to the variants. Can be path to query file
            or a pandas data. Requires the following columns:
            "var_id" "var_pos" "var_ref" "var_alt"
            'seq_start', 'seq_end', 'seq_strand',
            'group_id', 'center', 'strands_roi', 'positions_roi', 'bins_roi'
        fasta_file: str,
            Path to fasta file. Needs matching .fai index file.
        context_length: int,
            Context length of the DNA sequence windows in base pairs.
        var_pos_base: int,
            If the variant position is in 0 or 1 based format.
        seq_pos_base: int,
            If the variant position is in 0 or 1 based format.
        strict_mode: bool
            True/False if to run in strict mode (throw error if the provided
            reference does not match the reference genome base at the
            indicated position) or not in strict mode (overwrite the found
            base at the indicated position)
        strand_specific: bool = False,
            Default = False extracts the + strand sequence.
            True -> extracts the sequence according to the strand specified
            in the sequence query. Reverse complements minus strand sequences.
        return_seq_indices: Optional[bool] = False,
            If to return the sequence as indices rather than one hot encoded.
        filter_df_fn: Optional[Callable] = identity,
            A function to filter the queries.
        """
        super().__init__()

        self.strand_specific = strand_specific
        self.return_seq_indices = return_seq_indices
        self.strict_mode = strict_mode

        # assign reverse complement function
        self.reverse_complement: Callable = one_hot_reverse_complement
        if self.return_seq_indices:
            self.reverse_complement: Callable = seq_indices_reverse_complement

        if isinstance(var_query, pd.DataFrame):
            df = var_query
        else:
            # read from file
            query_path = Path(var_query)
            assert query_path.exists(), "path to variant_query.tsv file must exist"

            df = pd.read_csv(str(query_path), sep="\t")

        # filter dataframe based on custom function
        df = filter_df_fn(df)

        # strip dataframe of not needed columns
        df = df.loc[
            :,
            [
                "var_id",
                "var_chr",
                "var_pos",
                "var_ref",
                "var_alt",
                "var_linked_gene_strip",
                "chr",
                "seq_start",
                "seq_end",
                "seq_strand",
                "bins_roi",
            ],
        ]

        # ensure right base index is used
        assert var_pos_base in [0, 1], "var_pos_base must be 0 or 1!"
        assert seq_pos_base in [0, 1], "seq_pos_base must be 0 or 1!"

        # remove 1 from start coordinates to conform with bed format if needed
        df["var_pos"] -= var_pos_base
        df["seq_start"] -= seq_pos_base

        self.df = df

        self.fasta = FastaIntervalPysam(
            fasta_file=fasta_file,
            context_length=context_length,
            return_seq_indices=return_seq_indices,
        )

    def __len__(self) -> int:
        """Get the length of the interval dataset/frame"""
        return len(self.df)

    def __getitem__(self, ind: int) -> Tuple[Tensor, Tensor, int]:
        """Retrieve sequence and targets matching the interval by index

        Parameter
        ---------
        ind: int
            Index in the variant query dataframe to retrieve.

        Returns
        -------
        seq_ref: Tensor
            Sequence of the reference sequence matching the
            seq_window length. Either as one-hot-coded or as sequence indices.
        seq_var: Tensor
            Sequence of the variant/alternative sequence matching the
            seq_window length. Either as one-hot-coded or as sequence indices.
        roi_bin_idx: int
            The index of the bin of interest of the trunk embedding,
            e.g. where the gene tss is located.
        """
        seq_ref, roi_bin_ref = self.get_seq(ind, var=False)
        seq_var, roi_bin_var = self.get_seq(ind, var=True)

        # assert that the same bin index is desired fom the trunk output
        assert roi_bin_ref == roi_bin_var, (
            f"Different bins of the trunk output are desired for index {ind}\n"
            f"ref: {roi_bin_ref} and alt:{roi_bin_var}"
            f"They must be equal!"
        )

        return seq_ref, seq_var, roi_bin_ref

    def get_seq(self, ind: int, var: bool) -> Tensor:
        """Get the sequence with ref or var base

        Parameter
        ---------
        ind: int
            Index in the variant query dataframe to retrieve.
        var: bool
            False - extract the reference genome sequence matched with the
            reference base provided. (inserted in non strict mode, asserted
            in strict mode).
            True - insert the recorded variant into the retrieved sequence.

        Returns
        -------
        seq: Tensor
            Sequence of the reference or variant sequence matching the
            seq_window length. Either as one-hot-coded or as sequence indices.
        roi_bin_idx: int
            The index of the bin of interest of the trunk embedding,
            e.g. where the gene tss is located.
        """
        interval = self.get_query_by_index(ind)

        # grep reference or variant base
        if var:
            allele_of_interest: str = interval["var_alt"]
        else:
            allele_of_interest: str = interval["var_ref"]

        # fetch forward or strand specific sequence
        if self.strand_specific:
            use_strand = interval["seq_strand"]
        else:
            use_strand = "+"

        # extract (+ strand always)
        seq = self.fasta(
            interval["chr"],
            interval["seq_start"],
            interval["seq_end"],
        )

        # match with provided ref
        # distance from seq_start to var_pos
        var_pos_rel = interval["var_pos"] - interval["seq_start"]

        if self.return_seq_indices:
            allele_of_interest = str_to_seq_indices(allele_of_interest)
        else:
            allele_of_interest = str_to_one_hot(allele_of_interest)

        # if in strict mode ensure base matches
        if self.strict_mode and not var:
            assert all(allele_of_interest[0] == seq[var_pos_rel]), (
                f"The reference base {allele_of_interest} provided does not "
                f"match the reference genome which is {seq[var_pos_rel]} at "
                f"variant {interval}\n Stopping since in strict mode!"
            )

        # replace the base at the position of interest with the one provided
        seq[var_pos_rel] = allele_of_interest

        # reverse complement the seq if on minus strand and strand_specific mode
        if self.strand_specific and use_strand == "-":
            seq = self.reverse_complement(seq)

        # get the roi bin of interest to use to extract the correct embedding
        # from the trunk output
        roi_bin_idx = interval["bins_roi"]
        # ensure it is a single roi_bin_idx, stop if multiple found for now
        if isinstance(roi_bin_idx, list):
            assert len(roi_bin_idx) == 1, (
                "Only a single bin of interest to extract from the trunk "
                "output is supported at the moment. Check your 'bins_roi' "
                "column in the var_query_df!"
            )
            roi_bin_idx = roi_bin_idx[0]

        return seq, roi_bin_idx

    def get_query_by_index(self, ind: int) -> pd.Series:
        """Get entry in the query dataframe"""
        return self.df.loc[ind, :]

    def get_query_by_var_id(self, id: str) -> pd.DataFrame:
        """Get entry in the query dataframe"""
        return self.df.loc[self.df["var_id"] == id, :]
