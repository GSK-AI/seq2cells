"""
Sequence dataset from genomic intervals
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

Implementation copied and adapted from enformer_pytorch
https://github.com/lucidrains/enformer-pytorch
Accessed 20/09/2022
"""
from pathlib import Path
from random import random, randrange
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# import polars as pl
import torch
import torch.nn.functional as F
from pyfaidx import Fasta
from pysam import FastaFile
from torch.utils.data import Dataset


# helper functions ==============
def check_is_not_none(val):
    """Exists helper function"""
    return val is not None


def identity(t):
    """Helper function to call the identity (no filtering)"""
    return t


def cast_list(t):
    """Return values as list if it is not a list already"""
    return t if isinstance(t, list) else [t]


# genomic function transforms ===========
seq_indices_embed = torch.zeros(256).long()
seq_indices_embed[ord("a")] = 0
seq_indices_embed[ord("c")] = 1
seq_indices_embed[ord("g")] = 2
seq_indices_embed[ord("t")] = 3
seq_indices_embed[ord("n")] = 4
seq_indices_embed[ord("A")] = 0
seq_indices_embed[ord("C")] = 1
seq_indices_embed[ord("G")] = 2
seq_indices_embed[ord("T")] = 3
seq_indices_embed[ord("N")] = 4
seq_indices_embed[ord(".")] = -1

one_hot_embed = torch.zeros(256, 4)
one_hot_embed[ord("a")] = torch.Tensor([1.0, 0.0, 0.0, 0.0])
one_hot_embed[ord("c")] = torch.Tensor([0.0, 1.0, 0.0, 0.0])
one_hot_embed[ord("g")] = torch.Tensor([0.0, 0.0, 1.0, 0.0])
one_hot_embed[ord("t")] = torch.Tensor([0.0, 0.0, 0.0, 1.0])
one_hot_embed[ord("n")] = torch.Tensor([0.0, 0.0, 0.0, 0.0])
one_hot_embed[ord("A")] = torch.Tensor([1.0, 0.0, 0.0, 0.0])
one_hot_embed[ord("C")] = torch.Tensor([0.0, 1.0, 0.0, 0.0])
one_hot_embed[ord("G")] = torch.Tensor([0.0, 0.0, 1.0, 0.0])
one_hot_embed[ord("T")] = torch.Tensor([0.0, 0.0, 0.0, 1.0])
one_hot_embed[ord("N")] = torch.Tensor([0.0, 0.0, 0.0, 0.0])
one_hot_embed[ord(".")] = torch.Tensor([0.25, 0.25, 0.25, 0.25])

reverse_complement_map = torch.Tensor([3, 2, 1, 0, 4]).long()


def torch_fromstring(seq_strs):
    """Convert string to torch stack"""
    batched = not isinstance(seq_strs, str)
    seq_strs = cast_list(seq_strs)
    np_seq_chrs = list(map(lambda t: np.fromstring(t, dtype=np.uint8), seq_strs))
    seq_chrs = list(map(torch.from_numpy, np_seq_chrs))
    return torch.stack(seq_chrs) if batched else seq_chrs[0]


def str_to_seq_indices(seq_strs):
    """Convert DNA string to indices encoded"""
    seq_chrs = torch_fromstring(seq_strs)
    return seq_indices_embed[seq_chrs.long()]


def str_to_one_hot(seq_strs):
    """Convert DNA string to one hot encoded"""
    seq_chrs = torch_fromstring(seq_strs)
    return one_hot_embed[seq_chrs.long()]


def seq_indices_to_one_hot(t, padding=-1):
    """Convert sequence in indices format to one hot encoded"""
    is_padding = t == padding
    t = t.clamp(min=0)
    one_hot = F.one_hot(t, num_classes=5)
    out = one_hot[..., :4].float()
    out = out.masked_fill(is_padding[..., None], 0.25)
    return out


# augmentations ==============================
def seq_indices_reverse_complement(seq_indices):
    """Reverse complement a sequence in seq_indices format"""
    complement = reverse_complement_map[seq_indices.long()]
    return torch.flip(complement, dims=(-1,))


def one_hot_reverse_complement(one_hot):
    """Reverse complement a one-hot-coded sequence"""
    *_, n, d = one_hot.shape
    assert d == 4, "must be one hot encoding with last dimension equal to 4"
    return torch.flip(one_hot, (-1, -2))


# query file to genomic interval ==========
class FastaInterval:
    """Fasta interval for querying from a fasta file with pyfaidx"""

    def __init__(
        self,
        *,
        fasta_file: str,
        context_length: int,
        return_seq_indices: Optional[bool] = False,
        shift_augs: Optional[List[int]] = None,
        rc_aug: Optional[bool] = False,
        rc_force: Optional[bool] = False,
    ) -> None:

        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), "path to fasta file must exist"

        self.seqs = Fasta(str(fasta_file))
        self.return_seq_indices = return_seq_indices
        self.context_length = context_length
        self.shift_augs = shift_augs
        self.rc_aug = rc_aug
        self.rc_force = rc_force

    def __call__(
        self,
        chr_name: str,
        start: int,
        end: int,
        strand: Optional[str] = "+",
        return_augs: Optional[bool] = False,
    ) -> torch.Tensor:
        """Call interval class to retrieve a genomic window."""
        interval_length = end - start
        chromosome = self.seqs[chr_name]
        chromosome_length = len(chromosome)
        assert strand in ["+", "-"], "Strand token not supported!"

        rand_shift = 0

        if check_is_not_none(self.shift_augs):
            # handle single inputs
            if len(self.shift_augs) == 1:
                min_shift = self.shift_augs[0]
                max_shift = min_shift
            else:
                min_shift, max_shift = self.shift_augs
            max_shift += 1
            min_shift = max(start + min_shift, 0) - start
            max_shift = min(end + max_shift, chromosome_length) - end
            if min_shift == max_shift:  # handle edge cases where they end up
                # being the same
                max_shift += 1
            if max_shift < min_shift:  # handle augmentation shifts over the
                # chrom
                max_shift = min_shift + 1
            rand_shift = randrange(min_shift, max_shift)
            start += rand_shift
            end += rand_shift

        left_padding = right_padding = 0

        if (
            check_is_not_none(self.context_length)
            and interval_length < self.context_length
        ):
            extra_seq = self.context_length - interval_length

            extra_left_seq = extra_seq // 2
            extra_right_seq = extra_seq - extra_left_seq

            start -= extra_left_seq
            end += extra_right_seq

            if start < 0:
                left_padding = -start
                start = 0

            if end > chromosome_length:
                right_padding = end - chromosome_length
                end = chromosome_length

        seq = ("." * left_padding) + str(chromosome[start:end]) + ("." * right_padding)

        # use reverse complement if:
        # reverse_complement augmentation is desired
        # (either augmentation turned on and True coin flip result
        # or reverse_complement is forced
        # AND the ROI is on the + strand minus
        # or not reverse complement augmentation is to be applied but the
        # feature is on the - strand
        rc_applied = False  # rc augmentation to be applied

        if (self.rc_aug and random() > 0.5) or self.rc_force:
            rc_applied = True

        # if desired to return indices instead of one hot
        if self.return_seq_indices:
            seq = str_to_seq_indices(seq)
            if (rc_applied and strand == "+") or (not rc_applied and strand == "-"):

                seq = seq_indices_reverse_complement(seq)
            if return_augs:
                return seq, strand, rand_shift, rc_applied
            else:
                return seq

        # else convert to one hot encoded
        one_hot = str_to_one_hot(seq)

        if (rc_applied and strand == "+") or (not rc_applied and strand == "-"):
            one_hot = one_hot_reverse_complement(one_hot)

        if not return_augs:
            return one_hot

        return one_hot, strand, rand_shift, rc_applied


class FastaIntervalPysam:
    """Fasta interval for querying from a fasta file with pysam"""

    def __init__(
        self,
        *,
        fasta_file: str,
        context_length: int,
        return_seq_indices: Optional[bool] = False,
        shift_augs: Optional[List[int]] = None,
        rc_aug: Optional[bool] = False,
        rc_force: Optional[bool] = False,
    ) -> None:
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), "path to fasta file must exist"

        self.seqs = FastaFile(str(fasta_file))
        self.return_seq_indices = return_seq_indices
        self.context_length = context_length
        self.shift_augs = shift_augs
        self.rc_aug = rc_aug
        self.rc_force = rc_force

    def __call__(
        self,
        chr_name: str,
        start: int,
        end: int,
        strand: Optional[str] = "+",
        return_augs: Optional[bool] = False,
    ) -> torch.Tensor:
        """Call interval class to retrieve a genomic window."""
        interval_length = end - start
        chromosome_length = self.seqs.get_reference_length(chr_name)
        assert strand in ["+", "-"], "Strand token not supported!"

        rand_shift = 0

        if check_is_not_none(self.shift_augs):
            # handle single inputs
            if len(self.shift_augs) == 1:
                min_shift = self.shift_augs[0]
                max_shift = min_shift
            else:
                min_shift, max_shift = self.shift_augs
            max_shift += 1
            min_shift = max(start + min_shift, 0) - start
            max_shift = min(end + max_shift, chromosome_length) - end
            if min_shift == max_shift:  # handle edge cases where they end up
                # being the same
                max_shift += 1
            if max_shift < min_shift:  # handle augmentation shifts over the
                # chrom
                max_shift = min_shift + 1
            rand_shift = randrange(min_shift, max_shift)
            start += rand_shift
            end += rand_shift

        left_padding = right_padding = 0

        if (
            check_is_not_none(self.context_length)
            and interval_length < self.context_length
        ):
            extra_seq = self.context_length - interval_length

            extra_left_seq = extra_seq // 2
            extra_right_seq = extra_seq - extra_left_seq

            start -= extra_left_seq
            end += extra_right_seq

            if start < 0:
                left_padding = -start
                start = 0

            if end > chromosome_length:
                right_padding = end - chromosome_length
                end = chromosome_length

        fetched_seq = self.seqs.fetch(reference=chr_name, start=start, end=end)

        seq = ("." * left_padding) + fetched_seq + ("." * right_padding)

        # use reverse complement if:
        # reverse_complement augmentation is desired
        # (either augmentation turned on and True coin flip result
        # or reverse_complement is forced
        # AND the ROI is on the + strand minus
        # or not reverse complement augmentation is to be applied but the
        # feature is on the - strand
        rc_applied = False  # rc augmentation to be applied

        if (self.rc_aug and random() > 0.5) or self.rc_force:
            rc_applied = True

        # if desired to return indices instead of one hot
        if self.return_seq_indices:
            seq = str_to_seq_indices(seq)
            if (rc_applied and strand == "+") or (not rc_applied and strand == "-"):
                seq = seq_indices_reverse_complement(seq)
            if return_augs:
                return seq, strand, rand_shift, rc_applied
            else:
                return seq

        # else convert to one hot encoded
        one_hot = str_to_one_hot(seq)

        if (rc_applied and strand == "+") or (not rc_applied and strand == "-"):
            one_hot = one_hot_reverse_complement(one_hot)

        if not return_augs:
            return one_hot

        return one_hot, strand, rand_shift, rc_applied


# create genomic interval dataset from query =====================
# Adapted from original to:
# * use the already in memory query file
# * store the bins of interest for which to extract the embeddings and targets
# TODO add vcf/variant application to reference sequence
class GenomeIntervalDataset(Dataset):
    def __init__(
        self,
        query: Union[str, pd.DataFrame],  # can be path to query file or a
        # pandas data frame with the query loaded
        fasta_file: str,
        context_length: int,
        pos_base: int,
        filter_df_fn: Optional[Callable] = identity,
        query_to_fasta_map: Optional[dict] = dict(),
        return_seq_indices: Optional[bool] = False,
        shift_augs: Optional[List[int]] = None,
        rc_force: Optional[bool] = False,
        rc_aug: Optional[bool] = False,
        return_augs: Optional[bool] = False,
    ):
        super().__init__()
        if isinstance(query, pd.DataFrame):
            df = query
        else:
            # read from file
            query_path = Path(query)
            assert query_path.exists(), "path to .bed file must exist"

            df = pd.read_csv(str(query_path), sep="\t", headers=False)
            df = filter_df_fn(df)

        # ensure right base index is used
        assert pos_base in [0, 1]
        # remove 1 from start coordinate to  confirm with bed format
        if pos_base == 1:
            df.iloc[:, 1] -= 1

        self.df = df

        # if the chromosome name in the bed file is
        # different than the keyname in the fasta
        # can remap on the fly
        self.query_to_fasta_map = query_to_fasta_map

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

    def __getitem__(
        self, ind: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, str, int, bool]]:
        """Retrieve interval by index"""
        interval = self.df.iloc[ind, :]
        return self.fasta(
            interval["chr"],
            interval["seq_start"],
            interval["seq_end"],
            interval["seq_strand"],
            return_augs=self.return_augs,
        )
