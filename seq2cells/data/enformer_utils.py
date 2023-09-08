"""
Utilities for Enformer type data
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

from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd

# helper functions
from numpy import ndarray


def load_enformer_covered_tss(
    train_in: str, valid_in: str, test_in: str, return_list: Optional[bool] = False
) -> Union[
    Tuple[List[int], List[int], List[int], List[int]],
    Tuple[ndarray, ndarray, ndarray, ndarray],
]:
    """Load indices of TSS that are covered by Enformer windows.

    Parameters
    ----------
    train_in: str
        Path to training query file produced by extract_enformer_targets.py
    valid_in:
        Path to validation query file produced by extract_enformer_targets.py
    test_in: str
        Path to test query file produced by extract_enformer_targets.py
    return_list: bool
        True/False if to return as lists. Otherwise np.ndarrays are returned.

    Returns
    -------
    all_indices:  np.ndarray
        All indices covered combined.
    train_indices: np.ndarray
        Training indices covered.
    valid_indices: np.ndarray
        Validation indices covered.
    test_indices: np.ndarray
        Test indices covered.

    Notes
    -----
    TSS (or ROI) covered by Enformer windows are stored as queries
    from running  extract enformer_targets.py in tsv files.
    This functions reads all and extracts the 'query_index' column which
    refers to the index of the queries in the pre-computed embeddings -
    predicted targets file produced from cal_embeddings_and_targets.py
    """
    # target data tsv files
    train_enf_query = pd.read_csv(train_in, sep="\t")
    valid_enf_query = pd.read_csv(valid_in, sep="\t")
    test_enf_query = pd.read_csv(test_in, sep="\t")

    # grab the query patch indices from the TSS that could actually be
    # overlapped with Enformer windows
    train_indices = train_enf_query.loc[:, "query_index"].values
    valid_indices = valid_enf_query.loc[:, "query_index"].values
    test_indices = test_enf_query.loc[:, "query_index"].values

    all_indices = np.concatenate((train_indices, valid_indices, test_indices))

    if return_list:
        return (
            list(all_indices),
            list(train_indices),
            list(valid_indices),
            list(test_indices),
        )
    else:
        return all_indices, train_indices, valid_indices, test_indices


def filter_enf_targets(
    targets_file: str,
) -> Tuple[pd.DataFrame, List[int], List[int], List[int]]:
    """
    Filter the Enformer targets for open chromatin, k27ac and cage.

    Parameters
    ----------
    targets_file: str
        Path to targets file as downloaded from public Enformer repository.

    Returns
    -------
    cage_indices: List[int]
        Indices of the cage data in the filtered dataframe.
    open_indices: List[int]
        Indices of the open data in the filtered dataframe.
    k27ac_indices: List[int]
        Indices of the k27ac data in the filtered dataframe.

    Notes
    -----
    Only Open chromatin (DNase) H3K27ac and CAGE experiments were extracted
    and predicted.
    """
    df_targets_anno = pd.read_csv(targets_file, sep="\t")
    # grab the targets of interest (k27ac, DNase, CAGE)
    cage_targets = df_targets_anno.loc[
        df_targets_anno["description"].str.contains("CAGE")
    ]
    open_targets = df_targets_anno.loc[
        df_targets_anno["description"].str.contains("DNASE")
    ]
    k27ac_targets = df_targets_anno.loc[
        df_targets_anno["description"].str.contains("K27ac")
    ]
    all_targets = pd.concat([open_targets, k27ac_targets, cage_targets])
    all_targets = all_targets.rename(columns={"index": "enf_index"})
    # first reset the index
    all_targets = all_targets.reset_index(drop=True)
    # than get the reset index as extra column
    all_targets = all_targets.reset_index(drop=False)

    cage_indices = all_targets.loc[all_targets["description"].str.contains("CAGE")][
        "index"
    ].values
    open_indices = all_targets.loc[all_targets["description"].str.contains("DNASE")][
        "index"
    ].values
    k27ac_indices = all_targets.loc[all_targets["description"].str.contains("K27ac")][
        "index"
    ].values

    return all_targets, cage_indices, open_indices, k27ac_indices


def load_enformer_public_human_head_weights(
    weights_file: str, select_idx: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Load Enformer pre-trained weights from numpy (.npz) file

    Parameters
    ----------
    weights_file: str
        Gzipped numpy file containing the Enformer pre-trained
        head weights as numpy array.
    select_idx: List[int]
        List of integer indices selecting the pre-trained weights of interest.

    Returns
    -------
    enformer_weights: np.ndarray
        Numpy array of the head weights of selected indices.
    enformer_biases: np.ndarray
        Numpy array of the head biases of selected indices.

    Notes
    -----
    Load the pre-trained weights. Select the index/indices of interest.
    """
    enformer_in = np.load(weights_file)
    enformer_weights = enformer_in["human_head_weights"][select_idx, :]
    enformer_biases = enformer_in["human_head_biases"][select_idx]

    return enformer_weights, enformer_biases


def get_enf_stats(
    targets_file: str,
    target_index: List[int],
    set: Optional[str] = "train",
    log_transform=True,
    return_list: Optional[bool] = False,
):
    """Load Enf Training data and return mean and std per cell type

    Parameters
    ----------
    targets_file: str
        Path to .h5 file holding the observed targets in hdf5 format.
        Expects the keys: [] to be present.
        Each holding a numpy array.
    target_index: List[int]
        List of integers indicating which targets are trained against.
    set: str ['train', 'valid', 'test']
        String indicating which set to load in the dataset.
        Data are stored under separate keys in the hdf5 files.
    log_transform: Optional[bool]
        If to log transform data log(x+1) before calculating the std deviation.
    return_list: Optional[bool]
        Default False. If to return means and std as lists rather than
        numpy.ndarrays.

    Returns
    -------
    dic: dict
        Dictionary with mean and standard deviation per cell type
        for observed targets.
    """
    targets = h5py.File(targets_file, "r")[f"targets_{set}"]

    # subset to targets/classes of interest
    targets = targets[:, target_index]

    if log_transform:
        targets = np.log(targets + 1)

    dic = {"mean": np.mean(targets, axis=0), "std": np.std(targets, axis=0)}

    if return_list:
        dic["mean"] = list(dic["mean"])
        dic["std"] = list(dic["std"])

    return dic
