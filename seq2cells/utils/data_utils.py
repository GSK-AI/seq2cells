"""
Utilities for handling seq model data processing
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
from sys import exit
from typing import List, Optional, Union

import numpy as np


def aggregate_ndarray(
    in_array: np.ndarray, axis: Optional[int] = 0, method: Optional[str] = "mean"
) -> np.ndarray:
    """Aggregate a numpy array along a specified axis using a chosen method

    Parameters
    ----------
    in_array: np.ndarray
        Numpy array with enough axis to allow aggregation along the
        specified axis.
    axis: int
        Axis along which to aggregate the numpy array.
    method: str
        Aggregation method of choice. Valid options: 'sum' 'mean'
        'median' 'max' 'min'.

    Returns
    -------
    np.ndarray
        Numpy array of unique entries from provided list/array.

    Notes
    -----
    Will drop the axis along which the array was aggegated.
    """
    if method == "sum":
        aggr_array = np.sum(in_array, axis=axis)
    elif method == "mean":
        aggr_array = np.mean(in_array, axis=axis)
    elif method == "median":
        aggr_array = np.median(in_array, axis=axis)
    elif method == "max":
        aggr_array = np.max(in_array, axis=axis)
    elif method == "min":
        aggr_array = np.min(in_array, axis=axis)
    else:
        print("No valid aggregation method specified. Exiting!")
        exit()

    return aggr_array


def mirror_roi_bins(list_bins: List[int], bin_size: int, pred_window: int) -> List[int]:
    """Mirror the bin indices for extracting from reverse complement.

    Parameters
    ----------
    list_bins: list(int)
        List of integers containing the bin indices of interest.
    bin_size: int
        Size in bp of the sequence window over which the model actually
        predicts targets = bin_size * num_output_bins.
    pred_window: int
        Size of the prediction window over which the seq model actually
        produces outputs.

    Returns
    -------
    list(int)
        Sorted list of integers of the bin_ids mirrored on the sequence center.

    Notes
    -----
    If prediction window divided by bin_size is uneven will mirror on the
    center bin. If even will mirror on the junction between the two central
    bins.
    """
    num_bins = pred_window / bin_size
    mirrored_bins = [int(num_bins - b - 1) for b in list_bins]
    mirrored_bins.sort()

    return mirrored_bins


def get_roi_bins(
    bins_in: Union[str, int, np.int64], add_bins: Optional[int] = 0
) -> np.ndarray:
    """Get the desired bins indices as np.array from bins.

    Parameters
    ----------
    bins_in: Union[str, int, np.int64]
        Comma separated string of bins of interest.
        Or a single integer.
    add_bins: Optional[int]
        Integer specifying how many adjacent bins to each bin of interest should be
        included in the bin query.

    Returns
    -------
    np.ndarray
        Numpy array with individual bins of interest

    Notes
    -----
    Takes as input a comma separated string of integers or a single integer.
    Adds additional adjacent bins if desired.
    Bins below 0 will be dropped.
    Returns the sorted unique numpy array of bins of interest.

    Example
    -------
    test_ids = "1,2,3"

    data_utils.get_roi_bins(test_ids, add_bins=2)

    [0,1,2,3,4,5]

    """
    # ensure bins_in is string (convert if not)
    if not isinstance(bins_in, str):
        if isinstance(bins_in, np.int64):
            bins_in = int(bins_in)
        assert isinstance(bins_in, int), (
            f"`Incorrect type for bins_in " f"({type(bins_in)})"
        )
        bins_in = str(bins_in)
    bins = [int(i) for i in bins_in.split(",")]
    bins_to_query = []
    # add additional bins each side
    assert add_bins >= 0, "add_bins should be 0 or positive!"
    if add_bins > 0:
        for bin in bins:
            bins_to_query.append(bin)
            for add_idx in range(add_bins):
                add = add_idx + 1
                bins_to_query.append(bin + add)
                bins_to_query.append(bin - add)
    else:
        bins_to_query = bins
    # drop everything negative
    bins_to_query = [b for b in bins_to_query if b >= 0]

    return np.unique(np.sort(np.array(bins_to_query)))
