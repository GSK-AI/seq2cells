"""
Utilities for handling text processing needs for seq2cells
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
from collections import Counter
from typing import List, Union

import numpy as np


def unique(list1: Union[List[str], np.ndarray]) -> np.ndarray:
    """Get a unique numpy array from a list or numpy array

    Parameters
    ----------
    list1: list, np.ndarray
        List or numpy array of values.

    Returns
    -------
    np.ndarray
        Numpy array of unique entries from provided list/array.

    """
    x = np.array(list1)

    return np.unique(x)


def split_index_range_mix(input: Union[str, int]) -> np.ndarray:
    """Split a mixture of (0-based) indices and ranges into an np.array

    Parameters
    ----------
    input: str, int
        Comma separated string with a mixture of integers and ranges.
        May contain a single integer is no splitting is required.

    Returns
    -------
    np.ndarray
        Numpy array of the split up indices.

    Notes
    -----
    Ranges will be split and the last entry used.
    Example:
    '0,1,4:7,9,12:14' --> [0,1,4,5,6,7,9,12,13,14]
    """
    splitted_idx = input.split(",")
    assemble_idx = []
    for idx in splitted_idx:
        if ":" in idx:
            splitted = idx.split(":")
            start = int(splitted[0])
            end = int(splitted[1]) + 1
            start_to_end = [*range(start, end)]
            assemble_idx = assemble_idx + start_to_end
        else:
            assemble_idx.append(int(idx))
    assemble_idx.sort()

    return assemble_idx


def get_most_common(list_in: List[str]) -> str:
    """Get the most common str entry from a list of str.

    Parameters
    ----------
    list_in: list(str)
        List of strings.

    Returns
    -------
    str
        Most occurring string in the list.
    """
    # count occurrences grab top or fist if equal
    entry_counts = Counter(list_in)
    most_common = entry_counts.most_common()[0][0]

    return most_common
