"""
Cross cell correlation loss
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
from typing import Literal

import anndata as ad
import numpy as np


def get_across_x_correlation(
    ad: ad.AnnData,
    axis: Literal[0, 1] = 0,
    obs_layer: str = "X",
    pred_layer: str = "predicted",
    return_per_entry: bool = False,
):
    """Calculate correlation pred vs obs across given axis

    Arguments
    ---------
    ad: ad.AnnData
        AnnDataFrame hosting observations,
        should be in the orientation gene(obs) x cells(var).
    axis: int [0,1]
        Select the axis across which to correlate
        observations with predictions.
        0 - across genes; 1 - across cells
    obs_layer: str = 'X'
        Name of the anndata layer in which the observations are
        stored. If 'X' is supplied will use the default anndata.X
        matrix. Default = 'X'
    pred_layer: str = 'predicted'
        Name of the anndata layer in which the observations are
        stored. If 'X' is supplied will use the default anndata.X
        matrix. Default = 'predicted'
    return_per_entry: bool = False
        Toggle if to return a dataframe with the per entry correlations.

    Output
    ------
    mean_cor: float
        The mean Pearson correlation across all entries.
    cor_per_entry: [Optional] pd.DataFrame
        Listing the correlation per entry along the non selected axis.

    Notes
    -----
    Will compute the correlation of observations vs. predictions
    per entry. Depending on the axis selected:
    axis=0 -> across genes -> entries are cells
    axis=1 -> across cell (types) -> entries are cells
    """
    if obs_layer == "X":
        obs = ad.to_df()
    else:
        obs = ad.to_df(layer=obs_layer)

    if pred_layer == "X":
        pred = ad.to_df()
    else:
        pred = ad.to_df(layer=pred_layer)

    # correlation across genes per cell
    cor_per_entry = obs.corrwith(pred, axis=axis)

    mean_cor = np.nanmean(cor_per_entry)

    if return_per_entry:
        return mean_cor, cor_per_entry
    else:
        return mean_cor
