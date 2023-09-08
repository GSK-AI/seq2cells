"""
Utilities for handling variant effect predictions with anndata objects
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

import anndata
import numpy as np
import pandas as pd


def get_var_specific_anndata(
    var_idx: int,
    ad: anndata.AnnData,
    var_df: pd.DataFrame,
    ref_pred: np.ndarray,
    var_pred: np.ndarray,
) -> anndata.AnnData:
    """Get a copy of an anndata object for variant visualization

    Parameters
    ----------
    ad: anndata.Anndata
        anndata object storing the single cell observed counts and
        desired umap coordinates. Shape [n_cells x n_genes]
    var_idx: int
        Index in the variant Dataframe of the variant to extract.
    var_df: pd.DataFrame,
        Dataframe storing the variants matched against seq_windows.
    ref_pred: numpy.ndarray
        Reference sequence predictions of shape [n_variants x n_cells]
    var_pred: numpy.ndarray
        Variant sequence predictions of shape [n_variants x n_cells]

    Returns
    -------
    adata_var: anndata.AnnData
        Anndata object storing observed counts and variant
        effect predictions for a single variant - gene.
    """
    # get the variant id and gene symbol in the variant matched datafram
    var_id = var_df.loc[:, "var_linked_gene_strip"][var_idx]

    # subset anndata
    adata_var = ad[:, ad.var["ens_id_strip"] == var_id].copy()

    assert adata_var.shape[1] == 1, "No unique matching gene entry found!"

    # add prediction layers
    adata_var.layers["ref_pred"] = ref_pred[var_idx, None].T
    adata_var.layers["var_pred"] = var_pred[var_idx, None].T
    adata_var.layers["delta_pred"] = (
        adata_var.layers["var_pred"] - adata_var.layers["ref_pred"]
    )

    # add variant amtch summary
    adata_var.uns["variant_summary"] = var_df.iloc[var_idx, :]

    # calc and add summary stats for variant
    adata_var.uns["sum_abs_delta_pred"] = np.sum(np.abs(adata_var.layers["delta_pred"]))
    adata_var.uns["mean_abs_delta_pred"] = np.mean(
        np.abs(adata_var.layers["delta_pred"])
    )

    return adata_var
