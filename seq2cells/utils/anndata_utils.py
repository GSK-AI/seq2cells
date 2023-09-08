"""
Utilities for handling anndata objects
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
from typing import List, Literal, Optional

import anndata
import numpy as np
import pandas as pd


def get_layer(x: anndata.AnnData, layer: Optional[str] = None) -> np.ndarray:
    """Get specific layer of an anndata object"""
    if layer is not None:
        get_x = x.layers[layer]
    else:
        get_x = x.X

    return get_x


def grouped_obs_mean(
    adata: anndata.AnnData,
    group_key: str,
    layer: Optional[str] = None,
    gene_symbols: Optional[str] = None,
) -> pd.DataFrame:
    """Calc mean of observations by group

    Parameters
    ----------
    adata: anndata.AnnData
        Anndata object with 'group_key' as observation.
    group_key: str
        String indicating the observation column to use.
    layer: Optional[str] = None
        Either None (use adata.X) or the name of a layer to use.
    gene_symbols: Optional[str] = None
        String index of the variables column holding gene_symbols to use as
        new index of the resulting data frame, otherwise the index of
        adata.var is used.

    Returns
    -------
    out_df: pd.DataFrame
        Data frame indexed over adata.var or gene_symbols indicating the
        mean over grouped observations (e.g. mean expression per gene within
        each cell type).

    Notes
    -----
    Calculates the mean over groups of observations grouped by group_key,
    either on a provided layer or on adata.X.
    """
    if gene_symbols is not None:
        new_idx = adata.var[gene_symbols]
    else:
        new_idx = adata.var_names

    grouped = adata.obs.groupby(group_key)
    out_df = pd.DataFrame(
        np.zeros((adata.shape[1], len(grouped)), dtype=np.float32),
        columns=list(grouped.groups.keys()),
        index=new_idx,
    )

    for group, idx in grouped.indices.items():
        x = get_layer(adata[idx], layer=layer)
        out_df[group] = np.ravel(x.mean(axis=0, dtype=np.float32))

    return out_df


def grouped_var_mean(
    adata: anndata.AnnData,
    group_key: str,
    layer: Optional[str] = None,
    gene_symbols: Optional[str] = None,
) -> pd.DataFrame:
    """Calc mean of observations by group

    Parameters
    ----------
    adata: anndata.AnnData
        Anndata object with 'group_key' as observation.
    group_key: str
        String indicating the observation column to use.
    layer: Optional[str] = None
        Either None (use adata.X) or the name of a layer to use.
    gene_symbols: Optional[str] = None
        String index of the observations column holding gene_symbols to use as
        new index of the resulting data frame, otherwise the index of
        adata.obs is used.

    Returns
    -------
    out_df: pd.DataFrame
        Data frame indexed over adata.vars or gene_symbols indicating the
        mean over grouped observations (e.g. mean expression per gene within
        each cell type).

    Notes
    -----
    Calculates the mean over groups of observations grouped by group_key,
    either on a provided layer or on adata.X.
    """
    if gene_symbols is not None:
        new_idx = adata.obs[gene_symbols]
    else:
        new_idx = adata.obs_names

    grouped = adata.var.groupby(group_key)
    out_df = pd.DataFrame(
        np.zeros((adata.shape[0], len(grouped)), dtype=np.float32),
        columns=list(grouped.groups.keys()),
        index=new_idx,
    )

    for group, idx in grouped.indices.items():
        x = get_layer(adata[:, idx], layer=layer)
        out_df[group] = np.ravel(x.mean(axis=1, dtype=np.float32))

    return out_df


def pseudo_bulk(
    ad: anndata.AnnData,
    genes: List[str],
    cell_type_col: str,
    gene_col: str,
    mode: Literal["mean", "count_exp", "perc_exp"] = "mean",
    expr_threshold: float = 1.0,
    mem_efficient_mode: bool = False,
    layer: Optional[str] = None,
) -> anndata.AnnData:
    """Pseudo bulk observed expression by cell type

    Parameter
    ---------
    ad: anndata.AnnData
        annData object with an observed read counts (or transformed) matrix in .X
    genes: List[str]
        List of genes to pseudobulk. If list is empty will use all genes.
    cell_type_col: str
        Name of cell type column to use for defining the pseudo bulks.
    gene_col:
        Column name to be used a gene names / symbols for grouping the expression.
        If supplied as "'Index' will use the index of the anndata frames obs.
    mode: Literal['mean', 'sum', 'perc_exp'] = 'mean'
        The mode to use for pseudobulking. See Notes for details
    expr_threshold: float = 1.0
        Threshold of observed (transformed) read counts at and above which a
        genes is considered to be expressed.
    mem_efficient_mode: bool = False
        Toggle if to run in memory efficient mode. True - saves memeory but
        is slower omn the order of 10-100.
    layer: Optional[str] = None
        If provided use the layer instead of the .X data.

    Return
    ------
        Anndata object with same observations as the anndata frame provided or obs
        subsetted if only a selection of genes was extracted.
        .vars are summarised (seudo-bulke) by the chosen method.

    Notes
    -----
    Different pseudo-bulking modes are supported. All are based on the observed
    and optionally transformed read counts, following the doctrine of
    transform first, summarise after.
    Modes:
        "mean" - calculate the mean over all cells of the same cell type
        "sum" - sum up all observed read counts over cells of the same cell type
        "perc_expr" - calculate fraction of cell within the same cell type that
            express each gene respectiely.
        "perc_expr" - sum the number of cells within the same cell type that
            express each gene respectiely.
    """
    # assert a supported mode was selected
    assert mode in [
        "mean",
        "sum",
        "perc_exp",
        "count_exp",
    ], 'Only ["mean",  "sum", "perc_exp", "count_exp"] modes are supported!'

    # use index as new gene name column if running based on obs index
    if gene_col in ["index"]:
        ad.obs[gene_col] = ad.obs.index

    # get data for genes of interest
    if len(genes) >= 1:
        ad = ad[ad.obs[gene_col].isin(genes), :]

    # get unique cell types
    cell_types = np.unique(ad.var[cell_type_col])

    # mem friendly version =================
    # much slower!
    if mem_efficient_mode:
        # save number of cells
        num_cells = len(ad.var)

        # create zeros df to fill
        psd = pd.DataFrame(
            np.zeros(
                [
                    len(ad),
                    len(cell_types),
                ]
            )
        )

        psd.columns = cell_types
        psd.index = ad.obs.index

        # for every gene and every cell type calculate the pseudobulk aggregate
        for g in range(len(psd.index)):
            for c in range(len(cell_types)):

                gene = psd.index[g]

                cell_type = cell_types[c]

                if mode == "mean":
                    psd.iloc[g, c] = (
                        ad[ad.obs.index == gene, ad.var[cell_type_col] == cell_type]
                        .to_df()
                        .mean(numeric_only=True, axis=1)[0]
                    )
                elif mode == "sum":
                    psd.iloc[g, c] = (
                        ad[ad.obs.index == gene, ad.var[cell_type_col] == cell_type]
                        .to_df()
                        .sum(numeric_only=True, axis=1)[0]
                    )
                else:
                    temp_df = (
                        ad[
                            ad.obs.index == gene, ad.var[cell_type_col] == cell_type
                        ].to_df()
                        > expr_threshold
                    )
                    temp_count = temp_df.sum(axis=1)[0]
                    if mode == "count_exp":
                        psd.iloc[g, c] = temp_count
                    else:
                        psd.iloc[g, c] = temp_count / num_cells

    # mem demanding version =================
    else:
        psd = ad

        if layer:
            psd = psd.to_df(layer=layer)
        else:
            psd = psd.to_df()

        # add cell types as columns instead of cells
        psd.columns = ad.var[cell_type_col].values

        # add gene symbol
        if len(genes) >= 1:
            psd[gene_col] = ad[ad.obs[gene_col].isin(genes), :].obs[gene_col].values
        else:
            psd[gene_col] = ad.obs[gene_col].values

        # melt
        psd = psd.melt(id_vars=[gene_col], var_name=cell_type_col)

        # group_by and summarise
        if mode == "mean":
            psd = psd.groupby([gene_col, cell_type_col]).mean(numeric_only=True)
        elif mode == "sum":
            psd = psd.groupby([gene_col, cell_type_col]).sum(numeric_only=True)
        elif mode == "perc_exp":
            # filter only keep cells where expressed
            psd_all = psd.copy()
            psd = psd.loc[psd["value"] > expr_threshold, :]

            # count
            psd_all = psd_all.groupby([gene_col, cell_type_col]).count()
            psd = psd.groupby([gene_col, cell_type_col]).count()
            psd = psd / psd_all
            # psd = psd.loc[~psd['value'].isna(),:]

        else:
            psd = psd.loc[psd["value"] > expr_threshold, :]
            psd = psd.groupby([gene_col, cell_type_col]).count()

        # spread / unstack
        psd = psd.unstack(level=1)

        # sort to match anndata observations
        psd = psd.loc[ad.obs[gene_col].values]

    # create anndata frame
    psd_ad = anndata.AnnData(psd, dtype=np.float32)

    # add observations
    psd_ad.obs = ad.obs

    # add other data
    psd_ad.obsm = ad.obsm
    psd_ad.uns = ad.uns

    psd_ad.var = psd_ad.var.reset_index(level=0)

    if "level_0" in psd_ad.var.columns:
        # reset .var index
        psd_ad.var = psd_ad.var.drop(labels="level_0", axis=1)

    psd_ad.var[cell_type_col] = psd_ad.var.index
    psd_ad.var.index.name = None

    return psd_ad
