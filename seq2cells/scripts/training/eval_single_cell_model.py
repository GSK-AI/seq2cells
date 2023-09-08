"""
Predict and evaluate embedding-to-expression models
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
* Load anndata frame with precomputed (TSS) embeddings
* Load a model Checkpoint
* Predict the model targets for all queries.
* Save anndata frame with predicted layer
* Basic evaluation: cross-cell and cross-gene correlation

Can be run on (pseudo)bulk data or single cell data.
"""
import logging
import os
from typing import Any, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import torch
import yaml
from natsort import natsorted
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from seq2cells.data.enformer_tss_based import AnnDataEmbeddingTssDataset
from seq2cells.metrics_and_losses.eval_anndata import get_across_x_correlation
from seq2cells.models.embedding2target import Embedding2Target
from seq2cells.utils.predict_utils import BatchWiseWriter

# default behaviours ==========================
# define scanpy behaviour and default plotting options
sc.settings.verbosity = 2
# verbosity: errors (0), warnings (1), info (2),# hints (3)
sc.logging.print_header()

# set logger
logger = logging.getLogger(__name__)


# Process functions =================
def load_anndata(config: dict) -> ad.AnnData:
    """Load anndata frame

    Parameters
    ----------
    config: dict
        Configuration dictionary as parsed form the config_anndata_eval.yml
        config file.

    Returns
    -------
    adata: ad.Anndata
        Anndata object loaded from file.
    """
    logger.info("Loaded AnnData object:")
    if config["resource"]["backed_mode"]:
        adata = sc.read_h5ad(config["data"]["ann_data_file"], backed="r")
    else:
        adata = sc.read_h5ad(config["data"]["ann_data_file"])
    logger.info(adata)

    # set observed counts to selected layer
    if config["data"]["observed_counts_layer"] != "X":
        assert (
            config["data"]["observed_counts_layer"] in adata.layers
        ), "Specified layer is not in the provided adata object!"
        adata.X = adata.layers[config["data"]["observed_counts_layer"]]

    return adata


def get_dataloader(
    config: dict, adata: ad.AnnData
) -> torch.utils.data.dataloader.DataLoader:
    """Construct data loader

    Parameters
    ----------
    config: dict
        Configuration dictionary as parsed form the config_anndata_eval.yml
        config file.
    adata: ad.AnnData
        Anndata object in gene (obs) x cells (var) orientation, requires
        sequence embeddings stored as 'seq_embedding' slot in .obsm.

    Returns
    -------
    all_data_loader: torch.DataLoader
        Dataloader to return sequence embedding and expression across cells
        per instance.
    """
    all_data = AnnDataEmbeddingTssDataset(
        ann_in=adata,
        data_split=config["task"]["pred_on"],
        split_name=config["data"]["split_name"],
        subset_genes_col=config["task"]["subset_genes_column"],
    )

    if config["resource"]["pred_batch_size"] != 0:
        pred_batch_size = config["resource"]["pred_batch_size"]
        assert isinstance(pred_batch_size, int), (
            f"Expected integer or None as input "
            f"for pred_batch_size, got {pred_batch_size}"
        )
    else:
        # if set to None will use all data
        pred_batch_size = len(all_data)

    all_data_loader = DataLoader(all_data, batch_size=pred_batch_size, shuffle=False)

    return all_data_loader


def run_prediction(
    config: dict,
    model: Any,
    data_loader: torch.utils.data.dataloader.DataLoader,
    profiler: Any = None,
) -> torch.Tensor:
    """Run prediction and return results or resume last interrupted run.

    Parameters
    ----------
    config: dict
        Configuration dictionary as parsed form the config_anndata_eval.yml
        config file.
    model: torch.Module
        loaded torch model (expects Embedding2Target)
    data_loader: torch.DataLoader
        dataloader providing seq embeddings
    profiler:
        None or an initialised torch profiler.

    Returns
    -------
    predictions: torch.Tensor
        Tensor of single cell expression predicitons.
        Only returned if not run in temp_write mode.
    """
    logger.info("Running predictions ...")

    if config["resource"]["temp_write_pred"]:

        # check if existing .pt files in output directory
        # meaning an incomplete prediction run in the past
        check_for_pt = [x for x in os.listdir(output_path) if x.endswith(".pt")]

        if len(check_for_pt) > 0:
            logger.warning(
                f"Pre-existing .pt files found in output "
                f"directory {config['data']['output_path']}"
                f" make sure you are not mixing "
                f"predictions! found: {check_for_pt}"
                f"Continuing loading predictions."
                f"But not (over) writing new ones..."
            )

        else:
            logger.info("Writing predictions in batches to disk ...")
            # set up batch wise prediction writing to disk
            pred_writer = BatchWiseWriter(
                output_dir=output_path,
                write_interval="batch",
            )

            # setup pl trainer with writing to disk callback
            trainer = pl.Trainer(
                accelerator=config["resource"]["device"],
                callbacks=[pred_writer],
                devices=1,
                profiler=profiler,
                inference_mode=True,
            )

            # predict: writing to disk
            trainer.predict(model, dataloaders=data_loader, return_predictions=False)

            return None

    else:
        # set up pl trainer (no callbacks)
        trainer = pl.Trainer(
            accelerator=config["resource"]["device"],
            devices=1,
            profiler=profiler,
        )

        # predict storing in memory
        predictions = trainer.predict(
            model, dataloaders=data_loader, return_predictions=True
        )

        return predictions


def format_predictions(config: dict, predictions: torch.Tensor) -> np.ndarray:
    """Format the predictions

    Parameters
    ----------
    config: dict
        Configuration dictionary as parsed form the config_anndata_eval.yml
        config file.
    predictions: torch.Tensor
        Supplied as None if not available yet because they have to be read
        and formatted from disk.

    Returns
    -------
    predictions: np.ndarray
        Predictions formatted as numpy array (genes x cells).
    """
    if config["resource"]["temp_write_pred"]:
        logger.info("Loading temp saved prediction files ...")
        # read temp pred_ and batch_ files from disk
        pred_files = [
            x
            for x in os.listdir(output_path)
            if x.endswith(".pt") and x.startswith("pred_")
        ]

        # get matching batch indices
        batch_files = [
            x
            for x in os.listdir(output_path)
            if x.endswith(".pt") and x.startswith("batch_")
        ]

        # sort files
        pred_files = natsorted(pred_files)
        batch_files = natsorted(batch_files)

        # assemble to prediction matrix
        ba_idx = []
        for ba_file in batch_files:
            ba_idx_in = torch.load(os.path.join(output_path, ba_file))
            ba_idx = ba_idx + ba_idx_in

        predictions = None
        for pred_file in pred_files:
            pred_in = torch.load(os.path.join(output_path, pred_file)).cpu().numpy()
            if predictions is not None:
                predictions = np.vstack([predictions, pred_in])
            else:
                predictions = pred_in

        # sort by index
        predictions = predictions[ba_idx, :]

        # remove tmp .pt files
        for file in pred_files:
            os.remove(os.path.join(output_path, file))
        for file in batch_files:
            os.remove(os.path.join(output_path, file))

    else:
        # stack list of prediction tensors
        predictions = torch.vstack(predictions).detach().numpy()

    return predictions


def predict(adata: ad.AnnData, config: dict) -> ad.AnnData:
    """Wrapper function to predict expression given a model checkpoint

    Parameters
    ----------
    adata: ad.AnnData
        Anndata object in gene (obs) x cells (var) orientation, requires
        sequence embeddings stored as 'seq_embedding' slot in .obsm.
    config: dict
        Configuration dictionary as parsed form the config_anndata_eval.yml
        config file.

    Returns
    -------
    adata: ad.Anndata
        Anndata object supplied but with predictions stores in new layer.

    Notes
    -----
    Expects the global variabale 'prediction_layer' to be defined as the
    anndata layer in which to store predictions.
    """
    # use profiler?
    profiler = None
    if config["profile"]:
        from pytorch_lightning.profilers import PyTorchProfiler

        profiler = PyTorchProfiler()

    # if running in backed mode predictions need to be saved to
    assert config["resource"]["backed_mode"] is False, (
        "Running prediction is currently not supported when using the "
        "anndata in backed_mode. Please set 'backed_mode' to False! "
        "Stopping ..."
    )

    # prevent overwriting the original anndata object
    assert config["data"]["save_anndata_path"] != config["data"]["ann_data_file"], (
        "Files to read from and store anndata object with predictions to "
        "are the same! This is prevented when running predictions to not "
        "overwrite the original data!"
    )

    # Check if previous predictions were supplied
    if prediction_layer in adata.layers:
        if config["task"]["overwrite_pred"]:
            logger.info(
                "Previously computed predictions provided in anndata object ..."
                "Deleting predictions as 'overwrite_predictions was set to "
                "True ..."
            )
            del adata.layers[prediction_layer]
        else:
            logger.error(
                "Previously computed predictions provided in anndata object ..."
                "Skipping running predictions ..."
                "If predictions were incomplete, remove the predicted layer!"
            )

            return adata

    model = Embedding2Target.load_from_checkpoint(config["data"]["model_chkpt_path"])
    model.eval()

    all_data_loader = get_dataloader(config, adata)

    predictions = run_prediction(config, model, all_data_loader, profiler)

    predictions = format_predictions(config, predictions)

    adata = add_predictions_to_anndata(config, predictions, adata)

    # Store predictions ========================
    if config["data"]["save_predictions"]:
        logger.info("Storing anndata object with predictions ...")
        adata.write_h5ad(config["data"]["save_anndata_path"])

    logger.info(adata)
    assert (
        prediction_layer in adata.layers
    ), "No predictions layer in anndata object, seems predictions have failed!"

    return adata


def add_predictions_to_anndata(
    config: dict,
    predictions: np.ndarray,
    adata: ad.AnnData,
) -> ad.AnnData:
    """Add predictions as new layer to anndata

    Parameters
    ----------
    config: dict
        Configuration dictionary as parsed form the config_anndata_eval.yml
        config file.
    predictions: np.ndarray
        Predictions formatted as numpy array (genes x cells). Must match the
        dimensions of the provided anndata object.
    adata: ad.AnnData
        Anndata object in gene (obs) x cells (var) orientation, requires
        sequence embeddings stored as 'seq_embedding' slot in .obsm.

    Returns
    -------
    adata: ad.Anndata
        Same as anndata frame input with predictions added as layer.
    """
    # subset adata object if subset predictions were run
    if config["task"]["pred_on"] != "all":
        assert config["task"]["pred_on"] in ["train", "test", "valid"], (
            "If not predicting on 'all' data, 'pred_on' needs to be "
            "set to 'train' 'test' or 'valid'!"
        )
        logger.info(
            f"Subsetting the anndata object to store to {config['task']['pred_on']}"
        )

        if genes_subset_col is None:
            adata = adata[adata.obs[set_split_obs] == config["task"]["pred_on"], :]
        else:
            adata = adata[
                (adata.obs[set_split_obs] == config["task"]["pred_on"])
                & adata.obs[genes_subset_col],
                :,
            ]
    # if run on all data but subset on genes -> subset anndata frame
    if genes_subset_col is not None:
        adata = adata[adata.obs[genes_subset_col], :]

    # add predictions as new anndata layer
    adata.layers[prediction_layer] = predictions

    return adata


def prepare_metric_adata(
    config: dict, adata: ad.AnnData
) -> Tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
    """Subset anndata object for separate evaluation

    Parameters
    ----------
    config: dict
        Configuration dictionary as parsed form the config_anndata_eval.yml
        config file.
    adata: ad.AnnData
        Anndata object in gene (obs) x cells (var) orientation, requires
        sequence embeddings stored as 'seq_embedding' slot in .obsm and
        predictions in the layer indicated by 'predictions_layer'.

    Returns
    -------
    Tuple[ad.AnnData, ad.AnnData, ad.AnnData]
        Tuple of subset anndata objects
        [all, highly_variable, highly_expressed] genes with the later to
        being returned as None if evaluation on highly_variable and
        highly_expressed genes is not asked for accoridng to the config.
    """
    # to return None in the Tuple if not evaluated separately
    metric_adata_hv = None
    metric_adata_he = None

    # for writing temp files (if needed)
    if config["resource"]["backed_mode"]:
        backed_mode_temp_file = config["resource"]["backed_mode_temp_h5ad"]
        backed_mode_temp_file_stem = os.path.splitext(backed_mode_temp_file)[0]
        # Note the *predict.h5ad file will have predictions in the .X slot and
        # observed counts in the layer "observed"
        backed_mode_temp_file_hv = backed_mode_temp_file_stem + "_hv.h5ad"
        backed_mode_temp_file_he = backed_mode_temp_file_stem + "_he.h5ad"

    # Subset anndata object
    if config["task"]["eval_on"] == "all":
        metric_adata = adata
    else:
        # if running in backed mode temp h5ad files are required to
        # create views from
        if config["resource"]["backed_mode"]:
            logger.info(f"Writing subset temp h5ad file to {backed_mode_temp_file}")
            # all data
            metric_adata = adata[
                adata.obs[set_split_obs] == config["task"]["eval_on"], :
            ].copy(filename=backed_mode_temp_file)

        else:
            # in non backed mode views will suffice
            metric_adata = adata[
                adata.obs[set_split_obs] == config["task"]["eval_on"], :
            ]

    # regardless of set evaluated on create views (or temp backed files)
    # for highly exp and highly variable genes if they shall be evaluated
    # on separately
    if config["resource"]["backed_mode"]:
        # highly variable
        logger.info(
            "Writing temp files for backed mode operations:\n"
            f"{backed_mode_temp_file_hv}\n"
            f"{backed_mode_temp_file_he} ..."
        )
        if config["task"]["eval_highly_variable"]:
            metric_adata_hv = metric_adata[metric_adata.obs["highly_variable"], :].copy(
                filename=backed_mode_temp_file_hv
            )
        if config["task"]["eval_highly_expressed"]:
            # highly expressed
            metric_adata_he = metric_adata[
                metric_adata.obs["highly_expressed"], :
            ].copy(filename=backed_mode_temp_file_he)

    else:
        if config["task"]["eval_highly_variable"]:
            metric_adata_hv = metric_adata[metric_adata.obs["highly_variable"], :]
        if config["task"]["eval_highly_expressed"]:
            metric_adata_he = metric_adata[metric_adata.obs["highly_expressed"], :]

    # Report Number of genes in set for metric calculation
    num_of_g_in_set = len(metric_adata)
    logger.info(f"Num of genes in metric set: {num_of_g_in_set}")

    if config["task"]["eval_highly_variable"]:
        num_of_hv_in_set = sum(metric_adata.obs["highly_variable"])
        logger.info(
            f"Num of highly variable genes in metric set:" f" {num_of_hv_in_set}"
        )
    if config["task"]["eval_highly_expressed"]:
        num_of_he_in_set = sum(metric_adata.obs["highly_expressed"])
        logger.info(
            f"Num of highly expressed genes in metric set:" f" {num_of_he_in_set}"
        )

    return (metric_adata, metric_adata_hv, metric_adata_he)


def calculate_cross_gene_correlation(
    config: dict,
    metric_adata: ad.AnnData,
    metric_adata_hv: ad.AnnData,
    metric_adata_he: ad.AnnData,
) -> Tuple[float, float, float]:
    """Calculate the cross gene correlation

    Parameters
    ----------
    config: dict
        Configuration dictionary as parsed form the config_anndata_eval.yml
        config file.
    metric_adata: ad.AnnData
        Anndata object in gene (obs) x cells (var) orientation, subset to
        desired genes over which to calculate the correlations. Requires
        predictions in the layer indicated by 'predictions_layer'.
    metric_adata_hv: ad.AnnData
        Same as metric_adata but subset to genes flagged as highly_variable.
    metric_adata_he: ad.AnnData
        Same as metric_adata but subset to genes flagged as highly_expressed.
    metric_adata: ad.AnnData
        Anndata object in gene (obs) x cells (var) orientation, subset to
        desired genes over which to calulcate the correlations. Requires
        predictions in the layer indicated by 'predictions_layer'.

    Returns
    -------
    Tuple[float, float, float]
        Cross-gene correlation for all genes and highly_variable and
        highly_expressed if supplied as AnnDataFrame views.

    Notes
    -----
    If specified to not evaluate on highly variable or highly expressed
    separately, None values will be supplied to the function not triggering
    the respective eval run.
    """
    logger.info("Correlate obs and pred across genes ...")

    cor_all = get_across_x_correlation(
        metric_adata, axis=0, pred_layer=prediction_layer, return_per_entry=False
    )

    logger.info(
        f"Mean Pearson correlation across genes: " f"{np.round(cor_all, decimals=3)}"
    )

    cor_return = {
        'cor_all': cor_all,
        'cor_hv': 0.0,
        'cor_he': 0.0
    }

    if config["task"]["eval_highly_variable"]:
        cor_hv = get_across_x_correlation(
            metric_adata_hv,
            axis=0,
            pred_layer=prediction_layer,
            return_per_entry=False,
        )

        cor_return['cor_hv'] = cor_hv

        logger.info(
            f"Mean Pearson correlation across highly "
            f"variable genes: "
            f"{np.round(cor_hv, decimals=3)}"
        )

    if config["task"]["eval_highly_expressed"]:
        cor_he = get_across_x_correlation(
            metric_adata_he,
            axis=0,
            pred_layer=prediction_layer,
            return_per_entry=False,
        )

        cor_return['cor_he'] = cor_he

        logger.info(
            f"Mean Pearson correlation across highly "
            f"expressed genes: "
            f"{np.round(cor_he, decimals=3)}"
        )

    return cor_return


def calculate_cross_cell_correlation(
    config: dict,
    metric_adata: ad.AnnData,
    metric_adata_hv: ad.AnnData,
    metric_adata_he: ad.AnnData,
) -> Tuple[float, pd.DataFrame, float, float]:
    """Calculate the cross cell correlation

    Parameters
    ----------
    config: dict
        Configuration dictionary as parsed form the config_anndata_eval.yml
        config file.
    metric_adata: ad.AnnData
        Anndata object in gene (obs) x cells (var) orientation, subset to
        desired genes over which to calculate the correlations. Requires
        predictions in the layer indicated by 'predictions_layer'.
    metric_adata_hv: ad.AnnData
        Same as metric_adata but subset to genes flagged as highly_variable.
    metric_adata_he: ad.AnnData
        Same as metric_adata but subset to genes flagged as highly_expressed.
    metric_adata: ad.AnnData
        Anndata object in gene (obs) x cells (var) orientation, subset to
        desired genes over which to calulcate the correlations. Requires
        predictions in the layer indicated by 'predictions_layer'.

    Returns
    -------
    Tuple[float, float, float]
        Cross-cell correlation for all genes and highly_variable and
        highly_expressed if supplied as AnnDataFrame views.
    cor_per_gene: pd.DataFrame
        For all genes pd.DataFrame listing the cross-cell correlation
        per gene.

    Notes
    -----
    If specified to not evaluate on highly variable or highly expressed
    separately, None values will be supplied to the function not triggering
    the respective eval run.
    """
    logger.info("Correlate obs and pred across cells ...")

    (cor_all, cor_per_gene,) = get_across_x_correlation(
        metric_adata, axis=1, pred_layer=prediction_layer, return_per_entry=True
    )

    logger.info(
        f"Mean Pearson correlation across cell(type)s: "
        f"{np.round(cor_all, decimals=3)}"
    )

    cor_return = {
        'cor_all': cor_all,
        'cor_hv': 0.0,
        'cor_he': 0.1
    }

    #  for highly variable genes
    if config["task"]["eval_highly_variable"]:
        cor_hv = get_across_x_correlation(
            metric_adata_hv,
            axis=1,
            pred_layer=prediction_layer,
            return_per_entry=False,
        )

        cor_return['cor_hv'] = cor_hv

        logger.info(
            f"Mean Pearson correlation across cells for "
            f"highly variable genes: "
            f"{np.round(cor_hv, decimals=3)}"
        )

    #  for highly expressed genes
    if config["task"]["eval_highly_expressed"]:
        cor_he = get_across_x_correlation(
            metric_adata_he,
            axis=1,
            pred_layer=prediction_layer,
            return_per_entry=False,
        )

        cor_return['cor_he'] = cor_he

        logger.info(
            f"Mean Pearson correlation across cells for "
            f"highly expressed genes: "
            f"{np.round(cor_he, decimals=3)}"
        )

    return cor_return, cor_per_gene

def calculate_correlations(config: dict, adata: ad.AnnData) -> dict:
    """Calculate cross-gene and cross-cell correlations

    Parameters
    ----------
    config: dict
        Configuration dictionary as parsed form the config_anndata_eval.yml
        config file.
    adata: ad.AnnData
        Anndata object in gene (obs) x cells (var) orientation, requires
        predictions in the layer indicated by 'predictions_layer'.

    Returns
    -------
    metric_dict: dict
        Dictionary harboring the caluclated correlation metrics and per gene
        cross cell correlation pandas data frame.
    """
    assert prediction_layer in adata.layers, (
        f"Requires a predicted " f"counts layer under " f"provided {prediction_layer}!"
    )

    assert config["task"]["eval_on"] in [
        "train",
        "test",
        "valid",
        "all",
    ], "Select 'train' 'test' 'valid' or 'all' as 'eval_on' parameter."

    logger.info("Calculating evaluation metrics ...")

    metric_dict = {}

    metric_adata, metric_adata_hv, metric_adata_he = prepare_metric_adata(config, adata)

    cross_gene_correlations = calculate_cross_gene_correlation(
        config, metric_adata, metric_adata_hv, metric_adata_he
    )

    cross_cell_correlations, cross_cell_cor_per_gene = calculate_cross_cell_correlation(
        config, metric_adata, metric_adata_hv, metric_adata_he
    )

    metric_dict["mean_cross_gene_cor_all"] = cross_gene_correlations['cor_all']
    metric_dict["mean_cross_gene_cor_hv"] = cross_gene_correlations['cor_hv']
    metric_dict["mean_cross_gene_cor_he"] = cross_gene_correlations['cor_he']

    metric_dict["mean_cross_cell_cor_all"] = cross_cell_correlations['cor_all']
    metric_dict["cross_cell_cor_per_gene_all"] = cross_cell_cor_per_gene
    metric_dict["mean_cross_cell_cor_hv"] = cross_cell_correlations['cor_hv']
    metric_dict["mean_cross_cell_cor_he"] = cross_cell_correlations['cor_he']

    return metric_dict


def save_correlations(
        config: dict,
        metric_dict: dict,
        output_path: str
) -> None:
    """Save calculated correlations

    Parameters
    ----------
    config: dict
        Configuration dictionary as parsed form the config_anndata_eval.yml
        config file.
    metric_dict: dict
        Dictionary harboring the caluclated correlation metrics and per gene
        cross cell correlation pandas data frame.
    output_path: str,
        Directory to store the results.
    """
    # assemble metrics in data frame
    assemble_gene_set = ["all"]
    assemble_cross_gene = [metric_dict["mean_cross_gene_cor_all"]]
    assemble_cross_cell = [metric_dict["mean_cross_cell_cor_all"]]

    if config["task"]["eval_highly_variable"]:
        assemble_gene_set.append("highly_variable")
        assemble_cross_gene.append(metric_dict["mean_cross_gene_cor_hv"])
        assemble_cross_cell.append(metric_dict["mean_cross_cell_cor_hv"])

    if config["task"]["eval_highly_expressed"]:
        assemble_gene_set.append("highly_expressed")
        assemble_cross_gene.append(metric_dict["mean_cross_gene_cor_he"])
        assemble_cross_cell.append(metric_dict["mean_cross_cell_cor_he"])

    df_matrix_metrics = pd.DataFrame(
        {
            "gene_set": assemble_gene_set,
            "cross_gene": assemble_cross_gene,
            "cross_cell": assemble_cross_cell,
        },
    )

    # save per gene cross-cell correlation
    metric_dict["cross_cell_cor_per_gene_all"].to_csv(
        f"{output_path}/df_eval_cross_cell_correlation_per_gene" 
        f"_{config['task']['eval_on']}.tsv",
        sep="\t",
        header=False,
        index=True,
    )

    # save summary df of mean correlations
    df_matrix_metrics.to_csv(
        f"{output_path}/df_eval_mean_cor_summary"
        f"_{config['task']['eval_on']}.tsv",
        sep="\t",
        index=False,
    )


if __name__ == "__main__":

    # 1) Prep ============================
    # load config
    cli_config = OmegaConf.from_cli()
    with open(cli_config.config_file, "r") as f:
        config = yaml.safe_load(f)
    config = OmegaConf.to_object(OmegaConf.merge(config, cli_config))

    # overwrite if cli supplied checkpoint location
    if 'checkpoint_file' in cli_config:
        config['data']['model_chkpt_path'] = cli_config.checkpoint_file

    # log in debug mode?
    if config["debug"]:
        logging.basicConfig(level=logging.INFO)

    # store shorthands for frequently used config params
    prediction_layer = config["data"]["predictions_layer"]
    output_path = config["data"]["output_path"]
    # data_split expects anndata observation column name that indicates which
    # gene is train, test and valid
    set_split_obs = config["data"]["split_name"]

    # check and set if observations are to be subset by .obs column
    # e.g. highly expressed / highly variable
    genes_subset_col = config["task"]["subset_genes_column"]
    if genes_subset_col == "None":
        genes_subset_col = None

    os.makedirs(config["data"]["output_path"], exist_ok=True)

    adata = load_anndata(config)

    # os.chdir(output_path)

    # 2) Run model predictions ===========
    if config["task"]["run_pred"]:
        adata = predict(adata, config)

    # 3) Calculate cross-cell(type) and cross-gene correlation ========
    if config["task"]["run_eval"]:
        metric_dict = calculate_correlations(config, adata)

        # 4) Save metrics ====================
        save_correlations(config, metric_dict, output_path)

    # 5)  ================================
    if os.path.exists(config["resource"]["backed_mode_temp_h5ad"]):
        os.remove(config["resource"]["backed_mode_temp_h5ad"])
