"""
Embeddings2Target model wrapper

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

Model wrapper for pytorch lightning models
from Enformer pre-computed embeddings or pre-computed targets to
observed Enformer targets
"""

from typing import Literal, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchmetrics.functional import pearson_corrcoef

from seq2cells.metrics_and_losses.losses import (
    BalancedPearsonCorrelationLoss,
    poisson_loss,
)
from seq2cells.utils.training_utils import (
    get_linear_warm_up_cosine_decay_scheduler,
)


class Embedding2Target(pl.LightningModule):
    """A single FC layer model to predict targets from embeddings."""

    def __init__(
        self,
        emb_dim: int,
        target_dim: int,
        loss: str,
        learning_rate: float,
        lr_schedule: str,
        lr_warmup_epochs: int,
        lr_max_epochs: int,
        optimizer: str,
        weight_decay: float,
        dropout_prob: Optional[float] = 0.0,
        model_trunk: Optional[torch.nn.Module] = None,
        softplus: Optional[bool] = True,
        target_is_log: Optional[bool] = False,
        log_train: Optional[bool] = False,
        log_validate: Optional[bool] = True,
        std_validate: Optional[bool] = False,
        model_type: Optional[Literal["linear", "provided"]] = "linear",
        mode: Optional[Literal["bulk", "single_cell"]] = "bulk",
        freeze: Optional[str] = None,
        bottleneck_dim: Optional[int] = None,
        bottleneck_nonlin: Optional[str] = None,
        rel_weight_gene: Optional[float] = 1.0,
        rel_weight_cell: Optional[float] = 1.0,
        pears_norm_mode: Literal["mean", "nonzero_median"] = "mean",
        train_eval: bool = True,
    ) -> None:
        """Initialize the model

        Parameters
        ----------
            emb_dim: int
                Dimension of the embedding prior to the linear mapping.
            target_dim : int
                Number of targets
            loss : str
                'mse' or 'poisson' or 'poissonnll' or 'pearson'
                to apply mean-squared-error or poisson loss or
                poisson-nll-loss or a balanced Pearson correlation
                based loss respectively
            learning_rate: float
                Learning rate for optimizer default 1e-3
            lr_schedule:
                Select learning rate schedule: 'linear_warm_up_cosine_decay'
                or 'linear_warm_up'
            lr_warmup_epochs: int
                Warmup epochs to reach the provided learning_rate.
            lr_max_epochs: int
                Maximum number of epochs for lr schedule.
            optimizer: str
                Select desired optimizer to run.
            weight_decay: float
                Weight decay parameter for AdamW.
            dropout_prob: float
                Dropout probability > 0 will add a dropout layer in front of
                bottleneck layers. Will not apply in linear layer only models.
            model_trunk: Optional[torch.nn.Module],
                A torch module to use as model trunk
            softplus: bool
                If to append a softplus layer after the linear layer.
            target_is_log: bool
                Specify if the target counts/coverage have already been log
                transformed.
            log_train: bool
                Train against log(x+1) transformed data.
            log_validate: bool
                Validate using log(x+1) transformed data.
            std_validate: bool
                Standardize data across ROI / tss for validation.
            model_type: Optional[str] = "linear"
                Define if the model is a simple 'linear' model to  be
                applied on the embeddings or a provided model that goes from
                whatever input to intermediate embedding layer to output.
                Or a "bottleneck" model applying a bottleneck layer before the
                final linear layer.
            mode: Optional[str] = 'bulk'
                Model mode running against 'bulk' or 'single_cell' data.
            freeze: Optional[str] = None
                Specify if to freeze parts of the network. Must be None
                or 'trunk'.
            bottleneck_dim: Optional[int] = None
                Hidden dimension of the bottleneck layer.
            bottleneck_nonlin: Optional[str] = None
                Which non linearity to apply in bottleneck layer.
                None --> Apply no non linearity
                'RELU' --> apply RELU
            rel_weight_gene: Optional[float] = 1.0,
                For Balanced Pearson loss: relative weight to place on across
                gene correlation.
            rel_weight_cell: Optional[float] = 1.0,
                For Balanced Pearson loss: relative weight to place on across
                cell correlation.
            pears_norm_mode: Literal["mean", "nonzero_median"] = "mean"
                For Balanced Pearson loss: which average mode to use for
                norming the batches. ['mean', 'nonzero_median'] Default = mean
            train_eval: bool = True
                Set False to skip training set evaluation (e.g. for large
                models).
        """
        super().__init__()

        self.save_hyperparameters(ignore=["model_trunk"])
        self.model_trunk = model_trunk

        # save freeze hyperparam
        self.freeze = freeze

        # log if model tpye needs a trunk
        self.use_trunk = self.hparams.model_type in ["provided", "provided_bottleneck"]

        if self.model_trunk is not None:
            if self.freeze == "trunk":
                # set model trunk parameters to frozen
                for p in self.model_trunk.parameters():
                    p.requires_grad = False

        assert self.hparams.loss in [
            "mse",
            "poisson",
            "poissonnll",
            "pearson",
        ], "Select a valid loss function: 'mse' or 'poisson' or 'poissonnll'!"

        # init pearson loss if specified
        if self.hparams.loss == "pearson":
            self.pearson_loss = BalancedPearsonCorrelationLoss(
                rel_weight_gene=self.hparams.rel_weight_gene,
                rel_weight_cell=self.hparams.rel_weight_cell,
                norm_by=self.hparams.pears_norm_mode,
            )

        # setup the model =================
        assert self.hparams.model_type in [
            "linear",
            "provided",
            "bottleneck",
            "provided_bottleneck",
        ], (
            "Select a valid model_type: 'linear', 'provided', 'bottleneck', "
            "'provided_bottleneck!'"
        )
        modules = []

        if self.hparams.model_type in ["linear", "provided"]:
            modules.append(nn.Linear(self.hparams.emb_dim, self.hparams.target_dim))
        elif self.hparams.model_type in ["bottleneck", "provided_bottleneck"]:

            # from emb to bottleneck dim
            modules.append(nn.Linear(self.hparams.emb_dim, self.hparams.bottleneck_dim))

            # dropout if drop prob > 0
            if self.hparams.dropout_prob > 0.0:
                modules.append(nn.Dropout(p=self.hparams.dropout_prob))

            # add non linearity
            if self.hparams.bottleneck_nonlin == "RELU":
                modules.append(nn.ReLU())
            else:
                print("No non-linearity added!")

            # map from bottleneck to output dim
            modules.append(
                nn.Linear(self.hparams.bottleneck_dim, self.hparams.target_dim)
            )

        if self.hparams.softplus:
            modules.append(nn.Softplus())
        self.model_head = nn.Sequential(*modules)

        # check if optimizer supported
        assert self.hparams.optimizer in ["AdamW", "SGD", "RMSprop"], (
            "The selected optimizer is not supported. "
            "Select 'SGD', 'AdamW' or 'RMSprop'!"
        )

        # mode ['single_cell' or 'bulk'] defined what is reported on the
        # progress bar
        self.report_bar_across_genes = True
        if self.hparams.mode == "bulk":
            self.report_bar_across_celltypes = True
            self.report_bar_matrx = False
        else:
            self.report_bar_across_celltypes = False
            self.report_bar_matrx = True

    def forward(self, inputs):
        """Forward step separated."""
        if self.use_trunk:
            emb = self.model_trunk(inputs[0])
            y_hat = self.model_head(emb)
        else:
            y_hat = self.model_head(inputs[0])

        return y_hat

    def training_step(self, batch, batch_idx):
        """PL training step definition"""
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        if self.use_trunk:
            emb = self.model_trunk(x)
            y_hat = self.model_head(emb)
        else:
            y_hat = self.model_head(x)

        if self.hparams.target_is_log and not self.hparams.log_train:
            # transform back to exponentials
            y = torch.exp(y) - 1
            y_hat = torch.exp(y_hat) - 1

        if self.hparams.log_train and not self.hparams.target_is_log:
            # log transform expression data for correlation
            y = torch.log(y + 1)
            y_hat = torch.log(y_hat + 1)

        if self.hparams.loss == "mse":
            loss = F.mse_loss(y_hat, y)
        elif self.hparams.loss == "poisson":
            loss = poisson_loss(y_hat, y)
        elif self.hparams.loss == "poissonnll":
            loss = F.poisson_nll_loss(y_hat, y, log_input=False)
        elif self.hparams.loss == "pearson":
            loss = self.pearson_loss(y_hat, y)
        else:
            raise Exception("Select mse, poisson or poissonnll as loss hyperparam")

        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        return {"loss": loss, "y": y, "y_hat": y_hat}

    def validation_step(self, batch, batch_idx):
        """PL validation step definition"""
        x, y = batch
        if self.use_trunk:
            emb = self.model_trunk(x)
            y_hat = self.model_head(emb)
        else:
            y_hat = self.model_head(x)

        return {"y": y, "y_hat": y_hat}

    def training_epoch_end(self, outs):
        """PL to run training set eval after epoch"""
        if not self.hparams.train_eval:
            return

        y = torch.cat([x["y"] for x in outs])
        y_hat = torch.cat([x["y_hat"] for x in outs])

        if (
            self.hparams.log_validate
            and not self.hparams.log_train
            and not self.hparams.target_is_log
        ):
            # log transform expression data for correlation
            y = torch.log(y + 1)
            y_hat = torch.log(y_hat + 1)

        if self.hparams.std_validate:
            # calculate mean and std of observed and predicted targets
            # update class state
            self.target_means_obs = torch.mean(y, dim=0)
            self.target_stds_obs = torch.std(y, dim=0)
            self.target_means_pred = torch.mean(y_hat, dim=0)
            self.target_stds_pred = torch.std(y_hat, dim=0)
            # normalize output
            y = y - self.target_means_obs
            y = y / self.target_stds_obs
            y_hat = y_hat - self.target_means_pred
            y_hat = y_hat / self.target_stds_pred

        pc_across_tss = torch.stack(
            [pearson_corrcoef(y_hat[:, i], y[:, i]) for i in range(y_hat.shape[1])]
        )
        pc_across_tss = torch.nan_to_num(
            pc_across_tss
        )  # handle correlation NaN : default to 0
        mean_pc_across_tss = pc_across_tss.mean()

        pc_across_celltypes = torch.stack(
            [pearson_corrcoef(y_hat[i, :], y[i, :]) for i in range(y_hat.shape[0])]
        )
        pc_across_celltypes = torch.nan_to_num(
            pc_across_celltypes
        )  # handle correlation NaN : default to 0
        mean_pc_across_celltypes = pc_across_celltypes.mean()

        # calculate correlation of whole gene x cell(type) matrix
        pc_whole_matrix = pearson_corrcoef(torch.ravel(y_hat), torch.ravel(y))

        # outs is a list of whatever you returned in `validation_step`
        self.log(
            "train_corr_across_tss",
            mean_pc_across_tss,
            prog_bar=self.report_bar_across_genes,
        )
        self.log(
            "train_corr_across_celltypes",
            mean_pc_across_celltypes,
            prog_bar=self.report_bar_across_celltypes,
        )
        self.log("train_corr", pc_whole_matrix, prog_bar=self.report_bar_matrx)

    def validation_epoch_end(self, outs):
        """PL to run validation set eval after epoch"""
        y = torch.cat([outs[i]["y"] for i in range(len(outs))])
        y_hat = torch.cat([outs[i]["y_hat"] for i in range(len(outs))])

        if self.hparams.target_is_log and not self.hparams.log_validate:
            # transform back to exponentials
            y = torch.exp(y + 1)
            y_hat = torch.exp(y_hat + 1)

        if self.hparams.log_validate and not self.hparams.target_is_log:
            # log transform expression data for correlation
            y = torch.log(y + 1)
            y_hat = torch.log(y_hat + 1)

        if self.hparams.loss == "mse":
            val_loss = F.mse_loss(y_hat, y)
        elif self.hparams.loss == "poisson":
            val_loss = poisson_loss(y_hat, y)
        elif self.hparams.loss == "poissonnll":
            val_loss = F.poisson_nll_loss(y_hat, y, log_input=False)
        elif self.hparams.loss == "pearson":
            val_loss = self.pearson_loss(y_hat, y)
        else:
            raise Exception("Select mse, poisson or poissonnll as loss hyperparam")

        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, sync_dist=True)

        if self.hparams.std_validate:
            # normalize output
            if hasattr(self, "target_means_obs"):
                y = y - self.target_means_obs
                y = y / self.target_stds_obs
                y_hat = y_hat - self.target_means_pred
                y_hat = y_hat / self.target_stds_pred
            else:
                # for first epoch take mean and std of validation set
                y = y - torch.mean(y, dim=0)
                y = y / torch.std(y, dim=0)
                y_hat = y_hat - torch.mean(y_hat, dim=0)
                y_hat = y_hat / torch.std(y_hat, dim=0)

        pc_across_tss = torch.stack(
            [pearson_corrcoef(y_hat[:, i], y[:, i]) for i in range(y_hat.shape[1])]
        )
        pc_across_tss = torch.nan_to_num(
            pc_across_tss
        )  # handle correlation NaN : default to 0
        mean_pc_across_tss = pc_across_tss.mean()

        pc_across_celltypes = torch.stack(
            [pearson_corrcoef(y_hat[i, :], y[i, :]) for i in range(y_hat.shape[0])]
        )
        pc_across_celltypes = torch.nan_to_num(
            pc_across_celltypes
        )  # handle correl NaN : default to 0
        mean_pc_across_celltypes = pc_across_celltypes.mean()

        # calculate correlation of whole gene x cell(type) matrix
        pc_whole_matrix = pearson_corrcoef(torch.ravel(y_hat), torch.ravel(y))

        # outs is a list of whatever you returned in `validation_step`
        self.log(
            "valid_corr_across_tss",
            mean_pc_across_tss,
            prog_bar=self.report_bar_across_genes,
        )
        self.log(
            "valid_corr_across_celltypes",
            mean_pc_across_celltypes,
            prog_bar=self.report_bar_across_celltypes,
        )
        self.log("valid_corr", pc_whole_matrix, prog_bar=self.report_bar_matrx)
        self.log("hp_metric", val_loss)

    def test_step(self, batch, batch_idx):
        """PL test step definition"""
        """PL validation step definition"""
        x, y = batch
        if self.use_trunk:
            emb = self.model_trunk(x)
            y_hat = self.model_head(emb)
        else:
            y_hat = self.model_head(x)

        return {"y": y, "y_hat": y_hat}

    def test_epoch_end(self, outs):
        """PL to run validation set eval after epoch"""
        y = torch.cat([outs[i]["y"] for i in range(len(outs))])
        y_hat = torch.cat([outs[i]["y_hat"] for i in range(len(outs))])

        if self.hparams.log_validate:
            # log transform expression data for correlation
            y = torch.log(y + 1)
            y_hat = torch.log(y_hat + 1)

        if self.hparams.std_validate:
            # normalize output
            y = y - self.target_means_obs
            y = y / self.target_stds_obs
            y_hat = y_hat - self.target_means_pred
            y_hat = y_hat / self.target_stds_pred

        pc_across_tss = torch.stack(
            [pearson_corrcoef(y_hat[:, i], y[:, i]) for i in range(y_hat.shape[1])]
        )
        pc_across_tss = torch.nan_to_num(
            pc_across_tss
        )  # handle correlation NaN : default to 0
        mean_pc_across_tss = pc_across_tss.mean()

        pc_across_celltypes = torch.stack(
            [pearson_corrcoef(y_hat[i, :], y[i, :]) for i in range(y_hat.shape[0])]
        )
        pc_across_celltypes = torch.nan_to_num(
            pc_across_celltypes
        )  # handle correl NaN : default to 0
        mean_pc_across_celltypes = pc_across_celltypes.mean()

        # calculate correlation of whole gene x cell(type) matrix
        pc_whole_matrix = pearson_corrcoef(torch.ravel(y_hat), torch.ravel(y))

        # outs is a list of whatever you returned in `validation_step`
        self.log("test_corr_across_tss", mean_pc_across_tss)
        self.log("test_corr_across_celltypes", mean_pc_across_celltypes)
        self.log("test_corr", pc_whole_matrix)

    def configure_optimizers(self):
        """PL configure optimizer"""
        if self.hparams.optimizer == "AdamW":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "SGD":
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "RMSprop":
            optimizer = optim.RMSprop(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )

        if self.hparams.lr_schedule == "linear_warm_up":
            # mimicking the Enformer learning rate scheduler
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1e-04,
                    end_factor=1.0,
                    total_iters=self.hparams.lr_warmup_epochs,
                ),
                "name": "lr_logging",
            }

        elif self.hparams.lr_schedule == "linear_warm_up_cosine_decay":
            # linear warump with cosine decay
            lr_scheduler = {
                "scheduler": get_linear_warm_up_cosine_decay_scheduler(
                    optimizer,
                    lr_warmup_epochs=self.hparams.lr_warmup_epochs,
                    lr_max_epochs=self.hparams.lr_max_epochs,
                ),
                "name": "lr_logging",
            }
        elif self.hparams.lr_schedule == "constant":
            # constant learning rate
            lr_scheduler = {
                "scheduler": optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=lambda epoch: 1
                ),
                "name": "lr_logging",
            }
        elif self.hparams.lr_schedule == "reduce_on_plateau":
            lr_scheduler = {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=0.25,
                    patience=2,
                ),
                "monitor": "valid_corr_across_tss",
                "name": "lr_logging",
            }

        return [optimizer], [lr_scheduler]


class EnformerTrunk(nn.Module):
    """Trunk of the Enformer model

    Notes
    -----
    This class handles the enformer trunk from the pretrained model to make
    it available as module for other models.
    Important, rather than extracting all sequence embeddings over the
    sequence dimension, a single sequence embedding over a bin id of
    interest is extracted.
    """

    def __init__(self, enformer: torch.nn.Module) -> None:
        """Initialize an Enformer trunk from the pretrained model

        Parameters
        ----------
        enformer: torch.nn.Module
            Full Enformer model from pytorch public implementation.
        """
        super().__init__()
        self.enf = enformer._trunk

    def forward(
        self,
        seq: torch.Tensor,
        bin_to_extract: Union[int, torch.Tensor],
        add_bin: int = 0,
        bin_merge_mode: Literal["mean", "sum", "nomerge"] = "mean",
    ) -> torch.Tensor:
        """Forward pass of the enformer trunk given a DNA sequence as input

        Parameters
        ----------
        seq: torch.Tensor
            One hot encoded sequence of shape [N x seq_length x 4] with N
            indexing in the batch size.
        bin_to_extract: Union[int, torch.Tensor]
            Which single prediction bin to extract from the Enformer
            prediction vector. If Tensor expects shape of [n] where n are the
            number of bins of interest to extract.
        add_bin: int
            Number of adjacent bins in either sequence direction that should
            be merged with the selected (central) bin to extract from the
            Enformer sequence embeddings.
            Default = 0 -> no adjacent bins are combined with the selected.
        bin_merge_mode: Literal['mean', 'sum']
            Select which aggregation method to use for merging adjacent bins
            to extract. Default = 'mean' | 'nomerge' only for testing

        Returns
        -------
        out: torch.Tensor
            Tensor with the embedding of shape 1 x embedding dim

        Notes
        -----
        Passes the one hot encoded sequence through the Enformer trunk
        to produce embeddings. Extracts the embedding of a single bin of
        interest from the Enformer trunk output. For example, for gene TSS
        workflow this will be the central bin of the sequence.
        """
        out = self.enf(seq)
        # handle integer or tensor input
        # to extract same or different embeddings along batch dimension
        if isinstance(bin_to_extract, int):
            # slice the same sequence dimension embedding for all
            out = out[
                :, range(bin_to_extract - add_bin, bin_to_extract + add_bin + 1), :
            ]
        else:
            # extract embedding for each entry individually
            bin_to_extract = bin_to_extract.long()
            out_list = [
                out[
                    i,
                    range(bin_to_extract[i] - add_bin, bin_to_extract[i] + add_bin + 1),
                    :,
                ]
                for i in range(out.shape[0])
            ]
            out = torch.stack(out_list)

        if add_bin != 0:
            if bin_merge_mode == "mean":
                out = out.mean(axis=1, keepdim=True)
            elif bin_merge_mode == "sum":
                out = out.sum(axis=1, keepdim=True)

        return out


class Sequence2Embedding2Target(pl.LightningModule):
    """Combine a model trunk and head

    Notes
    -----
    Trunk takes sequence and outputs embedding.
    Head takes embedding and outputs predictions per cell(type).
    """

    def __init__(
        self,
        trunk: torch.nn.Module,
        head: torch.nn.Module,
        add_bin: int = 0,
    ) -> None:
        """Init model with trunk and head"""
        super().__init__()
        self.trunk = trunk
        self.head = head
        self.add_bin = add_bin

    def forward(
        self, seq: torch.Tensor, bin_to_extract: int, return_emb: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward pass

        Parameters
        ----------
        seq: torch.Tensor
            One hot encoded sequence of shape [N x seq_length x 4] with N
            indexing in the batch size.
        bin_to_extract: Union[int, torch.Tensor]
            Which single prediction bin to extract from the trunk
            prediction vector.
        return_emb: bool = False
            If True will return predictions and the embeddings extracted from
            the trunk.

        Returns
        -------
        out: torch.Tensor
            Tensor with the of shape 1 x target dimension
        (emb): torch.Tensor
            If return_emb == True will also output the embedding (output of
            the model trunk) of shape [1 x embedding_dimension]
        """
        emb = self.trunk(seq, bin_to_extract, add_bin=self.add_bin)

        out = self.head(emb)

        if return_emb:
            return out, emb
        else:
            return out


class Variant2Embedding2Target(Sequence2Embedding2Target):
    """Combine a model trunk and head for VEP

    Notes
    -----
    Trunk takes sequence and outputs embedding.
    Head takes embedding and outputs predictions per cell(type).
    Model takes as input a reference and a variant sequence and the bin at
    which to extract the embedding from the trunk output.
    """

    def __init__(
        self,
        trunk: torch.nn.Module,
        head: torch.nn.Module,
        return_emb: bool = False,
        add_bin: int = 0,
    ) -> None:
        """Init model with trunk and head"""
        super().__init__(trunk, head)
        self.trunk = trunk
        self.head = head
        self.return_emb = return_emb
        # control adding bins
        self.add_bin = add_bin

    def predict_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """Modifying the default predict step

        Notes
        -----
        Runs trunk and head for reference and variant sequence.
        Outputs the predicted targets and optionally the trunk embeddings.
        """
        ref_seq = batch[0]
        var_seq = batch[1]
        roi_bin_idx = batch[2]

        ref_emb = self.trunk(ref_seq, roi_bin_idx, add_bin=self.add_bin)
        var_emb = self.trunk(var_seq, roi_bin_idx, add_bin=self.add_bin)

        ref_out = self.head(ref_emb)
        var_out = self.head(var_emb)

        if self.return_emb:
            return ref_out, var_out, ref_emb, var_emb
        else:
            return ref_out, var_out
