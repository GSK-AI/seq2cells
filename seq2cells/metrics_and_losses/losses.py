"""
Custom losses for aiml-seq2cells
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

import torch


def log(t, eps=1e-20):
    """Custom log function clamped to minimum epsilon."""
    return torch.log(t.clamp(min=eps))


def poisson_loss(pred: torch.Tensor, target: torch.Tensor):
    """Poisson loss"""
    return (pred - target * log(pred)).mean()


def nonzero_median(tensor: torch.Tensor, axis: int, keepdim: bool) -> torch.Tensor:
    """Compute the median across non-zero float elements.

    Notes
    -----
    Modifies the tensor in place to avoid making a copy.
    """
    tensor = torch.where(tensor != 0.0, tensor.double(), float("nan"))

    # returns values and indices - we only want the value(s)
    medians = torch.nanmedian(tensor, dim=axis, keepdim=keepdim)[0]

    medians = medians.nan_to_num(0)

    return medians


class BalancedPearsonCorrelationLoss(torch.nn.Module):
    """Pearson Corr balances between across gene and cell performance"""

    def __init__(
        self,
        rel_weight_gene: float = 1.0,
        rel_weight_cell: float = 1.0,
        norm_by: Literal["mean", "nonzero_median"] = "mean",
        eps: float = 1e-8,
    ):
        """Initialise PearsonCorrelationLoss.

        Parameter
        ---------
        rel_weight_gene: float = 1.0
            The relative weight to put on the across gene/tss correlation.
        rel_weight_cell: float = 1.0
            The relative weight to put on the across cells correlation.
        norm_by:  Literal['mean', 'nonzero_median'] = 'nonzero_median'
            What to use as across gene / cell average to subtract from the
            signal to normalise it. Mean or the Median of the non zero entries.
        eps: float 1e-8
            epsilon
        """
        super().__init__()
        self.eps = eps
        self.norm_by = norm_by
        self.rel_weight_gene = rel_weight_gene
        self.rel_weight_cell = rel_weight_cell

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward.

        Notes
        -----
        preds: torch.Tensor
            2D torch tensor [genes x cells], batched over genes.
        targets: torch.Tensor
            2D torch tensor [genes x cells], batched over genes.
        """
        if self.norm_by == "mean":
            preds_avg_gene = preds.mean(dim=0, keepdim=True)
            targets_avg_gene = targets.mean(dim=0, keepdim=True)
            preds_avg_cell = preds.mean(dim=1, keepdim=True)
            targets_avg_cell = targets.mean(dim=1, keepdim=True)
        else:
            preds_avg_gene = nonzero_median(preds, 0, keepdim=True)
            targets_avg_gene = nonzero_median(targets, 0, keepdim=True)
            preds_avg_cell = nonzero_median(preds, 1, keepdim=True)
            targets_avg_cell = nonzero_median(targets, 1, keepdim=True)

        r_tss = torch.nn.functional.cosine_similarity(
            preds - preds_avg_gene,
            targets - targets_avg_gene,
            eps=self.eps,
            dim=0,
        )

        r_celltype = torch.nn.functional.cosine_similarity(
            preds - preds_avg_cell,
            targets - targets_avg_cell,
            eps=self.eps,
        )

        loss = self.rel_weight_gene * (1 - r_tss.mean()) + self.rel_weight_cell * (
            1 - r_celltype.mean()
        )

        # norm the loss to 2 by half the sum of the relative weights
        loss = (loss * 2) / (self.rel_weight_gene + self.rel_weight_cell)

        return loss
