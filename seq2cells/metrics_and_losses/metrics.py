"""
Custom Metrics for seq2cells
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

import torch
from numpy import dot
from numpy.linalg import norm
from torch import Tensor
from torchmetrics import Metric


def cosine_sim(a, b):
    """Cosine similarity"""
    return dot(a, b) / (norm(a) * norm(b))


class MeanPearsonCorrCoefPerChannel(Metric):
    """Pearson correlation coefficient accumulated over updates.

    Parameters
    ----------
    n_channels: int
        The number of channels to record correlation along.
    dist_sync_on_step: bool = False
        Whether to synchronise metrics across processes at each update step.
    """

    is_differentiable: bool = False
    full_state_update: bool = False
    higher_is_better: bool = True

    def __init__(
        self, n_channels: int, dist_sync_on_step: bool = False, **kwargs
    ) -> None:
        """Initialze the class.

        Returns
        -------
        Tensor:
            The aggregate Pearson correlation.
        """
        super().__init__(
            dist_sync_on_step=dist_sync_on_step, full_state_update=False, **kwargs
        )
        self.reduce_dims = (0, 1)

        zeros = torch.zeros(n_channels, dtype=torch.float32)
        self.add_state("product", default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("true", default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("true_squared", default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("pred", default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("pred_squared", default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("count", default=zeros.clone(), dist_reduce_fx="sum")

    def update(self, pred: Tensor, target: Tensor) -> None:
        """Update the running values.

        Parameters
        ----------
        pred: Tensor
            Prediction Tensor.
        target: Tensor
            Target Tensor.
        """
        assert pred.shape == target.shape

        self.product += torch.sum(pred * target, dim=self.reduce_dims)
        self.true += torch.sum(target, dim=self.reduce_dims)
        self.true_squared += torch.sum(torch.square(target), dim=self.reduce_dims)
        self.pred += torch.sum(pred, dim=self.reduce_dims)
        self.pred_squared += torch.sum(torch.square(pred), dim=self.reduce_dims)
        self.count += torch.sum(torch.ones_like(target), dim=self.reduce_dims)

    def compute(self) -> Tensor:
        """Compute the metric at the current point, given the aggregated states.

        Returns
        -------
        Tensor:
            The aggregate Pearson correlation.
        """
        true_mean = self.true / self.count
        pred_mean = self.pred / self.count

        covariance = (
            self.product
            - true_mean * self.pred
            - pred_mean * self.true
            + self.count * true_mean * pred_mean
        )

        true_var = self.true_squared - self.count * torch.square(true_mean)
        pred_var = self.pred_squared - self.count * torch.square(pred_mean)
        tp_var = torch.sqrt(true_var) * torch.sqrt(pred_var)
        correlation = covariance / tp_var

        return correlation
