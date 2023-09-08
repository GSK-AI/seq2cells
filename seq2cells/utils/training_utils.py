"""
Training utilities
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

import math

from torch import optim


# Learning rate scheduling ==========================
# adapted from GSK internal: Jacob Deasy aiml-seq-scale
def linear_warm_up_cosine_decay(
    lr_warmup_epochs: int, lr_max_epochs: int, epoch: int
) -> float:
    """Cosine decay with linear warmup.

    Parameters
    ----------
    lr_warmup_epochs: int
        Number of linear warmup steps to perform.
    lr_max_epochs: int
        Maximum step number to scale cosine decay to.
    step: int
        The current optimization step.

    Returns
    -------
    float:
        The learning rate scaling value.
    """
    if epoch < lr_warmup_epochs:
        lr_scale = epoch / lr_warmup_epochs
    else:
        rel_epoch = epoch - lr_warmup_epochs
        rel_max_epoch = lr_max_epochs - lr_warmup_epochs
        # Cosine decay to 0.1 * optimizer_config["lr"]
        lr_scale = 0.1 + 0.9 * (
            0.5 * (1 + math.cos(math.pi * rel_epoch / rel_max_epoch))
        )

    return lr_scale


def get_linear_warm_up_cosine_decay_scheduler(
    optimizer: optim.Optimizer, lr_warmup_epochs: int, lr_max_epochs: int
):
    """Get a standard learning rate optimizer.

    Parameters
    ----------
    optimizer: optim.Optimizer
        The optimizer to be wrapped with a learning rate scaling function.
    lr_warmup_epochs: int
        Number of linear warmup epochs to perform.
    lr_max_epochs: int
        Maximum epochs number to scale cosine decay to.

    Notes
    -----
    ``linear_warm_up_cosine_decay`` is chosen because the warmup is friendly to
    Transformers and the final learning rates aren't inefficiently small.
    """
    return optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: linear_warm_up_cosine_decay(
            lr_warmup_epochs, lr_max_epochs, epoch
        ),
    )
