"""
Utilities for predictions at inference time
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
import os

import pytorch_lightning as pl
import torch


class BatchWiseWriter(pl.callbacks.BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        """Write predictions and batch indices to disk per batch"""
        torch.save(
            batch_indices,
            os.path.join(
                self.output_dir,
                f"batch_dl{dataloader_idx}"
                f"_ba{batch_idx}"
                f"_tr{trainer.global_rank}.pt",
            ),
        )

        # save batch indices
        torch.save(
            prediction,
            os.path.join(
                self.output_dir,
                f"pred_dl{dataloader_idx}"
                f"_ba{batch_idx}"
                f"_tr{trainer.global_rank}.pt",
            ),
        )
