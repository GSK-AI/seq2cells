"""
Fine-tune a model on AnnData stored targets.
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
Provides flexibility to train from precomputed embedding or DNA sequence.
Can use the enformer model trunk to process / train from DNA sequence or
from pre-computed outputs (embeddings). The enformer trunk can be frozen or
fine-tuned jointly.
See the config archetype for explanations on different options:
aiml-seq2cells/research/configs/config_anndata_fine_tune.yml
"""

import logging
import os
import sys

import pytorch_lightning as pl
import yaml
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from seq2cells.data.enformer_tss_based import AnnDataEmbeddingTssDataset
from seq2cells.models.embedding2target import Embedding2Target, EnformerTrunk

# for debugging
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    # load config ==============================
    # Overwrite CLI args
    cli_config = OmegaConf.from_cli()
    with open(cli_config.config_file, "r") as f:
        config = yaml.safe_load(f)
    config = OmegaConf.to_object(OmegaConf.merge(config, cli_config))

    # debug?
    if config["debug"]:
        logging.basicConfig(level=logging.INFO)

    # pre asserts ==============================
    assert config["task"]["input"]["input_type"] in [
        "embeddings",
        "sequence",
    ], "No valid input_type selected!"

    # Prep data ================================
    # set up training and validation dataset according to task
    if config["task"]["input"]["input_type"] == "embeddings":
        train_data = AnnDataEmbeddingTssDataset(
            ann_in=config["data"]["dataset"]["ann_data_file"],
            data_split="train",
            split_name=config["data"]["dataset"]["split_name"],
            subset_genes_col=config["task"]["input"]["subset_genes_column"],
            use_layer=config["data"]["dataset"]["use_layer"],
            backed_mode=config["resource"]["backed_mode"],
        )
        valid_data = AnnDataEmbeddingTssDataset(
            ann_in=config["data"]["dataset"]["ann_data_file"],
            data_split="valid",
            split_name=config["data"]["dataset"]["split_name"],
            subset_genes_col=config["task"]["input"]["subset_genes_column"],
            use_layer=config["data"]["dataset"]["use_layer"],
            backed_mode=config["resource"]["backed_mode"],
        )
        if config["test"]["test_on"] != "test":
            test_data = AnnDataEmbeddingTssDataset(
                ann_in=config["data"]["dataset"]["ann_data_file"],
                data_split="test",
                split_name=config["data"]["dataset"]["split_name"],
                subset_genes_col=config["task"]["input"]["subset_genes_column"],
                use_layer=config["data"]["dataset"]["use_layer"],
                backed_mode=config["resource"]["backed_mode"],
            )

    elif config["task"]["input"]["input_type"] == "sequence":
        logger.error("Sequence input type not implemented for AnnData " "workflow yet.")
        sys.exit()

    # Seed and device settings ===========================
    pl.utilities.seed.seed_everything(config["seed"])

    # if only running testing skip many steps
    if config["optimization"]["run_train"]:
        # set up checkpoint saving ==================
        # combine a model tag with characterizing hparams
        model_tag = (
            f"{config['saving']['tb_log_prefix']}_"
            f"bs_{config['data']['loader']['batch_size']}_"
            f"drop_{config['optimization']['dropout_prob']}_"
            f"lr_{config['optimization']['optimizer']['lr']}_"
            f"wd_{config['optimization']['optimizer']['weight_decay']}_"
            f"we_{config['optimization']['scheduler']['warmup_epochs']}_"
            f"lrs_{config['optimization']['optimizer']['lr_schedule']}"
        )
        if not os.path.exists(config["saving"]["head_dir"]):
            os.mkdir(config["saving"]["head_dir"])
        ckpt_dir = os.path.join(
            config["saving"]["head_dir"], config["saving"]["checkpoint_dir"], model_tag
        )
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        # save config copy to checkpoint path
        OmegaConf.save(config=config, f=os.path.join(ckpt_dir, "config_hparams.yml"))

        # set up the checkpoint callback
        filename_string = (
            "{epoch}-{valid_corr_across_tss:.3f}"
            + "-{valid_corr_across_celltypes:.3f}-{val_loss:.3f}"
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            save_top_k=3,
            monitor="valid_corr_across_celltypes",
            filename=filename_string,
            mode="max",
            auto_insert_metric_name=True,
        )
        # early stopping
        early_stop_callback = EarlyStopping(
            monitor="valid_corr_across_celltypes",
            min_delta=0.001,
            patience=5,
            verbose=False,
            mode="max",
        )

        # Setup Tensorboard logging ==========================
        # create tensorboard log dir if not present
        tb_log_dir = os.path.join(
            config["saving"]["head_dir"], config["saving"]["tb_log_dir"], model_tag
        )
        if not os.path.exists(tb_log_dir):
            os.makedirs(tb_log_dir)
        logger.info(f"Saving tensorboard logs under {tb_log_dir}/{model_tag}")
        tb_logger = TensorBoardLogger(save_dir=tb_log_dir, name=model_tag)

        # Set up LearningRateMonitor =========================
        lr_monitor = LearningRateMonitor(logging_interval="step")

    # LOAD / BUILD A MODEL==================================
    # Load an existing model ==============================
    if config["model"]["load_trained_model"]:
        # Load an existing model
        logger.info("Loading existing model model:")
        assert os.path.exists(
            config["model"]["model_path"]
        ), f"{config['model']['model_path']} does not exists!"
        model = Embedding2Target.load_from_checkpoint(
            config["model"]["model_path"],
            loss=config["optimization"]["loss"],
            learning_rate=config["optimization"]["optimizer"]["lr"],
            lr_schedule=config["optimization"]["optimizer"]["lr_schedule"],
            lr_warmup_epochs=config["optimization"]["scheduler"]["warmup_epochs"],
            lr_max_epochs=config["optimization"]["epochs"],
            optimizer=config["optimization"]["optimizer"]["optimizer"],
            weight_decay=config["optimization"]["optimizer"]["weight_decay"],
            dropout_prob=config["optimization"]["dropout_prob"],
            softplus=config["model"]["softplus"],
            target_is_log=config["task"]["target"]["log_target_in"],
            log_train=config["task"]["target"]["log_transform_train"],
            log_validate=config["task"]["target"]["log_transform_validate"],
            std_validate=config["task"]["target"]["std_validate"],
            rel_weight_gene=config["optimization"]["rel_weights_gene"],
            rel_weight_cell=config["optimization"]["rel_weights_cell"],
            pears_norm_mode=config["optimization"]["pears_norm_mode"],
            train_eval=config["resource"]["run_train_eval"],
        )
        logger.info(model)
    else:
        # OR Construct the model ================================
        # If using a head on pre-computed embeddings/targets ===
        if config["task"]["input"]["input_type"] == "embeddings":
            model = Embedding2Target(
                emb_dim=config["task"]["input"]["emb_dim"],
                target_dim=len(train_data[0][1]),
                loss=config["optimization"]["loss"],
                learning_rate=config["optimization"]["optimizer"]["lr"],
                lr_schedule=config["optimization"]["optimizer"]["lr_schedule"],
                lr_warmup_epochs=config["optimization"]["scheduler"]["warmup_epochs"],
                lr_max_epochs=config["optimization"]["epochs"],
                optimizer=config["optimization"]["optimizer"]["optimizer"],
                weight_decay=config["optimization"]["optimizer"]["weight_decay"],
                dropout_prob=config["optimization"]["dropout_prob"],
                softplus=config["model"]["softplus"],
                target_is_log=config["task"]["target"]["log_target_in"],
                log_train=config["task"]["target"]["log_transform_train"],
                log_validate=config["task"]["target"]["log_transform_validate"],
                std_validate=config["task"]["target"]["std_validate"],
                model_type=config["model"]["model_type"],
                mode="single_cell",
                bottleneck_dim=config["model"]["bottleneck_dim"],
                bottleneck_nonlin=config["model"]["bottleneck_nonlin"],
                rel_weight_gene=config["optimization"]["rel_weights_gene"],
                rel_weight_cell=config["optimization"]["rel_weights_cell"],
                pears_norm_mode=config["optimization"]["pears_norm_mode"],
                train_eval=config["resource"]["run_train_eval"],
            )
        else:
            # If using the sequence input ===========
            # If using enformer trunk
            if config["enformer"]["enformer_trunk"]["use_enformer_trunk"]:
                from enformer_pytorch import Enformer

                # load/build an enformer trunk
                enf_model = Enformer.from_pretrained(
                    config["enformer"]["enformer_trunk"]["enformer_copy_path"]
                )
                trunk = EnformerTrunk(
                    enf_model, config["enformer"]["enformer_trunk"]["central_bin"]
                )
                if config["device"] in ["cuda", "gpu"]:
                    trunk.cuda()
            else:
                logger.warning(
                    "This is placeholder for future "
                    "implementations. If using input_type = "
                    "'sequence' then use_enformer_trunk = True is "
                    "required for now."
                )

            model = Embedding2Target(
                emb_dim=config["task"]["input"]["emb_dim"],
                target_dim=len(train_data[0][1]),
                loss=config["optimization"]["loss"],
                learning_rate=config["optimization"]["optimizer"]["lr"],
                lr_schedule=config["optimization"]["optimizer"]["lr_schedule"],
                lr_warmup_epochs=config["optimization"]["scheduler"]["warmup_epochs"],
                lr_max_epochs=config["optimization"]["epochs"],
                optimizer=config["optimization"]["optimizer"]["optimizer"],
                weight_decay=config["optimization"]["optimizer"]["weight_decay"],
                dropout_prob=config["optimization"]["dropout_prob"],
                softplus=config["model"]["softplus"],
                log_train=config["task"]["target"]["log_transform_train"],
                log_validate=config["task"]["target"]["log_transform_validate"],
                std_validate=config["task"]["target"]["std_validate"],
                model_type=config["model"]["model_type"],
                model_trunk=trunk,
                mode="single_cell",
                freeze=config["enformer"]["enformer_trunk"]["freeze"],
                bottleneck_dim=config["model"]["bottleneck_dim"],
                bottleneck_nonlin=config["model"]["bottleneck_nonlin"],
                rel_weight_gene=config["optimization"]["rel_weights_gene"],
                rel_weight_cell=config["optimization"]["rel_weights_cell"],
                pears_norm_mode=config["optimization"]["pears_norm_mode"],
                train_eval=config["resource"]["run_train_eval"],
            )

        logger.info("Initialized model:")
        if config["task"]["input"]["input_type"] == "embeddings":
            # print model summary if not using Enformer (to big to print neatly
            logger.info(model)

    # Set up Dataloaders ===================================
    train_data_loader = DataLoader(
        train_data,
        batch_size=config["data"]["loader"]["batch_size"],
        shuffle=config["data"]["loader"]["shuffle"],
        num_workers=config["resource"]["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )
    valid_data_loader = DataLoader(
        valid_data,
        batch_size=config["data"]["loader"]["batch_size"],
        shuffle=False,
        num_workers=config["resource"]["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )
    if config["test"]["run_test"]:
        # set up test data loader
        test_data_loader = DataLoader(
            test_data,
            batch_size=config["data"]["loader"]["batch_size"],
            shuffle=config["data"]["loader"]["shuffle"],
            num_workers=config["resource"]["num_workers"],
            pin_memory=True,
            persistent_workers=True,
        )

    # Training =================================================
    if config["optimization"]["run_train"]:

        # assemble callbacks
        collect_callbacks = [lr_monitor, checkpoint_callback]

        if config["optimization"]["swa"]["use_swa"]:
            swa_callback = StochasticWeightAveraging(
                swa_lrs=config["optimization"]["swa"]["swa_lr"],
                swa_epoch_start=config["optimization"]["swa"]["swa_epoch_start"],
                annealing_epochs=config["optimization"]["swa"]["swa_anneal_epochs"],
                annealing_strategy=config["optimization"]["swa"]["swa_anneal_strategy"],
            )

            collect_callbacks.append(swa_callback)

        else:
            # use early stopping
            collect_callbacks.append(early_stop_callback)

        if config["resource"]["num_devices"] <= 1:
            trainer = pl.Trainer(
                accelerator=config["resource"]["device"],
                max_epochs=config["optimization"]["epochs"],
                logger=tb_logger,
                callbacks=collect_callbacks,
            )
        else:
            # train in ddp mode
            trainer = pl.Trainer(
                accelerator=config["resource"]["device"],
                max_epochs=config["optimization"]["epochs"],
                logger=tb_logger,
                callbacks=collect_callbacks,
                devices=config["resource"]["num_devices"],
                strategy="ddp",
            )

        trainer.fit(model, train_data_loader, valid_data_loader)

    # TEST MODE ===============================================================
    if config["test"]["run_test"]:

        # assert that valid value for testing set was selected
        assert config["test"]["test_on"] in ["train", "test", "valid", "all"], (
            "Select a 'test_on' value from 'train', 'test', 'valid', " "'all'"
        )

        model.eval()
        # run one training epoch to update training means
        trainer = pl.Trainer(max_epochs=1)
        model.learning_rate = 0
        model.hparams.learning_rate = 0
        logger.info(
            "Running one epoch with lr = 0 to accumulate train "
            "set mean and standard deviations."
        )
        trainer.fit(model, train_data_loader, valid_data_loader)

        # test on all sets
        if config["test"]["test_on"] in ["all", "train"]:
            trainer.test(model, dataloaders=train_data_loader)
            print("^^^ Performance on Training set ^^^")
        if config["test"]["test_on"] in ["all", "valid"]:
            trainer.test(model, dataloaders=valid_data_loader)
            print("^^^ Performance on Validation set ^^^^")
        if config["test"]["test_on"] in ["all", "test"]:
            trainer.test(model, dataloaders=test_data_loader)
            print("^^^ Performance on Test set ^^^")
