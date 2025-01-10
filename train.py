import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything, strategies
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_info
from omegaconf import OmegaConf

from src.models import (
    DrugTargetCoembeddingLightning,
    MorganAttention,
    MorganChemBertAttention,
    MorganChemBertMhAttention,
    MorganChemBertMlp
)
from src.datamodule import (
    BaselineDataModule,
    MorganChembertDataModule,
    BacteriaDataModule
)
from src.callback.metrics_callback import MetricsCallback


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def init_config() -> OmegaConf:
    parser = ArgumentParser(description="PLM_DTI Training.")
    parser.add_argument("--config", help="YAML config file", default="configs/chembert_muilt.yaml")
    parser.add_argument("--ds", help="Dataset to select", default="")
    parser.add_argument("--dev", action='store_true', help="fast dev run")
    parser.add_argument("--stage", help="Stage", default='fit')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.update({k: v for k, v in vars(args).items() if v is not None})

    if config.classify:
        config.watch_metric = "val/F1Score_Average"
    else:
        config.watch_metric = "val/pcc"

    if config.dev:
        config.ds = 'test'

    return config


def create_model_and_datamodule(config):
    model_architecture = config.model_architecture
    stage = config.stage

    model_mapping = {
        "DrugTargetCoembedding": DrugTargetCoembeddingLightning,
        "MorganAttention": MorganAttention,
        "MorganChemBertaAttention": MorganChemBertAttention,
        "MorganChemBertaMhAttention": MorganChemBertMhAttention,
        "MorganChemBertMlp": MorganChemBertMlp
    }

    model_class = model_mapping.get(model_architecture)
    if model_class is None:
        raise ValueError(f"Unknown model architecture: {model_architecture}")

    if stage == 'Train':
        model = model_class(
            latent_dim=config.latent_dimension,
            classify=config.classify,
            drug_dim=config.drug_shape,
            target_dim=config.target_shape,
            num_classes=config.num_classes,
            loss_type=config.loss_type,
            ensemble_learn=config.ensemble_learn,
        )
    else:
        model = model_class.load_from_checkpoint(
            checkpoint_path=config.checkpoint_path,
            latent_dim=config.latent_dimension,
            classify=config.classify,
            drug_dim=config.drug_shape,
            target_dim=config.target_shape,
            num_classes=config.num_classes,
            loss_type=config.loss_type,
            ensemble_learn=config.ensemble_learn,
        )

    if config.task == "Bacteria":
        dm_class = BacteriaDataModule
    else:
        datamodule_mapping = {
            "DrugTargetCoembedding": BaselineDataModule,
            "MorganAttention": BaselineDataModule,
            "MorganChemBertAttention": MorganChembertDataModule,
            "MorganChemBertMhAttention": MorganChembertDataModule,
            "MorganChemBertMlp": MorganChembertDataModule
        }

        dm_class = datamodule_mapping.get(model_architecture)

    if dm_class is None:
        raise ValueError(f"Unknown DataModule for model: {model_architecture}")

    datamodule = dm_class(config)

    return model, datamodule


def main():
    config = init_config()
    rank_zero_info(config)
    seed_everything(config.replicate, workers=True)
    if config.ds == 'test':
        config.batch_size = 4
        config.num_workers = 2
    model, datamodule = create_model_and_datamodule(config)
    strategy = strategies.DDPStrategy(find_unused_parameters=True)
    metrics_callback = MetricsCallback(num_classes=config.num_classes, classify=config.classify)
    fn = f"{config.experiment_id}-epoch={{epoch}}-metric={{{config.watch_metric}:.2f}}"
    metric_save_callback = ModelCheckpoint(
        monitor=config.watch_metric,
        dirpath='ckpts',
        filename=fn,
        save_top_k=3,
        mode='max',
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False
    )
    trainer = Trainer(
        strategy=strategy,
        accelerator="gpu",
        devices=[0, 1],
        fast_dev_run=config.dev,
        callbacks=[metrics_callback, metric_save_callback]
    )
    if config.stage == 'fit':
        trainer.fit(model, datamodule=datamodule)
    elif config.stage == 'validate' or config.stage == 'test':
        trainer.validate(model, datamodule=datamodule)
    else:
        trainer.predict(model=model, dataloaders=datamodule)


if __name__ == "__main__":
    main()
