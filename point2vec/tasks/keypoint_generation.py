from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

from point2vec.datasets import LArNetDataModule
from point2vec.models import Point2VecKeyPointGeneration
import torch

if __name__ == "__main__":
    cli = LightningCLI(
        Point2VecKeyPointGeneration,
        trainer_defaults={
            "default_root_dir": "artifacts",
            "accelerator": "gpu",
            "devices": 1,
            "precision": 16,
            "max_epochs": 300,
            "track_grad_norm": -1,
            "log_every_n_steps": 10,
            "callbacks": [
                LearningRateMonitor(),
                ModelCheckpoint(save_on_train_epoch_end=True),
            ],
        },
        seed_everything_default=0,
        save_config_callback=None,  # https://github.com/Lightning-AI/lightning/issues/12028#issuecomment-1088325894
    )

