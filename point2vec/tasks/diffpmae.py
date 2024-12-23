from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

from point2vec.datasets import LArNetDataModule
from point2vec.models import DiffPMAE
import torch

if __name__ == "__main__":
    cli = LightningCLI(
        DiffPMAE,
        trainer_defaults={
            "default_root_dir": "artifacts",
            "accelerator": "gpu",
            "devices": 4,
            "precision": 16,
            "max_epochs": 800,
            "track_grad_norm": -1,
            "log_every_n_steps": 10,
            "check_val_every_n_epoch": 200,
            "callbacks": [
                LearningRateMonitor(),
                ModelCheckpoint(save_on_train_epoch_end=True),
                ModelCheckpoint(
                    filename="{epoch}-{step}-{val_loss:.3f}",
                    monitor="val_loss",
                ),
                # ModelCheckpoint(
                #     save_top_k=4,
                #     monitor="epoch", # checked every `check_val_every_n_epoch` epochss
                #     mode="max",
                #     filename="{epoch}-{step}-intermediate",
                # ),
            ],
            # 'profiler': 'advanced'
        },
        seed_everything_default=123,
        save_config_callback=None,  # https://github.com/Lightning-AI/lightning/issues/12028#issuecomment-1088325894
    )

