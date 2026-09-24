import time
import torch
from pathlib import Path
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from .backbone import AttentionDenseNet
from ..data.kul import NomadicLocalizationDataModule


def load_adn(checkpoint_path, device=None):
    model = AttentionDenseNet.load_from_checkpoint(checkpoint_path)
    model.eval()
    if device is not None:
        model.to(device)
    return model


def train_adn(data_dir, output_dir="output/adn", batch_size=32, num_workers=0,
              num_users=4, num_samples=240, augment_method="random_attenuation",
              max_epochs=200, seed=42):
    seed_everything(seed, workers=True)
    dm = NomadicLocalizationDataModule(data_dir=data_dir, batch_size=batch_size,
        num_workers=num_workers, num_users=num_users, num_samples=num_samples,
        mode="train_static", augment_method=augment_method)
    model = AttentionDenseNet()
    ckpt = ModelCheckpoint(monitor="val_loss", mode="min", filename="best_adn_backbone", save_top_k=1,
                           dirpath=str(Path(output_dir) / "checkpoints"))
    trainer = Trainer(max_epochs=max_epochs, callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=20),
                          LearningRateMonitor(), ckpt], accelerator="auto", devices=1)
    start = time.time()
    trainer.fit(model, datamodule=dm)
    path = ckpt.best_model_path
    model = AttentionDenseNet.load_from_checkpoint(path)
    model.eval()
    return {"model": model, "checkpoint_path": path, "runtime_seconds": time.time() - start}
