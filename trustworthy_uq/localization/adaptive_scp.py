import torch
import torch.nn.functional as F
import lightning as L
from torch import nn


class FrozenBackboneScaleModel(L.LightningModule):
    def __init__(self, backbone, lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.lr = lr
        self.weight_decay = weight_decay
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.scale_head = nn.Linear(128, 1)

    def extract_features(self, x):
        with torch.no_grad():
            x = self.backbone.denseBlock1(x)
            x = [self.backbone.subcarrier_attentions[i](x[:, :, i, :]).unsqueeze(2)
                 for i in range(self.backbone.num_antennas)]
            x = torch.cat(x, dim=2)
            x = self.backbone.ap1(x)
            x = self.backbone.denseBlock2(x)
            x = self.backbone.ap2(x)
            x = self.backbone.denseBlock3(x)
            x = self.backbone.denseBlock4(x)
            x = self.backbone.ap3(x)
            x = self.backbone.flat1(x.transpose(2, 3))
            x = self.backbone.antenna_attention(x)
            return self.backbone.flat2(x)

    def forward(self, x):
        h = F.relu(self.fc1(self.extract_features(x)))
        h = F.relu(self.fc2(h))
        return torch.clamp(F.softplus(self.scale_head(h)).squeeze(-1), min=1e-3)

    def _step(self, batch, stage):
        x, y = batch
        with torch.no_grad():
            center = self.backbone(x)
            error = torch.sqrt(torch.sum((center - y) ** 2, dim=1) + 1e-8)
        sigma = self(x)
        loss = F.l1_loss(sigma, error)
        self.log(f"{stage}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}
