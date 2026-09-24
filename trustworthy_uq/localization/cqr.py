import torch
import torch.nn.functional as F
from torch import nn
import lightning as L


class OneSidedMultiSLAQuantileHead(nn.Module):
    def __init__(self, in_channels=512, num_levels=3):
        super().__init__()
        h1, h2, h3 = in_channels // 2, in_channels // 4, in_channels // 8
        self.fc1 = nn.Linear(in_channels, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.rhi_head = nn.Linear(h3, num_levels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        rhi = F.softplus(self.rhi_head(x))
        return torch.cummax(rhi, dim=1)[0]


class FrozenBackboneOneSidedCQR(L.LightningModule):
    def __init__(self, backbone, sla_levels=None, lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.sla_levels = list(sla_levels or [0.90, 0.95, 0.99])
        self.alphas = [1.0 - x for x in self.sla_levels]
        self.lr = lr
        self.weight_decay = weight_decay
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.quantile_head = OneSidedMultiSLAQuantileHead(512, len(self.sla_levels))

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
            x = self.backbone.flat2(x)
        return x

    def forward(self, x):
        with torch.no_grad():
            center = self.backbone(x)
        return center, self.quantile_head(self.extract_features(x))

    def _loss_components(self, batch):
        x, y = batch
        center, rhi = self(x)
        error = torch.sqrt(torch.sum((center - y) ** 2, dim=1) + 1e-8)
        losses, coverages, radii = [], [], []
        for i, alpha in enumerate(self.alphas):
            q = 1.0 - alpha
            err = error - rhi[:, i]
            losses.append(torch.mean(torch.maximum(q * err, (q - 1.0) * err)))
            coverages.append((error <= rhi[:, i]).float().mean())
            radii.append(rhi[:, i].mean())
        return torch.stack(losses).mean(), error, torch.stack(coverages), torch.stack(radii)

    def training_step(self, batch, batch_idx):
        loss, error, coverage, radii = self._loss_components(batch)
        self.log("train_loss", loss)
        self.log("train_mean_radial_error", error.mean())
        for i, level in enumerate(self.sla_levels):
            name = str(level).replace(".", "p")
            self.log(f"train_raw_cov_{name}", coverage[i])
            self.log(f"train_rhi_{name}", radii[i])
        return loss

    def validation_step(self, batch, batch_idx):
        loss, error, coverage, radii = self._loss_components(batch)
        self.log("val_loss", loss)
        self.log("val_mean_radial_error", error.mean())
        for i, level in enumerate(self.sla_levels):
            name = str(level).replace(".", "p")
            self.log(f"val_raw_cov_{name}", coverage[i])
            self.log(f"val_rhi_{name}", radii[i])
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}
