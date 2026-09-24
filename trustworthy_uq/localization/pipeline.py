import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from torch.utils.data import DataLoader, Subset, ConcatDataset
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from .adn import load_adn, train_adn
from .cqr import FrozenBackboneOneSidedCQR
from .adaptive_scp import FrozenBackboneScaleModel
from ..data.kul import NomadicLocalizationDataModule
from ..calibration.conformal import conformal_quantile
from ..evaluation.metrics import euclidean_error
from ..evaluation.visualization import write_fig, plot_single_sample, plot_prediction
from ..artifacts.io import save_single_sample_result, save_test_results


class LocalizationUQ:
    """Simple workflow wrapper for ADN + CQR/Adaptive-SCP localization UQ."""

    def __init__(self, backbone=None, device=None, sla_levels=None, seed=42,
                 batch_size=32, num_workers=0, split=(0.25, 0.45, 0.30)):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sla_levels = list(sla_levels or [0.90, 0.95, 0.99])
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split
        self.backbone = backbone.to(self.device).eval() if backbone is not None else None
        self.cqr_model = None
        self.cqr_qhats = {}
        self.ascp_model = None
        self.ascp_qhats = {}
        self.cqr_calibration_info = {}
        self.ascp_calibration_info = {}

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def build_scenario_dataset(self, data_dir, scenario_id, num_users=4, num_samples=240):
        sample_ids = [scenario_id * 10000 + user * 1000 + s
                      for user in range(num_users) for s in range(num_samples)]
        dm = NomadicLocalizationDataModule(
            data_dir=data_dir, batch_size=self.batch_size, num_workers=self.num_workers,
            num_users=num_users, num_samples=num_samples, mode="test_only", sample_ids=sample_ids)
        dm.setup()
        return dm.test_dataset

    def build_pooled_dataset(self, data_dir, scenario_ids=range(6), num_users=4, num_samples=240):
        return ConcatDataset([
            self.build_scenario_dataset(data_dir, sid, num_users, num_samples)
            for sid in scenario_ids
        ])

    def split_dataset(self, dataset):
        n = len(dataset)
        idx = np.arange(n)
        np.random.default_rng(self.seed).shuffle(idx)
        n_head = int(round(n * self.split[0]))
        n_cal = int(round(n * self.split[1]))
        return (Subset(dataset, idx[:n_head]),
                Subset(dataset, idx[n_head:n_head + n_cal]),
                Subset(dataset, idx[n_head + n_cal:]))

    def _loader(self, dataset, shuffle=False, batch_size=64):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=self.num_workers)

    # ------------------------------------------------------------------
    # ADN
    # ------------------------------------------------------------------
    def load_adn(self, checkpoint_path):
        self.backbone = load_adn(checkpoint_path, self.device)
        return self.backbone

    def train_adn(self, data_dir, **kwargs):
        result = train_adn(data_dir, **kwargs)
        self.backbone = result["model"].to(self.device).eval()
        return result

    # ------------------------------------------------------------------
    # CQR
    # ------------------------------------------------------------------
    def train_cqr(self, train_dataset, validation_dataset=None, sla_levels=None,
                  max_epochs=100, lr=1e-3, weight_decay=1e-5, output_dir=None):
        levels = list(sla_levels or self.sla_levels)
        self.sla_levels = levels
        self.cqr_model = FrozenBackboneOneSidedCQR(self.backbone, levels, lr, weight_decay)
        callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=15), LearningRateMonitor()]
        checkpoint = None
        if output_dir:
            checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", filename="best_cqr_head",
                                        save_top_k=1, dirpath=str(Path(output_dir) / "checkpoints"))
            callbacks.append(checkpoint)
        trainer = Trainer(max_epochs=max_epochs, callbacks=callbacks,
                          accelerator="auto", devices=1, log_every_n_steps=10)
        start = time.time()
        trainer.fit(self.cqr_model,
                    train_dataloaders=self._loader(train_dataset, True, self.batch_size),
                    val_dataloaders=self._loader(validation_dataset or train_dataset, False, 64))
        path = checkpoint.best_model_path if checkpoint else ""
        if path:
            self.cqr_model = FrozenBackboneOneSidedCQR.load_from_checkpoint(path, backbone=self.backbone)
        self.cqr_model.to(self.device).eval()
        return {"model": self.cqr_model, "checkpoint_path": path,
                "runtime_seconds": time.time() - start, "sla_levels": levels}

    def load_cqr(self, checkpoint_path):
        self.cqr_model = FrozenBackboneOneSidedCQR.load_from_checkpoint(
            checkpoint_path, backbone=self.backbone)
        self.cqr_model.to(self.device).eval()
        self.sla_levels = list(self.cqr_model.sla_levels)
        return self.cqr_model

    def _predict_cqr_loader(self, loader):
        self.cqr_model.eval().to(self.device)
        centers, radii, targets = [], [], []
        with torch.no_grad():
            for x, y in loader:
                center, rhi = self.cqr_model(x.to(self.device))
                centers.append(center.cpu().numpy())
                radii.append(rhi.cpu().numpy())
                targets.append(y.numpy())
        return np.vstack(centers), np.vstack(radii), np.vstack(targets)

    def calibrate_cqr(self, calibration_dataset, sla_levels=None, save_path=None):
        levels = [sla_levels] if isinstance(sla_levels, float) else list(sla_levels or self.sla_levels)
        available = list(self.cqr_model.sla_levels)
        center, base, true = self._predict_cqr_loader(self._loader(calibration_dataset))
        errors = euclidean_error(center, true)
        self.cqr_qhats = {}
        for level in levels:
            i = available.index(level)
            self.cqr_qhats[str(level)] = conformal_quantile(errors - base[:, i], 1.0 - level)
        self.cqr_calibration_info = {"num_calibration_samples": len(calibration_dataset), "sla_levels": levels, "qhat_mm": self.cqr_qhats}
        result = {"method": "CQR-OneSided", "sla_levels": levels, "qhat_mm": self.cqr_qhats}
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            Path(save_path).write_text(json.dumps(result, indent=2))
        return result

    # ------------------------------------------------------------------
    # Adaptive SCP
    # ------------------------------------------------------------------
    def train_adaptive_scp(self, train_dataset, validation_dataset=None,
                           max_epochs=60, lr=1e-3, weight_decay=1e-5, output_dir=None):
        self.ascp_model = FrozenBackboneScaleModel(self.backbone, lr, weight_decay)
        callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=15), LearningRateMonitor()]
        checkpoint = None
        if output_dir:
            checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", filename="best_ascp_scale",
                                        save_top_k=1, dirpath=str(Path(output_dir) / "checkpoints"))
            callbacks.append(checkpoint)
        trainer = Trainer(max_epochs=max_epochs, callbacks=callbacks,
                          accelerator="auto", devices=1, log_every_n_steps=10)
        start = time.time()
        trainer.fit(self.ascp_model,
                    train_dataloaders=self._loader(train_dataset, True, self.batch_size),
                    val_dataloaders=self._loader(validation_dataset or train_dataset, False, 64))
        path = checkpoint.best_model_path if checkpoint else ""
        if path:
            self.ascp_model = FrozenBackboneScaleModel.load_from_checkpoint(path, backbone=self.backbone)
        self.ascp_model.to(self.device).eval()
        return {"model": self.ascp_model, "checkpoint_path": path,
                "runtime_seconds": time.time() - start}

    def calibrate_adaptive_scp(self, calibration_dataset, sla_levels=None, save_path=None):
        levels = [sla_levels] if isinstance(sla_levels, float) else list(sla_levels or self.sla_levels)
        self.ascp_model.eval().to(self.device)
        centers, sigmas, true = [], [], []
        loader = self._loader(calibration_dataset)
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                centers.append(self.backbone(x).cpu().numpy())
                sigmas.append(self.ascp_model(x).cpu().numpy())
                true.append(y.numpy())
        centers, sigmas, true = np.vstack(centers), np.concatenate(sigmas), np.vstack(true)
        errors = euclidean_error(centers, true)
        self.ascp_qhats = {}
        for level in levels:
            self.ascp_qhats[str(level)] = conformal_quantile(errors / sigmas, 1.0 - level)
        self.ascp_calibration_info = {"num_calibration_samples": len(calibration_dataset), "sla_levels": levels, "qhat": self.ascp_qhats}
        result = {"method": "AdaptiveSCP-Scale", "sla_levels": levels, "qhat": self.ascp_qhats}
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            Path(save_path).write_text(json.dumps(result, indent=2))
        return result

    def load_calibration(self, path):
        data = json.loads(Path(path).read_text())
        self.cqr_qhats = data.get("qhat_mm", {})
        self.ascp_qhats = data.get("qhat", {})
        if "qhat_mm" in data:
            self.cqr_calibration_info = {
                "sla_levels": data.get("sla_levels", []),
                "qhat_mm": self.cqr_qhats,
            }
        if "qhat" in data:
            self.ascp_calibration_info = {
                "sla_levels": data.get("sla_levels", []),
                "qhat": self.ascp_qhats,
            }
        return data

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _predict_backbone(self, samples):
        x = torch.as_tensor(samples, dtype=torch.float32)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 64):
                out.append(self.backbone(x[i:i+64].to(self.device)).cpu().numpy())
        return x, np.vstack(out)

    def predict(self, samples, sla_levels=None, y_true=None, method="CQR", output_path=None):
        levels = [sla_levels] if isinstance(sla_levels, float) else list(sla_levels or self.sla_levels)
        x, center = self._predict_backbone(samples)
        radii = []
        if method.upper() == "CQR":
            _, base = self._predict_cqr_tensor(x)
            for level in levels:
                i = self.cqr_model.sla_levels.index(level)
                radii.append(base[:, i] + float(self.cqr_qhats[str(level)]))
        elif method.upper() in ("ASCP", "ADAPTIVESCP"):
            with torch.no_grad():
                sigma = self.ascp_model(x.to(self.device)).cpu().numpy()
            for level in levels:
                radii.append(sigma * float(self.ascp_qhats[str(level)]))
        else:
            raise ValueError("method must be CQR or ASCP")
        radii = np.column_stack(radii)
        result = {"method": method, "sla_levels": levels,
                  "point_prediction": center, "radius_mm": radii}
        if y_true is not None:
            true = np.asarray(y_true)
            if true.ndim == 1:
                true = true[None, :]
            errors = euclidean_error(center, true)
            result["error_mm"] = errors
            result["covered"] = {str(level): (errors <= radii[:, j]).tolist()
                                 for j, level in enumerate(levels)}
        if output_path is not None:
            true_position = None
            if y_true is not None:
                true = np.asarray(y_true)
                if true.ndim == 1:
                    true = true[None, :]
                true_position = true[0] if len(true) == 1 else true
            save_single_sample_result(
                output_path,
                method=method,
                task="2D-localization",
                sample_index=0,
                sla_results={
                    str(level): {
                        "target_coverage": float(level),
                        "radius_mm": float(radii[0, j]),
                        "covered": (None if "covered" not in result else bool(result["covered"][str(level)][0])),
                        "error_mm": (None if "error_mm" not in result else float(result["error_mm"][0])),
                    }
                    for j, level in enumerate(levels)
                },
                point_prediction=center[0] if len(center) == 1 else center,
                true_position=true_position,
                model_info={"backbone": "ADN", "uq_model": method},
            )
        return result

    def _predict_cqr_tensor(self, x):
        self.cqr_model.eval().to(self.device)
        centers, radii = [], []
        with torch.no_grad():
            for i in range(0, len(x), 64):
                c, r = self.cqr_model(x[i:i+64].to(self.device))
                centers.append(c.cpu().numpy()); radii.append(r.cpu().numpy())
        return np.vstack(centers), np.vstack(radii)

    # ------------------------------------------------------------------
    # Evaluation / plots
    # ------------------------------------------------------------------
    def evaluate(self, test_dataset, sla_levels=None, method="CQR", output_dir=None):
        levels = [sla_levels] if isinstance(sla_levels, float) else list(sla_levels or self.sla_levels)
        loader = self._loader(test_dataset)
        if method.upper() == "CQR":
            center, base, true = self._predict_cqr_loader(loader)
            errors = euclidean_error(center, true)
            rows, predictions = [], []
            for level in levels:
                i = self.cqr_model.sla_levels.index(level)
                q = float(self.cqr_qhats[str(level)])
                radius = base[:, i] + q
                covered = errors <= radius
                rows.append({"method":"CQR-OneSided", "target_assurance":level, "alpha":1-level,
                    "calibration_size":None, "test_size":len(errors), "qhat_mm":q,
                    "test_mean_error_mm":float(errors.mean()), "test_median_error_mm":float(np.median(errors)),
                    "test_p90_error_mm":float(np.quantile(errors,.90)), "test_p95_error_mm":float(np.quantile(errors,.95)),
                    "true_coverage":float(covered.mean()), "breach_rate":float(1-covered.mean()),
                    "mean_base_radius_mm":float(base[:,i].mean()), "mean_final_radius_mm":float(radius.mean())})
                for j in range(len(errors)):
                    predictions.append({"method":"CQR-OneSided", "target_assurance":level,
                        "pred_x":float(center[j,0]), "pred_y":float(center[j,1]), "true_x":float(true[j,0]),
                        "true_y":float(true[j,1]), "euclidean_error_mm":float(errors[j]),
                        "base_radius_mm":float(base[j,i]), "qhat_mm":q, "final_radius_mm":float(radius[j]),
                        "covered":float(covered[j])})
            metrics, preds = pd.DataFrame(rows), pd.DataFrame(predictions)
            prefix = "cqr"
        else:
            self.ascp_model.eval().to(self.device)
            centers, sigmas, true = [], [], []
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(self.device)
                    centers.append(self.backbone(x).cpu().numpy())
                    sigmas.append(self.ascp_model(x).cpu().numpy())
                    true.append(y.numpy())
            center, sigma, true = np.vstack(centers), np.concatenate(sigmas), np.vstack(true)
            errors = euclidean_error(center, true)
            rows, predictions = [], []
            for level in levels:
                q = float(self.ascp_qhats[str(level)])
                radius = sigma*q; covered = errors <= radius
                rows.append({"method":"AdaptiveSCP-Scale", "target_assurance":level, "alpha":1-level,
                    "calibration_size":None, "test_size":len(errors), "qhat_mm":q,
                    "test_mean_error_mm":float(errors.mean()), "test_median_error_mm":float(np.median(errors)),
                    "test_p90_error_mm":float(np.quantile(errors,.90)), "test_p95_error_mm":float(np.quantile(errors,.95)),
                    "true_coverage":float(covered.mean()), "breach_rate":float(1-covered.mean()),
                    "mean_interval_radius_mm":float(radius.mean()), "median_interval_radius_mm":float(np.median(radius))})
                for j in range(len(errors)):
                    predictions.append({"method":"AdaptiveSCP-Scale", "target_assurance":level,
                        "pred_x":float(center[j,0]), "pred_y":float(center[j,1]), "true_x":float(true[j,0]),
                        "true_y":float(true[j,1]), "euclidean_error_mm":float(errors[j]),
                        "base_radius_mm":float(sigma[j]), "qhat_mm":q, "final_radius_mm":float(radius[j]),
                        "covered":float(covered[j])})
            metrics, preds = pd.DataFrame(rows), pd.DataFrame(predictions)
            prefix = "adaptive_scp"
        if output_dir:
            out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
            metrics.to_csv(out/f"{prefix}_metrics.csv",index=False)
            preds.to_csv(out/f"{prefix}_predictions.csv",index=False)
            fig=go.Figure(); fig.add_bar(x=metrics.target_assurance,y=metrics.true_coverage,name="empirical_coverage")
            fig.add_scatter(x=metrics.target_assurance,y=metrics.target_assurance,mode="lines",name="ideal")
            write_fig(fig,out/f"{prefix}_coverage_vs_target")
            radius_col="mean_final_radius_mm" if prefix=="cqr" else "mean_interval_radius_mm"
            fig=go.Figure(); fig.add_bar(x=metrics.target_assurance,y=metrics[radius_col],name="mean_radius")
            write_fig(fig,out/f"{prefix}_mean_radius_vs_target")

            calibration_info = self.cqr_calibration_info if prefix == "cqr" else self.ascp_calibration_info
            save_test_results(
                out / "uq_test_results.json",
                method=metrics.iloc[0]["method"],
                task="2D-localization",
                metrics_df=metrics,
                num_test_samples=len(test_dataset),
                calibration_info=calibration_info,
                model_info={"backbone": "ADN", "uq_model": method},
            )
        return metrics, preds

    def plot_sample(self, samples, y_true, sample_index=0, sla_level=0.95,
                    method="CQR", output_path=None):
        """Run UQ inference for raw CSI samples and plot one sample.

        This method is intended for callers that have raw CSI input.  For a
        prediction that has already been computed, use ``plot_prediction``
        instead so that the backbone is not evaluated a second time.
        """
        result = self.predict(samples, [sla_level], y_true, method)
        true = np.asarray(y_true)
        if true.ndim == 1:
            true = true[None, :]
        if output_path is None:
            output_path = Path("output") / f"single_sample_{method}_idx{sample_index}_sla{str(sla_level).replace('.','p')}"
        return plot_single_sample(result["point_prediction"][sample_index],
                                  true[sample_index],
                                  float(result["radius_mm"][sample_index,0]),
                                  sla_level, method_name=method,
                                  output_path=output_path)

    def plot_prediction(self, prediction, true_position, radius, sla_level=0.95,
                        method="CQR", output_path=None):
        """Plot an already-computed localization prediction and UQ radius.

        Unlike :meth:`plot_sample`, this method does not run the backbone or
        UQ model again.  It is therefore the appropriate visualization path
        for results returned by :meth:`evaluate` or :meth:`predict`.
        """
        if output_path is None:
            output_path = (Path("output") /
                           f"single_sample_{method}_sla{str(sla_level).replace('.','p')}")
        return plot_prediction(prediction, true_position, radius, sla_level,
                               method_name=method, output_path=output_path)

    # ------------------------------------------------------------------
    # One-call reproduction of the original experiment workflow
    # ------------------------------------------------------------------
    def load_ascp(self, checkpoint_path):
        """Load a previously trained Adaptive-SCP scale model."""
        self.ascp_model = FrozenBackboneScaleModel.load_from_checkpoint(
            checkpoint_path, backbone=self.backbone)
        self.ascp_model.to(self.device).eval()
        return self.ascp_model

    def run_experiment(self, data_dir, method="CQR", eval_mode="pooled", scenario_id=0,
                       scenario_ids=range(6), adn_checkpoint=None, cqr_checkpoint=None,
                       ascp_checkpoint=None,
                       train_new_adn=False, train_new_cqr=False, train_new_ascp=False,
                       sla_levels=None, output_dir="output/unified_conformal_experiments",
                       enable_single_sample=True, single_sample_index=0, single_sample_sla=0.95,
                       **train_kwargs):
        levels = list(sla_levels or self.sla_levels)
        if adn_checkpoint:
            self.load_adn(adn_checkpoint)
        elif train_new_adn:
            self.train_adn(data_dir, **train_kwargs)
        dataset = (self.build_pooled_dataset(data_dir, scenario_ids)
                   if eval_mode.lower()=="pooled"
                   else self.build_scenario_dataset(data_dir, scenario_id))
        head, cal, test = self.split_dataset(dataset)
        out = Path(output_dir) / ("pooled_scenarios" if eval_mode.lower()=="pooled" else f"single_scenario_{scenario_id}") / method.upper()
        if method.upper() == "CQR":
            if cqr_checkpoint:
                self.load_cqr(cqr_checkpoint)
            elif train_new_cqr:
                self.train_cqr(head, cal, levels, output_dir=out, **{k:v for k,v in train_kwargs.items() if k in {"max_epochs","lr","weight_decay"}})
            self.calibrate_cqr(cal, levels, save_path=out/"cqr_calibration.json")
        else:
            if ascp_checkpoint:
                self.load_ascp(ascp_checkpoint)
            elif train_new_ascp:
                self.train_adaptive_scp(
                    head, cal, output_dir=out,
                    **{k: v for k, v in train_kwargs.items()
                       if k in {"max_epochs", "lr", "weight_decay"}}
                )
            self.calibrate_adaptive_scp(cal, levels, save_path=out/"ascp_calibration.json")
        metrics, preds = self.evaluate(test, levels, method, out)
        if enable_single_sample:
            # ``evaluate`` has already performed inference.  Select the
            # requested SLA/sample from its results and visualize those
            # values directly instead of feeding predicted coordinates back
            # into ``plot_sample`` as if they were raw CSI.
            single = preds[preds["target_assurance"] == single_sample_sla].reset_index(drop=True)
            if single_sample_index < 0 or single_sample_index >= len(single):
                raise IndexError(
                    f"single_sample_index={single_sample_index} is out of range "
                    f"for {len(single)} test samples at SLA={single_sample_sla}."
                )
            row = single.iloc[single_sample_index]

            # Save a compact single-sample UQ artifact containing all requested
            # SLA levels. The plot remains focused on the selected SLA.
            sample_sla_results = {}
            for level in levels:
                level_rows = preds[preds["target_assurance"] == level].reset_index(drop=True)
                level_row = level_rows.iloc[single_sample_index]
                sample_sla_results[str(level)] = {
                    "target_coverage": float(level),
                    "radius_mm": float(level_row["final_radius_mm"]),
                    "covered": bool(level_row["covered"]),
                    "error_mm": float(level_row["euclidean_error_mm"]),
                    "base_radius_mm": float(level_row["base_radius_mm"]),
                    "qhat_mm": float(level_row["qhat_mm"]),
                }
            save_single_sample_result(
                out / "uq_single_sample.json",
                method=method,
                task="2D-localization",
                sample_index=single_sample_index,
                sla_results=sample_sla_results,
                point_prediction=np.array([row["pred_x"], row["pred_y"]], dtype=float),
                true_position=np.array([row["true_x"], row["true_y"]], dtype=float),
                model_info={"backbone": "ADN", "uq_model": method},
            )

            self.plot_prediction(
                prediction=np.array([row["pred_x"], row["pred_y"]], dtype=float),
                true_position=np.array([row["true_x"], row["true_y"]], dtype=float),
                radius=float(row["final_radius_mm"]),
                sla_level=single_sample_sla,
                method=method,
                output_path=out/"single_sample"
            )
        return metrics, preds
