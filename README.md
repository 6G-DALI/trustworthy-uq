# trustworthy-uq

A simple Python package for uncertainty-aware CSI localization experiments used in the 6G-DALI work.

## Included

- AttentionDenseNet (ADN) localization backbone
- One-sided multi-SLA CQR
- Adaptive SCP with learned scale model
- KUL/Nomadic CSI dataset loader
- Training/loading of ADN and CQR models
- Conformal calibration for one or multiple SLA levels
- Batch inference and single-sample prediction disks
- Coverage, breach rate and radius evaluation
- CSV and Plotly artifact generation
- Optional MLflow adapter

## Basic workflow

```python
from trustworthy_uq import LocalizationUQ

uq = LocalizationUQ.from_artifacts(
    adn_model="best_model.ckpt",
    cqr_model="best_cqr_head.ckpt"
)

uq.calibrate_cqr(calibration_dataset, sla_levels=[0.95])
result = uq.predict(X_test, sla_levels=[0.95])
metrics, predictions = uq.evaluate(test_dataset, sla_levels=[0.95])
```

The package keeps the CQR head multi-SLA: it is trained once for the configured SLA levels, while conformal calibration can be performed for one or more of those levels.
