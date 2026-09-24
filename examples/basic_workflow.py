from trustworthy_uq import LocalizationUQ
from trustworthy_uq.data.kul import NomadicLocalizationDataModule

DATA_DIR = "data/nomadic_dataset/ULA_lab_LoS"
ADN_CHECKPOINT = "path/to/best_model.ckpt"
CQR_CHECKPOINT = "path/to/best_cqr_head.ckpt"

uq = LocalizationUQ.from_artifacts(
    adn_model=ADN_CHECKPOINT,
    cqr_model=CQR_CHECKPOINT,
    sla_levels=[0.90, 0.95, 0.99],
)

# Build the same pooled dynamic dataset used by the original experiments.
sample_ids = [scenario * 10000 + user * 1000 + sample
              for scenario in range(6)
              for user in range(4)
              for sample in range(240)]

dm = NomadicLocalizationDataModule(
    data_dir=DATA_DIR, batch_size=32, num_workers=0,
    num_users=4, num_samples=240, mode="test_only", sample_ids=sample_ids
)
dm.setup()

head, calibration, test = uq._split_dataset(dm.test_dataset)
uq.calibrate_cqr(calibration, sla_levels=[0.90, 0.95, 0.99], save_path="cqr_calibration.json")
metrics, predictions = uq.evaluate(test, sla_levels=[0.90, 0.95, 0.99], output_dir="output/cqr")
print(metrics)
