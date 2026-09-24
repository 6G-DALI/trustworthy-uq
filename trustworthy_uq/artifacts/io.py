import json
from pathlib import Path

import numpy as np


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), indent=2))
    return path


def load_json(path):
    return json.loads(Path(path).read_text())


def save_test_results(path, method, task, metrics_df, num_test_samples,
                      calibration_info=None, model_info=None):
    """Save a human-readable summary of whole-test-set UQ results.

    The existing CSV files remain the detailed tabular artifacts. This JSON is
    a compact, self-contained summary intended for experimenters/MLOps tools.
    """
    sla_results = {}
    for _, row in metrics_df.iterrows():
        row_data = row.to_dict()
        level = row_data.pop("target_assurance")
        sla_results[str(level)] = _json_safe(row_data)

    result = {
        "method": method,
        "task": task,
        "num_test_samples": int(num_test_samples),
        "model": model_info or {},
        "calibration": calibration_info or {},
        "sla_results": sla_results,
    }
    return save_json(result, path)


def save_single_sample_result(path, method, task, sample_index,
                              sla_results, point_prediction=None,
                              true_position=None, model_info=None):
    """Save the UQ result for one test/input sample across SLA levels."""
    result = {
        "method": method,
        "task": task,
        "sample_index": int(sample_index),
        "model": model_info or {},
        "point_prediction": _json_safe(point_prediction),
        "true_position": _json_safe(true_position),
        "sla_results": _json_safe(sla_results),
    }
    return save_json(result, path)
