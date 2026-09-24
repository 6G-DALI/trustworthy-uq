def log_evaluation(mlflow, metrics_df, predictions_csv=None, metrics_csv=None, figures_dir=None):
    if metrics_csv:
        mlflow.log_artifact(str(metrics_csv))
    if predictions_csv:
        mlflow.log_artifact(str(predictions_csv))
    if figures_dir:
        for path in figures_dir.rglob("*"):
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path="figures")
