"""Record the frozen results in MLflow (local SQLite). Fits nothing; reads saved artifacts only.

    python -m src.track
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import csv
import json

import mlflow

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    fz = json.loads((ROOT / "models/FREEZE.json").read_text(encoding="utf-8"))
    meta = json.loads((ROOT / "models/metadata.json").read_text(encoding="utf-8"))
    test = json.loads((ROOT / "artifacts/results/test_results.json").read_text(encoding="utf-8"))
    for name in ("strokesense-development", "strokesense-final-test"):
        if mlflow.get_experiment_by_name(name) is None:
            mlflow.create_experiment(name, artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment("strokesense-development")
    if not mlflow.search_runs(output_format="list", max_results=1):
        for r in csv.DictReader(open(ROOT / "artifacts/results/FINAL_RESULTS.csv", encoding="utf-8")):
            with mlflow.start_run(run_name=r["model"]):
                mlflow.set_tags({"evaluation": "4x5-fold CV, nested tuning and threshold", "selected": str(r["model"] == "Logistic 4 (selected)")})
                mlflow.log_metrics({k: float(v) for k, v in r.items() if k != "model" and v != ""})
    mlflow.set_experiment("strokesense-final-test")
    if not mlflow.search_runs(output_format="list", max_results=1):
        for name, m in test["models"].items():
            with mlflow.start_run(run_name=name):
                mlflow.log_params({"threshold": m["threshold"], **({"C": .1, "features": ",".join(meta["features"]), "model_sha256": fz["sha256"]["models/model.joblib"]} if "4" in name else {})})
                mlflow.set_tags({"scored_at": test["scored_at"], "data_sha256": meta["data_sha256"]})
                mlflow.log_metrics({k: float(v["value"] if isinstance(v, dict) else v) for k, v in m.items() if k != "threshold"})
    print("runs:", sum(len(mlflow.search_runs(experiment_names=[n], output_format="list")) for n in ("strokesense-development", "strokesense-final-test")))


if __name__ == "__main__":
    main()
