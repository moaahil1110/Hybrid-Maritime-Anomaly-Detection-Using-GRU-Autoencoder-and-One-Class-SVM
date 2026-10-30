"""End-to-end run of both training scripts on a tiny synthetic dataset (about a minute)."""
import json
import os
import subprocess
import sys
from pathlib import Path

from conftest import make_tracks

ROOT = Path(__file__).resolve().parent.parent


def run(script, env, *args):
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                            env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


def test_training_scripts_produce_api_artifacts(tmp_path):
    csv = tmp_path / "training_set.csv"
    tracks = make_tracks([200] * 30).rename(columns={"latitude": "lat", "longitude": "lon"})
    tracks["lat"] = 55 + tracks["lat"] * 0.1
    tracks.sample(frac=1.0, random_state=0).to_csv(csv, index=False)

    models, results = tmp_path / "models", tmp_path / "results"
    env = {**os.environ, "MARITIME_DATA_CSV": str(csv), "MARITIME_MODEL_DIR": str(models),
           "MARITIME_RESULTS_DIR": str(results), "TF_CPP_MIN_LOG_LEVEL": "3"}

    run("train_autoencoder.py", env, "--epochs", "1", "--sample-vessels", "0.8")
    run("train_detector.py", env, "--svm-train-samples", "500", "--eval-samples", "1000")

    for name in ("gru_encoder.keras", "final_anomaly_detector.pkl", "final_data_scaler.pkl",
                 "final_autoencoder_model.keras", "training_config.json"):
        assert (models / name).exists(), name
    assert json.loads((models / "training_config.json").read_text())["sample_vessels"] == 0.8

    metrics = json.loads((results / "detector_metrics.json").read_text())
    assert 0 < metrics["test_anomaly_share"] < 0.3
    assert (results / "detector_evaluation.png").exists()
    assert (results / "autoencoder_loss.png").exists()
