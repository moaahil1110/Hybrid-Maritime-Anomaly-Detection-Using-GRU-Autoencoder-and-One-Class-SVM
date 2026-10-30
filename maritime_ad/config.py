"""Settings shared by the training scripts and the API.

Paths default to folders inside the repository and can be overridden with
environment variables, so nothing depends on one machine's directory layout.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Baltic Sea AIS training data (see README "Dataset")
DATA_CSV = Path(os.environ.get("MARITIME_DATA_CSV", REPO_ROOT / "data" / "training_set.csv"))
# Trained artifacts; app/api.py loads from the same folder
MODEL_DIR = Path(os.environ.get("MARITIME_MODEL_DIR", REPO_ROOT / "models" / "latest-model"))
# Plots and metrics written by the training scripts
RESULTS_DIR = Path(os.environ.get("MARITIME_RESULTS_DIR", REPO_ROOT / "results"))

SEQ_LEN = 10
FEATURES = ["latitude", "longitude", "speed", "course", "heading"]
SEED = 42

# Artifact file names (app/api.py expects ENCODER_FILE, SCALER_FILE and DETECTOR_FILE)
AUTOENCODER_FILE = "final_autoencoder_model.keras"
ENCODER_FILE = "gru_encoder.keras"
SCALER_FILE = "final_data_scaler.pkl"
DETECTOR_FILE = "final_anomaly_detector.pkl"
# Written by scripts/train_autoencoder.py; scripts/train_detector.py reuses its vessel sample
TRAINING_CONFIG_FILE = "training_config.json"
