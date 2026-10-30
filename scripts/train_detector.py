"""Train and evaluate the One-Class SVM detector.

Encodes windows with the encoder from train_autoencoder.py, trains a One-Class
SVM on latent vectors of clean training windows, picks nu on the validation
vessels, and reports the final scores once on the untouched test vessels.

Synthetic anomalies (speed spike, direction change, location jump, slow
movement) are injected only into the validation and test vessels, as 10-point
segments of each vessel's time-ordered track.

Input (from train_autoencoder.py, in MARITIME_MODEL_DIR):
  final_data_scaler.pkl, gru_encoder.keras, training_config.json
Output:
  final_anomaly_detector.pkl in MARITIME_MODEL_DIR
  detector_metrics.json and detector_evaluation.png in MARITIME_RESULTS_DIR

Usage:  python scripts/train_detector.py [--svm-train-samples N] [--eval-samples N]
The "# %%" markers let editors such as VS Code run each section as an interactive cell.
"""

# %% Setup
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
if not (ROOT / "maritime_ad").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

INTERACTIVE = "ipykernel" in sys.modules

parser = argparse.ArgumentParser(description="Train and evaluate the One-Class SVM detector.")
# A One-Class SVM scales roughly quadratically with training size, so it is fitted on a sample
parser.add_argument("--svm-train-samples", type=int, default=20_000)
parser.add_argument("--eval-samples", type=int, default=200_000, help="windows sampled from each of val and test")
parser.add_argument("--segments-per-1000", type=float, default=5,
                    help="injected anomalous segments per 1,000 points")
args = parser.parse_args([] if INTERACTIVE else None)

import joblib
import matplotlib
if not INTERACTIVE:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.svm import OneClassSVM

from maritime_ad import config
from maritime_ad.data import (
    inject_anomalies,
    load_ais,
    make_windows,
    sample_vessels,
    scale_windows,
    split_by_vessel,
)

config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NU_GRID = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
GAMMA = "auto"

# Reuse the autoencoder's vessel sample so its training vessels never reach the test set
training_config = json.loads((config.MODEL_DIR / config.TRAINING_CONFIG_FILE).read_text())

# %% Same vessel split as train_autoencoder.py (same sample and seed); anomalies only in val and test
df = sample_vessels(load_ais(config.DATA_CSV), training_config["sample_vessels"], seed=training_config["seed"])
df_train, df_val, df_test = split_by_vessel(df, seed=training_config["seed"])
del df

X_train, _ = make_windows(df_train)
X_val, y_val = make_windows(inject_anomalies(df_val, args.segments_per_1000, seed=config.SEED + 1))
X_test, y_test = make_windows(inject_anomalies(df_test, args.segments_per_1000, seed=config.SEED + 2))

rng = np.random.default_rng(config.SEED)


def sample(X, y, n):
    keep = rng.choice(len(X), size=min(n, len(X)), replace=False)
    return X[keep], (None if y is None else y[keep])


X_train, _ = sample(X_train, None, args.svm_train_samples)
X_val, y_val = sample(X_val, y_val, args.eval_samples)
X_test, y_test = sample(X_test, y_test, args.eval_samples)

for name, y in [("val", y_val), ("test", y_test)]:
    print(f"{name}: {len(y)} windows, {y.mean():.1%} anomalous")

# %% Scale with the scaler the encoder was trained with, then encode to latent vectors
scaler = joblib.load(config.MODEL_DIR / config.SCALER_FILE)
encoder = tf.keras.models.load_model(config.MODEL_DIR / config.ENCODER_FILE)


def encode(X):
    return encoder.predict(scale_windows(X, scaler), batch_size=1024, verbose=0)


Z_train, Z_val, Z_test = encode(X_train), encode(X_val), encode(X_test)
print("Latent vectors:", Z_train.shape, Z_val.shape, Z_test.shape)


# %% Choose nu on the validation vessels only
def scores(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def flagged(model, Z):
    return (model.predict(Z) == -1).astype(int)  # -1 = outlier = anomaly


tuning = []
for nu in NU_GRID:
    svm = OneClassSVM(kernel="rbf", gamma=GAMMA, nu=nu).fit(Z_train)
    tuning.append({"nu": nu, **scores(y_val, flagged(svm, Z_val))})
tuning = pd.DataFrame(tuning)
print(tuning.round(4).to_string(index=False))

best_nu = float(tuning.loc[tuning["f1"].idxmax(), "nu"])
detector = OneClassSVM(kernel="rbf", gamma=GAMMA, nu=best_nu).fit(Z_train)
joblib.dump(detector, config.MODEL_DIR / config.DETECTOR_FILE)
print(f"\nSelected nu = {best_nu} (best validation F1). Saved", config.MODEL_DIR / config.DETECTOR_FILE)

# %% Final evaluation on the test vessels, which played no part in any choice above
test_pred = flagged(detector, Z_test)
results = {
    "one_class_svm": {"nu": best_nu, **scores(y_test, test_pred)},
    "isolation_forest": scores(
        y_test, flagged(IsolationForest(contamination="auto", random_state=config.SEED).fit(Z_train), Z_test)
    ),
    # Reference point: what flagging every window would score
    "flag_everything": scores(y_test, np.ones_like(y_test)),
    "test_windows": int(len(y_test)),
    "test_anomaly_share": float(y_test.mean()),
    "validation_tuning": tuning.to_dict(orient="records"),
}
print(classification_report(y_test, test_pred, target_names=["Normal", "Anomaly"], digits=4))
(config.RESULTS_DIR / "detector_metrics.json").write_text(json.dumps(results, indent=2))
print("Metrics saved to", config.RESULTS_DIR / "detector_metrics.json")

# %% Plots: confusion matrix, latent space, tuning curve
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

sns.heatmap(confusion_matrix(y_test, test_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"], ax=axes[0])
axes[0].set(title=f"Test confusion matrix (nu={best_nu})", xlabel="Predicted", ylabel="True")

Z_2d = PCA(n_components=2, random_state=config.SEED).fit_transform(Z_test)
axes[1].scatter(*Z_2d[y_test == 0].T, s=4, alpha=0.4, label="Normal")
axes[1].scatter(*Z_2d[y_test == 1].T, s=4, alpha=0.4, label="Anomaly", color="tab:red")
axes[1].set(title="Test latent vectors (PCA)", xlabel="PC 1", ylabel="PC 2")
axes[1].legend()

axes[2].plot(tuning["nu"], tuning["f1"], marker="o")
axes[2].axvline(best_nu, color="grey", linestyle="--", label=f"selected nu = {best_nu}")
axes[2].set(title="Validation F1 by nu", xlabel="nu", ylabel="Anomaly F1")
axes[2].legend()

plt.tight_layout()
plt.savefig(config.RESULTS_DIR / "detector_evaluation.png", bbox_inches="tight")
if INTERACTIVE:
    plt.show()
plt.close()
print("Plot saved to", config.RESULTS_DIR / "detector_evaluation.png")
