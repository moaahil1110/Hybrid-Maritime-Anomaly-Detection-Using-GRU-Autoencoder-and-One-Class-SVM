"""Train the Bidirectional GRU autoencoder.

Trains on clean 10-step windows of [latitude, longitude, speed, course, heading]
from the training vessels and validates on a separate set of vessels.

Input:  the Baltic AIS CSV at MARITIME_DATA_CSV (default data/training_set.csv)
Output, in MARITIME_MODEL_DIR (default models/latest-model/):
  final_data_scaler.pkl          StandardScaler the encoder was trained with
  final_autoencoder_model.keras  full autoencoder
  gru_encoder.keras              encoder half (window -> 128-d latent vector)
  training_config.json           settings and vessel sample, reused by train_detector.py

Usage:  python scripts/train_autoencoder.py [--epochs N] [--sample-vessels F]
Run this before scripts/train_detector.py. The "# %%" markers let editors such
as VS Code run each section as an interactive cell.
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

parser = argparse.ArgumentParser(description="Train the GRU autoencoder.")
parser.add_argument("--epochs", type=int, default=150, help="maximum epochs; early stopping usually ends sooner")
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--sample-vessels", type=float, default=1.0,
                    help="fraction of vessels to use, e.g. 0.2 for a quicker run")
args = parser.parse_args([] if INTERACTIVE else None)

import joblib
import matplotlib
if not INTERACTIVE:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import GRU, Bidirectional, Dense, Input, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from maritime_ad import config
from maritime_ad.data import load_ais, make_windows, sample_vessels, scale_windows, split_by_vessel

np.random.seed(config.SEED)
tf.random.set_seed(config.SEED)
config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LEARNING_RATE = 1e-3
GRU_UNITS = 64          # per direction, so the latent vector is 128-d
NOISE_FACTOR = 0.05     # denoising: train on noisy input, reconstruct the clean window

print("Data:  ", config.DATA_CSV)
print("Models:", config.MODEL_DIR)

# %% Load, then split by vessel so no vessel's windows appear in more than one split
df = sample_vessels(load_ais(config.DATA_CSV), args.sample_vessels)
df_train, df_val, _ = split_by_vessel(df)  # the test vessels are only used by train_detector.py
del df

X_train, _ = make_windows(df_train)
X_val, _ = make_windows(df_val)
print(f"Vessels: train {df_train['mmsi'].nunique()}, val {df_val['mmsi'].nunique()}")
print(f"Windows: train {X_train.shape}, val {X_val.shape}")

# %% Fit the scaler on training windows only, and save it for train_detector.py and the API
scaler = StandardScaler().fit(X_train.reshape(-1, X_train.shape[-1]))
joblib.dump(scaler, config.MODEL_DIR / config.SCALER_FILE)

X_train_scaled = scale_windows(X_train, scaler)
X_val_scaled = scale_windows(X_val, scaler)
del X_train, X_val
print("Scaler saved to", config.MODEL_DIR / config.SCALER_FILE)

# %% Build and train the autoencoder
inputs = Input(shape=(config.SEQ_LEN, len(config.FEATURES)), name="window")
latent = Bidirectional(GRU(GRU_UNITS, activation="relu"), name="encoder")(inputs)
repeated = RepeatVector(config.SEQ_LEN)(latent)
decoded = Bidirectional(GRU(GRU_UNITS, activation="relu", return_sequences=True), name="decoder")(repeated)
outputs = TimeDistributed(Dense(len(config.FEATURES), activation="linear"))(decoded)

autoencoder = Model(inputs, outputs, name="gru_autoencoder")
autoencoder.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mean_squared_error")
autoencoder.summary()

rng = np.random.default_rng(config.SEED)
X_train_noisy = X_train_scaled + NOISE_FACTOR * rng.standard_normal(X_train_scaled.shape, dtype=np.float32)

history = autoencoder.fit(
    X_train_noisy,
    X_train_scaled,
    epochs=args.epochs,
    batch_size=args.batch_size,
    validation_data=(X_val_scaled, X_val_scaled),
    callbacks=[
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
    ],
    verbose=1 if INTERACTIVE else 2,
)

# %% Save the models (EarlyStopping restored the best weights) and the training settings
encoder = Model(inputs, latent, name="gru_encoder")
autoencoder.save(config.MODEL_DIR / config.AUTOENCODER_FILE)
encoder.save(config.MODEL_DIR / config.ENCODER_FILE)

val_loss = history.history["val_loss"]
best_epoch = int(np.argmin(val_loss)) + 1
training_config = {
    "seed": config.SEED,
    "sample_vessels": args.sample_vessels,
    "epochs_run": len(val_loss),
    "best_epoch": best_epoch,
    "best_val_loss": float(min(val_loss)),
    "batch_size": args.batch_size,
}
(config.MODEL_DIR / config.TRAINING_CONFIG_FILE).write_text(json.dumps(training_config, indent=2))
print(f"Best epoch {best_epoch} of {len(val_loss)}, val_loss {min(val_loss):.5f}")
print("Models saved to", config.MODEL_DIR)

# %% Plot the training curve
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="train")
plt.plot(val_loss, label="validation")
plt.axvline(best_epoch - 1, color="grey", linestyle="--", label=f"best (epoch {best_epoch})")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Autoencoder training loss")
plt.legend()
plt.savefig(config.RESULTS_DIR / "autoencoder_loss.png", bbox_inches="tight")
if INTERACTIVE:
    plt.show()
plt.close()
print("Plot saved to", config.RESULTS_DIR / "autoencoder_loss.png")
