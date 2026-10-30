"""Data preparation shared by both training notebooks.

Every function here works on a DataFrame sorted by vessel (mmsi) and then by
time, with a fresh RangeIndex, which is what load_ais returns. Under that
ordering each vessel's track is one contiguous block of rows, so windows and
injected anomalies can be addressed by position without crossing vessels.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURES, SEED, SEQ_LEN

HEADING_NOT_AVAILABLE = 511  # AIS code for "no heading"
ANOMALY_PATTERNS = ("speed_spike", "direction_change", "location_jump", "slow_movement")


def load_ais(csv_path, clip_quantiles=(0.01, 0.99)):
    """Load the AIS CSV (columns timestamp, mmsi, lat, lon, speed, course, heading).

    Returns one row per (mmsi, timestamp), sorted by vessel then time.
    """
    df = pd.read_csv(
        csv_path,
        usecols=["timestamp", "mmsi", "lat", "lon", "speed", "course", "heading"],
        low_memory=False,
    )
    df = df.rename(columns={"lat": "latitude", "lon": "longitude"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Fall back to course when heading is missing, as app/live_client.py does
    no_heading = df["heading"].isna() | (df["heading"] == HEADING_NOT_AVAILABLE)
    df.loc[no_heading, "heading"] = df.loc[no_heading, "course"]

    df = df.dropna(subset=["timestamp"] + FEATURES)
    df = df.drop_duplicates(subset=["mmsi", "timestamp"])

    if clip_quantiles is not None:
        low, high = clip_quantiles
        for col in FEATURES:
            df[col] = df[col].clip(df[col].quantile(low), df[col].quantile(high))

    return df.sort_values(["mmsi", "timestamp"], kind="stable").reset_index(drop=True)


def split_by_vessel(df, val_size=0.2, test_size=0.2, seed=SEED):
    """Split into train/val/test DataFrames so each vessel lands in exactly one split."""
    vessels = df["mmsi"].unique()
    train_vessels, rest = train_test_split(vessels, test_size=val_size + test_size, random_state=seed)
    val_vessels, test_vessels = train_test_split(
        rest, test_size=test_size / (val_size + test_size), random_state=seed
    )

    def rows_for(selected):
        return df[df["mmsi"].isin(selected)].reset_index(drop=True)

    return rows_for(train_vessels), rows_for(val_vessels), rows_for(test_vessels)


def vessel_bounds(mmsi):
    """Start and end (exclusive) row positions of each vessel's block of rows."""
    mmsi = np.asarray(mmsi)
    if len(mmsi) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    change = np.flatnonzero(mmsi[1:] != mmsi[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(mmsi)]))
    return starts, ends


def inject_anomalies(df, segments_per_1000=5, seq_len=SEQ_LEN, seed=SEED):
    """Return a copy of df with synthetic anomalous segments and a 0/1 'label' column.

    Each vessel with at least seq_len points gets about segments_per_1000
    segments per 1,000 points (at least one). A segment is seq_len consecutive
    points of that vessel's time-ordered track, altered with one pattern from
    ANOMALY_PATTERNS. Only the points inside a segment are labelled 1.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    lat = out["latitude"].to_numpy(dtype=float, copy=True)
    lon = out["longitude"].to_numpy(dtype=float, copy=True)
    speed = out["speed"].to_numpy(dtype=float, copy=True)
    course = out["course"].to_numpy(dtype=float, copy=True)
    heading = out["heading"].to_numpy(dtype=float, copy=True)
    label = np.zeros(len(out), dtype=np.int8)

    for start, end in zip(*vessel_bounds(out["mmsi"])):
        n_points = end - start
        if n_points < seq_len:
            continue
        n_segments = max(1, round(n_points * segments_per_1000 / 1000))
        for seg_start in rng.integers(start, end - seq_len + 1, size=n_segments):
            seg = slice(seg_start, seg_start + seq_len)
            pattern = rng.choice(ANOMALY_PATTERNS)
            if pattern == "speed_spike":
                speed[seg] *= rng.uniform(3, 5)
            elif pattern == "direction_change":
                course[seg] = (course[seg] + 180) % 360
                heading[seg] = (heading[seg] + 180) % 360
            elif pattern == "location_jump":
                lat[seg] += rng.uniform(0.5, 1.0)
                lon[seg] -= rng.uniform(0.5, 1.0)
            else:  # slow_movement
                speed[seg] = rng.uniform(0.1, 0.5)
            label[seg] = 1

    out["latitude"], out["longitude"], out["speed"] = lat, lon, speed
    out["course"], out["heading"], out["label"] = course, heading, label
    return out


def make_windows(df, seq_len=SEQ_LEN, features=FEATURES):
    """Cut each vessel's track into overlapping windows of seq_len points.

    Windows never span two vessels. Returns (X, y): X is float32 with shape
    (n_windows, seq_len, n_features); y is 1 when any point in the window is
    labelled anomalous (all zeros if df has no 'label' column).
    """
    data = df[features].to_numpy(dtype=np.float32)
    point_labels = df["label"].to_numpy() if "label" in df else np.zeros(len(df), dtype=np.int8)

    starts = [np.arange(s, e - seq_len + 1) for s, e in zip(*vessel_bounds(df["mmsi"])) if e - s >= seq_len]
    window_starts = np.concatenate(starts) if starts else np.empty(0, dtype=np.int64)
    idx = window_starts[:, None] + np.arange(seq_len)

    X = data[idx]
    y = point_labels[idx].any(axis=1).astype(np.int8)

    # Drop frozen windows: identical points while reporting more than 1 knot
    frozen = (X.var(axis=1).sum(axis=1) == 0) & (X[:, :, features.index("speed")].mean(axis=1) > 1)
    return X[~frozen], y[~frozen]


def scale_windows(X, scaler):
    """Apply a scaler fitted on individual points (n_features columns) to 3-D windows."""
    n_windows, seq_len, n_features = X.shape
    return scaler.transform(X.reshape(-1, n_features)).reshape(n_windows, seq_len, n_features).astype(np.float32)
