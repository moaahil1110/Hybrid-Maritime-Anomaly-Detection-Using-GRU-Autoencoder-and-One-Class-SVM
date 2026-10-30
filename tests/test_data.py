import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from maritime_ad.config import FEATURES, SEQ_LEN
from maritime_ad.data import (
    inject_anomalies,
    load_ais,
    make_windows,
    scale_windows,
    split_by_vessel,
    vessel_bounds,
)


def make_tracks(points_per_vessel, seed=0):
    """Time-sorted tracks; latitude encodes the vessel so windows can be traced back."""
    rng = np.random.default_rng(seed)
    frames = []
    for vessel, n in enumerate(points_per_vessel):
        frames.append(pd.DataFrame({
            "mmsi": 100 + vessel,
            "timestamp": pd.date_range("2018-01-01", periods=n, freq="5min"),
            "latitude": float(vessel),
            "longitude": rng.uniform(10, 20, n),
            "speed": rng.uniform(5, 15, n),
            "course": rng.uniform(0, 360, n),
            "heading": rng.uniform(0, 360, n),
        }))
    return pd.concat(frames, ignore_index=True)


def test_windows_never_cross_vessels():
    df = make_tracks([15, 12, 5])
    X, y = make_windows(df)
    # 15-point track -> 6 windows, 12 -> 3, 5 -> none
    assert X.shape == (9, SEQ_LEN, len(FEATURES))
    assert np.all(X[:, :, 0] == X[:, :1, 0]), "a window mixes two vessels"
    assert not y.any()


def test_injected_labels_stay_inside_their_segments():
    df = make_tracks([400] * 20)
    injected = inject_anomalies(df, segments_per_1000=5)

    # Points are only labelled inside runs that belong to one vessel
    for start, end in zip(*vessel_bounds(injected["mmsi"])):
        assert injected["label"].iloc[start:end].sum() <= 2 * SEQ_LEN

    # 2 segments per vessel -> at most 2 * (2 * SEQ_LEN - 1) anomalous windows each
    _, y = make_windows(injected)
    assert 0 < y.mean() < 0.15

    changed = ~np.isclose(injected[FEATURES].to_numpy(), df[FEATURES].to_numpy()).all(axis=1)
    assert not changed[injected["label"].to_numpy() == 0].any(), "unlabelled point was modified"


def test_shuffled_input_cannot_spread_labels(tmp_path):
    # The old notebook labelled df.loc[start:end] on a shuffled index, which tagged ~99% of rows.
    df = make_tracks([300] * 30)
    shuffled = df.sample(frac=1.0, random_state=1).rename(columns={"latitude": "lat", "longitude": "lon"})
    csv = tmp_path / "ais.csv"
    shuffled.to_csv(csv, index=False)

    loaded = load_ais(csv, clip_quantiles=None)
    assert loaded["mmsi"].is_monotonic_increasing
    for _, track in loaded.groupby("mmsi"):
        assert track["timestamp"].is_monotonic_increasing

    labelled_share = inject_anomalies(loaded, segments_per_1000=5)["label"].mean()
    assert labelled_share <= 2 * SEQ_LEN / 300


def test_load_ais_replaces_unavailable_heading(tmp_path):
    df = make_tracks([3]).rename(columns={"latitude": "lat", "longitude": "lon"})
    df.loc[1, "heading"] = 511
    csv = tmp_path / "ais.csv"
    df.to_csv(csv, index=False)

    loaded = load_ais(csv, clip_quantiles=None)
    assert loaded.loc[1, "heading"] == pytest.approx(loaded.loc[1, "course"])


def test_split_by_vessel_has_no_overlap():
    df = make_tracks([20] * 50)
    train, val, test = split_by_vessel(df)
    sets = [set(part["mmsi"]) for part in (train, val, test)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert sum(len(s) for s in sets) == 50
    assert (len(sets[1]), len(sets[2])) == (10, 10)


def test_scale_windows_matches_point_scaling():
    X, _ = make_windows(make_tracks([30, 30]))
    scaler = StandardScaler().fit(X.reshape(-1, X.shape[-1]))
    scaled = scale_windows(X, scaler)
    assert scaled.shape == X.shape and scaled.dtype == np.float32
    np.testing.assert_allclose(scaled[3], scaler.transform(X[3]), rtol=1e-5, atol=1e-5)
