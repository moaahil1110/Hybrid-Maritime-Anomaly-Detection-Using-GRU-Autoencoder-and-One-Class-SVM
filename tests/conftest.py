import numpy as np
import pandas as pd


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
