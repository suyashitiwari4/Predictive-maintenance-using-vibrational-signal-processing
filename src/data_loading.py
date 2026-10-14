"""
data_loading.py
----------------
Loaders for the two raw datasets:

1. CWRU Bearing Dataset (.mat files) -> used for multiclass FAULT-TYPE
   classification (Healthy / Inner Race / Outer Race / Ball).

2. NASA IMS Bearing Dataset (plain text files, one per timestamp) ->
   used for DEGRADATION-STAGE classification (Normal / Suspect /
   Degraded / Near-Failure), based on where each file sits in a
   bearing's run-to-failure timeline.

Download links:
  CWRU -> https://engineering.case.edu/bearingdatacenter/download-data-file
  IMS  -> https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
          (direct zip mirror: https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip)
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat


# =================================================================
# CWRU loader
# =================================================================

# CWRU sampling rate is either 12kHz or 48kHz depending on the file set.
CWRU_FS_12K = 12000
CWRU_FS_48K = 48000

# Map filename keywords -> fault label. Adjust to match the exact
# filenames you download (CWRU naming isn't perfectly consistent).
CWRU_LABEL_KEYWORDS = {
    "normal": "Healthy",
    "ir": "InnerRace",
    "or": "OuterRace",
    "b0": "Ball",       # e.g. B007, B014...
    "ball": "Ball",
}


def _infer_cwru_label(filename):
    fname = filename.lower()
    for key, label in CWRU_LABEL_KEYWORDS.items():
        if key in fname:
            return label
    return "Unknown"


def _find_de_signal_key(mat_dict):
    """
    CWRU .mat files store the drive-end signal under a key like
    'X097_DE_time'. This grabs the first key containing 'DE_time'.
    """
    for k in mat_dict.keys():
        if "DE_time" in k:
            return k
    raise KeyError("No '*_DE_time' key found in this .mat file")


def load_cwru_folder(folder_path, fs=CWRU_FS_12K):
    """
    Loads every .mat file in folder_path, extracts the drive-end (DE)
    vibration channel, and returns a list of dicts:
        {"signal": np.array, "label": str, "file": str, "fs": fs}
    """
    records = []
    for filepath in sorted(glob.glob(os.path.join(folder_path, "*.mat"))):
        mat = loadmat(filepath)
        key = _find_de_signal_key(mat)
        sig = mat[key].squeeze().astype(float)
        label = _infer_cwru_label(os.path.basename(filepath))
        records.append({
            "signal": sig,
            "label": label,
            "file": os.path.basename(filepath),
            "fs": fs,
        })
    return records


# =================================================================
# IMS loader
# =================================================================

# IMS test set sampling rate
IMS_FS = 20480


def load_ims_folder(folder_path, channel=0):
    """
    IMS data ships as one plain-text file per short recording
    (filename = timestamp, e.g. '2003.10.22.12.06.24'), each
    containing columns for each bearing/channel. Files are sorted
    chronologically by filename, which represents the bearing's
    life timeline from healthy -> failure.

    channel : which column (0-indexed) to use as the vibration signal
              for a specific bearing/accelerometer.

    Returns a DataFrame with columns: [file, timestamp_rank, signal]
    ordered from earliest (healthiest) to latest (failure).
    """
    files = sorted(glob.glob(os.path.join(folder_path, "*")))
    rows = []
    for i, filepath in enumerate(files):
        data = np.loadtxt(filepath)
        sig = data[:, channel] if data.ndim > 1 else data
        rows.append({
            "file": os.path.basename(filepath),
            "timestamp_rank": i,
            "signal": sig,
        })
    return pd.DataFrame(rows)


def assign_degradation_stage(df, stage_boundaries=(0.6, 0.8, 0.95)):
    """
    Splits a chronologically-ordered IMS run (output of load_ims_folder)
    into degradation stages based on position in the timeline:

        Normal        : 0%   -> stage_boundaries[0]
        Suspect       : stage_boundaries[0] -> stage_boundaries[1]
        Degraded      : stage_boundaries[1] -> stage_boundaries[2]
        Near-Failure  : stage_boundaries[2] -> 100%

    This is a simple heuristic starting point. For a more rigorous
    cut, replace this with thresholds on RMS/kurtosis trend instead
    of pure time position (see notebooks/eda for guidance).
    """
    n = len(df)
    ranks = df["timestamp_rank"].values / (n - 1)

    labels = np.empty(n, dtype=object)
    labels[ranks <= stage_boundaries[0]] = "Normal"
    labels[(ranks > stage_boundaries[0]) & (ranks <= stage_boundaries[1])] = "Suspect"
    labels[(ranks > stage_boundaries[1]) & (ranks <= stage_boundaries[2])] = "Degraded"
    labels[ranks > stage_boundaries[2]] = "Near-Failure"

    df = df.copy()
    df["stage"] = labels
    return df
