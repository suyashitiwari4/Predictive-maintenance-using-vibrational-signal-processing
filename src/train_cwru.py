"""
train_cwru.py
-------------
Trains a multiclass fault-type classifier (Healthy / InnerRace /
OuterRace / Ball) on the CWRU Bearing Dataset.

Usage:
    python src/train_cwru.py --data_dir data/cwru --fs 12000
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from features import bandpass_filter, segment_signal, extract_features_dataframe
from data_loading import load_cwru_folder


WINDOW_SIZE = 2048  # samples per window; ~170ms at 12kHz


def build_dataset(data_dir, fs):
    records = load_cwru_folder(data_dir, fs=fs)
    if not records:
        raise RuntimeError(
            f"No .mat files found in {data_dir}. "
            "Download the CWRU dataset from "
            "https://engineering.case.edu/bearingdatacenter/download-data-file "
            "and place the .mat files there."
        )

    all_feats = []
    all_labels = []
    for rec in records:
        clean = bandpass_filter(rec["signal"], fs=rec["fs"])
        windows = segment_signal(clean, WINDOW_SIZE)
        feats_df = extract_features_dataframe(windows, rec["fs"])
        all_feats.append(feats_df)
        all_labels.extend([rec["label"]] * len(feats_df))

    X = pd.concat(all_feats, ignore_index=True)
    y = np.array(all_labels)
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/cwru")
    parser.add_argument("--fs", type=int, default=12000)
    parser.add_argument("--model", choices=["rf", "xgb"], default="xgb")
    parser.add_argument("--out", default="models/cwru_model.joblib")
    args = parser.parse_args()

    print("Loading + featurizing CWRU data...")
    X, y = build_dataset(args.data_dir, args.fs)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    if args.model == "rf":
        clf = RandomForestClassifier(n_estimators=300, random_state=42)
    else:
        clf = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric="mlogloss", random_state=42
        )

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    print("\nClassification report:")
    print(classification_report(y_test, preds, target_names=le.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({"model": clf, "label_encoder": le, "feature_names": list(X.columns)}, args.out)
    print(f"\nSaved model to {args.out}")


if __name__ == "__main__":
    main()
