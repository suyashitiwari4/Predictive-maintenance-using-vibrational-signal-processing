"""
train_ims.py
------------
Trains a multiclass degradation-stage classifier (Normal / Suspect /
Degraded / Near-Failure) on the NASA IMS Bearing Dataset, using each
bearing's position in its run-to-failure timeline as the label source.

Usage:
    python src/train_ims.py --data_dir data/ims/1st_test --channel 0
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

from features import bandpass_filter, extract_features
from data_loading import load_ims_folder, assign_degradation_stage, IMS_FS


def build_dataset(data_dir, channel, fs):
    df = load_ims_folder(data_dir, channel=channel)
    if df.empty:
        raise RuntimeError(
            f"No files found in {data_dir}. Download the IMS dataset from "
            "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/"
            "pcoe/pcoe-data-set-repository/ and point --data_dir at one test folder "
            "(e.g. '1st_test')."
        )
    df = assign_degradation_stage(df)

    rows = []
    for _, r in df.iterrows():
        clean = bandpass_filter(r["signal"], fs=fs)
        feats = extract_features(clean, fs)
        feats["stage"] = r["stage"]
        rows.append(feats)

    full = pd.DataFrame(rows)
    X = full.drop(columns=["stage"])
    y = full["stage"].values
    return X, y, df  # df returned too, for plotting the health trend curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/ims/1st_test")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--fs", type=int, default=IMS_FS)
    parser.add_argument("--model", choices=["rf", "xgb"], default="rf")
    parser.add_argument("--out", default="models/ims_model.joblib")
    args = parser.parse_args()

    print("Loading + featurizing IMS data (this can take a while for long runs)...")
    X, y, df = build_dataset(args.data_dir, args.channel, args.fs)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    if args.model == "rf":
        clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
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
