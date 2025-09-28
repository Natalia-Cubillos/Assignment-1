"""
Name: Natalia Andrea Cubillos Villegas
Student ID: U3246979 

Data Science Technology and Systems
Assignment 1: Predictive Modelling of Eating-Out Problem
"""
# src/preprocess.py
# Minimal, DVC-friendly preprocessing script

import argparse, os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

def map_rating_from_number(x):
    if pd.isna(x): return np.nan
    if x >= 4.5:   return "Excellent"
    if x >= 4.0:   return "Very Good"
    if x >= 3.5:   return "Good"
    if x >= 2.5:   return "Average"
    return "Poor"

def main(src, dst):
    df = pd.read_csv(src)

    # 1) Impute cost fields with median
    for col in ["cost", "cost_2"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # 2) KNN impute rating_number & votes using cost/coords as predictors
    targets     = [c for c in ["rating_number", "votes"] if c in df.columns]
    predictors  = [c for c in ["cost", "cost_2", "lat", "lng"] if c in df.columns]
    cols = list(dict.fromkeys(targets + predictors))
    if targets and predictors and all(c in df.columns for c in cols):
        work = df[cols].copy()
        nan_mask = work[targets].isna()
        imputer = KNNImputer(n_neighbors=5, weights="distance")
        imputed = pd.DataFrame(imputer.fit_transform(work), columns=cols, index=df.index)
        for t in targets:
            df.loc[nan_mask[t], t] = imputed.loc[nan_mask[t], t]

    # 3) Drop rows missing lat/lng (spatially sensitive)
    must_have = [c for c in ["lat", "lng"] if c in df.columns]
    if must_have:
        df = df.dropna(subset=must_have)

    # 4) Impute 'type' with mode
    if "type" in df.columns:
        mode_type = df["type"].dropna().mode()
        if not mode_type.empty:
            df["type"] = df["type"].fillna(mode_type.iloc[0])

    # 5) rating_text: fill only the missing ones from rating_number
    if "rating_text" in df.columns and "rating_number" in df.columns:
        mask = df["rating_text"].isna()
        df.loc[mask, "rating_text"] = df.loc[mask, "rating_number"].apply(map_rating_from_number)

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    df.to_csv(dst, index=False)
    print(f"[preprocess] saved {dst} with shape={df.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    args = ap.parse_args()
    main(args.src, args.dst)
