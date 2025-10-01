#!/usr/bin/env python3
"""
Clean the crop production dataset.

Steps performed:
1) Drop junk index columns (e.g., 'Unnamed: 0')
2) Normalize categoricals: strip + lowercase for State_Name, Season, Crop
3) Remove NA and negative values in numeric columns
4) Enforce yield consistency: keep rows where Yield_ton_per_hec == Production_in_tons / Area_in_hectares (±1e-6)
5) Drop ALL rows with Production_in_tons == 0 (global decision)
6) Per-crop upper-tail yield trim at the 99.5th percentile to remove absurd outliers
7) Save cleaned CSV and print a summary

Adjust INPUT/OUTPUT paths and YIELD_TRIM_QUANTILE as needed.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ===== CONFIG =====
INPUT_CSV = "../Data/RawData/Crop_production.csv"
OUTPUT_CSV = "../Data/CleanedData/Crop_production_cleaned.csv"
YIELD_TRIM_QUANTILE = 0.995                  # 99.5th pct per-crop upper trim
NUM_TOL = 1e-6                               # tolerance for yield consistency check

# Columns we expect (script is robust if some are missing)
CAT_COLS = ["State_Name", "Season", "Crop"]
NUM_COLS = [
    "N","P","K","pH","rainfall","temperature",
    "Area_in_hectares","Production_in_tons","Yield_ton_per_hec"
]

def drop_junk_index_columns(df: pd.DataFrame) -> pd.DataFrame:
    junk = [c for c in df.columns if c.lower().startswith("unnamed")]
    return df.drop(columns=junk, errors="ignore")

def normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()
    return df

def sanitize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in NUM_COLS:
        if c in df.columns:
            df = df[df[c].notna()]     # drop NA
            df = df[df[c] >= 0]        # drop negatives
    return df

def enforce_yield_consistency(df: pd.DataFrame, tol: float = NUM_TOL):
    if {"Production_in_tons","Area_in_hectares","Yield_ton_per_hec"}.issubset(df.columns):
        recomputed = df["Production_in_tons"] / df["Area_in_hectares"]
        agree = (df["Yield_ton_per_hec"] - recomputed).abs() <= tol
        n_bad = int((~agree).sum())
        df = df[agree]
        return df, n_bad
    return df, 0

def drop_zero_production(df: pd.DataFrame):
    if "Production_in_tons" in df.columns:
        n_zero = int((df["Production_in_tons"] == 0).sum())
        df = df[df["Production_in_tons"] > 0]
        return df, n_zero
    return df, 0

def trim_yield_outliers_per_crop(df: pd.DataFrame, q: float = YIELD_TRIM_QUANTILE):
    if {"Crop","Yield_ton_per_hec"}.issubset(df.columns) and len(df) > 0:
        # Per-crop threshold
        thr = df.groupby("Crop", observed=True)["Yield_ton_per_hec"].quantile(q)
        cut = df["Crop"].map(thr)
        mask_out = df["Yield_ton_per_hec"] > cut
        n_out = int(mask_out.sum())
        df = df[~mask_out]
        return df, n_out
    return df, 0

def main():
    # Load
    df = pd.read_csv(INPUT_CSV)

    # 1) Drop junk index columns
    df = drop_junk_index_columns(df)

    # 1.5) Rename Crop_Type to Season
    if 'Crop_Type' in df.columns:
        df = df.rename(columns={'Crop_Type': 'Season'})
        print(f"Renamed 'Crop_Type' column to 'Season'")

    # 2) Normalize categorical text
    df = normalize_categoricals(df)

    # 3) Remove NA/negatives in numerics
    df = sanitize_numeric(df)

    # 4) Enforce yield consistency
    df, n_inconsistent = enforce_yield_consistency(df, tol=NUM_TOL)

    # 5) Drop ALL zero-production rows
    df, n_zero_prod = drop_zero_production(df)

    # 6) Per-crop upper-tail yield trim
    df, n_outliers = trim_yield_outliers_per_crop(df, q=YIELD_TRIM_QUANTILE)

    # 7) Save
    df = df.reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)

    # Summary
    print("====== CLEANING SUMMARY ======")
    print(f"Input file: {INPUT_CSV}")
    print(f"Output file: {OUTPUT_CSV}")
    print(f"Dropped rows (yield inconsistency): {n_inconsistent}")
    print(f"Dropped rows (Production_in_tons == 0): {n_zero_prod}")
    print(f"Dropped rows (yield > {int(YIELD_TRIM_QUANTILE*1000)/10}% per-crop): {n_outliers}")
    print(f"Final rows: {len(df):,}")
    print("Categoricals normalized:", [c for c in CAT_COLS if c in df.columns])

if __name__ == "__main__":
    main()
