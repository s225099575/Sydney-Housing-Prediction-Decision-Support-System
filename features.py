"""Shared cleaning and feature engineering for the Sydney housing project.
Used by both the notebook (training) and app.py (prediction)."""
import numpy as np
import pandas as pd

PROPERTY_GROUPS = {
    "Apartment": "Apartment", "Unit": "Apartment", "Studio": "Apartment",
    "House": "House",
    "Townhouse": "Townhouse/Other", "Duplex/semi-detached": "Townhouse/Other",
    "Villa": "Townhouse/Other", "Retirement living": "Townhouse/Other",
}
PROPERTY_TYPES = list(PROPERTY_GROUPS)
SUBURBS = ["Mosman", "Parramatta", "Penrith"]

CAT_COLS = ["suburb", "property_group"]
NUM_COLS = ["bedrooms", "bathrooms", "parking", "land_size", "total_rooms",
            "log_land", "is_landed", "has_parking", "land_missing"]
FEATURES = CAT_COLS + NUM_COLS


def clean(df):
    """Training-data cleaning: fix types, drop exact duplicates, fill gaps."""
    df = df.drop_duplicates().copy()
    df["land_size"] = pd.to_numeric(df["land_size"], errors="coerce")
    df["bedrooms"] = df["bedrooms"].fillna(0)      # studios have no separate bedroom
    df["parking"] = df["parking"].fillna(0)        # blank = no car space listed
    df["property_group"] = df["property_type"].map(PROPERTY_GROUPS)
    # Landed properties with no listed land size: flag, then fill with suburb median
    landed_missing = df["land_size"].isna()
    df["land_missing"] = landed_missing.astype(int)
    medians = df[df["land_size"] > 0].groupby("suburb")["land_size"].median()
    df.loc[landed_missing, "land_size"] = df.loc[landed_missing, "suburb"].map(medians)
    return df.reset_index(drop=True)


def add_features(df):
    """Engineered features (works on a cleaned frame or a single app input row)."""
    df = df.copy()
    if "property_group" not in df:
        df["property_group"] = df["property_type"].map(PROPERTY_GROUPS)
    if "land_missing" not in df:
        df["land_missing"] = 0
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    df["log_land"] = np.log1p(df["land_size"])
    df["is_landed"] = (df["property_group"] != "Apartment").astype(int)
    df["has_parking"] = (df["parking"] > 0).astype(int)
    return df
