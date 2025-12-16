"""Data preprocessing utilities for clustering."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def encode_categorical(df):
    """
    Encode categorical columns using LabelEncoder.

    Args:
        df: pandas DataFrame

    Returns:
        DataFrame with categorical columns encoded as integers
    """
    df_encoded = df.copy()
    encoders = {}

    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le

    return df_encoded, encoders


def scale_features(df):
    """
    Standardize features using StandardScaler.

    Args:
        df: pandas DataFrame with numeric columns

    Returns:
        tuple: (scaled_array, scaler) - numpy array and fitted scaler
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return scaled, scaler


def preprocess_features(df, features):
    """
    Complete preprocessing pipeline: select features, encode, and scale.

    Args:
        df: pandas DataFrame
        features: list of column names to use

    Returns:
        dict with keys:
            - 'X_raw': original feature DataFrame
            - 'X_encoded': encoded DataFrame
            - 'X_scaled': scaled numpy array
            - 'scaler': fitted StandardScaler
            - 'encoders': dict of LabelEncoders
            - 'feature_names': list of feature names
    """
    X_raw = df[features].copy()

    # Handle missing values with median/mode imputation
    for col in X_raw.columns:
        if X_raw[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X_raw[col]):
                X_raw[col].fillna(X_raw[col].median(), inplace=True)
            else:
                X_raw[col].fillna(X_raw[col].mode().iloc[0] if len(X_raw[col].mode()) > 0 else 'Unknown', inplace=True)

    X_encoded, encoders = encode_categorical(X_raw)
    X_scaled, scaler = scale_features(X_encoded)

    return {
        'X_raw': X_raw,
        'X_encoded': X_encoded,
        'X_scaled': X_scaled,
        'scaler': scaler,
        'encoders': encoders,
        'feature_names': features
    }
