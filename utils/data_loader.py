"""Data loading utilities for ZE-Workbench."""

import pandas as pd
import tempfile
import os
import streamlit as st


def load_file(uploaded_file):
    """
    Load CSV or SAV files.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        tuple: (DataFrame, metadata) - metadata is None for CSV files
    """
    if uploaded_file is None:
        return None, None

    name = uploaded_file.name

    if name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        return df, None

    elif name.endswith('.sav'):
        try:
            import pyreadstat
            with tempfile.NamedTemporaryFile(delete=False, suffix=".sav") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            df, meta = pyreadstat.read_sav(tmp_path)
            os.remove(tmp_path)
            return df, meta
        except ImportError:
            st.error(
                "Please install pyreadstat to open .sav files: `pip install pyreadstat`")
            return None, None
    else:
        st.error("Unsupported file format. Please upload CSV or SAV files.")
        return None, None


def download_df(df):
    """
    Convert DataFrame to CSV bytes for download.

    Args:
        df: pandas DataFrame

    Returns:
        bytes: CSV encoded as UTF-8
    """
    return df.to_csv(index=False).encode('utf-8')


def get_variable_metadata(df, meta=None):
    """
    Generate a 'Variable View' dataframe with column metadata.

    Args:
        df: pandas DataFrame
        meta: pyreadstat metadata object (optional)

    Returns:
        DataFrame with columns: Name, Type, Label, Missing, Unique, Measure
    """
    data = []

    for col in df.columns:
        col_name = col
        col_type = str(df[col].dtype)

        # Label from metadata if available
        col_label = ""
        if meta and hasattr(meta, 'column_names_to_labels'):
            col_label = meta.column_names_to_labels.get(col, "")

        missing_count = df[col].isnull().sum()
        unique_count = df[col].nunique()

        # Measure heuristic
        if pd.api.types.is_numeric_dtype(df[col]):
            if unique_count < 15:
                measure = "Nominal/Ordinal"
            else:
                measure = "Scale"
        else:
            measure = "Nominal"

        data.append([col_name, col_type, col_label,
                    missing_count, unique_count, measure])

    return pd.DataFrame(data, columns=["Name", "Type", "Label", "Missing", "Unique", "Measure"])
