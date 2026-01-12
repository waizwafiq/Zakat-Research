"""
Binary Data Analysis Module for MPI Dataset.

This module provides specialized analysis for binary/categorical survey data:
- Tetrachoric Correlation Matrix
- Multiple Correspondence Analysis (MCA)
- Binary data utilities

PhD-level methodology for poverty data analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
import warnings


def calculate_tetrachoric_correlation(x, y):
    """
    Calculate tetrachoric correlation between two binary variables.

    Tetrachoric correlation estimates the Pearson correlation between
    two latent continuous variables that underlie the observed binary data.

    Args:
        x: Binary array (0/1)
        y: Binary array (0/1)

    Returns:
        float: Tetrachoric correlation coefficient
    """
    # Create 2x2 contingency table
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    # Ensure binary
    x = (x > 0).astype(int)
    y = (y > 0).astype(int)

    # Count frequencies
    n = len(x)
    a = np.sum((x == 1) & (y == 1))  # Both 1
    b = np.sum((x == 1) & (y == 0))  # x=1, y=0
    c = np.sum((x == 0) & (y == 1))  # x=0, y=1
    d = np.sum((x == 0) & (y == 0))  # Both 0

    # Edge cases
    if a == 0 or b == 0 or c == 0 or d == 0:
        # Add small constant for continuity correction
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    total = a + b + c + d

    # Calculate marginal proportions
    p1 = (a + b) / total  # P(X=1)
    p2 = (a + c) / total  # P(Y=1)

    # Thresholds for underlying normal distributions
    try:
        h = stats.norm.ppf(1 - p1)
        k = stats.norm.ppf(1 - p2)
    except:
        return 0.0

    # Cell proportion
    p11 = a / total  # P(X=1, Y=1)

    # Objective function: find rho such that bivariate normal CDF matches observed proportion
    def objective(rho):
        if abs(rho) >= 1:
            return float('inf')
        try:
            # Use approximation for bivariate normal integral
            from scipy.stats import mvn
            lower = np.array([h, k])
            upper = np.array([np.inf, np.inf])
            cov = np.array([[1, rho], [rho, 1]])
            prob, _ = mvn.mvnun(lower, upper, np.zeros(2), cov)
            return prob - p11
        except:
            # Fallback: use cosine approximation
            cos_val = np.cos(np.pi * (1 - p11) / (p1 * p2 + (1-p1)*(1-p2) + 0.001))
            return rho - cos_val

    # Try to find root
    try:
        rho = brentq(objective, -0.999, 0.999)
    except:
        # Fallback: Digby's approximation
        odds_ratio = (a * d) / (b * c + 0.001)
        c_val = odds_ratio ** 0.25
        rho = (c_val - 1) / (c_val + 1)
        rho = np.clip(rho, -1, 1)

    return rho


def tetrachoric_correlation_matrix(df, columns=None):
    """
    Calculate tetrachoric correlation matrix for binary variables.

    Args:
        df: DataFrame with binary (0/1) columns
        columns: List of column names (uses all if None)

    Returns:
        DataFrame: Tetrachoric correlation matrix
    """
    if columns is None:
        columns = df.columns.tolist()

    n_vars = len(columns)
    corr_matrix = np.eye(n_vars)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            rho = calculate_tetrachoric_correlation(
                df[columns[i]].values,
                df[columns[j]].values
            )
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho

    return pd.DataFrame(corr_matrix, index=columns, columns=columns)


def identify_redundant_features(corr_matrix, threshold=0.9):
    """
    Identify redundant features based on high tetrachoric correlation.

    Args:
        corr_matrix: Tetrachoric correlation matrix
        threshold: Correlation threshold for redundancy

    Returns:
        list: List of tuples (var1, var2, correlation) for redundant pairs
    """
    redundant = []
    columns = corr_matrix.columns.tolist()

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > threshold:
                redundant.append((columns[i], columns[j], corr))

    return redundant


def perform_mca(df, columns=None, n_components=None):
    """
    Perform Multiple Correspondence Analysis (MCA) on categorical/binary data.

    MCA is the appropriate dimensionality reduction technique for categorical
    data, analogous to PCA for continuous data.

    Args:
        df: DataFrame with categorical/binary columns
        columns: List of column names (uses all if None)
        n_components: Number of components to return (default: min(n_vars-1, 10))

    Returns:
        dict with:
            - 'coordinates': Row coordinates in MCA space
            - 'explained_inertia': Explained inertia per component
            - 'total_inertia': Total inertia
            - 'column_coordinates': Column (variable) coordinates
            - 'eigenvalues': Eigenvalues
    """
    if columns is None:
        columns = df.columns.tolist()

    data = df[columns].copy()

    # Convert to binary indicator matrix (one-hot encoding)
    indicator_df = pd.get_dummies(data, prefix_sep='_')
    Z = indicator_df.values.astype(float)

    n_rows, n_cols = Z.shape
    n_categories = n_cols
    n_variables = len(columns)

    # Row and column masses
    total = Z.sum()
    row_masses = Z.sum(axis=1) / total
    col_masses = Z.sum(axis=0) / total

    # Expected frequencies under independence
    expected = np.outer(row_masses, col_masses) * total

    # Standardized residuals matrix
    S = (Z - expected) / np.sqrt(expected + 1e-10)

    # Weight matrices
    Dr_inv_sqrt = np.diag(1.0 / np.sqrt(row_masses + 1e-10))
    Dc_inv_sqrt = np.diag(1.0 / np.sqrt(col_masses + 1e-10))

    # Weighted matrix for SVD
    weighted_S = Dr_inv_sqrt @ S @ Dc_inv_sqrt / np.sqrt(total)

    # SVD
    try:
        U, singular_values, Vt = np.linalg.svd(weighted_S, full_matrices=False)
    except:
        # Fallback for numerical issues
        U, singular_values, Vt = np.linalg.svd(weighted_S + 1e-10, full_matrices=False)

    # Eigenvalues (squared singular values)
    eigenvalues = singular_values ** 2

    # Total inertia
    total_inertia = eigenvalues.sum()

    # Explained inertia
    explained_inertia = eigenvalues / total_inertia if total_inertia > 0 else eigenvalues

    # Determine number of components
    if n_components is None:
        n_components = min(n_variables - 1, 10, len(eigenvalues) - 1)
    n_components = max(1, min(n_components, len(eigenvalues) - 1))

    # Row coordinates (principal coordinates)
    row_coords = Dr_inv_sqrt @ U[:, 1:n_components+1] * singular_values[1:n_components+1]

    # Column coordinates
    col_coords = Dc_inv_sqrt @ Vt[1:n_components+1, :].T * singular_values[1:n_components+1]

    # Create DataFrames
    component_names = [f'Dim{i+1}' for i in range(n_components)]

    row_coord_df = pd.DataFrame(
        row_coords,
        index=df.index,
        columns=component_names
    )

    col_coord_df = pd.DataFrame(
        col_coords,
        index=indicator_df.columns,
        columns=component_names
    )

    return {
        'coordinates': row_coord_df,
        'explained_inertia': explained_inertia[1:n_components+1],
        'total_inertia': total_inertia,
        'column_coordinates': col_coord_df,
        'eigenvalues': eigenvalues[1:n_components+1],
        'n_components': n_components
    }


def is_binary_column(series, threshold=0.1):
    """
    Check if a column is binary or nearly binary.

    Args:
        series: pandas Series
        threshold: Maximum proportion of values outside {0,1} to allow

    Returns:
        bool: True if column is binary
    """
    unique_vals = series.dropna().unique()

    # Check if only 0 and 1
    if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
        return True

    # Check if only 2 unique values
    if len(unique_vals) == 2:
        return True

    return False


def get_binary_columns(df):
    """
    Identify binary columns in a DataFrame.

    Args:
        df: pandas DataFrame

    Returns:
        list: Column names that are binary
    """
    binary_cols = []
    for col in df.columns:
        if is_binary_column(df[col]):
            binary_cols.append(col)
    return binary_cols


def binarize_dataframe(df, columns=None):
    """
    Convert DataFrame columns to binary (0/1) format.

    Args:
        df: pandas DataFrame
        columns: Columns to convert (uses all if None)

    Returns:
        DataFrame with binary values
    """
    if columns is None:
        columns = df.columns.tolist()

    result = df[columns].copy()

    for col in columns:
        unique_vals = result[col].dropna().unique()

        if len(unique_vals) <= 2:
            # Already binary or can be made binary
            if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                result[col] = result[col].astype(int)
            else:
                # Map to 0/1
                sorted_vals = sorted(unique_vals)
                result[col] = result[col].map({sorted_vals[0]: 0, sorted_vals[-1]: 1})
        else:
            # Convert to binary based on median
            median_val = result[col].median()
            result[col] = (result[col] > median_val).astype(int)

    return result
