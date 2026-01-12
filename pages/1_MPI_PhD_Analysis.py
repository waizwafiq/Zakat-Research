"""
MPI PhD-Level Clustering Analysis

A rigorous experimental design and methodology for clustering binary/categorical
deprivation data from the Multidimensional Poverty Index (MPI) survey.

This page implements the four-phase methodology:
- Phase 1: Data Audit & Mathematical Representation
- Phase 2: Defining Similarity in Discrete Space
- Phase 3: Experimental Models (K-Modes, LCA, HAC, Spectral)
- Phase 4: Evaluation & Benchmarking Strategy

Author: Data Science Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyreadstat
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.binary_analysis import (
    tetrachoric_correlation_matrix,
    identify_redundant_features,
    perform_mca,
    get_binary_columns,
    binarize_dataframe
)
from utils.cluster_profiling import (
    calculate_item_response_probabilities,
    calculate_relative_risk_ratios,
    generate_cluster_profiles
)
from utils.validation import (
    bootstrap_stability_analysis,
    external_validation
)
from models import LatentClassModel, KModesModel, HierarchicalJaccardModel

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="MPI PhD Analysis | ZE-Workbench",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DATA LOADING
# =============================================================================
DATA_FILE = Path(__file__).parent.parent / "data" / "DATA MPI SPSS.sav"

@st.cache_data
def load_mpi_data():
    """Load and cache the MPI dataset."""
    df, meta = pyreadstat.read_sav(str(DATA_FILE))
    return df, meta

# Load data
df, meta = load_mpi_data()

# =============================================================================
# FEATURE GROUPINGS (Based on MPI Framework)
# =============================================================================
# Define feature groups based on MPI dimensions and Maqasid Syariah
# Column names match the actual SPSS variable names
FEATURE_GROUPS = {
    'Economic (Income & Employment)': [
        'PendapatanbulananpurataisirumahkurangdaripadaPGK',  # Income below PGK
        'Isirumahtidakmampumenyewadanmemilikirumah'  # Cannot afford rent/own home
    ],
    'Living Standards (Housing)': [
        'Keadaantempattinggalusangataumulaiburuk',  # Dilapidated housing
        'Lebih2ahliisirumahbilik',  # Overcrowding
        'Menggunakantandasselaindaripadatandaspam',  # No flush toilet
        'TiadakemudahanKutipanSampah',  # No waste collection
        'Semuaahliisirumahsamaadatidakmenggunakanpengangkutanpersendirian'  # No private transport
    ],
    'Digital & Communication': [
        'Tidakmemilikitelefontaliantetapatautelefonbimbit',  # No phone
        'Tidakdapatmengaksesinternet',  # No internet access
        'TidakmendapatkemudahanawamTeknologiICT'  # No ICT facilities
    ],
    'Education': [
        'Semuaahliisirumahberumur1760tahunmempunyaikurangdaripada11tahunp',  # Low education
        'Kanakkanakantara616tahunyangtidakbersekolah'  # Children not in school
    ],
    'Maqasid Syariah': [
        'Pemeliharaanagama',  # Preservation of religion
        'Pemeliharaanakal',  # Preservation of intellect
        'Pemeliharaannyawa',  # Preservation of life
        'Pemeliharaanketurunan',  # Preservation of lineage
        'Pemeliharaanharta'  # Preservation of wealth
    ],
    'Food & Health': [
        'Pengambilanmakanankurangdaripadatiga3kalisehari',  # Less than 3 meals/day
        'Selaindaripadabekalanairterawatdidalamrumahdanpaipairawampaipber',  # No treated water
        'IndikatorKemudahanJarakkepadakemudahankesihatanmelebihi3kmdantia'  # Health facility >3km
    ]
}

# Short display names for visualization
COLUMN_SHORT_NAMES = {
    'PendapatanbulananpurataisirumahkurangdaripadaPGK': 'Income < PGK',
    'Isirumahtidakmampumenyewadanmemilikirumah': 'Housing Afford',
    'Keadaantempattinggalusangataumulaiburuk': 'Dilapidated',
    'Lebih2ahliisirumahbilik': 'Overcrowding',
    'Menggunakantandasselaindaripadatandaspam': 'No Flush Toilet',
    'TiadakemudahanKutipanSampah': 'No Waste Collect',
    'Semuaahliisirumahsamaadatidakmenggunakanpengangkutanpersendirian': 'No Transport',
    'Tidakmemilikitelefontaliantetapatautelefonbimbit': 'No Phone',
    'Tidakdapatmengaksesinternet': 'No Internet',
    'TidakmendapatkemudahanawamTeknologiICT': 'No ICT',
    'Semuaahliisirumahberumur1760tahunmempunyaikurangdaripada11tahunp': 'Low Education',
    'Kanakkanakantara616tahunyangtidakbersekolah': 'Child No School',
    'Pemeliharaanagama': 'Maqasid: Religion',
    'Pemeliharaanakal': 'Maqasid: Intellect',
    'Pemeliharaannyawa': 'Maqasid: Life',
    'Pemeliharaanketurunan': 'Maqasid: Lineage',
    'Pemeliharaanharta': 'Maqasid: Wealth',
    'Pengambilanmakanankurangdaripadatiga3kalisehari': 'Food < 3 meals',
    'Selaindaripadabekalanairterawatdidalamrumahdanpaipairawampaipber': 'No Clean Water',
    'IndikatorKemudahanJarakkepadakemudahankesihatanmelebihi3kmdantia': 'Health > 3km'
}

# Helper function to get short display name
def get_short_name(col):
    """Get short display name for a column."""
    return COLUMN_SHORT_NAMES.get(col, col[:20] + '...' if len(col) > 20 else col)

# All binary deprivation indicators (from predefined groups)
ALL_INDICATORS = [col for cols in FEATURE_GROUPS.values() for col in cols if col in df.columns]

# Fallback: if no predefined indicators found, detect binary columns automatically
if not ALL_INDICATORS:
    ALL_INDICATORS = get_binary_columns(df)

# Create short names list for display
ALL_INDICATORS_SHORT = [get_short_name(col) for col in ALL_INDICATORS]

# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
st.sidebar.title("MPI PhD Analysis")
st.sidebar.caption("Rigorous Clustering Methodology")
st.sidebar.divider()

phase = st.sidebar.radio(
    "Analysis Phase",
    [
        "Overview",
        "Phase 1: Data Audit",
        "Phase 2: Distance Metrics",
        "Phase 3: Clustering Models",
        "Phase 4: Validation",
        "Final Report"
    ],
    label_visibility="collapsed"
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def compute_distance_matrix(X, metric='jaccard'):
    """Compute pairwise distance matrix."""
    if metric == 'jaccard':
        dist = pdist(X, metric='jaccard')
    elif metric == 'hamming':
        dist = pdist(X, metric='hamming')
    elif metric == 'dice':
        dist = pdist(X, metric='dice')
    else:
        dist = pdist(X, metric='jaccard')
    # Handle NaN values
    dist = np.nan_to_num(dist, nan=0.0)
    return squareform(dist)


def silhouette_with_precomputed(X, labels, metric='jaccard'):
    """Calculate silhouette score with precomputed distance matrix."""
    try:
        dist_matrix = compute_distance_matrix(X, metric)
        return silhouette_score(dist_matrix, labels, metric='precomputed')
    except:
        return np.nan


# =============================================================================
# OVERVIEW PAGE
# =============================================================================
if phase == "Overview":
    st.title("MPI PhD-Level Clustering Analysis")
    st.caption("Rigorous Experimental Design for Binary Deprivation Data")

    st.markdown("""
    ---
    ### Research Context

    This analysis addresses the challenge of clustering **Multidimensional Poverty Index (MPI)** data,
    which consists of binary deprivation indicators. Standard geometric clustering (K-Means with
    Euclidean distance) is **mathematically invalid** for such data because:

    1. The "mean" of a binary vector is interpretable only as a probability, not a centroid location
    2. Euclidean distance suffers from the curse of dimensionality in sparse binary spaces
    3. The assumption of spherical clusters is violated

    ---
    ### Methodology Overview
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### Phase 1: Data Audit
        - Reconstruct binary matrix X
        - Tetrachoric correlation analysis
        - Feature group segregation
        - Redundancy identification

        #### Phase 2: Distance Metrics
        - Hamming Distance (symmetric)
        - Jaccard Dissimilarity (asymmetric)
        - Gower's Coefficient
        - Metric benchmarking
        """)

    with col2:
        st.markdown("""
        #### Phase 3: Experimental Models
        - **Model A**: K-Modes (Partitional)
        - **Model B**: Latent Class Analysis (Probabilistic)
        - **Model C**: Hierarchical + MCA
        - **Model D**: Spectral Clustering

        #### Phase 4: Validation
        - Bootstrap Stability (ARI)
        - External Validation (Cramer's V)
        - Model Comparison (BIC/AIC)
        - MPI Profile Heatmaps
        """)

    st.divider()

    # Data Summary
    st.subheader("Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observations (N)", len(df))
    col2.metric("Total Variables", len(df.columns))
    col3.metric("Binary Indicators (P)", len(ALL_INDICATORS))
    col4.metric("Feature Groups", len(FEATURE_GROUPS))

    # Feature groups table
    st.subheader("Feature Groups (Multi-View Structure)")

    groups_data = []
    for group, features in FEATURE_GROUPS.items():
        available = [f for f in features if f in df.columns]
        groups_data.append({
            'Domain': group,
            'Variables': ', '.join(available),
            'Count': len(available)
        })

    groups_df = pd.DataFrame(groups_data)
    st.dataframe(groups_df, use_container_width=True, hide_index=True)

    # Mathematical formulation
    st.subheader("Mathematical Formulation")

    st.latex(r"""
    X \in \{0, 1\}^{N \times P} \quad \text{where} \quad
    x_{ij} = \begin{cases}
    1 & \text{if household } i \text{ is deprived in indicator } j \\
    0 & \text{otherwise}
    \end{cases}
    """)

# =============================================================================
# PHASE 1: DATA AUDIT
# =============================================================================
elif phase == "Phase 1: Data Audit":
    st.title("Phase 1: Data Audit & Mathematical Representation")
    st.caption("Formalizing the input space and analyzing feature correlations")

    # Sub-navigation
    audit_section = st.radio(
        "Section",
        ["1.1 Binary Matrix", "1.2 Tetrachoric Correlation", "1.3 Feature Selection"],
        horizontal=True,
        label_visibility="collapsed"
    )

    # Get binary columns and create binary matrix
    binary_cols = [col for col in ALL_INDICATORS if col in df.columns]
    X_binary = binarize_dataframe(df, binary_cols)

    # -------------------------------------------------------------------------
    if audit_section == "1.1 Binary Matrix":
        st.subheader("1.1 Binary Matrix Reconstruction")

        st.info("""
        **Objective:** Reconstruct the matrix X where each element represents the
        deprivation status of a household on a specific indicator.
        """)

        # Show binary matrix preview
        st.markdown("#### Binary Deprivation Matrix Preview")
        st.dataframe(X_binary.head(20), use_container_width=True, height=400)

        # Summary statistics
        st.markdown("#### Deprivation Rates by Indicator")

        deprivation_rates = X_binary.mean().sort_values(ascending=False)
        rates_df = pd.DataFrame({
            'Indicator': [get_short_name(col) for col in deprivation_rates.index],
            'Full Name': deprivation_rates.index,
            'Deprivation Rate': deprivation_rates.values,
            'N Deprived': (X_binary.sum()).astype(int).values,
            'N Non-Deprived': (len(X_binary) - X_binary.sum()).astype(int).values
        })

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(rates_df, use_container_width=True, hide_index=True)

        with col2:
            fig = px.bar(
                rates_df,
                x='Indicator',
                y='Deprivation Rate',
                title='Deprivation Rate by Indicator',
                color='Deprivation Rate',
                color_continuous_scale='YlOrRd'
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        # Feature group breakdown
        st.markdown("#### Deprivation by Feature Group")

        group_rates = []
        for group, features in FEATURE_GROUPS.items():
            available = [f for f in features if f in X_binary.columns]
            if available:
                mean_rate = X_binary[available].mean().mean()
                group_rates.append({'Group': group, 'Mean Deprivation Rate': mean_rate})

        group_df = pd.DataFrame(group_rates)

        if len(group_df) > 0:
            fig = px.bar(
                group_df,
                x='Group',
                y='Mean Deprivation Rate',
                title='Mean Deprivation Rate by Domain',
                color='Mean Deprivation Rate',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=350, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No feature groups found matching the data columns.")

    # -------------------------------------------------------------------------
    elif audit_section == "1.2 Tetrachoric Correlation":
        st.subheader("1.2 Tetrachoric Correlation Analysis")

        st.info("""
        **Why Tetrachoric Correlation?**

        Pearson correlation is inappropriate for binary data. Tetrachoric correlation
        estimates the correlation between two latent continuous variables that underlie
        the observed binary responses. High correlations (|r| > 0.9) indicate redundant
        indicators that should be removed before clustering.
        """)

        # Select variables
        selected_vars = st.multiselect(
            "Select Variables for Analysis",
            binary_cols,
            default=binary_cols[:min(15, len(binary_cols))]
        )

        if len(selected_vars) >= 2:
            with st.spinner("Computing tetrachoric correlation matrix..."):
                tetra_corr = tetrachoric_correlation_matrix(X_binary, selected_vars)

            # Heatmap
            fig = px.imshow(
                tetra_corr,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                aspect="auto",
                zmin=-1, zmax=1,
                title="Tetrachoric Correlation Matrix"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Redundant pairs
            st.markdown("#### Highly Correlated Pairs (|r| > 0.9)")

            threshold = st.slider("Redundancy Threshold", 0.7, 0.99, 0.9, 0.05)
            redundant = identify_redundant_features(tetra_corr, threshold=threshold)

            if redundant:
                redundant_df = pd.DataFrame(
                    redundant,
                    columns=['Variable 1', 'Variable 2', 'Correlation']
                )
                st.dataframe(redundant_df, use_container_width=True, hide_index=True)
                st.warning(f"Found {len(redundant)} redundant pairs. Consider removing one from each pair.")
            else:
                st.success("No highly redundant pairs found at the current threshold.")
        else:
            st.info("Select at least 2 variables for correlation analysis.")

    # -------------------------------------------------------------------------
    elif audit_section == "1.3 Feature Selection":
        st.subheader("1.3 Feature Selection for Clustering")

        st.info("""
        **Selection Criteria:**

        1. Remove features with correlation > 0.9 (redundant information)
        2. Remove features with near-zero variance (no discriminating power)
        3. Balance representation across feature domains
        """)

        # Compute redundancy
        with st.spinner("Analyzing feature redundancy..."):
            tetra_corr = tetrachoric_correlation_matrix(X_binary, binary_cols)
            redundant = identify_redundant_features(tetra_corr, threshold=0.9)

        # Features to remove
        remove_candidates = set()
        for v1, v2, corr in redundant:
            var1 = X_binary[v1].var()
            var2 = X_binary[v2].var()
            remove_candidates.add(v2 if var1 >= var2 else v1)

        recommended = [c for c in binary_cols if c not in remove_candidates]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Variables to Remove")
            if remove_candidates:
                for v in remove_candidates:
                    st.write(f"- {v}")
            else:
                st.success("No variables need to be removed.")

        with col2:
            st.markdown("#### Recommended Variables")
            st.write(f"**{len(recommended)}** variables selected:")
            for v in recommended:
                st.write(f"- {v}")

        # Store in session state
        st.session_state.selected_features = recommended
        st.session_state.X_binary = X_binary[recommended]

        st.success(f"Feature selection complete. {len(recommended)} features saved for clustering.")

# =============================================================================
# PHASE 2: DISTANCE METRICS
# =============================================================================
elif phase == "Phase 2: Distance Metrics":
    st.title("Phase 2: Defining Similarity in Discrete Space")
    st.caption("Benchmarking distance metrics for binary poverty data")

    # Get binary data
    binary_cols = [col for col in ALL_INDICATORS if col in df.columns]
    X_binary = binarize_dataframe(df, binary_cols)
    X_array = X_binary.values.astype(float)

    st.info("""
    **Core Insight:** Two households are "close" if they share specific patterns of deprivation.
    The choice of distance metric fundamentally affects cluster interpretation.
    """)

    # Distance metric explanations
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### Hamming Distance
        **Symmetric metric**

        Counts the proportion of positions where two vectors differ.

        **Use case:** When absence of deprivation (0-0) is equally informative as presence (1-1).
        """)
        st.latex(r"d_H(x_i, x_k) = \frac{1}{P}\sum_{j=1}^P |x_{ij} - x_{kj}|")

    with col2:
        st.markdown("""
        #### Jaccard Dissimilarity
        **Asymmetric metric**

        Focuses only on shared deprivations, ignoring mutual non-deprivations.

        **Use case:** Critical for poverty data where shared deprivations matter more.
        """)
        st.latex(r"d_J(x_i, x_k) = 1 - \frac{|M_{11}|}{|M_{01}| + |M_{10}| + |M_{11}|}")

    with col3:
        st.markdown("""
        #### Dice Coefficient
        **Asymmetric metric**

        Similar to Jaccard but weights shared deprivations more heavily.

        **Use case:** When emphasizing commonality over total presence.
        """)
        st.latex(r"d_D(x_i, x_k) = 1 - \frac{2|M_{11}|}{2|M_{11}| + |M_{01}| + |M_{10}|}")

    st.divider()

    # Distance matrix comparison
    st.subheader("Distance Matrix Comparison")

    # Subsample for visualization
    n_sample = min(100, len(X_array))
    sample_idx = np.random.RandomState(42).choice(len(X_array), n_sample, replace=False)
    X_sample = X_array[sample_idx]

    with st.spinner("Computing distance matrices..."):
        dist_hamming = compute_distance_matrix(X_sample, 'hamming')
        dist_jaccard = compute_distance_matrix(X_sample, 'jaccard')
        dist_dice = compute_distance_matrix(X_sample, 'dice')

    # Heatmaps
    col1, col2, col3 = st.columns(3)

    with col1:
        fig = px.imshow(
            dist_hamming,
            color_continuous_scale='Viridis',
            title='Hamming Distance',
            aspect='equal'
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.imshow(
            dist_jaccard,
            color_continuous_scale='Viridis',
            title='Jaccard Distance',
            aspect='equal'
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fig = px.imshow(
            dist_dice,
            color_continuous_scale='Viridis',
            title='Dice Distance',
            aspect='equal'
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Distribution comparison
    st.subheader("Distance Distribution Analysis")

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=dist_hamming.flatten(), name='Hamming', opacity=0.7, nbinsx=50))
    fig.add_trace(go.Histogram(x=dist_jaccard.flatten(), name='Jaccard', opacity=0.7, nbinsx=50))
    fig.add_trace(go.Histogram(x=dist_dice.flatten(), name='Dice', opacity=0.7, nbinsx=50))
    fig.update_layout(
        barmode='overlay',
        title='Distribution of Pairwise Distances',
        xaxis_title='Distance',
        yaxis_title='Frequency',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary statistics
    st.subheader("Distance Statistics")

    stats_data = []
    for name, dist in [('Hamming', dist_hamming), ('Jaccard', dist_jaccard), ('Dice', dist_dice)]:
        flat = dist[np.triu_indices_from(dist, k=1)]
        stats_data.append({
            'Metric': name,
            'Mean': f"{flat.mean():.4f}",
            'Std': f"{flat.std():.4f}",
            'Min': f"{flat.min():.4f}",
            'Max': f"{flat.max():.4f}",
            'Median': f"{np.median(flat):.4f}"
        })

    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

    st.markdown("""
    ---
    **Recommendation:** For MPI deprivation data, **Jaccard distance** is preferred because:
    - It ignores mutual non-deprivations (both households NOT having internet is less informative)
    - It emphasizes shared deprivation patterns
    - It handles the asymmetric nature of poverty indicators
    """)

# =============================================================================
# PHASE 3: CLUSTERING MODELS
# =============================================================================
elif phase == "Phase 3: Clustering Models":
    st.title("Phase 3: Experimental Clustering Models")
    st.caption("Comparing partitional, probabilistic, and hierarchical approaches")

    # Get binary data
    binary_cols = [col for col in ALL_INDICATORS if col in df.columns]
    X_binary = binarize_dataframe(df, binary_cols)
    X_array = X_binary.values.astype(float)

    # Model selection
    model_tab = st.radio(
        "Select Model",
        ["A. K-Modes", "B. Latent Class Analysis", "C. Hierarchical + MCA", "D. Model Comparison"],
        horizontal=True,
        label_visibility="collapsed"
    )

    # -------------------------------------------------------------------------
    if model_tab == "A. K-Modes":
        st.subheader("Model A: K-Modes Clustering")

        st.info("""
        **K-Modes** is the categorical analogue of K-Means. It uses:
        - Mode (most frequent value) instead of mean for centroids
        - Matching distance or Jaccard distance instead of Euclidean
        - Frequency-based centroid updates
        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Configuration")
            k_kmodes = st.slider("Number of Clusters (K)", 2, 10, 4, key="kmodes_k")
            distance_metric = st.selectbox(
                "Distance Metric",
                ['jaccard', 'hamming', 'dice'],
                index=0
            )
            init_method = st.selectbox(
                "Initialization",
                ['huang', 'cao', 'random'],
                help="Huang: frequency-based, Cao: diversity-maximizing"
            )
            n_init = st.slider("Random Initializations", 5, 50, 10)

            run_kmodes = st.button("Run K-Modes", type="primary", use_container_width=True)

        with col2:
            if run_kmodes:
                with st.spinner("Running K-Modes clustering..."):
                    model = KModesModel()
                    model.set_params(
                        n_clusters=k_kmodes,
                        distance_metric=distance_metric,
                        init_method=init_method,
                        n_init=n_init
                    )
                    labels = model.fit_predict(X_array)

                # Store results
                st.session_state.kmodes_labels = labels
                st.session_state.kmodes_model = model

                st.success(f"K-Modes completed with {k_kmodes} clusters.")

                # Cluster sizes
                sizes = pd.Series(labels).value_counts().sort_index()
                sizes_df = pd.DataFrame({
                    'Cluster': [f'Cluster {k}' for k in sizes.index],
                    'Count': sizes.values,
                    'Proportion': (sizes.values / len(labels) * 100).round(1)
                })

                st.markdown("#### Cluster Distribution")
                st.dataframe(sizes_df, use_container_width=True, hide_index=True)

                # Cluster modes heatmap
                st.markdown("#### Cluster Modes (Centroids)")
                modes = model.get_cluster_modes()
                short_cols = [get_short_name(c) for c in binary_cols]
                modes_df = pd.DataFrame(modes, columns=short_cols)
                modes_df.index = [f'Cluster {i}' for i in range(k_kmodes)]

                fig = px.imshow(
                    modes_df,
                    text_auto='.0f',
                    color_continuous_scale='YlOrRd',
                    aspect='auto',
                    title='Cluster Modes (1 = Deprived, 0 = Non-Deprived)'
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

                # Metrics
                try:
                    sil = silhouette_with_precomputed(X_array, labels, distance_metric)
                    st.metric("Silhouette Score (Jaccard)", f"{sil:.4f}")
                except:
                    pass

    # -------------------------------------------------------------------------
    elif model_tab == "B. Latent Class Analysis":
        st.subheader("Model B: Latent Class Analysis (LCA)")

        st.info("""
        **Latent Class Analysis** is the gold standard for categorical survey data.
        It assumes observed responses are independent conditional on latent class membership.

        **Model:** P(Y) = Sum_k [pi_k * Product_j P(Y_j | C_k)]

        Use BIC/AIC for model selection (lower = better).
        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Configuration")
            k_min, k_max = st.slider("Class Range", 2, 10, (2, 6), key="lca_range")
            max_iter = st.slider("Max EM Iterations", 50, 500, 100)
            n_init_lca = st.slider("Random Initializations", 5, 30, 10)

            run_lca = st.button("Run LCA Model Selection", type="primary", use_container_width=True)

        with col2:
            if run_lca:
                results = []
                progress = st.progress(0)
                status = st.empty()

                k_range = range(k_min, k_max + 1)

                for i, k in enumerate(k_range):
                    status.text(f"Fitting LCA with {k} classes...")
                    progress.progress((i + 1) / len(k_range))

                    try:
                        model = LatentClassModel()
                        model.set_params(n_classes=k, max_iter=max_iter, n_init=n_init_lca)
                        labels = model.fit_predict(X_array)
                        stats = model.get_model_fit_statistics()

                        results.append({
                            'Classes': k,
                            'BIC': stats['bic'],
                            'AIC': stats['aic'],
                            'Log-Likelihood': stats['log_likelihood'],
                            'Entropy': stats['entropy'],
                            'model': model,
                            'labels': labels
                        })
                    except Exception as e:
                        st.warning(f"LCA with {k} classes failed: {e}")

                progress.empty()
                status.empty()

                if results:
                    st.session_state.lca_results = results

                    results_df = pd.DataFrame([
                        {k: v for k, v in r.items() if k not in ['model', 'labels']}
                        for r in results
                    ])

                    # Find optimal
                    optimal_idx = results_df['BIC'].idxmin()
                    optimal_k = int(results_df.loc[optimal_idx, 'Classes'])

                    st.success(f"Optimal number of classes: **{optimal_k}** (minimum BIC)")

                    # Comparison table
                    st.markdown("#### Model Comparison")
                    st.dataframe(
                        results_df.style.highlight_min(subset=['BIC', 'AIC']),
                        use_container_width=True,
                        hide_index=True
                    )

                    # BIC/AIC plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results_df['Classes'], y=results_df['BIC'],
                        name='BIC', mode='lines+markers'
                    ))
                    fig.add_trace(go.Scatter(
                        x=results_df['Classes'], y=results_df['AIC'],
                        name='AIC', mode='lines+markers'
                    ))
                    fig.add_vline(x=optimal_k, line_dash="dash", line_color="green",
                                  annotation_text=f"Optimal K={optimal_k}")
                    fig.update_layout(
                        title="Information Criteria by Number of Classes",
                        xaxis_title="Number of Classes",
                        yaxis_title="Information Criterion (lower = better)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Store best model
                    best_model = results[optimal_idx]['model']
                    best_labels = results[optimal_idx]['labels']
                    st.session_state.lca_labels = best_labels
                    st.session_state.lca_model = best_model

                    # Item response probabilities
                    st.markdown(f"#### Item Response Probabilities ({optimal_k} Classes)")
                    item_probs = best_model.get_item_response_probabilities()
                    short_cols = [get_short_name(c) for c in binary_cols]
                    item_probs_df = pd.DataFrame(
                        item_probs,
                        index=[f'Class {i}' for i in range(optimal_k)],
                        columns=short_cols
                    )

                    fig = px.imshow(
                        item_probs_df,
                        text_auto='.2f',
                        color_continuous_scale='YlOrRd',
                        aspect='auto',
                        title='P(Deprived | Class)'
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    elif model_tab == "C. Hierarchical + MCA":
        st.subheader("Model C: Hierarchical Clustering with MCA")

        st.info("""
        **Methodology:**
        1. Apply Multiple Correspondence Analysis (MCA) to project binary data into continuous space
        2. Apply hierarchical agglomerative clustering on MCA components

        This allows visualization of the "poverty hierarchy" and dendrogram interpretation.
        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Configuration")
            n_components = st.slider("MCA Components", 2, 10, 5)
            linkage_method = st.selectbox(
                "Linkage Method",
                ['ward', 'average', 'complete', 'single'],
                help="Ward: minimizes variance, Average: uses mean distance"
            )
            n_clusters_hac = st.slider("Number of Clusters", 2, 10, 4, key="hac_k")

            run_hac = st.button("Run HAC + MCA", type="primary", use_container_width=True)

        with col2:
            if run_hac:
                with st.spinner("Performing MCA..."):
                    mca_result = perform_mca(X_binary, binary_cols, n_components=n_components)
                    coords = mca_result['coordinates'].values

                with st.spinner("Running hierarchical clustering..."):
                    Z = linkage(coords, method=linkage_method)
                    labels = fcluster(Z, t=n_clusters_hac, criterion='maxclust') - 1

                st.session_state.hac_labels = labels

                st.success(f"HAC completed with {n_clusters_hac} clusters.")

                # MCA variance
                st.markdown("#### MCA Explained Inertia")
                inertia = mca_result['explained_inertia']
                inertia_df = pd.DataFrame({
                    'Component': [f'Dim{i+1}' for i in range(len(inertia))],
                    'Inertia': inertia,
                    'Cumulative': np.cumsum(inertia)
                })
                st.dataframe(inertia_df, use_container_width=True, hide_index=True)

                # 2D MCA scatter with clusters
                st.markdown("#### MCA Projection with Clusters")
                coords_df = mca_result['coordinates'].copy()
                coords_df['Cluster'] = [f'Cluster {l}' for l in labels]

                fig = px.scatter(
                    coords_df,
                    x='Dim1', y='Dim2',
                    color='Cluster',
                    title=f'MCA Space (Dim1: {inertia[0]:.1%}, Dim2: {inertia[1]:.1%})',
                    opacity=0.7
                )
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)

                # Cluster sizes
                sizes = pd.Series(labels).value_counts().sort_index()
                sizes_df = pd.DataFrame({
                    'Cluster': [f'Cluster {k}' for k in sizes.index],
                    'Count': sizes.values,
                    'Proportion': (sizes.values / len(labels) * 100).round(1)
                })
                st.dataframe(sizes_df, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    elif model_tab == "D. Model Comparison":
        st.subheader("Model Comparison Dashboard")

        st.info("""
        Compare all clustering solutions using internal validation metrics computed
        with the appropriate distance metric (Jaccard for binary data).
        """)

        # Check for available results
        available_models = []

        if 'kmodes_labels' in st.session_state:
            available_models.append(('K-Modes', st.session_state.kmodes_labels))
        if 'lca_labels' in st.session_state:
            available_models.append(('LCA', st.session_state.lca_labels))
        if 'hac_labels' in st.session_state:
            available_models.append(('HAC + MCA', st.session_state.hac_labels))

        if not available_models:
            st.warning("No clustering results available. Run models in previous tabs first.")
        else:
            # Compute metrics for all models
            comparison_data = []

            for name, labels in available_models:
                try:
                    sil_jaccard = silhouette_with_precomputed(X_array, labels, 'jaccard')
                    sil_euclidean = silhouette_score(X_array, labels)
                    db = davies_bouldin_score(X_array, labels)
                    ch = calinski_harabasz_score(X_array, labels)
                    n_clusters = len(np.unique(labels))

                    comparison_data.append({
                        'Model': name,
                        'K': n_clusters,
                        'Silhouette (Jaccard)': sil_jaccard,
                        'Silhouette (Euclidean)': sil_euclidean,
                        'Davies-Bouldin': db,
                        'Calinski-Harabasz': ch
                    })
                except Exception as e:
                    st.warning(f"Metrics failed for {name}: {e}")

            if comparison_data:
                comp_df = pd.DataFrame(comparison_data)

                st.markdown("#### Validation Metrics Comparison")
                st.dataframe(
                    comp_df.style
                    .highlight_max(subset=['Silhouette (Jaccard)', 'Silhouette (Euclidean)', 'Calinski-Harabasz'])
                    .highlight_min(subset=['Davies-Bouldin']),
                    use_container_width=True,
                    hide_index=True
                )

                # Visual comparison
                st.markdown("#### Visual Comparison")

                fig = make_subplots(rows=1, cols=2, subplot_titles=[
                    'Silhouette Score (higher = better)',
                    'Davies-Bouldin Index (lower = better)'
                ])

                fig.add_trace(
                    go.Bar(x=comp_df['Model'], y=comp_df['Silhouette (Jaccard)'], name='Jaccard'),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Bar(x=comp_df['Model'], y=comp_df['Davies-Bouldin'], name='DB Index'),
                    row=1, col=2
                )

                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                # Recommendation
                best_model = comp_df.loc[comp_df['Silhouette (Jaccard)'].idxmax(), 'Model']
                st.success(f"**Recommended Model:** {best_model} (highest Silhouette with Jaccard distance)")

# =============================================================================
# PHASE 4: VALIDATION
# =============================================================================
elif phase == "Phase 4: Validation":
    st.title("Phase 4: Evaluation & Benchmarking Strategy")
    st.caption("Internal validation, stability analysis, and external profiling")

    # Get binary data
    binary_cols = [col for col in ALL_INDICATORS if col in df.columns]
    X_binary = binarize_dataframe(df, binary_cols)
    X_array = X_binary.values.astype(float)

    # Select model for validation
    available_models = {}
    if 'kmodes_labels' in st.session_state:
        available_models['K-Modes'] = st.session_state.kmodes_labels
    if 'lca_labels' in st.session_state:
        available_models['LCA'] = st.session_state.lca_labels
    if 'hac_labels' in st.session_state:
        available_models['HAC + MCA'] = st.session_state.hac_labels

    if not available_models:
        st.warning("No clustering results available. Run models in Phase 3 first.")
        st.stop()

    selected_model = st.selectbox("Select Model for Validation", list(available_models.keys()))
    labels = available_models[selected_model]

    validation_tab = st.radio(
        "Validation Type",
        ["Bootstrap Stability", "External Validation", "Cluster Profiles"],
        horizontal=True,
        label_visibility="collapsed"
    )

    # -------------------------------------------------------------------------
    if validation_tab == "Bootstrap Stability":
        st.subheader("Bootstrap Stability Analysis")

        st.info("""
        **Methodology:**

        1. Resample data with replacement (bootstrap)
        2. Re-cluster each bootstrap sample
        3. Compare to original using Adjusted Rand Index (ARI)

        **Interpretation:**
        - ARI > 0.9: Excellent stability
        - ARI > 0.7: Good stability (publishable)
        - ARI > 0.5: Fair stability
        - ARI < 0.5: Poor stability
        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            n_bootstrap = st.slider("Bootstrap Iterations", 20, 200, 50)
            sample_ratio = st.slider("Sample Ratio", 0.5, 0.95, 0.8, 0.05)

            run_stability = st.button("Run Stability Analysis", type="primary", use_container_width=True)

        with col2:
            if run_stability:
                with st.spinner(f"Running {n_bootstrap} bootstrap iterations..."):
                    # Use K-Modes for stability test
                    n_clusters = len(np.unique(labels))

                    def clustering_func(X):
                        model = KModesModel()
                        model.set_params(n_clusters=n_clusters, n_init=5)
                        return model.fit_predict(X)

                    stability = bootstrap_stability_analysis(
                        X_array,
                        clustering_func,
                        n_bootstrap=n_bootstrap,
                        sample_ratio=sample_ratio
                    )

                # Results
                st.markdown("#### Stability Results")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Mean ARI", f"{stability['mean_ari']:.3f}")
                m2.metric("Std ARI", f"{stability['std_ari']:.3f}")
                m3.metric("Grade", stability['stability_grade'])
                m4.metric("Stable?", "Yes" if stability['is_stable'] else "No")

                # Distribution plot
                if stability['ari_values']:
                    fig = px.histogram(
                        x=stability['ari_values'],
                        nbins=20,
                        title='Distribution of Bootstrap ARI Values',
                        labels={'x': 'Adjusted Rand Index', 'y': 'Count'}
                    )
                    fig.add_vline(x=0.7, line_dash="dash", line_color="green",
                                  annotation_text="Good (0.7)")
                    fig.add_vline(x=stability['mean_ari'], line_dash="solid", line_color="red",
                                  annotation_text=f"Mean ({stability['mean_ari']:.3f})")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                if stability['is_stable']:
                    st.success("Clusters are **STABLE** and suitable for publication.")
                else:
                    st.warning("Clusters may be **UNSTABLE**. Consider alternative K or method.")

    # -------------------------------------------------------------------------
    elif validation_tab == "External Validation":
        st.subheader("External Validation (Cramer's V)")

        st.info("""
        **Methodology:**

        Cross-tabulate clusters with external categorical variables (e.g., Urban/Rural).

        **Metrics:**
        - Chi-Square: Statistical significance
        - Cramer's V: Effect size (0.1 = small, 0.3 = medium, 0.5 = large)
        - NMI: Information-theoretic measure
        """)

        # Find categorical columns
        cat_columns = [col for col in df.columns if df[col].dtype == 'object' or df[col].nunique() < 10]

        external_var = st.selectbox("External Variable", cat_columns)

        if st.button("Run External Validation", type="primary"):
            with st.spinner("Computing external validation..."):
                ext_val = external_validation(labels, df[external_var].values, external_var)

            # Metrics
            st.markdown("#### Results")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Chi-Square", f"{ext_val['chi_square']:.2f}")
            m2.metric("P-Value", f"{ext_val['p_value']:.4f}")
            m3.metric("Cramer's V", f"{ext_val['cramers_v']:.3f}")
            m4.metric("NMI", f"{ext_val['nmi']:.3f}")

            # Interpretation
            if ext_val['p_value'] < 0.05:
                if ext_val['cramers_v'] > 0.5:
                    effect = "Strong"
                elif ext_val['cramers_v'] > 0.3:
                    effect = "Medium"
                else:
                    effect = "Small"
                st.success(f"**Significant association** with {effect.lower()} effect size.")
            else:
                st.warning("No significant association found.")

            # Crosstab
            st.markdown("#### Cross-tabulation (Row %)")
            st.dataframe(ext_val['crosstab_pct'], use_container_width=True)

            # Visualization
            ct_pct = ext_val['crosstab_pct']
            fig = px.bar(
                ct_pct.reset_index().melt(id_vars='Cluster'),
                x='Cluster',
                y='value',
                color='variable',
                barmode='group',
                title=f'Distribution of {external_var} by Cluster',
                labels={'value': 'Percentage (%)', 'variable': external_var}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    elif validation_tab == "Cluster Profiles":
        st.subheader("MPI Profile Heatmap (Faces of Poverty)")

        st.info("""
        **Item Response Probabilities** show P(Deprived | Cluster).
        **Relative Risk Ratios** compare cluster rates to population average.

        Expected archetypes:
        1. **Deeply Deprived**: High across all indicators
        2. **Infrastructure Poor**: High services deprivation, low income deprivation
        3. **Digital Excluded**: Only technology indicators
        4. **Near Average**: Close to population baseline
        """)

        # Calculate profiles
        with st.spinner("Generating cluster profiles..."):
            short_cols = [get_short_name(c) for c in binary_cols]
            irp = calculate_item_response_probabilities(X_array, labels, short_cols)
            rr = calculate_relative_risk_ratios(X_array, labels, short_cols)
            profiles = generate_cluster_profiles(X_array, labels, short_cols)

        # Item Response Probabilities
        st.markdown("#### Item Response Probabilities")

        fig = px.imshow(
            irp,
            text_auto='.2f',
            color_continuous_scale='YlOrRd',
            aspect='auto',
            title='P(Deprived | Cluster)'
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        # Relative Risk Ratios
        st.markdown("#### Relative Risk Ratios")

        fig = px.imshow(
            rr,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=1.0,
            aspect='auto',
            title='RR = P(Deprived | Cluster) / P(Deprived | Population)'
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        # Cluster descriptions
        st.markdown("#### Cluster Archetypes")

        for cluster, desc in profiles['cluster_descriptions'].items():
            with st.expander(f"{cluster}", expanded=True):
                st.write(desc)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**High Risk Indicators**")
                    high = profiles['high_risk_features'].get(cluster, {})
                    if high:
                        for feat, val in list(high.items())[:5]:
                            st.write(f"- {feat}: RR = {val:.2f}")
                    else:
                        st.write("None identified")

                with col2:
                    st.markdown("**Protected Indicators**")
                    low = profiles['low_risk_features'].get(cluster, {})
                    if low:
                        for feat, val in list(low.items())[:5]:
                            st.write(f"- {feat}: RR = {val:.2f}")
                    else:
                        st.write("None identified")

# =============================================================================
# FINAL REPORT
# =============================================================================
elif phase == "Final Report":
    st.title("Final Analysis Report")
    st.caption("Comprehensive summary of PhD-level MPI clustering analysis")

    # Get binary data
    binary_cols = [col for col in ALL_INDICATORS if col in df.columns]
    X_binary = binarize_dataframe(df, binary_cols)
    X_array = X_binary.values.astype(float)

    st.markdown("---")

    # Data Summary
    st.header("1. Data Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observations", len(df))
    col2.metric("Indicators", len(binary_cols))
    col3.metric("Feature Groups", len(FEATURE_GROUPS))
    col4.metric("Mean Deprivation Rate", f"{X_binary.mean().mean():.1%}")

    # Model Results
    st.header("2. Clustering Results")

    available_models = {}
    if 'kmodes_labels' in st.session_state:
        available_models['K-Modes'] = st.session_state.kmodes_labels
    if 'lca_labels' in st.session_state:
        available_models['LCA'] = st.session_state.lca_labels
    if 'hac_labels' in st.session_state:
        available_models['HAC + MCA'] = st.session_state.hac_labels

    if available_models:
        # Metrics comparison
        comparison_data = []

        for name, labels in available_models.items():
            try:
                sil = silhouette_with_precomputed(X_array, labels, 'jaccard')
                db = davies_bouldin_score(X_array, labels)
                n_k = len(np.unique(labels))

                comparison_data.append({
                    'Model': name,
                    'K': n_k,
                    'Silhouette (Jaccard)': round(sil, 4),
                    'Davies-Bouldin': round(db, 4)
                })
            except:
                pass

        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Determine best model
            best_idx = comp_df['Silhouette (Jaccard)'].idxmax()
            best_model = comp_df.loc[best_idx, 'Model']
            best_labels = available_models[best_model]

            st.success(f"**Recommended Solution:** {best_model}")

            # Final cluster profiles
            st.header("3. Final Cluster Profiles")

            short_cols = [get_short_name(c) for c in binary_cols]
            profiles = generate_cluster_profiles(X_array, best_labels, short_cols)

            # Cluster sizes
            sizes_df = pd.DataFrame({
                'Cluster': list(profiles['cluster_sizes'].keys()),
                'N': list(profiles['cluster_sizes'].values()),
                'Proportion': [f"{v:.1%}" for v in profiles['cluster_proportions'].values()]
            })
            st.dataframe(sizes_df, use_container_width=True, hide_index=True)

            # Heatmap
            irp = profiles['item_response_probs']

            fig = px.imshow(
                irp,
                text_auto='.2f',
                color_continuous_scale='YlOrRd',
                aspect='auto',
                title='Final Item Response Probability Heatmap'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Descriptions
            st.header("4. Poverty Archetypes")

            for cluster, desc in profiles['cluster_descriptions'].items():
                st.markdown(f"**{cluster}:** {desc}")

            # Export options
            st.header("5. Export Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                # Export labels
                export_df = df.copy()
                export_df['Cluster'] = best_labels
                st.download_button(
                    "Download Data with Clusters",
                    export_df.to_csv(index=False).encode('utf-8'),
                    "MPI_Clustering_Results.csv",
                    "text/csv"
                )

            with col2:
                # Export item response probabilities
                st.download_button(
                    "Download Item Response Probabilities",
                    irp.to_csv().encode('utf-8'),
                    "Cluster_Item_Response_Probabilities.csv",
                    "text/csv"
                )

            with col3:
                # Export relative risk ratios
                rr = profiles['relative_risks']
                st.download_button(
                    "Download Relative Risk Ratios",
                    rr.to_csv().encode('utf-8'),
                    "Cluster_Relative_Risk_Ratios.csv",
                    "text/csv"
                )
    else:
        st.warning("No clustering results available. Complete Phase 3 first.")

    # Methodology Notes
    st.header("6. Methodology Notes")

    st.markdown("""
    **Key Methodological Decisions:**

    1. **Distance Metric:** Jaccard distance is used for all binary data analysis,
       as it appropriately handles the asymmetric nature of deprivation indicators.

    2. **Correlation Analysis:** Tetrachoric correlation is used instead of Pearson
       to account for the binary nature of the data.

    3. **Model Selection:** BIC/AIC criteria for LCA, Silhouette (Jaccard) for
       distance-based methods.

    4. **Validation:** Bootstrap stability analysis (ARI > 0.7 threshold) and
       external validation with demographic variables.

    **Limitations:**

    - Results depend on the selected number of clusters
    - Bootstrap stability may vary with different random seeds
    - External validation requires meaningful demographic variables

    **References:**

    - Alkire, S. & Foster, J. (2011). Counting and Multidimensional Poverty Measurement
    - Vermunt, J.K. & Magidson, J. (2002). Latent Class Cluster Analysis
    - Huang, Z. (1998). Extensions to the K-Means Algorithm for Clustering Large Data Sets
    """)

# =============================================================================
# FOOTER
# =============================================================================
st.sidebar.divider()
st.sidebar.caption("MPI PhD Analysis v1.0")
st.sidebar.caption("ZE-Workbench")
