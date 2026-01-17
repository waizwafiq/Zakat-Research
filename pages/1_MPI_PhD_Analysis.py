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
    st.caption("Rigorous Experimental Design for Zakat Eligibility Assessment")

    st.markdown("""
    ---
    ### Research Context: Why This Matters for Zakat

    Since we are dealing with **Zakat eligibility**, the stakes are incredibly high. We are not just
    optimizing a business metric; we are trying to mathematically ensure justice (*'Adl*) in distribution.
    If our model is wrong, a deserving family might be excluded, or funds might be misallocated.

    Because of this, standard "out-of-the-box" approaches like **K-Means are dangerous here**. We need
    methods that respect the specific nature of our data (binary "Yes/No" checklists) and the
    **Maqasid Syariah** framework embedded in our variables.

    ---
    ### Why Standard Clustering Fails

    Standard geometric clustering (K-Means with Euclidean distance) is **mathematically invalid** for this data:

    1. The "mean" of a binary vector is interpretable only as a probability, not a centroid location
    2. Euclidean distance suffers from the curse of dimensionality in sparse binary spaces
    3. The assumption of spherical clusters is violated

    ---
    ### Methodology Overview
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### Phase 1: The "Digital Checklist"
        - Represent households as Deprivation Vectors
        - Tetrachoric correlation analysis
        - Feature group segregation
        - Redundancy identification

        #### Phase 2: Measuring Similarity
        - Jaccard Distance (asymmetric - ignores shared non-poverty)
        - Hamming Distance (symmetric)
        - Focus on shared *problems*, not shared wealth
        """)

    with col2:
        st.markdown("""
        #### Phase 3: Finding Hidden Groups
        - **Model A**: K-Modes (interpretable archetypes)
        - **Model B**: LCA (probability of membership)
        - **Model C**: Hierarchical + MCA

        #### Phase 4: The 'Adl Check
        - Bootstrap Stability (ARI)
        - External Validation (Cramer's V)
        - Maqasid Syariah Profiling
        - Tailored intervention recommendations
        """)

    # Summary table
    st.divider()
    st.subheader("Summary: Why Each Approach Was Chosen")

    summary_data = pd.DataFrame({
        'Approach': [
            'Binary Vectors',
            'Jaccard Distance',
            'Latent Class Analysis',
            'K-Modes (not K-Means)',
            'Maqasid Profiling'
        ],
        'Why We Chose It for Zakat': [
            'Because poverty is a checklist of specific lacks, not a single average number.',
            'Because we care about shared *problems*, not shared wealth.',
            'Because it handles uncertainty and tells us the *probability* of a family being in a specific poverty category.',
            'Because K-Means gives "0.5 toilets" - physically impossible. K-Modes gives interpretable archetypes.',
            'To ensure the Zakat type (Cash vs. Education vs. Housing) matches the actual deficit.'
        ]
    })
    st.dataframe(summary_data, use_container_width=True, hide_index=True)

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
        st.subheader("1.1 The 'Digital Checklist' - Binary Matrix Reconstruction")

        st.info("""
        **What is a Deprivation Vector?**

        Our dataset contains variables like Income < PGK, No Waste Collection, and Preservation of Intellect.
        These are **Binary Variables**. A household either *is* deprived (1) or *is not* deprived (0).

        We represent every household as a "Deprivation Vector":
        - **Household A:** `[1, 0, 1, ...]` (Poor Income, Good Toilet, No Internet...)
        """)

        with st.expander("Why not just average these into a single Poverty Score?", expanded=False):
            st.warning("""
            **The Trap:** You might be tempted to average these 0s and 1s to get a "Poverty Score" immediately.

            **The Risk:** If Family X lacks **Food** but has a **Phone**, and Family Y has **Food** but no **Phone**,
            their "average score" might be the same. But for Zakat, Family X (starving) is critically different
            from Family Y (disconnected).

            **The Fix:** We keep the data as *vectors* (patterns) rather than smashing them into a single
            average score too early. This preserves the *type* of poverty.
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
    st.title("Phase 2: Measuring Similarity - The 'Symptom' Match")
    st.caption("Why Jaccard Distance is critical for Zakat eligibility")

    # Get binary data
    binary_cols = [col for col in ALL_INDICATORS if col in df.columns]
    X_binary = binarize_dataframe(df, binary_cols)
    X_array = X_binary.values.astype(float)

    st.info("""
    **Core Insight:** To cluster people, we have to measure how "close" Household A is to Household B.
    For Zakat, we don't care about clustering the non-poor. We care about the **presence of deprivation**.
    """)

    # Educational example - Hospital Triage Analogy
    with st.expander("The Hospital Triage Analogy - Why We Ignore the Non-Poor", expanded=True):
        st.markdown("""
        This is a crucial concept in **Unbalanced Data Analytics**. We use an analogy from emergency
        medicine, which shares the same "triage" philosophy as Zakat distribution.

        ---

        **Imagine running a busy hospital emergency room:**

        - **Patient A** is healthy. No fever, no broken bones, no pain.
        - **Patient B** is also healthy. No fever, no broken bones, no pain.
        - **Patient C** has a broken leg.
        - **Patient D** has a heart attack.

        If we use a standard clustering algorithm that treats "Health" and "Sickness" as equally important,
        the computer would look at Patient A and Patient B and say:

        > *"Wow! Patient A and Patient B are 100% identical! They match on every single symptom (having none).
        > This is the strongest cluster in the dataset!"*

        **Why is this useless?**

        Because the hospital's goal is not to identify healthy people. We don't need a "Cluster of Healthy People"
        to assign a doctor to. We need to distinguish Patient C (Orthopedics) from Patient D (Cardiology).

        **In Zakat, the "Non-Poor" are the healthy patients.** If we treat "Not Poor" (0) as a similarity trait,
        our algorithm will spend most of its energy discovering that rich people are rich, which is a waste of
        time and distorts the math.
        """)

    # Mathematical explanation
    with st.expander("The Mathematical Explanation - The 'Sparse Data' Problem", expanded=False):
        st.markdown("""
        Let's look at the actual math of why this happens using hypothetical households:

        - **Household Rich 1:** `[0, 0, 0, 0, 0]` (No deprivations)
        - **Household Rich 2:** `[0, 0, 0, 0, 0]` (No deprivations)
        - **Household Poor A:** `[1, 1, 0, 0, 0]` (Lacks Food, Lacks Income)
        - **Household Poor B:** `[0, 0, 1, 1, 0]` (Lacks Education, Lacks Internet)

        ---

        #### Scenario 1: Using "Symmetric" Similarity (Standard Approach)

        This approach counts **everything** that matches, whether it's a 1 or a 0.

        - **Rich 1 vs. Rich 2:** They match on 5 out of 5 items. **100% Similarity.**
        - **Poor A vs. Poor B:** They match on 1 item (the last zero). **20% Similarity.**

        **Result:** The algorithm sees the Rich households as the most "important" and "tight" cluster.
        The Poor households look like noise because they are different from each other.
        The model essentially says, *"I found a huge group of people who are fine!"*

        ---

        #### Scenario 2: Using "Asymmetric" Similarity (Jaccard Approach)

        This approach **ignores** the 0-0 matches. It asks: *"Out of the problems they HAVE, how many do they SHARE?"*

        - **Rich 1 vs. Rich 2:** They have 0 problems. They share 0 problems. **Undefined/Irrelevant.** The algorithm ignores them.
        - **Poor A vs. Poor B:** They have distinct problems. The algorithm sees them as **distinct types of poverty**.

        **Result:** The algorithm forces itself to look *only* at the people with 1s (Deprivation).
        It ignores the massive "background noise" of the non-poor.

        ---

        #### The Zakat Implication

        If we don't ignore the non-poor (the 0-0 matches), two risks emerge:

        1. **The "Average" Problem:** The sheer number of non-poor people (zeros) will drown out the signal
           of the poor. The algorithm might group a "Hardcore Poor" family and a "Near Poor" family into one
           big "Generally Poor" cluster just because they are both "Not Rich," missing the nuance that one
           needs food and the other needs a laptop.

        2. **Resource Allocation:** Zakat is about **specificity**. We need to know *why* someone is poor.
           - If they are in a cluster defined by `Income=1`, they get cash (*Fakir/Miskin*).
           - If they are in a cluster defined by `Debt=1`, they get debt relief (*Al-Gharimin*).
           - If they are in a cluster defined by `Education=0` (Not deprived), we shouldn't waste scholarship funds on them.

        By "not caring about the non-poor," we are mathematically forcing the model to zoom in strictly on the
        **deficits**, ensuring that every cluster we generate represents a specific **need** that Zakat can solve.
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

    # Detailed Jaccard explanation
    with st.expander("Deep Dive: Why Jaccard is Critical for Our Analysis", expanded=False):
        st.markdown("""
        This is the most critical mathematical concept in our entire research design.
        If this point is misunderstood, the whole clustering experiment will fail to produce useful Zakat categories.

        ---

        ### 1. The Data Definition (The Trap)

        In our dataset, the variables are coded like this:

        - **0 = "Okay"** (Has toilet, has income, has food).
        - **1 = "Deprived"** (No toilet, no income, no food).

        In a normal dataset (like clustering flowers or customers), "0" and "1" are just two different colors.
        Being "Blue" (0) is just as important as being "Red" (1).

        **But in poverty data, "0" is essentially "Nothing to see here."**

        - A "0" means the family is fine in that area.
        - A "1" means the family has a problem.

        ---

        ### 2. The "0-0 Match" (The False Positive)

        When we say Jaccard **"ignores the 0-0 matches,"** we mean this scenario:

        Imagine two wealthy families, **Family A** and **Family B**.

        - **Family A:** Has Income (0), Has Toilet (0), Has Food (0), Has Internet (0). -> `[0, 0, 0, 0]`
        - **Family B:** Has Income (0), Has Toilet (0), Has Food (0), Has Internet (0). -> `[0, 0, 0, 0]`

        If we use a standard algorithm (like Euclidean distance or Simple Matching), it calculates:

        > *"Wow! Family A and Family B match on 4 out of 4 items! They are 100% identical!
        > This is the strongest cluster in the universe!"*

        **The Problem:** We have just successfully clustered people who **do not need Zakat**.
        The algorithm is wasting its energy finding similarities between people who are fine.
        This "0-0 match" (matching on *not* having a problem) dominates the data because most people
        are usually not deprived in everything.

        ---

        ### 3. The "1-1 Match" (The Signal)

        When we say Jaccard **"focuses strictly on the 1-1 matches (shared suffering),"** we mean this:

        Imagine two struggling families, **Family X** and **Family Y**.

        - **Family X:** No Income (1), Has Toilet (0), No Food (1), Has Internet (0).
        - **Family Y:** No Income (1), Has Toilet (0), No Food (1), Has Internet (0).

        **Jaccard Distance** looks at this and says:

        > *"I see they both have Zeros (toilets/internet) - I will IGNORE that. I don't care about what they have.*
        > *I see they both have Ones (No Income/No Food) - I will COUNT that.*
        > *Conclusion: These families are identical because they share the same specific pain."*

        ---

        ### 4. The "Specific Needs" (The Zakat Application)

        If we didn't use Jaccard, we might get a cluster called **"The Generally Poor."**

        - It would include people who lack Food (`1`) and people who lack Education (`1`) mixed together,
          simply because they are "Not Rich" (they don't have `0`s everywhere).

        By using Jaccard, we force the math to only group people if they match on the **specific problem**:

        - **Cluster A:** Everyone here matches because they all have `1` on **Income**. -> *Remedy: Cash Zakat.*
        - **Cluster B:** Everyone here matches because they all have `1` on **Education**. -> *Remedy: Scholarship Zakat.*

        ---

        ### Summary

        - **Standard Math:** "You are both similar because you both own shoes." (Useless for Zakat).
        - **Jaccard Math:** "You are both similar because you both have a broken leg." (Useful for Zakat).

        **We use Jaccard because Zakat is a hospital for society; we classify patients by their injuries,
        not by the organs that are healthy.**
        """)

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
    st.title("Phase 3: Finding the Hidden Groups")
    st.caption("Discovering 'Profiles of Poverty' using appropriate algorithms")

    # Get binary data
    binary_cols = [col for col in ALL_INDICATORS if col in df.columns]
    X_binary = binarize_dataframe(df, binary_cols)
    X_array = X_binary.values.astype(float)

    # Model selection
    model_tab = st.radio(
        "Select Model",
        ["A. K-Modes", "B. Latent Class Analysis", "C. Hierarchical + MCA", "D. Model Comparison", "E. AutoML Optimizer"],
        horizontal=True,
        label_visibility="collapsed"
    )

    # -------------------------------------------------------------------------
    if model_tab == "A. K-Modes":
        st.subheader("Model A: K-Modes Clustering (Not K-Means!)")

        st.info("""
        **K-Modes** is a hard partitioning algorithm designed for categorical data.
        It calculates the **Mode** (the most frequent answer) instead of the mean.
        """)

        with st.expander("Why NOT K-Means?", expanded=True):
            st.warning("""
            **K-Means calculates the "Centroid" (average) of a cluster.**

            **Example:** If Cluster 1 has 100 families, and 50 have a toilet and 50 don't,
            the "Average Family" has **0.5 toilets**.

            **The Problem:** "0.5 toilets" is physically impossible. It makes the cluster center uninterpretable.

            ---

            **Why K-Modes?**
            - It calculates the **Mode** (the most frequent answer).
            - If most families in Cluster 1 lack education, the "Center" of Cluster 1 will simply be "Lacks Education."
            - This creates **real, interpretable archetypes**: *"This is the cluster of families who live in cities
              but have no education."*
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
        st.subheader("Model B: Latent Class Analysis (LCA) - The Probabilistic Approach")

        st.info("""
        **Latent Class Analysis** is a probabilistic model that assumes there are hidden "classes"
        of poverty causing the observed data. It is the **gold standard** for categorical survey data.
        """)

        with st.expander("The Doctor Analogy - Understanding LCA", expanded=True):
            st.markdown("""
            Think of this like a **doctor diagnosing a flu**. The doctor doesn't see "The Flu";
            they see fever, cough, and fatigue.

            - **Symptoms** = Our Variables (Income < PGK, No Internet, etc.)
            - **The Disease** = The Latent Class (e.g., "Hardcore Poor" vs. "Situational Poor")

            ---

            **Why LCA is Best for Zakat:**

            1. It gives us a **Probability of Membership**. It won't just say "You are Poor."
               It will say "There is a 95% chance this household belongs to the Hardcore Poor group."

            2. This helps with **Borderline Cases** (e.g., a family with 45% probability).
               In Zakat, we often want to give the benefit of the doubt to these borderline cases;
               LCA lets us see them.

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

    # -------------------------------------------------------------------------
    elif model_tab == "E. AutoML Optimizer":
        st.subheader("Zakat AutoML Engine - Genetic Algorithm Optimizer")

        st.info("""
        **The Zakat AutoML Engine** uses a Genetic Algorithm (GA) to automatically discover
        the optimal clustering configuration. Instead of manually testing different models,
        K values, and feature subsets, the GA evolves toward the best solution.
        """)

        with st.expander("How the Genetic Algorithm Works", expanded=True):
            st.markdown("""
            #### The DNA of a Clustering Solution

            Each "individual" in the population is a complete clustering configuration encoded as:

            | Gene | Description | Values |
            |------|-------------|--------|
            | **Model Type** | Which algorithm | 0 = K-Modes, 1 = LCA |
            | **K** | Number of clusters | 2-15 |
            | **Feature Mask** | Which variables to use | Binary string (e.g., `11010...`) |

            ---

            #### The Zakat Effectiveness Score (ZES)

            The fitness function measures how "good" a clustering is for Zakat distribution:

            **ZES = w1*Silhouette_Jaccard + w2*Stability + w3*Balance + w4*MaqasidSeparation**

            - **Silhouette (Jaccard):** Cluster quality using appropriate distance
            - **Stability:** How consistent are clusters across bootstraps
            - **Balance:** Penalize solutions with tiny or dominant clusters
            - **Maqasid Separation:** Do clusters differentiate on Maqasid indicators?

            ---

            #### Evolution Process

            1. **Initialize:** Create 50 random clustering configurations
            2. **Evaluate:** Calculate ZES for each configuration
            3. **Select:** Keep top 10 performers (elite selection)
            4. **Crossover:** Combine genes from top performers
            5. **Mutate:** Random changes to explore new configurations
            6. **Repeat:** Run for 10-20 generations
            """)

        st.divider()

        # =================================================================
        # GENETIC ALGORITHM IMPLEMENTATION
        # =================================================================

        import random
        import time
        from copy import deepcopy

        class ZakatGeneticOptimizer:
            """Genetic Algorithm for optimizing Zakat clustering configurations."""

            def __init__(self, X, feature_names, maqasid_indices=None):
                self.X = X
                self.feature_names = feature_names
                self.n_features = len(feature_names)
                self.maqasid_indices = maqasid_indices or []
                self.history = []

            def _encode_chromosome(self, model_type, k, feature_mask):
                """Encode a clustering configuration as a chromosome."""
                return {
                    'model_type': model_type,  # 0 = K-Modes, 1 = LCA
                    'k': k,
                    'feature_mask': feature_mask  # List of 0/1
                }

            def _decode_chromosome(self, chrom):
                """Decode chromosome back to configuration."""
                return chrom['model_type'], chrom['k'], chrom['feature_mask']

            def _random_chromosome(self, k_min=2, k_max=10, min_features=5):
                """Generate a random chromosome."""
                model_type = random.randint(0, 1)
                k = random.randint(k_min, k_max)
                # Random feature mask with at least min_features
                feature_mask = [random.randint(0, 1) for _ in range(self.n_features)]
                while sum(feature_mask) < min_features:
                    idx = random.randint(0, self.n_features - 1)
                    feature_mask[idx] = 1
                return self._encode_chromosome(model_type, k, feature_mask)

            def _fitness(self, chrom, w1=0.35, w2=0.25, w3=0.20, w4=0.20, n_bootstrap=5):
                """Calculate Zakat Effectiveness Score (ZES)."""
                model_type, k, feature_mask = self._decode_chromosome(chrom)

                # Get selected features
                selected_idx = [i for i, m in enumerate(feature_mask) if m == 1]
                if len(selected_idx) < 3:
                    return 0.0, {}  # Invalid configuration

                X_subset = self.X[:, selected_idx]

                try:
                    # Fit model
                    if model_type == 0:  # K-Modes
                        model = KModesModel()
                        model.set_params(n_clusters=k, n_init=3)
                        labels = model.fit_predict(X_subset)
                    else:  # LCA
                        model = LatentClassModel()
                        model.set_params(n_classes=k, max_iter=50, n_init=3)
                        labels = model.fit_predict(X_subset)

                    # Component 1: Silhouette with Jaccard
                    try:
                        dist_matrix = compute_distance_matrix(X_subset, 'jaccard')
                        silhouette = silhouette_score(dist_matrix, labels, metric='precomputed')
                        silhouette = max(0, (silhouette + 1) / 2)  # Normalize to 0-1
                    except:
                        silhouette = 0.0

                    # Component 2: Stability (quick bootstrap)
                    ari_values = []
                    for _ in range(n_bootstrap):
                        boot_idx = np.random.choice(len(X_subset), size=int(0.8 * len(X_subset)), replace=True)
                        X_boot = X_subset[boot_idx]
                        try:
                            if model_type == 0:
                                boot_model = KModesModel()
                                boot_model.set_params(n_clusters=k, n_init=2)
                                boot_labels = boot_model.fit_predict(X_boot)
                            else:
                                boot_model = LatentClassModel()
                                boot_model.set_params(n_classes=k, max_iter=30, n_init=2)
                                boot_labels = boot_model.fit_predict(X_boot)
                            # Compare with original on same samples
                            orig_labels = labels[boot_idx]
                            ari = adjusted_rand_score(orig_labels, boot_labels)
                            ari_values.append(max(0, ari))
                        except:
                            pass
                    stability = np.mean(ari_values) if ari_values else 0.0

                    # Component 3: Balance (penalize imbalanced clusters)
                    cluster_sizes = np.bincount(labels)
                    min_ratio = cluster_sizes.min() / cluster_sizes.max()
                    balance = min_ratio  # Higher is better (0-1 range)

                    # Component 4: Maqasid Separation
                    maqasid_sep = 0.0
                    if self.maqasid_indices:
                        maq_selected = [i for i in range(len(selected_idx))
                                       if selected_idx[i] in self.maqasid_indices]
                        if len(maq_selected) >= 2:
                            # Measure variance in Maqasid deprivation across clusters
                            maq_rates = []
                            for cl in np.unique(labels):
                                mask = labels == cl
                                if mask.sum() > 0 and len(maq_selected) > 0:
                                    cl_maq_rate = X_subset[mask][:, maq_selected].mean()
                                    maq_rates.append(cl_maq_rate)
                            if len(maq_rates) > 1:
                                maqasid_sep = np.std(maq_rates) * 2  # Normalize
                                maqasid_sep = min(1.0, maqasid_sep)

                    # Calculate ZES
                    zes = w1 * silhouette + w2 * stability + w3 * balance + w4 * maqasid_sep

                    metrics = {
                        'silhouette': silhouette,
                        'stability': stability,
                        'balance': balance,
                        'maqasid_sep': maqasid_sep,
                        'zes': zes,
                        'n_features': len(selected_idx),
                        'model': 'K-Modes' if model_type == 0 else 'LCA',
                        'k': k,
                        'labels': labels
                    }

                    return zes, metrics

                except Exception as e:
                    return 0.0, {'error': str(e)}

            def _crossover(self, parent1, parent2):
                """Single-point crossover for chromosomes."""
                child = {}

                # Model type from random parent
                child['model_type'] = random.choice([parent1['model_type'], parent2['model_type']])

                # K: average or random parent
                if random.random() < 0.5:
                    child['k'] = (parent1['k'] + parent2['k']) // 2
                else:
                    child['k'] = random.choice([parent1['k'], parent2['k']])

                # Feature mask: crossover point
                crossover_point = random.randint(1, self.n_features - 1)
                child['feature_mask'] = (parent1['feature_mask'][:crossover_point] +
                                        parent2['feature_mask'][crossover_point:])

                return child

            def _mutate(self, chrom, mutation_rate=0.1):
                """Apply random mutations to chromosome."""
                mutated = deepcopy(chrom)

                # Mutate model type
                if random.random() < mutation_rate:
                    mutated['model_type'] = 1 - mutated['model_type']

                # Mutate K
                if random.random() < mutation_rate:
                    mutated['k'] = max(2, min(15, mutated['k'] + random.randint(-2, 2)))

                # Mutate feature mask
                for i in range(len(mutated['feature_mask'])):
                    if random.random() < mutation_rate:
                        mutated['feature_mask'][i] = 1 - mutated['feature_mask'][i]

                # Ensure minimum features
                while sum(mutated['feature_mask']) < 3:
                    idx = random.randint(0, len(mutated['feature_mask']) - 1)
                    mutated['feature_mask'][idx] = 1

                return mutated

            def evolve(self, population_size=30, n_generations=10, elite_size=5,
                      mutation_rate=0.15, k_min=2, k_max=10, progress_callback=None):
                """Run the genetic algorithm."""

                # Initialize population
                population = [self._random_chromosome(k_min, k_max)
                             for _ in range(population_size)]

                best_overall = None
                best_fitness = 0.0
                self.history = []

                for gen in range(n_generations):
                    # Evaluate fitness
                    fitness_scores = []
                    for chrom in population:
                        score, metrics = self._fitness(chrom)
                        fitness_scores.append((score, chrom, metrics))

                    # Sort by fitness
                    fitness_scores.sort(key=lambda x: x[0], reverse=True)

                    # Record best of generation
                    gen_best = fitness_scores[0]
                    self.history.append({
                        'generation': gen + 1,
                        'best_fitness': gen_best[0],
                        'avg_fitness': np.mean([f[0] for f in fitness_scores]),
                        'best_config': gen_best[2]
                    })

                    # Update overall best
                    if gen_best[0] > best_fitness:
                        best_fitness = gen_best[0]
                        best_overall = gen_best

                    # Progress callback
                    if progress_callback:
                        progress_callback(gen + 1, n_generations, gen_best[0], gen_best[2])

                    # Selection: keep elite
                    elite = [f[1] for f in fitness_scores[:elite_size]]

                    # Create new population
                    new_population = list(elite)  # Keep elite

                    # Fill rest with offspring
                    while len(new_population) < population_size:
                        parent1 = random.choice(elite)
                        parent2 = random.choice(elite)
                        child = self._crossover(parent1, parent2)
                        child = self._mutate(child, mutation_rate)
                        new_population.append(child)

                    population = new_population

                return best_overall, self.history

        # =================================================================
        # STREAMLIT UI - CONTROL ROOM
        # =================================================================

        st.markdown("### Control Room")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Population Settings")
            pop_size = st.slider("Population Size", 10, 100, 30,
                                help="Number of candidate solutions per generation")
            n_generations = st.slider("Generations", 5, 30, 10,
                                     help="Number of evolution cycles")
            elite_size = st.slider("Elite Size", 3, 15, 5,
                                  help="Top performers kept each generation")

        with col2:
            st.markdown("#### Search Space")
            k_min, k_max = st.slider("K Range", 2, 15, (2, 8),
                                    help="Range of cluster numbers to explore")
            mutation_rate = st.slider("Mutation Rate", 0.05, 0.30, 0.15,
                                     help="Probability of random gene changes")

        with col3:
            st.markdown("#### Fitness Weights")
            w_silhouette = st.slider("Silhouette Weight", 0.0, 1.0, 0.35)
            w_stability = st.slider("Stability Weight", 0.0, 1.0, 0.25)
            w_balance = st.slider("Balance Weight", 0.0, 1.0, 0.20)
            w_maqasid = st.slider("Maqasid Weight", 0.0, 1.0, 0.20)

        # Normalize weights
        total_w = w_silhouette + w_stability + w_balance + w_maqasid
        if total_w > 0:
            w_silhouette /= total_w
            w_stability /= total_w
            w_balance /= total_w
            w_maqasid /= total_w

        st.divider()

        # =================================================================
        # RUN OPTIMIZATION
        # =================================================================

        run_automl = st.button("Start Evolution", type="primary", use_container_width=True)

        if run_automl:
            st.markdown("### Live Evolution Dashboard")

            # Find Maqasid indices
            maqasid_cols = FEATURE_GROUPS.get('Maqasid Syariah', [])
            maqasid_indices = [i for i, col in enumerate(binary_cols) if col in maqasid_cols]

            # =================================================================
            # DETAILED PROGRESS UI LAYOUT
            # =================================================================

            # Top status bar
            status_header = st.empty()

            # Main progress
            col_prog1, col_prog2 = st.columns([3, 1])
            with col_prog1:
                progress_bar = st.progress(0)
            with col_prog2:
                time_display = st.empty()

            st.divider()

            # Two-column layout for detailed progress
            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.markdown("#### Generation Details")
                gen_status = st.empty()
                st.markdown("#### Current Best Configuration")
                best_config_display = st.empty()
                st.markdown("#### Fitness Components")
                fitness_components = st.empty()

            with col_right:
                st.markdown("#### Population Evaluation Log")
                eval_log_container = st.container(height=400)
                eval_log = eval_log_container.empty()

            st.divider()

            # Bottom section - charts and top performers
            chart_col, top_col = st.columns([2, 1])

            with chart_col:
                st.markdown("#### Evolution Progress")
                chart_placeholder = st.empty()

            with top_col:
                st.markdown("#### Top 5 This Generation")
                top_performers = st.empty()

            st.divider()

            # Detailed operations log
            st.markdown("#### Genetic Operations Log")
            ops_log_container = st.container(height=200)
            ops_log = ops_log_container.empty()

            # =================================================================
            # ENHANCED OPTIMIZER WITH DETAILED CALLBACKS
            # =================================================================

            class DetailedZakatOptimizer(ZakatGeneticOptimizer):
                """Enhanced optimizer with detailed progress reporting."""

                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.eval_logs = []
                    self.ops_logs = []
                    self.generation_stats = []
                    self.start_time = None

                def evolve_detailed(self, population_size=30, n_generations=10, elite_size=5,
                                   mutation_rate=0.15, k_min=2, k_max=10,
                                   ui_callbacks=None):
                    """Run GA with detailed UI updates."""

                    self.start_time = time.time()
                    self.eval_logs = []
                    self.ops_logs = []

                    # Initialize population
                    self._log_op("INIT", f"Creating initial population of {population_size} chromosomes...")
                    population = []
                    for i in range(population_size):
                        chrom = self._random_chromosome(k_min, k_max)
                        population.append(chrom)
                        model_type = "K-Modes" if chrom['model_type'] == 0 else "LCA"
                        n_feat = sum(chrom['feature_mask'])
                        self._log_op("INIT", f"  Chromosome #{i+1}: {model_type}, K={chrom['k']}, Features={n_feat}")

                    if ui_callbacks:
                        ui_callbacks['update_ops_log'](self.ops_logs)

                    best_overall = None
                    best_fitness = 0.0
                    self.history = []

                    for gen in range(n_generations):
                        gen_start = time.time()
                        self.eval_logs = []  # Reset per generation

                        self._log_op("GEN", f"=== GENERATION {gen+1}/{n_generations} ===")

                        # Update generation status
                        if ui_callbacks:
                            elapsed = time.time() - self.start_time
                            ui_callbacks['update_status'](gen + 1, n_generations, elapsed)

                        # Evaluate fitness for each chromosome
                        self._log_op("EVAL", f"Evaluating {len(population)} chromosomes...")
                        fitness_scores = []

                        for idx, chrom in enumerate(population):
                            eval_start = time.time()
                            score, metrics = self._fitness(chrom,
                                                          w1=ui_callbacks.get('weights', {}).get('w1', 0.35),
                                                          w2=ui_callbacks.get('weights', {}).get('w2', 0.25),
                                                          w3=ui_callbacks.get('weights', {}).get('w3', 0.20),
                                                          w4=ui_callbacks.get('weights', {}).get('w4', 0.20))
                            eval_time = time.time() - eval_start

                            model_type = "K-Modes" if chrom['model_type'] == 0 else "LCA"
                            n_feat = sum(chrom['feature_mask'])

                            # Detailed eval log entry
                            log_entry = {
                                'idx': idx + 1,
                                'model': model_type,
                                'k': chrom['k'],
                                'features': n_feat,
                                'zes': score,
                                'sil': metrics.get('silhouette', 0),
                                'stab': metrics.get('stability', 0),
                                'bal': metrics.get('balance', 0),
                                'maq': metrics.get('maqasid_sep', 0),
                                'time': eval_time
                            }
                            self.eval_logs.append(log_entry)
                            fitness_scores.append((score, chrom, metrics))

                            # Update UI every 3 evaluations or on last one
                            if ui_callbacks and (idx % 3 == 0 or idx == len(population) - 1):
                                ui_callbacks['update_eval_log'](self.eval_logs, idx + 1, len(population))
                                # Update progress within generation
                                gen_progress = (gen + (idx + 1) / len(population)) / n_generations
                                ui_callbacks['update_progress'](gen_progress)

                        # Sort by fitness
                        fitness_scores.sort(key=lambda x: x[0], reverse=True)

                        # Generation statistics
                        gen_fitness = [f[0] for f in fitness_scores]
                        gen_stats = {
                            'gen': gen + 1,
                            'best': max(gen_fitness),
                            'worst': min(gen_fitness),
                            'avg': np.mean(gen_fitness),
                            'std': np.std(gen_fitness),
                            'time': time.time() - gen_start
                        }
                        self.generation_stats.append(gen_stats)

                        self._log_op("STATS", f"Gen {gen+1} Stats: Best={gen_stats['best']:.4f}, Avg={gen_stats['avg']:.4f}, Worst={gen_stats['worst']:.4f}")

                        # Record best of generation
                        gen_best = fitness_scores[0]
                        self.history.append({
                            'generation': gen + 1,
                            'best_fitness': gen_best[0],
                            'avg_fitness': gen_stats['avg'],
                            'std_fitness': gen_stats['std'],
                            'best_config': gen_best[2]
                        })

                        # Update overall best
                        if gen_best[0] > best_fitness:
                            best_fitness = gen_best[0]
                            best_overall = gen_best
                            self._log_op("BEST", f"NEW BEST FOUND! ZES={best_fitness:.4f}")

                        # Update UI with generation results
                        if ui_callbacks:
                            ui_callbacks['update_gen_complete'](
                                gen + 1, gen_stats, gen_best,
                                fitness_scores[:5], self.history
                            )

                        # SELECTION
                        self._log_op("SELECT", f"Selecting top {elite_size} elite chromosomes...")
                        elite = []
                        for i, (score, chrom, metrics) in enumerate(fitness_scores[:elite_size]):
                            elite.append(chrom)
                            model_type = "K-Modes" if chrom['model_type'] == 0 else "LCA"
                            self._log_op("SELECT", f"  Elite #{i+1}: {model_type}, K={chrom['k']}, ZES={score:.4f}")

                        # Create new population
                        new_population = list(elite)  # Keep elite

                        # CROSSOVER & MUTATION
                        self._log_op("BREED", f"Creating {population_size - elite_size} offspring...")
                        offspring_count = 0

                        while len(new_population) < population_size:
                            # Select parents
                            p1_idx = random.randint(0, elite_size - 1)
                            p2_idx = random.randint(0, elite_size - 1)
                            parent1 = elite[p1_idx]
                            parent2 = elite[p2_idx]

                            # Crossover
                            child = self._crossover(parent1, parent2)
                            p1_model = "KM" if parent1['model_type'] == 0 else "LCA"
                            p2_model = "KM" if parent2['model_type'] == 0 else "LCA"
                            c_model = "KM" if child['model_type'] == 0 else "LCA"

                            crossover_log = f"  P1({p1_model},K={parent1['k']}) x P2({p2_model},K={parent2['k']}) -> Child({c_model},K={child['k']})"

                            # Mutation
                            old_model = child['model_type']
                            old_k = child['k']
                            old_feat = sum(child['feature_mask'])

                            child = self._mutate(child, mutation_rate)

                            new_model = child['model_type']
                            new_k = child['k']
                            new_feat = sum(child['feature_mask'])

                            mutations = []
                            if old_model != new_model:
                                mutations.append(f"Model: {'KM' if old_model == 0 else 'LCA'}->{'KM' if new_model == 0 else 'LCA'}")
                            if old_k != new_k:
                                mutations.append(f"K: {old_k}->{new_k}")
                            if old_feat != new_feat:
                                mutations.append(f"Feat: {old_feat}->{new_feat}")

                            if mutations:
                                self._log_op("MUTATE", crossover_log + f" | Mutations: {', '.join(mutations)}")
                            else:
                                self._log_op("CROSS", crossover_log)

                            new_population.append(child)
                            offspring_count += 1

                        population = new_population

                        # Update ops log
                        if ui_callbacks:
                            ui_callbacks['update_ops_log'](self.ops_logs[-20:])  # Keep last 20

                    return best_overall, self.history

                def _log_op(self, op_type, message):
                    """Log a genetic operation."""
                    timestamp = time.time() - self.start_time if self.start_time else 0
                    self.ops_logs.append({
                        'time': timestamp,
                        'type': op_type,
                        'message': message
                    })

            # =================================================================
            # UI UPDATE FUNCTIONS
            # =================================================================

            gen_history = []
            fitness_history = []
            avg_fitness_history = []

            def update_status(gen, total, elapsed):
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
                status_header.markdown(f"""
                ### Generation {gen} / {total}
                """)
                time_display.markdown(f"**Elapsed:** {mins}m {secs}s")

            def update_progress(progress):
                progress_bar.progress(min(progress, 1.0))

            def update_eval_log(logs, current, total):
                # Create a formatted table of evaluations
                if logs:
                    log_text = f"**Evaluating: {current}/{total}**\n\n"
                    log_text += "| # | Model | K | Feat | ZES | Sil | Stab | Bal | Maq | Time |\n"
                    log_text += "|---|-------|---|------|-----|-----|------|-----|-----|------|\n"

                    # Show last 12 entries
                    for entry in logs[-12:]:
                        log_text += f"| {entry['idx']} | {entry['model']} | {entry['k']} | {entry['features']} | "
                        log_text += f"**{entry['zes']:.3f}** | {entry['sil']:.2f} | {entry['stab']:.2f} | "
                        log_text += f"{entry['bal']:.2f} | {entry['maq']:.2f} | {entry['time']:.1f}s |\n"

                    eval_log.markdown(log_text)

            def update_gen_complete(gen, stats, best, top5, history):
                # Update generation status
                gen_status.markdown(f"""
                | Metric | Value |
                |--------|-------|
                | **Generation** | {gen} |
                | **Best ZES** | {stats['best']:.4f} |
                | **Average ZES** | {stats['avg']:.4f} |
                | **Std Dev** | {stats['std']:.4f} |
                | **Worst ZES** | {stats['worst']:.4f} |
                | **Gen Time** | {stats['time']:.1f}s |
                """)

                # Update best config display
                best_metrics = best[2]
                best_chrom = best[1]
                selected_feat_names = [get_short_name(binary_cols[i])
                                      for i, m in enumerate(best_chrom['feature_mask']) if m == 1]

                best_config_display.markdown(f"""
                **Model:** {best_metrics.get('model', 'N/A')}
                **Clusters (K):** {best_metrics.get('k', 'N/A')}
                **Features:** {best_metrics.get('n_features', 'N/A')}/{len(binary_cols)}

                **Selected Features:**
                {', '.join(selected_feat_names[:8])}{'...' if len(selected_feat_names) > 8 else ''}
                """)

                # Update fitness components
                fitness_components.markdown(f"""
                | Component | Score | Weight | Contribution |
                |-----------|-------|--------|--------------|
                | Silhouette | {best_metrics.get('silhouette', 0):.3f} | {w_silhouette:.2f} | {best_metrics.get('silhouette', 0) * w_silhouette:.3f} |
                | Stability | {best_metrics.get('stability', 0):.3f} | {w_stability:.2f} | {best_metrics.get('stability', 0) * w_stability:.3f} |
                | Balance | {best_metrics.get('balance', 0):.3f} | {w_balance:.2f} | {best_metrics.get('balance', 0) * w_balance:.3f} |
                | Maqasid Sep. | {best_metrics.get('maqasid_sep', 0):.3f} | {w_maqasid:.2f} | {best_metrics.get('maqasid_sep', 0) * w_maqasid:.3f} |
                | **TOTAL ZES** | | | **{best_metrics.get('zes', 0):.4f}** |
                """)

                # Update top 5 performers
                top5_text = ""
                for i, (score, chrom, metrics) in enumerate(top5):
                    model = "K-Modes" if chrom['model_type'] == 0 else "LCA"
                    medal = ["1st", "2nd", "3rd", "4th", "5th"][i]
                    top5_text += f"**{medal}:** {model}, K={chrom['k']}, F={sum(chrom['feature_mask'])}, ZES={score:.4f}\n\n"
                top_performers.markdown(top5_text)

                # Update chart
                gen_history.append(gen)
                fitness_history.append(stats['best'])
                avg_fitness_history.append(stats['avg'])

                if len(gen_history) > 1:
                    chart_df = pd.DataFrame({
                        'Generation': gen_history + gen_history,
                        'ZES': fitness_history + avg_fitness_history,
                        'Type': ['Best'] * len(gen_history) + ['Average'] * len(gen_history)
                    })
                    fig = px.line(chart_df, x='Generation', y='ZES', color='Type',
                                 title='Fitness Evolution',
                                 markers=True)
                    fig.update_layout(height=300)
                    chart_placeholder.plotly_chart(fig, use_container_width=True)

            def update_ops_log(logs):
                if logs:
                    log_text = ""
                    for entry in logs[-15:]:
                        time_str = f"{entry['time']:.1f}s"
                        type_icon = {
                            'INIT': '[INIT]',
                            'GEN': '[GEN]',
                            'EVAL': '[EVAL]',
                            'STATS': '[STATS]',
                            'BEST': '[BEST]',
                            'SELECT': '[SELECT]',
                            'BREED': '[BREED]',
                            'CROSS': '[CROSS]',
                            'MUTATE': '[MUTATE]'
                        }.get(entry['type'], '[???]')
                        log_text += f"`{time_str}` **{type_icon}** {entry['message']}\n\n"
                    ops_log.markdown(log_text)

            # =================================================================
            # RUN THE OPTIMIZER
            # =================================================================

            optimizer = DetailedZakatOptimizer(
                X_array,
                binary_cols,
                maqasid_indices=maqasid_indices
            )

            ui_callbacks = {
                'update_status': update_status,
                'update_progress': update_progress,
                'update_eval_log': update_eval_log,
                'update_gen_complete': update_gen_complete,
                'update_ops_log': update_ops_log,
                'weights': {
                    'w1': w_silhouette,
                    'w2': w_stability,
                    'w3': w_balance,
                    'w4': w_maqasid
                }
            }

            best_result, history = optimizer.evolve_detailed(
                population_size=pop_size,
                n_generations=n_generations,
                elite_size=elite_size,
                mutation_rate=mutation_rate,
                k_min=k_min,
                k_max=k_max,
                ui_callbacks=ui_callbacks
            )

            progress_bar.progress(1.0)
            status_header.success("### Evolution Complete!")

            # =================================================================
            # RESULTS PANEL
            # =================================================================

            st.divider()
            st.markdown("### Optimal Configuration Found")

            if best_result and best_result[2]:
                best_metrics = best_result[2]
                best_chrom = best_result[1]

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.markdown("#### Best Configuration")
                    st.metric("Zakat Effectiveness Score", f"{best_metrics.get('zes', 0):.4f}")
                    st.markdown(f"""
                    - **Model:** {best_metrics.get('model', 'N/A')}
                    - **K (Clusters):** {best_metrics.get('k', 'N/A')}
                    - **Features Used:** {best_metrics.get('n_features', 'N/A')} / {len(binary_cols)}
                    """)

                    # Show selected features
                    st.markdown("#### Selected Features")
                    selected_features = [binary_cols[i] for i, m in enumerate(best_chrom['feature_mask']) if m == 1]
                    for feat in selected_features:
                        st.write(f"- {get_short_name(feat)}")

                with col2:
                    st.markdown("#### Component Scores")

                    scores_df = pd.DataFrame({
                        'Component': ['Silhouette (Jaccard)', 'Stability (ARI)', 'Balance', 'Maqasid Separation'],
                        'Score': [
                            best_metrics.get('silhouette', 0),
                            best_metrics.get('stability', 0),
                            best_metrics.get('balance', 0),
                            best_metrics.get('maqasid_sep', 0)
                        ],
                        'Weight': [w_silhouette, w_stability, w_balance, w_maqasid],
                        'Contribution': [
                            best_metrics.get('silhouette', 0) * w_silhouette,
                            best_metrics.get('stability', 0) * w_stability,
                            best_metrics.get('balance', 0) * w_balance,
                            best_metrics.get('maqasid_sep', 0) * w_maqasid
                        ]
                    })

                    fig = px.bar(scores_df, x='Component', y='Score',
                                title='Fitness Component Breakdown',
                                color='Score', color_continuous_scale='Greens')
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)

                # Evolution history
                st.markdown("#### Evolution History")

                history_df = pd.DataFrame(history)
                fig = px.line(history_df, x='generation', y='best_fitness',
                             title='Best Fitness by Generation',
                             markers=True,
                             labels={'generation': 'Generation', 'best_fitness': 'Best ZES'})
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                # Store best labels for use in validation
                if 'labels' in best_metrics:
                    st.session_state.automl_labels = best_metrics['labels']
                    st.session_state.automl_config = {
                        'model': best_metrics.get('model'),
                        'k': best_metrics.get('k'),
                        'features': selected_features,
                        'zes': best_metrics.get('zes')
                    }

                    # Cluster distribution
                    st.markdown("#### Cluster Distribution")
                    labels = best_metrics['labels']
                    sizes = pd.Series(labels).value_counts().sort_index()
                    sizes_df = pd.DataFrame({
                        'Cluster': [f'Cluster {k}' for k in sizes.index],
                        'Count': sizes.values,
                        'Proportion': (sizes.values / len(labels) * 100).round(1)
                    })
                    st.dataframe(sizes_df, use_container_width=True, hide_index=True)

                    # Quick profile heatmap
                    st.markdown("#### Quick Cluster Profile")

                    # Get data for selected features only
                    selected_idx = [i for i, m in enumerate(best_chrom['feature_mask']) if m == 1]
                    X_selected = X_array[:, selected_idx]
                    selected_short = [get_short_name(binary_cols[i]) for i in selected_idx]

                    # Calculate item response probabilities
                    n_clusters = len(np.unique(labels))
                    irp_data = []
                    for cl in range(n_clusters):
                        mask = labels == cl
                        irp_data.append(X_selected[mask].mean(axis=0))

                    irp_df = pd.DataFrame(
                        irp_data,
                        index=[f'Cluster {i}' for i in range(n_clusters)],
                        columns=selected_short
                    )

                    fig = px.imshow(
                        irp_df,
                        text_auto='.2f',
                        color_continuous_scale='YlOrRd',
                        aspect='auto',
                        title='Item Response Probabilities (Optimized Configuration)'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    st.success("""
                    **Results saved!** You can now proceed to Phase 4 (Validation) to validate
                    this automatically optimized solution using Bootstrap Stability and Maqasid Profiling.
                    """)
            else:
                st.error("Evolution did not find a valid solution. Try adjusting parameters.")

# =============================================================================
# PHASE 4: VALIDATION
# =============================================================================
elif phase == "Phase 4: Validation":
    st.title("Phase 4: The 'Adl Check - Maqasid Syariah Validation")
    st.caption("Ensuring justice in Zakat distribution through rigorous validation")

    # Get binary data
    binary_cols = [col for col in ALL_INDICATORS if col in df.columns]
    X_binary = binarize_dataframe(df, binary_cols)
    X_array = X_binary.values.astype(float)

    st.info("""
    Our dataset uniquely includes **Maqasid Syariah** indicators:
    *Pemeliharaan Agama* (Religion), *Nyawa* (Life), *Akal* (Intellect),
    *Keturunan* (Lineage), *Harta* (Property).

    This phase validates the clusters and determines **tailored Zakat interventions**.
    """)

    with st.expander("Why Maqasid Profiling Matters", expanded=False):
        st.markdown("""
        **The Experiment: Supervised Profiling**

        1. **Cluster Interpretation:** We look at "Cluster 1."
        2. **Maqasid Check:** Does Cluster 1 suffer predominantly in *Nyawa* (Life/Health) and *Harta* (Wealth)?
           - **Zakat Action:** This group needs immediate cash (*Wang Zakat*) and medical aid.

        3. **Maqasid Check:** Does Cluster 2 suffer predominantly in *Akal* (Intellect/Education) and *Keturunan* (Lineage/Social)?
           - **Zakat Action:** Cash won't solve this long-term. This group needs *Bantuan Pendidikan* (Scholarships) or vocational training.

        ---

        **Data-Based Reasoning:**

        If we ignore this distinction and just give cash to everyone, we fail the goal of Zakat,
        which is to move them from *Mustahiq* (receiver) to *Muzakki* (payer).

        By clustering based on the *type* of deprivation, we **tailor the intervention**.
        """)

    # Select model for validation
    available_models = {}
    if 'kmodes_labels' in st.session_state:
        available_models['K-Modes'] = st.session_state.kmodes_labels
    if 'lca_labels' in st.session_state:
        available_models['LCA'] = st.session_state.lca_labels
    if 'hac_labels' in st.session_state:
        available_models['HAC + MCA'] = st.session_state.hac_labels
    if 'automl_labels' in st.session_state:
        config = st.session_state.get('automl_config', {})
        model_name = f"AutoML ({config.get('model', 'Optimized')}, K={config.get('k', '?')})"
        available_models[model_name] = st.session_state.automl_labels

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
        st.subheader("MPI Profile Heatmap - Faces of Poverty and Zakat Recommendations")

        st.info("""
        **Item Response Probabilities** show P(Deprived | Cluster).
        **Relative Risk Ratios** compare cluster rates to population average.

        Based on the Maqasid Syariah framework, we identify **tailored Zakat interventions** for each cluster.
        """)

        with st.expander("Expected Poverty Archetypes and Interventions", expanded=False):
            st.markdown("""
            | Archetype | Maqasid Deficit | Recommended Zakat Intervention |
            |-----------|-----------------|--------------------------------|
            | **Deeply Deprived** | All Maqasid | Immediate cash (*Wang Zakat*) + comprehensive support |
            | **Infrastructure Poor** | *Nyawa* (Life), *Harta* (Wealth) | Housing assistance, utilities support |
            | **Digital Excluded** | *Akal* (Intellect) | ICT training, device provision |
            | **Education Deficit** | *Akal* (Intellect), *Keturunan* (Lineage) | *Bantuan Pendidikan* (Scholarships), vocational training |
            | **Near Average** | Minimal | Preventive support, monitoring |
            """)

        st.markdown("""
        **Viewing the heatmaps below:**
        - **Red/Orange cells** = High deprivation probability (needs intervention)
        - **White/Light cells** = Low deprivation probability (protected)

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
    if 'automl_labels' in st.session_state:
        config = st.session_state.get('automl_config', {})
        model_name = f"AutoML ({config.get('model', 'Optimized')}, K={config.get('k', '?')})"
        available_models[model_name] = st.session_state.automl_labels

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
