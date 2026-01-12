"""
ZE-Workbench - Data Analysis & Machine Learning Platform
Main Streamlit Application

A modular data analysis and machine learning platform with
clustering, statistical analysis, and model comparison capabilities.

PhD-Level Methodology for MPI Binary Data:
- Tetrachoric Correlation for binary collinearity
- Multiple Correspondence Analysis (MCA) for dimensionality reduction
- Latent Class Analysis (LCA) for probabilistic clustering
- K-Modes with Jaccard distance for categorical clustering
- Item Response Probabilities and Relative Risk Ratios for profiling
- Bootstrap Stability Analysis for validation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Local imports
from utils import download_df, get_variable_metadata, preprocess_features
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
from models import (
    get_model, ModelRunner, create_experiment_configs, LatentClassModel,
    ENABLED_MODEL_NAMES
)
import pyreadstat
from pathlib import Path

# Data file path
DATA_FILE = Path(__file__).parent / "data" / "DATA MPI SPSS.sav"

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="ZE-Workbench | Data Analysis & ML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if 'data' not in st.session_state:
    # Load data directly from the file
    df, meta = pyreadstat.read_sav(str(DATA_FILE))
    st.session_state.data = df
    st.session_state.meta = meta
if 'meta' not in st.session_state:
    st.session_state.meta = None
if 'experiment_results' not in st.session_state:
    st.session_state.experiment_results = None
if 'experiment_results_df' not in st.session_state:
    st.session_state.experiment_results_df = None
if 'current_labels' not in st.session_state:
    st.session_state.current_labels = None

# =============================================================================
# SIDEBAR - NAVIGATION
# =============================================================================
st.sidebar.title("📊 ZE-Workbench")
st.sidebar.caption("Data Analysis & Machine Learning Platform")
st.sidebar.divider()

# Show loaded data info
st.sidebar.caption(f"📁 Data: DATA MPI SPSS.sav")

# Navigation (data is always loaded)
# Following the PhD-level methodology from the data scientist's report:
# 1. Preprocessing & Feature Engineering
# 2. Core Clustering Methodologies
# 3. Cluster Profiling ("Faces of Poverty")
# 4. Validation Framework
if st.session_state.data is not None:
    st.sidebar.divider()

    # Main section selection - follows report methodology
    section = st.sidebar.radio(
        "Section",
        ["📋 Data", "🔬 Preprocessing", "📊 Clustering", "📈 Profiling", "✓ Validation"],
        label_visibility="collapsed"
    )

    # Sub-navigation based on section
    if section == "📋 Data":
        page = st.sidebar.radio(
            "Data Pages",
            ["Data View", "Variable View"],
            label_visibility="collapsed"
        )
    elif section == "🔬 Preprocessing":
        # Step 1: Mathematical Preprocessing & Feature Engineering
        page = st.sidebar.radio(
            "Preprocessing Pages",
            ["1. Tetrachoric Correlation", "2. MCA Analysis", "3. Feature Selection"],
            label_visibility="collapsed"
        )
    elif section == "📊 Clustering":
        # Step 2: Core Clustering Methodologies
        page = st.sidebar.radio(
            "Clustering Pages",
            ["A. LCA (Gold Standard)", "B. Distance-Based (K-Modes)", "Model Comparison"],
            label_visibility="collapsed"
        )
    elif section == "📈 Profiling":
        # Step 3: Cluster Profiling & "Faces of Poverty"
        page = st.sidebar.radio(
            "Profiling Pages",
            ["Item Response Probabilities", "Relative Risk Ratios", "Poverty Archetypes"],
            label_visibility="collapsed"
        )
    else:  # Validation
        # Step 4: Validation Framework
        page = st.sidebar.radio(
            "Validation Pages",
            ["Bootstrap Stability (ARI)", "External Validation", "Summary Report"],
            label_visibility="collapsed"
        )
else:
    section = None
    page = None

# =============================================================================
# MAIN CONTENT
# =============================================================================

# Data should always be loaded now, but keep a fallback just in case
if st.session_state.data is None:
    st.error("Failed to load data file. Please check that 'data/DATA MPI SPSS.sav' exists.")
    st.stop()

# Get working copy of data
df = st.session_state.data.copy()

# =============================================================================
# DATA SECTION
# =============================================================================
if section == "📋 Data":

    if page == "Data View":
        st.title("Data Editor")
        st.caption(f"{len(df)} rows × {len(df.columns)} columns")

        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            height=500
        )

        if not df.equals(edited_df):
            st.session_state.data = edited_df
            st.success("Data updated!")

        col1, col2 = st.columns([1, 4])
        with col1:
            st.download_button(
                "📥 Download CSV",
                download_df(edited_df),
                "data.csv",
                "text/csv"
            )

    elif page == "Variable View":
        st.title("Variable View")
        var_view = get_variable_metadata(df, st.session_state.meta)
        st.dataframe(var_view, use_container_width=True, hide_index=True, height=500)

# =============================================================================
# PREPROCESSING SECTION (Step 1: Mathematical Preprocessing & Feature Engineering)
# =============================================================================
elif section == "🔬 Preprocessing":
    # Identify binary columns for all preprocessing pages
    binary_cols = get_binary_columns(df)

    if page == "1. Tetrachoric Correlation":
        st.title("Step 1: Tetrachoric Correlation Matrix")
        st.caption("Collinearity inspection for binary deprivation indicators")

        st.info("""
        **Why Tetrachoric Correlation?**
        Standard Pearson correlation is inappropriate for binary data. Tetrachoric correlation
        estimates the Pearson correlation between two latent continuous variables that underlie
        the observed binary data. Use this to identify redundant indicators (ρ > 0.9).
        """)

        selected_binary = st.multiselect(
            "Select Binary Variables",
            binary_cols,
            default=binary_cols[:min(15, len(binary_cols))],
            help="Variables with only 0/1 or 2 unique values"
        )

        if len(selected_binary) >= 2:
            with st.spinner("Calculating tetrachoric correlations..."):
                binary_df = binarize_dataframe(df, selected_binary)
                tetra_corr = tetrachoric_correlation_matrix(binary_df, selected_binary)

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

                # Redundant features
                st.subheader("Redundant Features (|r| > 0.9)")
                redundant = identify_redundant_features(tetra_corr, threshold=0.9)
                if redundant:
                    redundant_df = pd.DataFrame(redundant, columns=['Variable 1', 'Variable 2', 'Correlation'])
                    st.dataframe(redundant_df, use_container_width=True, hide_index=True)
                    st.warning("Consider removing one variable from each redundant pair for clustering.")
                else:
                    st.success("No highly redundant feature pairs found.")
        else:
            st.info("Select at least 2 binary variables.")

    elif page == "2. MCA Analysis":
        st.title("Step 2: Multiple Correspondence Analysis (MCA)")
        st.caption("Dimensionality reduction for categorical data (analogue of PCA)")

        st.info("""
        **Why MCA instead of PCA?**
        PCA assumes continuous data. MCA projects binary/categorical data into a continuous
        low-dimensional Euclidean space. This helps visualize the "poverty landscape" and
        identify if Maqasid variables load onto the same dimensions as material deprivation.
        """)

        selected_mca = st.multiselect(
            "Select Variables for MCA",
            binary_cols,
            default=binary_cols[:min(20, len(binary_cols))],
            key="mca_vars"
        )

        n_components = st.slider("Number of Components", 2, min(10, len(selected_mca)-1) if len(selected_mca) > 2 else 2, 3)

        if len(selected_mca) >= 3:
            if st.button("Run MCA", type="primary"):
                with st.spinner("Performing MCA..."):
                    binary_df = binarize_dataframe(df, selected_mca)
                    mca_result = perform_mca(binary_df, selected_mca, n_components=n_components)

                    # Store in session state
                    st.session_state.mca_result = mca_result

                    # Show explained inertia
                    st.subheader("Explained Inertia")
                    inertia_df = pd.DataFrame({
                        'Component': [f'Dim{i+1}' for i in range(len(mca_result['explained_inertia']))],
                        'Explained Inertia': mca_result['explained_inertia'],
                        'Cumulative': np.cumsum(mca_result['explained_inertia'])
                    })
                    st.dataframe(inertia_df, use_container_width=True, hide_index=True)

                    # 2D scatter plot
                    coords = mca_result['coordinates']
                    fig = px.scatter(
                        coords,
                        x='Dim1', y='Dim2',
                        title=f"MCA - First 2 Dimensions (Inertia: {mca_result['explained_inertia'][:2].sum():.1%})",
                        labels={'Dim1': f'Dimension 1 ({mca_result["explained_inertia"][0]:.1%})',
                                'Dim2': f'Dimension 2 ({mca_result["explained_inertia"][1]:.1%})'}
                    )
                    fig.update_traces(marker=dict(size=6, opacity=0.6))
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

                    # Variable contributions
                    st.subheader("Variable Loadings")
                    col_coords = mca_result['column_coordinates']
                    fig = px.scatter(
                        col_coords,
                        x='Dim1', y='Dim2',
                        text=col_coords.index,
                        title="Variable Coordinates in MCA Space"
                    )
                    fig.update_traces(textposition="top center", marker=dict(size=10))
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least 3 variables for MCA.")

    elif page == "3. Feature Selection":
        st.title("Step 3: Feature Selection for Clustering")
        st.caption("Select orthogonal features based on tetrachoric correlation analysis")

        st.info("""
        **Feature Selection Rationale:**
        Highly correlated indicators (ρ > 0.9) represent the same latent deprivation.
        Remove redundant features to ensure orthogonality and improve clustering quality.
        """)

        if len(binary_cols) >= 2:
            threshold = st.slider("Redundancy Threshold", 0.7, 0.99, 0.9, 0.05,
                                  help="Features with correlation above this are considered redundant")

            with st.spinner("Analyzing feature redundancy..."):
                binary_df = binarize_dataframe(df, binary_cols)
                tetra_corr = tetrachoric_correlation_matrix(binary_df, binary_cols)
                redundant = identify_redundant_features(tetra_corr, threshold=threshold)

                # Create list of features to potentially remove
                remove_candidates = set()
                for v1, v2, corr in redundant:
                    # Suggest removing the one with lower variance
                    var1 = binary_df[v1].var()
                    var2 = binary_df[v2].var()
                    remove_candidates.add(v2 if var1 >= var2 else v1)

                if remove_candidates:
                    st.warning(f"Suggested features to remove ({len(remove_candidates)}):")
                    st.write(list(remove_candidates))

                    recommended = [c for c in binary_cols if c not in remove_candidates]
                    st.success(f"Recommended features for clustering ({len(recommended)}):")
                    st.write(recommended)

                    # Store recommended features for clustering
                    st.session_state.recommended_features = recommended
                else:
                    st.success("All features appear to be non-redundant. Use all for clustering.")
                    st.session_state.recommended_features = binary_cols

# =============================================================================
# CLUSTERING SECTION (Step 2: Core Clustering Methodologies)
# =============================================================================
elif section == "📊 Clustering":

    # Feature selection (shared across clustering pages)
    st.sidebar.divider()
    st.sidebar.subheader("Feature Selection")

    # Filter out likely ID columns
    likely_ids = [c for c in df.columns if df[c].nunique() == len(df)]
    default_features = [c for c in df.columns if c not in likely_ids]

    features = st.sidebar.multiselect(
        "Features",
        df.columns.tolist(),
        default=default_features,
        help="Select features for clustering"
    )

    if not features:
        st.warning("Please select at least one feature for clustering.")
        st.stop()

    # Preprocess data
    try:
        prep = preprocess_features(df, features)
        X_scaled = prep['X_scaled']
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        st.stop()

    # -------------------------------------------------------------------------
    # A. LCA (GOLD STANDARD) PAGE
    # -------------------------------------------------------------------------
    if page == "A. LCA (Gold Standard)":
        st.title("Latent Class Analysis - Model Selection")
        st.caption("Determine optimal number of latent classes using BIC/AIC criteria")

        st.info("""
        **Latent Class Analysis (LCA)** is the gold standard for clustering binary/categorical survey data.
        It assumes observed responses are independent conditional on latent class membership.
        Use BIC to select the optimal number of classes (lower = better).
        """)

        col1, col2 = st.columns([1, 3])

        with col1:
            st.subheader("Configuration")

            k_min, k_max = st.slider(
                "Class Range",
                2, 10, (2, 6),
                help="Range of latent classes to test"
            )

            max_iter = st.slider("Max Iterations", 50, 500, 100)
            n_init = st.slider("Random Initializations", 5, 50, 10)

            run_lca = st.button("Run LCA Model Selection", type="primary", use_container_width=True)

        with col2:
            if run_lca:
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                k_range = range(k_min, k_max + 1)
                total_k = len(k_range)

                for i, k in enumerate(k_range):
                    status_text.text(f"Fitting LCA with {k} classes...")
                    progress_bar.progress((i + 1) / total_k)

                    try:
                        model = LatentClassModel()
                        model.set_params(n_classes=k, max_iter=max_iter, n_init=n_init)
                        labels = model.fit_predict(X_scaled)
                        fit_stats = model.get_model_fit_statistics()
                        metrics = model.get_metrics(X_scaled, labels)

                        results.append({
                            'Classes': k,
                            'BIC': fit_stats['bic'],
                            'AIC': fit_stats['aic'],
                            'Log-Likelihood': fit_stats['log_likelihood'],
                            'Entropy': fit_stats['entropy'],
                            'Silhouette': metrics['silhouette'],
                            'model': model,
                            'labels': labels
                        })
                    except Exception as e:
                        st.warning(f"Failed for k={k}: {e}")

                progress_bar.empty()
                status_text.empty()

                if results:
                    # Store results
                    st.session_state.lca_results = results

                    # Create results DataFrame
                    results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['model', 'labels']} for r in results])

                    # Find optimal k
                    optimal_idx = results_df['BIC'].idxmin()
                    optimal_k = results_df.loc[optimal_idx, 'Classes']

                    st.success(f"Optimal number of classes: **{optimal_k}** (minimum BIC)")

                    # Metrics display
                    st.subheader("Model Comparison")
                    st.dataframe(
                        results_df.style.highlight_min(subset=['BIC', 'AIC']).highlight_max(subset=['Entropy', 'Silhouette']),
                        use_container_width=True,
                        hide_index=True
                    )

                    # BIC/AIC plot
                    st.subheader("Information Criteria")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=results_df['Classes'], y=results_df['BIC'], name='BIC', mode='lines+markers'))
                    fig.add_trace(go.Scatter(x=results_df['Classes'], y=results_df['AIC'], name='AIC', mode='lines+markers'))
                    fig.update_layout(
                        title="BIC and AIC by Number of Classes",
                        xaxis_title="Number of Classes",
                        yaxis_title="Information Criterion (lower is better)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Entropy plot
                    st.subheader("Classification Certainty")
                    fig = px.bar(results_df, x='Classes', y='Entropy', title="Entropy by Number of Classes (higher = more certain)")
                    st.plotly_chart(fig, use_container_width=True)

                    # Use optimal model
                    st.divider()
                    st.subheader(f"Selected Model: {optimal_k} Classes")

                    best_model = results[optimal_idx]['model']
                    best_labels = results[optimal_idx]['labels']
                    st.session_state.current_labels = best_labels
                    st.session_state.current_model = best_model

                    # Show class sizes
                    class_sizes = pd.Series(best_labels).value_counts().sort_index()
                    class_sizes_df = pd.DataFrame({
                        'Class': class_sizes.index,
                        'Count': class_sizes.values,
                        'Proportion': (class_sizes.values / len(best_labels) * 100).round(1)
                    })
                    st.dataframe(class_sizes_df, use_container_width=True, hide_index=True)

                    # Item Response Probabilities heatmap
                    st.subheader("Item Response Probabilities")
                    st.caption("P(Deprived | Class) - Probability of deprivation for each indicator by class")

                    item_probs = best_model.get_item_response_probabilities()
                    item_probs_df = pd.DataFrame(
                        item_probs,
                        index=[f'Class {i}' for i in range(optimal_k)],
                        columns=features
                    )

                    fig = px.imshow(
                        item_probs_df,
                        text_auto='.2f',
                        color_continuous_scale='YlOrRd',
                        aspect="auto",
                        title="Item Response Probabilities Heatmap"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

            elif 'lca_results' in st.session_state:
                st.info("Previous LCA results available. Run again to update.")

    # -------------------------------------------------------------------------
    # B. DISTANCE-BASED (K-MODES) PAGE
    # -------------------------------------------------------------------------
    elif page == "B. Distance-Based (K-Modes)":
        st.title("Distance-Based Clustering Methods")
        st.caption("K-Modes and Hierarchical clustering with Jaccard distance for binary data")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Configuration")

            # Model selection
            selected_models = st.multiselect(
                "Models to Run",
                list(ENABLED_MODEL_NAMES.keys()),
                default=list(ENABLED_MODEL_NAMES.keys()),
                format_func=lambda x: ENABLED_MODEL_NAMES[x]
            )

            # K range for models that support it
            st.divider()
            k_min, k_max = st.slider(
                "Cluster Range (k)",
                2, 15, (2, 8),
                help="Range of cluster values to try"
            )
            k_range = range(k_min, k_max + 1)

            # Note: DBSCAN/OPTICS parameters removed (models disabled for binary data)
            eps_values = []
            min_samples_values = []

            run_batch = st.button(
                "🚀 Run All Experiments",
                type="primary",
                use_container_width=True
            )

        with col2:
            if run_batch and selected_models:
                # Create configurations
                configs = create_experiment_configs(
                    selected_models,
                    k_range,
                    eps_values,
                    min_samples_values
                )

                st.info(f"Running {len(configs)} experiments...")

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Run experiments
                runner = ModelRunner(df)

                # Run batch and get results DataFrame
                total_configs = len(configs)
                results_df = pd.DataFrame()

                for i, config in enumerate(configs):
                    status_text.text(f"Running experiment {i + 1}/{total_configs}")
                    progress_bar.progress((i + 1) / total_configs)

                    # Run single config as a batch of 1
                    single_result = runner.run_batch(X_scaled, [config.copy()])
                    results_df = pd.concat([results_df, single_result], ignore_index=True)

                progress_bar.empty()
                status_text.empty()

                # Store results
                st.session_state.experiment_results = runner
                st.session_state.experiment_results_df = results_df

                st.success(f"Completed {len(results_df)} experiments!")

                # Show summary
                valid_count = results_df['Valid'].sum() if 'Valid' in results_df.columns else len(results_df)

                st.metric("Valid Results", f"{valid_count}/{len(results_df)}")

                # Best results preview
                st.subheader("Top 5 Results (by Silhouette)")
                top5 = results_df[results_df['Valid']].head(5)
                st.dataframe(top5, use_container_width=True, hide_index=True)

            elif st.session_state.experiment_results:
                st.info("Previous experiment results available. Go to 'Model Comparison' to view.")

    # -------------------------------------------------------------------------
    # MODEL COMPARISON PAGE
    # -------------------------------------------------------------------------
    elif page == "Model Comparison":
        st.title("Results Comparison")

        if 'experiment_results_df' not in st.session_state or st.session_state.experiment_results_df is None:
            st.warning("No experiment results yet. Run a batch experiment first.")
            st.stop()

        results_df = st.session_state.experiment_results_df

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_model = st.multiselect(
                "Filter by Model",
                results_df['Model'].unique().tolist(),
                default=results_df['Model'].unique().tolist()
            )
        with col2:
            filter_valid = st.checkbox("Valid Only", value=True)
        with col3:
            sort_by = st.selectbox(
                "Sort By",
                ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Clusters"]
            )

        # Apply filters
        filtered_df = results_df[results_df['Model'].isin(filter_model)]
        if filter_valid:
            filtered_df = filtered_df[filtered_df['Valid']]

        # Sort
        ascending = sort_by == "Davies-Bouldin"  # Lower is better for DB
        filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)

        # Summary metrics
        if len(filtered_df) > 0:
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Experiments", len(filtered_df))
            m2.metric("Best Silhouette", f"{filtered_df['Silhouette'].max():.4f}")
            m3.metric("Best Davies-Bouldin", f"{filtered_df['Davies-Bouldin'].min():.4f}")

        # Results table with highlighting
        st.subheader("All Results")

        styled_df = filtered_df.style.background_gradient(
            subset=['Silhouette'],
            cmap='Greens'
        ).background_gradient(
            subset=['Davies-Bouldin'],
            cmap='Reds_r'
        )

        st.dataframe(styled_df, use_container_width=True, height=400)

        st.download_button(
            "📥 Download All Results",
            filtered_df.to_csv(index=False).encode('utf-8'),
            "experiment_results.csv",
            "text/csv"
        )

        # Comparison charts
        st.divider()
        st.subheader("Visual Comparison")

        tab1, tab2, tab3 = st.tabs(["By Model", "By Clusters", "Metrics Scatter"])

        with tab1:
            if len(filtered_df) > 0:
                fig = px.box(
                    filtered_df,
                    x='Model',
                    y='Silhouette',
                    title="Silhouette Score Distribution by Model",
                    color='Model'
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if len(filtered_df) > 0:
                fig = px.scatter(
                    filtered_df,
                    x='Clusters',
                    y='Silhouette',
                    color='Model',
                    size='Calinski-Harabasz',
                    hover_data=['Parameters'],
                    title="Silhouette vs Number of Clusters"
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            if len(filtered_df) > 0:
                fig = px.scatter(
                    filtered_df,
                    x='Silhouette',
                    y='Davies-Bouldin',
                    color='Model',
                    hover_data=['Parameters', 'Clusters'],
                    title="Silhouette vs Davies-Bouldin (ideal: top-left)"
                )
                st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PROFILING SECTION (Step 3: Cluster Profiling - "Faces of Poverty")
# =============================================================================
elif section == "📈 Profiling":

    # Check if we have clustering results
    if st.session_state.current_labels is None:
        st.warning("No clustering results yet. Run a model in the Clustering section first.")
        st.stop()

    labels = st.session_state.current_labels

    # Get features for profiling (use recommended or all binary)
    binary_cols = get_binary_columns(df)
    profile_features = st.session_state.get('recommended_features', binary_cols)

    # Preprocess for profiling
    try:
        prep = preprocess_features(df, profile_features)
        X_scaled = prep['X_scaled']
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        st.stop()

    # -------------------------------------------------------------------------
    # ITEM RESPONSE PROBABILITIES PAGE
    # -------------------------------------------------------------------------
    if page == "Item Response Probabilities":
        st.title("Item Response Probabilities")
        st.caption("P(Deprived | Cluster) - Probability of deprivation for each indicator by cluster")

        st.info("""
        **Item Response Probabilities** show the probability that a household in each cluster
        is deprived in each indicator. Higher values (darker red) indicate higher deprivation rates.
        This is the foundation of cluster interpretation in LCA.
        """)

        with st.spinner("Calculating item response probabilities..."):
            irp = calculate_item_response_probabilities(X_scaled, labels, profile_features)

        # Heatmap
        fig = px.imshow(
            irp,
            text_auto='.2f',
            color_continuous_scale='YlOrRd',
            aspect="auto",
            title="Item Response Probabilities Heatmap"
        )
        fig.update_layout(height=max(400, len(profile_features) * 25))
        st.plotly_chart(fig, use_container_width=True)

        # Download
        st.download_button(
            "📥 Download Item Response Probabilities",
            irp.to_csv().encode('utf-8'),
            "item_response_probabilities.csv",
            "text/csv"
        )

        # Summary table
        st.subheader("Summary by Cluster")
        summary_df = pd.DataFrame({
            'Cluster': irp.index,
            'Mean Probability': irp.mean(axis=1).round(3),
            'Max Probability': irp.max(axis=1).round(3),
            'Min Probability': irp.min(axis=1).round(3)
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # RELATIVE RISK RATIOS PAGE
    # -------------------------------------------------------------------------
    elif page == "Relative Risk Ratios":
        st.title("Relative Risk Ratios")
        st.caption("RR = P(Deprived | Cluster) / P(Deprived | Population)")

        st.info("""
        **Relative Risk Ratios** compare cluster-specific deprivation rates to the population average:
        - **RR > 1**: Cluster has HIGHER deprivation than population average (risk factor)
        - **RR = 1**: Same as population average
        - **RR < 1**: Cluster has LOWER deprivation than population average (protective factor)

        Values > 1.5 or < 0.67 are typically considered meaningful.
        """)

        with st.spinner("Calculating relative risk ratios..."):
            rr = calculate_relative_risk_ratios(X_scaled, labels, profile_features)

        # Diverging heatmap centered at 1
        fig = px.imshow(
            rr,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=1.0,
            aspect="auto",
            title="Relative Risk Ratios (Red = Higher Risk, Blue = Lower Risk)"
        )
        fig.update_layout(height=max(400, len(profile_features) * 25))
        st.plotly_chart(fig, use_container_width=True)

        # Download
        st.download_button(
            "📥 Download Relative Risk Ratios",
            rr.to_csv().encode('utf-8'),
            "relative_risk_ratios.csv",
            "text/csv"
        )

        # Risk summary by cluster
        st.subheader("Risk Summary by Cluster")
        rr_threshold = st.slider("Risk Threshold", 1.2, 2.0, 1.5, 0.1)

        for cluster_name in rr.index:
            cluster_rr = rr.loc[cluster_name]
            high_risk = cluster_rr[cluster_rr > rr_threshold].sort_values(ascending=False)
            low_risk = cluster_rr[cluster_rr < 1/rr_threshold].sort_values()

            with st.expander(f"{cluster_name}", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**High Risk** (RR > {rr_threshold})")
                    if len(high_risk) > 0:
                        for feat, val in high_risk.items():
                            st.write(f"• {feat}: {val:.2f}")
                    else:
                        st.info("No high-risk indicators")

                with col2:
                    st.markdown(f"**Low Risk** (RR < {1/rr_threshold:.2f})")
                    if len(low_risk) > 0:
                        for feat, val in low_risk.items():
                            st.write(f"• {feat}: {val:.2f}")
                    else:
                        st.info("No low-risk indicators")

    # -------------------------------------------------------------------------
    # POVERTY ARCHETYPES PAGE
    # -------------------------------------------------------------------------
    elif page == "Poverty Archetypes":
        st.title("Poverty Archetypes - Faces of Poverty")
        st.caption("Automated cluster interpretation and naming")

        st.info("""
        **Poverty Archetypes** provide interpretable names for each cluster based on
        their deprivation profiles. This helps translate statistical clusters into
        meaningful policy-relevant categories.
        """)

        with st.spinner("Generating cluster profiles..."):
            profiles = generate_cluster_profiles(X_scaled, labels, profile_features)

        # Cluster sizes
        st.subheader("Cluster Sizes")
        sizes_df = pd.DataFrame({
            'Cluster': list(profiles['cluster_sizes'].keys()),
            'Count': list(profiles['cluster_sizes'].values()),
            'Proportion (%)': [v * 100 for v in profiles['cluster_proportions'].values()]
        })
        st.dataframe(sizes_df, use_container_width=True, hide_index=True)

        # Pie chart of cluster proportions
        fig = px.pie(
            sizes_df,
            values='Proportion (%)',
            names='Cluster',
            title="Population Distribution by Cluster"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cluster descriptions
        st.subheader("Cluster Descriptions")
        for cluster, description in profiles['cluster_descriptions'].items():
            st.markdown(f"**{cluster}**: {description}")

        # High/Low risk summary
        st.subheader("Key Distinguishing Features")
        for cluster in profiles['high_risk_features'].keys():
            with st.expander(f"{cluster} - Key Features"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Elevated Risk Indicators**")
                    high_risk = profiles['high_risk_features'][cluster]
                    if high_risk:
                        for feat, rr in list(high_risk.items())[:5]:
                            st.write(f"• {feat}: RR = {rr:.2f}")
                    else:
                        st.info("No elevated indicators")

                with col2:
                    st.markdown("**Protected Indicators**")
                    low_risk = profiles['low_risk_features'][cluster]
                    if low_risk:
                        for feat, rr in list(low_risk.items())[:5]:
                            st.write(f"• {feat}: RR = {rr:.2f}")
                    else:
                        st.info("No protected indicators")

# =============================================================================
# VALIDATION SECTION (Step 4: Validation Framework)
# =============================================================================
elif section == "✓ Validation":

    # Check if we have clustering results
    if st.session_state.current_labels is None:
        st.warning("No clustering results yet. Run a model in the Clustering section first.")
        st.stop()

    labels = st.session_state.current_labels

    # Get features for validation
    binary_cols = get_binary_columns(df)
    val_features = st.session_state.get('recommended_features', binary_cols)

    # Preprocess for validation
    try:
        prep = preprocess_features(df, val_features)
        X_scaled = prep['X_scaled']
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        st.stop()

    # -------------------------------------------------------------------------
    # BOOTSTRAP STABILITY PAGE
    # -------------------------------------------------------------------------
    if page == "Bootstrap Stability (ARI)":
        st.title("Bootstrap Stability Analysis")
        st.caption("Measuring cluster reproducibility using Adjusted Rand Index")

        st.info("""
        **Bootstrap Stability** tests if clusters are reproducible by:
        1. Resampling the data multiple times (with replacement)
        2. Re-clustering each bootstrap sample
        3. Measuring agreement with original clusters using Adjusted Rand Index (ARI)

        **Interpretation:**
        - ARI > 0.9: Excellent stability (highly reproducible)
        - ARI > 0.7: Good stability (publishable results)
        - ARI > 0.5: Fair stability (interpret with caution)
        - ARI < 0.5: Poor stability (unstable clusters)
        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Configuration")

            n_bootstrap = st.slider("Bootstrap Iterations", 20, 200, 50,
                                    help="More iterations = more reliable estimate")
            sample_ratio = st.slider("Sample Ratio", 0.5, 0.95, 0.8, 0.05,
                                     help="Proportion of data to sample each iteration")

            # Select model for re-clustering
            model_key = st.selectbox(
                "Model for Stability Test",
                list(ENABLED_MODEL_NAMES.keys()),
                format_func=lambda x: ENABLED_MODEL_NAMES[x]
            )

            run_bootstrap = st.button("Run Bootstrap Analysis", type="primary", use_container_width=True)

        with col2:
            if run_bootstrap:
                with st.spinner(f"Running {n_bootstrap} bootstrap iterations..."):
                    model = get_model(model_key)

                    # Set default params to match current clustering
                    n_clusters = len(np.unique(labels[labels >= 0]))
                    if model.supports_n_clusters:
                        if 'n_clusters' in model.params:
                            model.set_params(n_clusters=n_clusters)
                        elif 'n_classes' in model.params:
                            model.set_params(n_classes=n_clusters)
                        elif 'n_components' in model.params:
                            model.set_params(n_components=n_clusters)

                    def clustering_func(X):
                        return model.fit_predict(X)

                    stability = bootstrap_stability_analysis(
                        X_scaled,
                        clustering_func,
                        n_bootstrap=n_bootstrap,
                        sample_ratio=sample_ratio
                    )

                # Display results
                st.subheader("Results")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Mean ARI", f"{stability['mean_ari']:.3f}")
                m2.metric("Std ARI", f"{stability['std_ari']:.3f}")
                m3.metric("Stability Grade", stability['stability_grade'])
                m4.metric("Stable?", "Yes" if stability['is_stable'] else "No")

                # ARI distribution histogram
                if stability['ari_values']:
                    fig = px.histogram(
                        x=stability['ari_values'],
                        nbins=20,
                        title="Distribution of Bootstrap ARI Values",
                        labels={'x': 'Adjusted Rand Index', 'y': 'Count'}
                    )
                    fig.add_vline(x=0.7, line_dash="dash", line_color="green",
                                  annotation_text="Good threshold (0.7)")
                    fig.add_vline(x=stability['mean_ari'], line_dash="solid", line_color="red",
                                  annotation_text=f"Mean ({stability['mean_ari']:.3f})")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                # Summary interpretation
                if stability['is_stable']:
                    st.success(f"""
                    **Clusters are STABLE** (Mean ARI = {stability['mean_ari']:.3f})

                    Your clustering results are reproducible and reliable for publication.
                    The clusters represent genuine structure in the data.
                    """)
                else:
                    st.warning(f"""
                    **Clusters may be UNSTABLE** (Mean ARI = {stability['mean_ari']:.3f})

                    Consider:
                    - Using a different number of clusters
                    - Trying a different clustering algorithm
                    - Checking for outliers or data quality issues
                    """)

    # -------------------------------------------------------------------------
    # EXTERNAL VALIDATION PAGE
    # -------------------------------------------------------------------------
    elif page == "External Validation":
        st.title("External Validation")
        st.caption("Comparing clusters with known external variables")

        st.info("""
        **External Validation** tests if clusters are meaningfully related to known variables
        (e.g., Urban/Rural, Region, Income quintile). This helps validate that clusters
        capture real-world distinctions.

        **Metrics:**
        - **Chi-Square Test**: Tests if cluster × external variable association is statistically significant
        - **Cramer's V**: Effect size measure (0.1 = small, 0.3 = medium, 0.5 = large)
        - **Normalized Mutual Information**: Information-theoretic measure of association
        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Configuration")

            # Select external variable
            all_columns = df.columns.tolist()
            external_var = st.selectbox(
                "External Variable",
                all_columns,
                help="Select a known variable (e.g., Urban/Rural, Region) to validate clusters"
            )

            run_validation = st.button("Run External Validation", type="primary", use_container_width=True)

        with col2:
            if run_validation and external_var:
                with st.spinner("Calculating external validation..."):
                    ext_val = external_validation(labels, df[external_var].values, external_var)

                # Metrics
                st.subheader("Statistical Tests")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Chi-Square", f"{ext_val['chi_square']:.2f}")
                m2.metric("P-Value", f"{ext_val['p_value']:.4f}")
                m3.metric("Cramer's V", f"{ext_val['cramers_v']:.3f}")
                m4.metric("NMI", f"{ext_val['nmi']:.3f}")

                # Interpretation
                st.subheader("Interpretation")
                if ext_val['p_value'] < 0.05:
                    st.success(f"**Significant association** between clusters and {external_var} (p < 0.05)")

                    if ext_val['cramers_v'] > 0.5:
                        effect = "Strong"
                    elif ext_val['cramers_v'] > 0.3:
                        effect = "Medium"
                    elif ext_val['cramers_v'] > 0.1:
                        effect = "Small"
                    else:
                        effect = "Negligible"

                    st.info(f"**{effect} effect size** (Cramer's V = {ext_val['cramers_v']:.3f})")
                else:
                    st.warning(f"No significant association between clusters and {external_var}")

                # Crosstabulation
                st.subheader("Crosstabulation (Row %)")
                st.dataframe(ext_val['crosstab_pct'], use_container_width=True)

                # Visualization
                ct_pct = ext_val['crosstab_pct']
                fig = px.bar(
                    ct_pct.reset_index().melt(id_vars='Cluster'),
                    x='Cluster',
                    y='value',
                    color='variable',
                    barmode='group',
                    title=f"Distribution of {external_var} by Cluster",
                    labels={'value': 'Percentage (%)', 'variable': external_var}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # SUMMARY REPORT PAGE
    # -------------------------------------------------------------------------
    elif page == "Summary Report":
        st.title("Validation Summary Report")
        st.caption("Comprehensive overview of clustering quality")

        # Cluster overview
        st.subheader("Cluster Overview")
        unique_labels = np.unique(labels[labels >= 0])
        n_clusters = len(unique_labels)
        cluster_sizes = pd.Series(labels).value_counts().sort_index()

        col1, col2, col3 = st.columns(3)
        col1.metric("Number of Clusters", n_clusters)
        col2.metric("Total Samples", len(labels))
        col3.metric("Noise Points", np.sum(labels < 0) if np.any(labels < 0) else 0)

        # Cluster size distribution
        st.subheader("Cluster Size Distribution")
        sizes_df = pd.DataFrame({
            'Cluster': [f'Cluster {k}' for k in cluster_sizes.index],
            'Count': cluster_sizes.values,
            'Proportion (%)': (cluster_sizes.values / len(labels) * 100).round(1)
        })
        st.dataframe(sizes_df, use_container_width=True, hide_index=True)

        # Internal metrics (if available)
        st.subheader("Internal Validation Metrics")

        if n_clusters > 1:
            from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

            try:
                sil = silhouette_score(X_scaled, labels)
                db = davies_bouldin_score(X_scaled, labels)
                ch = calinski_harabasz_score(X_scaled, labels)

                m1, m2, m3 = st.columns(3)
                m1.metric("Silhouette Score", f"{sil:.3f}",
                          help="Higher is better. Range: -1 to 1")
                m2.metric("Davies-Bouldin", f"{db:.3f}",
                          help="Lower is better. Measures cluster separation")
                m3.metric("Calinski-Harabasz", f"{ch:.1f}",
                          help="Higher is better. Variance ratio criterion")

                # Interpretation
                if sil > 0.5:
                    st.success("Strong cluster structure detected (Silhouette > 0.5)")
                elif sil > 0.25:
                    st.info("Moderate cluster structure detected (Silhouette 0.25-0.5)")
                else:
                    st.warning("Weak cluster structure (Silhouette < 0.25)")

            except Exception as e:
                st.warning(f"Could not calculate metrics: {e}")
        else:
            st.warning("Need at least 2 clusters to calculate metrics")

        # Recommendations
        st.subheader("Recommendations")

        st.markdown("""
        **Next Steps for Publication-Ready Analysis:**

        1. **Bootstrap Stability**: Run bootstrap analysis to confirm cluster reproducibility (target ARI > 0.7)
        2. **External Validation**: Cross-tabulate with known variables (Urban/Rural, Region)
        3. **Profile Interpretation**: Examine Item Response Probabilities and Relative Risk Ratios
        4. **Sensitivity Analysis**: Test different numbers of clusters using BIC/AIC
        5. **Documentation**: Record all parameter choices and random seeds for reproducibility
        """)

# =============================================================================
# FOOTER
# =============================================================================
st.sidebar.divider()
st.sidebar.caption("ZE-Workbench v3.0 | PhD-Level MPI Analysis")
