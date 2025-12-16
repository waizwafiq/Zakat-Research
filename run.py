"""
ZE-Workbench - Data Analysis & Machine Learning Platform
Main Streamlit Application

A modular data analysis and machine learning platform with
clustering, statistical analysis, and model comparison capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from utils import load_file, download_df, get_variable_metadata, preprocess_features
from utils.metrics import calculate_cluster_metrics
from models import (
    MODEL_REGISTRY, MODEL_NAMES, get_model,
    ModelRunner, create_experiment_configs
)

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
    st.session_state.data = None
if 'meta' not in st.session_state:
    st.session_state.meta = None
if 'experiment_results' not in st.session_state:
    st.session_state.experiment_results = None
if 'experiment_results_df' not in st.session_state:
    st.session_state.experiment_results_df = None
if 'current_labels' not in st.session_state:
    st.session_state.current_labels = None

# =============================================================================
# SIDEBAR - NAVIGATION & FILE UPLOAD
# =============================================================================
st.sidebar.title("📊 ZE-Workbench")
st.sidebar.caption("Data Analysis & Machine Learning Platform")
st.sidebar.divider()

# File Upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Data",
    type=['csv', 'sav'],
    help="Supported formats: CSV, SAV (.sav)"
)

if uploaded_file:
    df, meta = load_file(uploaded_file)
    if df is not None:
        st.session_state.data = df
        st.session_state.meta = meta

# Navigation
if st.session_state.data is not None:
    st.sidebar.divider()

    # Main section selection
    section = st.sidebar.radio(
        "Section",
        ["📋 Data", "📈 Analysis", "🔬 Clustering"],
        label_visibility="collapsed"
    )

    # Sub-navigation based on section
    if section == "📋 Data":
        page = st.sidebar.radio(
            "Data Pages",
            ["Data View", "Variable View"],
            label_visibility="collapsed"
        )
    elif section == "📈 Analysis":
        page = st.sidebar.radio(
            "Analysis Pages",
            ["Descriptive Statistics", "Correlation Matrix", "T-Tests", "Crosstabs"],
            label_visibility="collapsed"
        )
    else:  # Clustering
        page = st.sidebar.radio(
            "Clustering Pages",
            ["Single Model", "Batch Experiment", "Results Comparison", "Visualization"],
            label_visibility="collapsed"
        )
else:
    section = None
    page = None

# =============================================================================
# MAIN CONTENT
# =============================================================================

if st.session_state.data is None:
    # Welcome Screen
    st.markdown("""
    # Welcome to ZE-Workbench

    A data analysis and machine learning platform for clustering experiments.

    ### Features
    - **Data Editor**: View and edit your data with variable metadata view
    - **Statistical Analysis**: Descriptive stats, correlations, t-tests, crosstabs
    - **Clustering**: K-Means, Hierarchical, DBSCAN, Gaussian Mixture
    - **Batch Experiments**: Run all models with multiple parameters at once
    - **Model Comparison**: Compare results across models and configurations

    ### Getting Started
    👈 Upload a CSV or SAV file using the sidebar to begin.
    """)
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
# ANALYSIS SECTION
# =============================================================================
elif section == "📈 Analysis":
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_columns = df.columns.tolist()

    if page == "Descriptive Statistics":
        st.title("Descriptive Statistics")

        selected_cols = st.multiselect(
            "Select Variables",
            numeric_cols,
            default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
        )

        if selected_cols:
            desc = df[selected_cols].describe().T
            desc['skewness'] = df[selected_cols].skew()
            desc['kurtosis'] = df[selected_cols].kurtosis()
            st.dataframe(desc, use_container_width=True)

    elif page == "Correlation Matrix":
        st.title("Correlation Matrix")

        selected_cols = st.multiselect(
            "Select Variables",
            numeric_cols,
            default=numeric_cols
        )

        if len(selected_cols) > 1:
            method = st.selectbox("Method", ["pearson", "spearman", "kendall"])
            corr = df[selected_cols].corr(method=method)

            fig = px.imshow(
                corr,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                aspect="auto",
                zmin=-1, zmax=1
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

    elif page == "T-Tests":
        st.title("Independent Samples T-Test")

        col1, col2 = st.columns(2)
        with col1:
            target_var = st.selectbox("Test Variable (Numeric)", numeric_cols)
        with col2:
            group_var = st.selectbox("Grouping Variable", all_columns)

        if target_var and group_var:
            groups = df[group_var].dropna().unique()

            if len(groups) == 2:
                g1 = df[df[group_var] == groups[0]][target_var].dropna()
                g2 = df[df[group_var] == groups[1]][target_var].dropna()

                t_stat, p_val = stats.ttest_ind(g1, g2)

                col1, col2, col3 = st.columns(3)
                col1.metric("T-Statistic", f"{t_stat:.4f}")
                col2.metric("P-Value", f"{p_val:.4f}")
                col3.metric(
                    "Result",
                    "Significant" if p_val < 0.05 else "Not Significant"
                )

                st.divider()

                # Group comparison
                st.subheader("Group Comparison")
                comparison = pd.DataFrame({
                    'Group': [str(groups[0]), str(groups[1])],
                    'N': [len(g1), len(g2)],
                    'Mean': [g1.mean(), g2.mean()],
                    'Std': [g1.std(), g2.std()]
                })
                st.dataframe(comparison, hide_index=True)
            else:
                st.warning(f"Grouping variable must have exactly 2 unique values. Found: {len(groups)}")

    elif page == "Crosstabs":
        st.title("Crosstabs (Contingency Table)")

        col1, col2 = st.columns(2)
        with col1:
            row_var = st.selectbox("Row Variable", all_columns)
        with col2:
            col_var = st.selectbox("Column Variable", all_columns, index=min(1, len(all_columns)-1))

        if row_var and col_var and row_var != col_var:
            ct = pd.crosstab(df[row_var], df[col_var])
            st.subheader("Observed Counts")
            st.dataframe(ct, use_container_width=True)

            chi2, p, dof, expected = stats.chi2_contingency(ct)

            col1, col2, col3 = st.columns(3)
            col1.metric("Chi-Square", f"{chi2:.4f}")
            col2.metric("P-Value", f"{p:.4f}")
            col3.metric("Degrees of Freedom", dof)

# =============================================================================
# CLUSTERING SECTION
# =============================================================================
elif section == "🔬 Clustering":

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
    # SINGLE MODEL PAGE
    # -------------------------------------------------------------------------
    if page == "Single Model":
        st.title("Single Model Clustering")

        col1, col2 = st.columns([1, 3])

        with col1:
            model_key = st.selectbox(
                "Algorithm",
                list(MODEL_NAMES.keys()),
                format_func=lambda x: MODEL_NAMES[x]
            )

            model = get_model(model_key)
            st.caption(model.description)

            st.divider()

            # Dynamic parameter UI
            params = {}
            for param_config in model.get_param_config():
                name = param_config['name']
                ptype = param_config['type']

                if ptype == 'int':
                    params[name] = st.slider(
                        param_config.get('description', name),
                        param_config['min'],
                        param_config['max'],
                        param_config['default']
                    )
                elif ptype == 'float':
                    params[name] = st.slider(
                        param_config.get('description', name),
                        param_config['min'],
                        param_config['max'],
                        param_config['default'],
                        step=param_config.get('step', 0.1)
                    )
                elif ptype == 'select':
                    params[name] = st.selectbox(
                        param_config.get('description', name),
                        param_config['options'],
                        index=param_config['options'].index(param_config['default'])
                    )

            run_btn = st.button("🚀 Run Clustering", type="primary", use_container_width=True)

        with col2:
            if run_btn:
                with st.spinner("Running clustering..."):
                    model.set_params(**params)
                    labels = model.fit_predict(X_scaled)
                    metrics = model.get_metrics(X_scaled, labels)

                    st.session_state.current_labels = labels

                if metrics['valid']:
                    # Metrics display
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Clusters", metrics['n_clusters'])
                    m2.metric("Silhouette", f"{metrics['silhouette']:.4f}")
                    m3.metric("Davies-Bouldin", f"{metrics['davies_bouldin']:.4f}")
                    m4.metric("Calinski-Harabasz", f"{metrics['calinski_harabasz']:.2f}")

                    # Results table
                    results_df = df.copy()
                    results_df['Cluster'] = labels

                    st.subheader("Clustered Data")
                    st.dataframe(results_df, use_container_width=True, height=300)

                    st.download_button(
                        "📥 Download Results",
                        download_df(results_df),
                        "clustered_data.csv",
                        "text/csv"
                    )

                    # Cluster profiles
                    st.subheader("Cluster Profiles")
                    numeric_results = results_df.select_dtypes(include=np.number)
                    profile = numeric_results.groupby('Cluster').mean()

                    fig = plt.figure(figsize=(12, 6))
                    sns.heatmap(profile.T, cmap='YlOrRd', annot=True, fmt='.2f', linewidths=0.5)
                    plt.title("Mean Values by Cluster")
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.warning("Clustering produced only 1 cluster or all noise. Try adjusting parameters.")

    # -------------------------------------------------------------------------
    # BATCH EXPERIMENT PAGE
    # -------------------------------------------------------------------------
    elif page == "Batch Experiment":
        st.title("Batch Experiment")
        st.caption("Run multiple models with different parameters simultaneously")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Configuration")

            # Model selection
            selected_models = st.multiselect(
                "Models to Run",
                list(MODEL_NAMES.keys()),
                default=list(MODEL_NAMES.keys()),
                format_func=lambda x: MODEL_NAMES[x]
            )

            # K range for models that support it
            st.divider()
            k_min, k_max = st.slider(
                "Cluster Range (k)",
                2, 15, (2, 8),
                help="Range of cluster values to try"
            )
            k_range = range(k_min, k_max + 1)

            # DBSCAN parameters
            if 'dbscan' in selected_models:
                st.divider()
                st.caption("DBSCAN Parameters")
                eps_values = st.multiselect(
                    "Epsilon Values",
                    [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
                    default=[0.5, 1.0, 2.0]
                )
                min_samples_values = st.multiselect(
                    "Min Samples Values",
                    [3, 5, 10, 15, 20],
                    default=[5]
                )
            else:
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
                st.info("Previous experiment results available. Go to 'Results Comparison' to view.")

    # -------------------------------------------------------------------------
    # RESULTS COMPARISON PAGE
    # -------------------------------------------------------------------------
    elif page == "Results Comparison":
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

    # -------------------------------------------------------------------------
    # VISUALIZATION PAGE
    # -------------------------------------------------------------------------
    elif page == "Visualization":
        st.title("Cluster Visualization")

        has_single = st.session_state.current_labels is not None
        has_batch = st.session_state.experiment_results is not None

        if not has_single and not has_batch:
            st.warning("No clustering results yet. Run a model first.")
            st.stop()

        # Select which result to visualize
        options = []
        if has_single:
            options.append("Last Single Model")
        if has_batch:
            options.append("Best from Batch")

        source = st.radio(
            "Visualize",
            options,
            horizontal=True
        )

        if source == "Last Single Model":
            labels = st.session_state.current_labels
        else:
            best = st.session_state.experiment_results.get_best_model('Silhouette')
            if best is None:
                st.warning("No valid results found.")
                st.stop()
            labels = best.labels
            st.info(f"Showing: {best.model_name} ({best.params_string})")

        # PCA projection
        st.subheader("PCA Projection")

        pca = PCA(n_components=3)
        components = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame(
            components,
            columns=['PC1', 'PC2', 'PC3']
        )
        pca_df['Cluster'] = labels.astype(str)

        # Add original features for hover
        for col in features[:3]:  # Limit to first 3 for hover
            pca_df[col] = df[col].values

        tab1, tab2 = st.tabs(["3D View", "2D View"])

        with tab1:
            fig_3d = px.scatter_3d(
                pca_df,
                x='PC1', y='PC2', z='PC3',
                color='Cluster',
                title=f"3D PCA (Variance Explained: {sum(pca.explained_variance_ratio_):.1%})",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_3d.update_traces(marker=dict(size=5))
            fig_3d.update_layout(height=600)
            st.plotly_chart(fig_3d, use_container_width=True)

        with tab2:
            fig_2d = px.scatter(
                pca_df,
                x='PC1', y='PC2',
                color='Cluster',
                title=f"2D PCA (Variance Explained: {pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]:.1%})",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_2d.update_traces(marker=dict(size=8))
            fig_2d.update_layout(height=500)
            st.plotly_chart(fig_2d, use_container_width=True)

        # Feature Importance
        st.divider()
        st.subheader("Feature Importance (Random Forest)")

        unique_labels = np.unique(labels)
        if len(unique_labels) > 1 and -1 not in unique_labels:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(prep['X_encoded'], labels)

            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': rf.feature_importances_
            }).sort_values('Importance', ascending=True)

            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Which features determine cluster membership?",
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=max(300, len(features) * 25))
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# FOOTER
# =============================================================================
st.sidebar.divider()
st.sidebar.caption("ZE-Workbench v2.0 | Modular Clustering")
