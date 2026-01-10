"""
Streamlit Dashboard untuk EDA Dataset Dengue Virus
Dashboard komprehensif untuk ahli virologi
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Dengue Virus Mutation Detection - EDA Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_raw_data():
    """Load raw data dari dataset directory"""
    dataset_dir = Path('dataset')
    
    data = {}
    
    # Load semua CSV files
    files = {
        'metadata': 'sample_metadata.csv',
        'sequence_features': 'sequence_features.csv',
        'mutation_profile': 'mutation_profile.csv',
        'labels': 'label_table.csv'
    }
    
    for key, filename in files.items():
        filepath = dataset_dir / filename
        if filepath.exists():
            data[key] = pd.read_csv(filepath)
        else:
            data[key] = None
    
    return data


@st.cache_data
def load_cleaned_data():
    """Load cleaned data"""
    cleaned_path = Path('ml_dataset_raw.csv')
    if cleaned_path.exists():
        return pd.read_csv(cleaned_path)
    return None


def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<div class="main-header">🦠 Dengue Virus Mutation Detection - EDA Dashboard</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        [
            "📈 Overview",
            "🔬 Raw Data Analysis",
            "✨ Cleaned Data Analysis",
            "🧬 Sequence Features",
            "🔀 Mutation Analysis",
            "📊 Serotype & Genotype",
            "🌍 Geographic Analysis",
            "📅 Temporal Analysis",
            "🔍 Feature Importance",
            "📋 Summary Report"
        ]
    )
    
    # Load data
    with st.spinner("Loading data..."):
        raw_data = load_raw_data()
        cleaned_data = load_cleaned_data()
    
    # Page routing
    if page == "📈 Overview":
        show_overview(raw_data, cleaned_data)
    elif page == "🔬 Raw Data Analysis":
        show_raw_data_analysis(raw_data)
    elif page == "✨ Cleaned Data Analysis":
        show_cleaned_data_analysis(cleaned_data)
    elif page == "🧬 Sequence Features":
        show_sequence_features(raw_data, cleaned_data)
    elif page == "🔀 Mutation Analysis":
        show_mutation_analysis(raw_data, cleaned_data)
    elif page == "📊 Serotype & Genotype":
        show_serotype_genotype_analysis(cleaned_data)
    elif page == "🌍 Geographic Analysis":
        show_geographic_analysis(cleaned_data)
    elif page == "📅 Temporal Analysis":
        show_temporal_analysis(cleaned_data)
    elif page == "🔍 Feature Importance":
        show_feature_importance(cleaned_data)
    elif page == "📋 Summary Report":
        show_summary_report(raw_data, cleaned_data)


def show_overview(raw_data, cleaned_data):
    """Show overview dashboard"""
    st.header("📈 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Raw data metrics
    if raw_data.get('metadata') is not None:
        with col1:
            st.metric("Total Raw Samples", len(raw_data['metadata']))
        
        with col2:
            st.metric("Sequence Features", len(raw_data['sequence_features']) if raw_data.get('sequence_features') is not None else 0)
        
        with col3:
            st.metric("Mutation Profiles", len(raw_data['mutation_profile']) if raw_data.get('mutation_profile') is not None else 0)
    
    # Cleaned data metrics
    if cleaned_data is not None:
        with col4:
            st.metric("Cleaned Samples", len(cleaned_data))
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Quality Metrics")
            missing_pct = (cleaned_data.isnull().sum().sum() / (cleaned_data.shape[0] * cleaned_data.shape[1])) * 100
            st.metric("Missing Values (%)", f"{missing_pct:.2f}%")
            st.metric("Total Features", cleaned_data.shape[1])
            st.metric("Complete Samples", cleaned_data['is_complete'].sum() if 'is_complete' in cleaned_data.columns else "N/A")
        
        with col2:
            st.subheader("Serotype Distribution")
            if 'serotype' in cleaned_data.columns:
                serotype_counts = cleaned_data['serotype'].value_counts().head(10)
                fig = px.pie(
                    values=serotype_counts.values,
                    names=serotype_counts.index,
                    title="Top 10 Serotypes"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Data flow diagram
    st.markdown("---")
    st.subheader("Data Processing Pipeline")
    st.markdown("""
    ```
    Raw Data (GenBank)
        ↓
    Preprocessing & QC
        ↓
    Alignment & Mutation Calling
        ↓
    Feature Engineering
        ↓
    Cleaned Dataset (ML-ready)
    ```
    """)


def show_raw_data_analysis(raw_data):
    """Show raw data analysis"""
    st.header("🔬 Raw Data Analysis")
    
    if raw_data.get('metadata') is None:
        st.warning("Raw metadata not found!")
        return
    
    df_meta = raw_data['metadata']
    
    # Basic info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Info")
        st.write(f"**Shape:** {df_meta.shape[0]} rows × {df_meta.shape[1]} columns")
        st.write(f"**Columns:** {', '.join(df_meta.columns.tolist())}")
    
    with col2:
        st.subheader("Missing Values")
        missing = df_meta.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Missing %': (missing.values / len(df_meta) * 100).round(2)
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        st.dataframe(missing_df, use_container_width=True)
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df_meta.head(20), use_container_width=True)
    
    # Statistics
    st.subheader("Descriptive Statistics")
    numeric_cols = df_meta.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        st.dataframe(df_meta[numeric_cols].describe(), use_container_width=True)
    
    # Distribution plots
    if 'genome_length' in df_meta.columns:
        st.subheader("Genome Length Distribution")
        fig = px.histogram(
            df_meta, 
            x='genome_length',
            nbins=50,
            title="Distribution of Genome Length"
        )
        st.plotly_chart(fig, use_container_width=True)


def show_cleaned_data_analysis(cleaned_data):
    """Show cleaned data analysis"""
    st.header("✨ Cleaned Data Analysis")
    
    if cleaned_data is None:
        st.warning("Cleaned data not found! Please run the pipeline first.")
        return
    
    # Comparison: Raw vs Cleaned
    raw_data = load_raw_data()
    
    col1, col2, col3 = st.columns(3)
    
    if raw_data.get('metadata') is not None:
        with col1:
            st.metric("Raw Samples", len(raw_data['metadata']))
        with col2:
            st.metric("Cleaned Samples", len(cleaned_data))
        with col3:
            removed = len(raw_data['metadata']) - len(cleaned_data)
            st.metric("Removed Samples", removed)
    
    st.markdown("---")
    
    # Data quality improvements
    st.subheader("Data Quality Improvements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Before Cleaning:**")
        if raw_data.get('metadata') is not None:
            raw_missing = raw_data['metadata'].isnull().sum().sum()
            st.metric("Total Missing Values", raw_missing)
    
    with col2:
        st.write("**After Cleaning:**")
        cleaned_missing = cleaned_data.isnull().sum().sum()
        st.metric("Total Missing Values", cleaned_missing)
    
    # Feature comparison
    st.subheader("Feature Comparison")
    if raw_data.get('metadata') is not None:
        raw_features = len(raw_data['metadata'].columns)
        cleaned_features = len(cleaned_data.columns)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Raw Data', 'Cleaned Data'],
            y=[raw_features, cleaned_features],
            marker_color=['#ff7f0e', '#2ca02c']
        ))
        fig.update_layout(
            title="Number of Features",
            yaxis_title="Feature Count"
        )
        st.plotly_chart(fig, use_container_width=True)


def show_sequence_features(raw_data, cleaned_data):
    """Show sequence features analysis"""
    st.header("🧬 Sequence Features Analysis")
    
    if raw_data.get('sequence_features') is None:
        st.warning("Sequence features not found!")
        return
    
    df_seq = raw_data['sequence_features']
    
    # GC Content analysis
    if 'gc_content' in df_seq.columns:
        st.subheader("GC Content Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df_seq,
                x='gc_content',
                nbins=50,
                title="GC Content Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            gc_stats = df_seq['gc_content'].describe()
            st.dataframe(gc_stats.to_frame().T, use_container_width=True)
    
    # K-mer analysis
    st.subheader("K-mer Frequency Analysis")
    
    kmer_cols = [col for col in df_seq.columns if col.startswith('kmer_')]
    
    if kmer_cols:
        # Top k-mers
        kmer_means = df_seq[kmer_cols].mean().sort_values(ascending=False).head(20)
        
        # Create DataFrame for plotly
        kmer_df = pd.DataFrame({
            'K-mer': kmer_means.index,
            'Frequency': kmer_means.values
        })
        
        fig = px.bar(
            kmer_df,
            x='Frequency',
            y='K-mer',
            orientation='h',
            title="Top 20 Most Frequent K-mers",
            labels={'Frequency': 'Frequency', 'K-mer': 'K-mer'}
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # K-mer correlation heatmap (sample)
        if len(kmer_cols) > 10:
            st.subheader("K-mer Correlation (Sample)")
            sample_kmer_cols = kmer_cols[:20]  # Sample first 20
            corr_matrix = df_seq[sample_kmer_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                title="K-mer Correlation Matrix (Sample)",
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig, use_container_width=True)


def show_mutation_analysis(raw_data, cleaned_data):
    """Show mutation analysis"""
    st.header("🔀 Mutation Analysis")
    
    if raw_data.get('mutation_profile') is None:
        st.warning("Mutation profile not found!")
        return
    
    df_mut = raw_data['mutation_profile']
    
    # Mutation density
    if 'mutation_density' in df_mut.columns:
        st.subheader("Mutation Density Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df_mut,
                x='mutation_density',
                nbins=50,
                title="Mutation Density Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            mut_stats = df_mut['mutation_density'].describe()
            st.dataframe(mut_stats.to_frame().T, use_container_width=True)
            
            st.metric("Mean Mutation Density", f"{df_mut['mutation_density'].mean():.6f}")
            st.metric("Max Mutation Density", f"{df_mut['mutation_density'].max():.6f}")
    
    # Length difference
    if 'length_diff' in df_mut.columns:
        st.subheader("Genome Length Difference")
        fig = px.box(
            df_mut,
            y='length_diff',
            title="Distribution of Length Differences"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Combined analysis with serotype
    if cleaned_data is not None and 'serotype' in cleaned_data.columns and 'mutation_density' in cleaned_data.columns:
        st.subheader("Mutation Density by Serotype")
        
        mut_by_sero = cleaned_data.groupby('serotype')['mutation_density'].agg(['mean', 'std', 'count']).reset_index()
        mut_by_sero = mut_by_sero.sort_values('mean', ascending=False)
        
        fig = px.bar(
            mut_by_sero,
            x='serotype',
            y='mean',
            error_y='std',
            title="Average Mutation Density by Serotype"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)


def show_serotype_genotype_analysis(cleaned_data):
    """Show serotype and genotype analysis"""
    st.header("📊 Serotype & Genotype Analysis")
    
    if cleaned_data is None:
        st.warning("Cleaned data not found!")
        return
    
    col1, col2 = st.columns(2)
    
    # Serotype distribution
    if 'serotype' in cleaned_data.columns:
        with col1:
            st.subheader("Serotype Distribution")
            serotype_counts = cleaned_data['serotype'].value_counts()
            
            # Create DataFrame for plotly
            serotype_df = pd.DataFrame({
                'Serotype': serotype_counts.index,
                'Count': serotype_counts.values
            })
            
            fig = px.bar(
                serotype_df,
                x='Serotype',
                y='Count',
                title="Serotype Counts",
                labels={'Serotype': 'Serotype', 'Count': 'Count'}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(serotype_counts.to_frame('Count'), use_container_width=True)
    
    # Genotype distribution
    if 'genotype' in cleaned_data.columns:
        with col2:
            st.subheader("Genotype Distribution")
            genotype_counts = cleaned_data['genotype'].value_counts()
            
            if len(genotype_counts) > 0:
                fig = px.pie(
                    values=genotype_counts.values,
                    names=genotype_counts.index,
                    title="Genotype Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No genotype data available")
    
    # Serotype vs Genotype
    if 'serotype' in cleaned_data.columns and 'genotype' in cleaned_data.columns:
        st.subheader("Serotype vs Genotype Cross-tabulation")
        crosstab = pd.crosstab(cleaned_data['serotype'], cleaned_data['genotype'])
        st.dataframe(crosstab, use_container_width=True)
        
        # Heatmap
        fig = px.imshow(
            crosstab.values,
            x=crosstab.columns,
            y=crosstab.index,
            title="Serotype-Genotype Heatmap",
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)


def show_geographic_analysis(cleaned_data):
    """Show geographic analysis"""
    st.header("🌍 Geographic Analysis")
    
    if cleaned_data is None:
        st.warning("Cleaned data not found!")
        return
    
    if 'country' not in cleaned_data.columns:
        st.info("Country data not available")
        return
    
    # Country distribution
    st.subheader("Sample Distribution by Country")
    country_counts = cleaned_data['country'].value_counts().head(20)
    
    # Create DataFrame for plotly
    country_df = pd.DataFrame({
        'Country': country_counts.index,
        'Count': country_counts.values
    })
    
    fig = px.bar(
        country_df,
        x='Count',
        y='Country',
        orientation='h',
        title="Top 20 Countries by Sample Count",
        labels={'Count': 'Sample Count', 'Country': 'Country'}
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Country vs Serotype
    if 'serotype' in cleaned_data.columns:
        st.subheader("Serotype Distribution by Country")
        country_sero = pd.crosstab(cleaned_data['country'], cleaned_data['serotype'])
        st.dataframe(country_sero, use_container_width=True)
    
    # Region analysis
    if 'region' in cleaned_data.columns:
        st.subheader("Sample Distribution by Region")
        region_counts = cleaned_data['region'].value_counts()
        
        fig = px.pie(
            values=region_counts.values,
            names=region_counts.index,
            title="Regional Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)


def show_temporal_analysis(cleaned_data):
    """Show temporal analysis"""
    st.header("📅 Temporal Analysis")
    
    if cleaned_data is None:
        st.warning("Cleaned data not found!")
        return
    
    if 'year' not in cleaned_data.columns:
        st.info("Year data not available")
        return
    
    # Clean year data
    df_temp = cleaned_data.copy()
    df_temp['year'] = pd.to_numeric(df_temp['year'], errors='coerce')
    df_temp = df_temp[df_temp['year'].notna()]
    
    if len(df_temp) == 0:
        st.warning("No valid year data found")
        return
    
    # Year distribution
    st.subheader("Sample Collection Over Time")
    year_counts = df_temp['year'].value_counts().sort_index()
    
    # Create DataFrame for plotly
    year_df = pd.DataFrame({
        'Year': year_counts.index,
        'Sample Count': year_counts.values
    })
    
    fig = px.line(
        year_df,
        x='Year',
        y='Sample Count',
        title="Number of Samples Collected Over Time",
        labels={'Year': 'Year', 'Sample Count': 'Sample Count'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Year vs Serotype
    if 'serotype' in cleaned_data.columns:
        st.subheader("Serotype Trends Over Time")
        year_sero = pd.crosstab(df_temp['year'], df_temp['serotype'])
        
        fig = px.line(
            year_sero,
            title="Serotype Distribution Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)


def show_feature_importance(cleaned_data):
    """Show feature importance analysis"""
    st.header("🔍 Feature Importance Analysis")
    
    if cleaned_data is None:
        st.warning("Cleaned data not found!")
        return
    
    # Try to load feature importance from model
    model_path = Path('results/models/baseline_classifier.pkl')
    
    if model_path.exists():
        import pickle
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        
        if hasattr(model, 'feature_importances_'):
            st.subheader("Model Feature Importance")
            
            # Get feature names
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            else:
                feature_names = [f'feature_{i}' for i in range(len(model.feature_importances_))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(30)
            
            fig = px.bar(
                importance_df,
                x='importance',
                y='feature',
                orientation='h',
                title="Top 30 Most Important Features"
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(importance_df, use_container_width=True)
    else:
        st.info("Model not found. Please train the model first to see feature importance.")


def show_summary_report(raw_data, cleaned_data):
    """Show comprehensive summary report"""
    st.header("📋 Comprehensive Summary Report")
    
    st.subheader("Dataset Summary")
    
    # Raw data summary
    if raw_data.get('metadata') is not None:
        st.write("**Raw Data:**")
        st.write(f"- Total samples: {len(raw_data['metadata'])}")
        st.write(f"- Total features: {len(raw_data['metadata'].columns)}")
        st.write(f"- Missing values: {raw_data['metadata'].isnull().sum().sum()}")
    
    # Cleaned data summary
    if cleaned_data is not None:
        st.write("**Cleaned Data:**")
        st.write(f"- Total samples: {len(cleaned_data)}")
        st.write(f"- Total features: {len(cleaned_data.columns)}")
        st.write(f"- Missing values: {cleaned_data.isnull().sum().sum()}")
        
        if 'serotype' in cleaned_data.columns:
            st.write(f"- Unique serotypes: {cleaned_data['serotype'].nunique()}")
        
        if 'genotype' in cleaned_data.columns:
            st.write(f"- Unique genotypes: {cleaned_data['genotype'].nunique()}")
    
    # Key findings
    st.subheader("Key Findings")
    
    findings = []
    
    if cleaned_data is not None:
        if 'serotype' in cleaned_data.columns:
            top_serotype = cleaned_data['serotype'].value_counts().index[0]
            findings.append(f"Most common serotype: {top_serotype}")
        
        if 'mutation_density' in cleaned_data.columns:
            avg_mut_density = cleaned_data['mutation_density'].mean()
            findings.append(f"Average mutation density: {avg_mut_density:.6f}")
        
        if 'gc_content' in cleaned_data.columns:
            avg_gc = cleaned_data['gc_content'].mean()
            findings.append(f"Average GC content: {avg_gc:.4f}")
    
    for finding in findings:
        st.write(f"- {finding}")
    
    # Download report
    st.subheader("Export Report")
    st.info("Report export functionality can be added here")


if __name__ == "__main__":
    main()

