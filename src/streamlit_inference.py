"""
Streamlit App untuk Inference Model Dengue Virus
Aplikasi untuk melakukan prediksi menggunakan model yang sudah di-train
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

# Add src directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from inference import InferencePipeline

# Set page config
st.set_page_config(
    page_title="Dengue Virus Mutation Detection - Inference",
    page_icon="🔬",
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
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .novel-alert {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-alert {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'inference_pipeline' not in st.session_state:
    st.session_state.inference_pipeline = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'results' not in st.session_state:
    st.session_state.results = None


@st.cache_resource
def load_models(models_dir='results/models'):
    """Load models dengan caching"""
    try:
        inference = InferencePipeline(models_dir=models_dir)
        inference.load_models()
        return inference, True
    except Exception as e:
        return None, str(e)


def main():
    """Main function untuk inference app"""
    
    # Header
    st.markdown('<div class="main-header">🔬 Dengue Virus Mutation Detection - Inference</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    models_dir = st.sidebar.text_input(
        "Models Directory",
        value="results/models",
        help="Path ke directory tempat model disimpan"
    )
    
    # Load models button
    if st.sidebar.button("🔄 Load Models", type="primary"):
        with st.spinner("Loading models..."):
            inference, status = load_models(models_dir)
            if status is True:
                st.session_state.inference_pipeline = inference
                st.session_state.models_loaded = True
                st.sidebar.success("✅ Models loaded successfully!")
            else:
                st.session_state.models_loaded = False
                st.sidebar.error(f"❌ Error loading models: {status}")
    
    # Check if models are loaded
    if st.session_state.models_loaded and st.session_state.inference_pipeline:
        st.sidebar.success("✅ Models Ready")
        
        # Show available models
        st.sidebar.subheader("Available Models")
        pipeline = st.session_state.inference_pipeline
        
        if pipeline.baseline_model:
            st.sidebar.write("✅ Baseline Classifier")
        if pipeline.novelty_detector and pipeline.novelty_detector.best_model:
            st.sidebar.write("✅ Novelty Detector")
        if pipeline.open_set_detector and pipeline.open_set_detector.best_model:
            st.sidebar.write("✅ Open-set Detector")
    else:
        st.sidebar.warning("⚠️ Please load models first")
    
    st.markdown("---")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload Data", "📊 View Results", "ℹ️ About"])
    
    with tab1:
        st.header("Upload Data for Prediction")
        
        if not st.session_state.models_loaded:
            st.warning("⚠️ Please load models first using the sidebar before uploading data.")
            return
        
        # Input method selection
        input_method = st.radio(
            "Select input method:",
            ["Upload CSV File", "Use Sample Dataset"],
            horizontal=True
        )
        
        uploaded_file = None
        input_data = None
        
        if input_method == "Upload CSV File":
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload CSV file dengan format yang sama seperti dataset training"
            )
            
            if uploaded_file is not None:
                try:
                    input_data = pd.read_csv(uploaded_file)
                    st.success(f"✅ File loaded: {len(input_data)} samples")
                    st.dataframe(input_data.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        else:  # Use Sample Dataset
            sample_dataset_path = Path('dataset')
            if sample_dataset_path.exists():
                st.info("Using sample dataset from 'dataset' directory")
                try:
                    from data_cleaning import DataCleaner
                    cleaner = DataCleaner(dataset_dir='dataset')
                    cleaner.load_datasets()
                    cleaner.merge_tables()
                    input_data = cleaner.cleaned_data
                    st.success(f"✅ Dataset loaded: {len(input_data)} samples")
                    st.dataframe(input_data.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading dataset: {e}")
            else:
                st.warning("Sample dataset directory not found!")
        
        # Task selection
        if input_data is not None:
            st.markdown("---")
            st.subheader("Select Prediction Tasks")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                run_baseline = st.checkbox("Baseline Classification", value=True, 
                                         help="Predict serotype/genotype")
            
            with col2:
                run_novelty = st.checkbox("Novelty Detection", value=True,
                                        help="Detect novel genotypes")
            
            with col3:
                run_open_set = st.checkbox("Open-set Detection", value=True,
                                         help="Detect potential new serotypes")
            
            # Run inference button
            if st.button("🚀 Run Inference", type="primary", use_container_width=True):
                tasks = []
                if run_baseline:
                    tasks.append('baseline')
                if run_novelty:
                    tasks.append('novelty')
                if run_open_set:
                    tasks.append('open_set')
                
                if len(tasks) == 0:
                    st.warning("Please select at least one task!")
                else:
                    with st.spinner("Running inference..."):
                        try:
                            pipeline = st.session_state.inference_pipeline
                            results = pipeline.run_full_inference(
                                input_data,
                                is_dataframe=True,
                                tasks=tasks
                            )
                            
                            # Format results
                            df_results = pipeline.format_results(results, output_format='dataframe')
                            st.session_state.results = df_results
                            
                            st.success("✅ Inference completed! Check the 'View Results' tab.")
                            st.balloons()
                            
                        except Exception as e:
                            st.error(f"Error during inference: {e}")
                            import traceback
                            st.code(traceback.format_exc())
    
    with tab2:
        st.header("📊 Prediction Results")
        
        if st.session_state.results is None:
            st.info("👈 Please run inference first in the 'Upload Data' tab")
        else:
            df_results = st.session_state.results
            
            # Summary statistics
            st.subheader("Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Samples", len(df_results))
            
            if 'predicted_serotype' in df_results.columns:
                with col2:
                    unique_serotypes = df_results['predicted_serotype'].nunique()
                    st.metric("Unique Serotypes", unique_serotypes)
            
            if 'is_novel_genotype' in df_results.columns:
                with col3:
                    novel_count = df_results['is_novel_genotype'].sum()
                    st.metric("Novel Genotypes", novel_count)
            
            if 'is_potential_new_serotype' in df_results.columns:
                with col4:
                    open_set_count = df_results['is_potential_new_serotype'].sum()
                    st.metric("Potential New Serotypes", open_set_count)
            
            st.markdown("---")
            
            # Detailed results table
            st.subheader("Detailed Results")
            st.dataframe(df_results, use_container_width=True, height=400)
            
            # Download results
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="inference_results.csv",
                mime="text/csv"
            )
            
            st.markdown("---")
            
            # Visualizations
            st.subheader("Visualizations")
            
            # Baseline predictions
            if 'predicted_serotype' in df_results.columns:
                st.write("**Serotype Predictions Distribution**")
                serotype_counts = df_results['predicted_serotype'].value_counts()
                
                import plotly.express as px
                serotype_df = pd.DataFrame({
                    'Serotype': serotype_counts.index,
                    'Count': serotype_counts.values
                })
                
                fig = px.bar(
                    serotype_df,
                    x='Serotype',
                    y='Count',
                    title="Predicted Serotype Distribution",
                    color='Count',
                    color_continuous_scale='Viridis'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Novelty detection
            if 'is_novel_genotype' in df_results.columns:
                st.write("**Novel Genotype Detection**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    novel_pie = pd.DataFrame({
                        'Category': ['Known', 'Novel'],
                        'Count': [
                            (~df_results['is_novel_genotype']).sum(),
                            df_results['is_novel_genotype'].sum()
                        ]
                    })
                    
                    fig = px.pie(
                        novel_pie,
                        values='Count',
                        names='Category',
                        title="Novel vs Known Genotypes"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'anomaly_score' in df_results.columns:
                        fig = px.histogram(
                            df_results,
                            x='anomaly_score',
                            color='is_novel_genotype',
                            nbins=30,
                            title="Anomaly Score Distribution",
                            labels={'anomaly_score': 'Anomaly Score', 'count': 'Count'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Open-set detection
            if 'is_potential_new_serotype' in df_results.columns:
                st.write("**Open-set Detection (Potential New Serotypes)**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    open_set_pie = pd.DataFrame({
                        'Category': ['Known Serotype', 'Potential New'],
                        'Count': [
                            (~df_results['is_potential_new_serotype']).sum(),
                            df_results['is_potential_new_serotype'].sum()
                        ]
                    })
                    
                    fig = px.pie(
                        open_set_pie,
                        values='Count',
                        names='Category',
                        title="Open-set Detection Results"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'reconstruction_error' in df_results.columns:
                        fig = px.histogram(
                            df_results,
                            x='reconstruction_error',
                            color='is_potential_new_serotype',
                            nbins=30,
                            title="Reconstruction Error Distribution",
                            labels={'reconstruction_error': 'Reconstruction Error', 'count': 'Count'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Alert for novel samples
            if 'is_novel_genotype' in df_results.columns:
                novel_samples = df_results[df_results['is_novel_genotype'] == True]
                if len(novel_samples) > 0:
                    st.markdown('<div class="novel-alert">', unsafe_allow_html=True)
                    st.warning(f"⚠️ **{len(novel_samples)} novel genotype(s) detected!**")
                    st.dataframe(novel_samples[['sample_id', 'anomaly_score']].head(10), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            if 'is_potential_new_serotype' in df_results.columns:
                open_set_samples = df_results[df_results['is_potential_new_serotype'] == True]
                if len(open_set_samples) > 0:
                    st.markdown('<div class="novel-alert">', unsafe_allow_html=True)
                    st.warning(f"⚠️ **{len(open_set_samples)} potential new serotype(s) detected!**")
                    st.dataframe(open_set_samples[['sample_id', 'reconstruction_error']].head(10), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.header("ℹ️ About Inference App")
        
        st.markdown("""
        ### 📋 Overview
        
        Aplikasi ini digunakan untuk melakukan prediksi menggunakan model Machine Learning yang sudah di-train 
        untuk deteksi mutasi Dengue Virus.
        
        ### 🔬 Available Tasks
        
        1. **Baseline Classification**
           - Memprediksi serotipe/genotipe dari sample
           - Menggunakan Random Forest atau XGBoost
           - Output: Predicted serotype dengan confidence score
        
        2. **Novelty Detection**
           - Mendeteksi genotipe yang tidak dikenal saat training
           - Menggunakan Isolation Forest
           - Output: Anomaly score dan flag novel/known
        
        3. **Open-set Detection**
           - Mendeteksi potensi serotipe baru
           - Menggunakan Autoencoder
           - Output: Reconstruction error dan flag potential new serotype
        
        ### 📤 Input Format
        
        Input data harus memiliki format yang sama dengan dataset training:
        - `sample_id`: ID unik untuk setiap sample
        - Features yang digunakan saat training (sequence features, mutation profile, dll)
        
        ### 📊 Output Format
        
        Output berupa CSV dengan kolom:
        - `sample_id`: ID sample
        - `predicted_serotype`: Prediksi serotipe (jika baseline task dijalankan)
        - `prediction_confidence`: Confidence score prediksi
        - `is_novel_genotype`: Apakah sample adalah novel genotype (jika novelty task dijalankan)
        - `anomaly_score`: Anomaly score untuk novelty detection
        - `is_potential_new_serotype`: Apakah sample berpotensi serotipe baru (jika open-set task dijalankan)
        - `reconstruction_error`: Reconstruction error untuk open-set detection
        
        ### ⚠️ Important Notes
        
        - Pastikan model sudah di-train dan tersimpan di `results/models/`
        - Preprocessor harus ada di `results/preprocessor.pkl`
        - Input data harus memiliki features yang sama dengan training data
        """)
        
        st.markdown("---")
        
        st.subheader("📚 Model Information")
        
        if st.session_state.models_loaded and st.session_state.inference_pipeline:
            pipeline = st.session_state.inference_pipeline
            
            if pipeline.baseline_model:
                st.write("**Baseline Classifier:**")
                st.write(f"- Model Type: {type(pipeline.baseline_model).__name__}")
                if hasattr(pipeline.baseline_model, 'n_estimators'):
                    st.write(f"- Number of Estimators: {pipeline.baseline_model.n_estimators}")
            
            if pipeline.novelty_detector and pipeline.novelty_detector.best_model:
                st.write("**Novelty Detector:**")
                st.write(f"- Model Type: {pipeline.novelty_detector.best_model_name}")
                if pipeline.novelty_detector.threshold:
                    st.write(f"- Threshold: {pipeline.novelty_detector.threshold:.4f}")
            
            if pipeline.open_set_detector and pipeline.open_set_detector.best_model_name:
                st.write("**Open-set Detector:**")
                st.write(f"- Model Type: {pipeline.open_set_detector.best_model_name}")
                if pipeline.open_set_detector.threshold:
                    st.write(f"- Threshold: {pipeline.open_set_detector.threshold:.4f}")
        else:
            st.info("Load models first to see model information")


if __name__ == "__main__":
    main()

