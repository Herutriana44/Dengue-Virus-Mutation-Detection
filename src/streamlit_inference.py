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
from sequence_feature_extractor import extract_features_from_sequences, add_mutation_features

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
if 'input_data' not in st.session_state:
    st.session_state.input_data = None
if 'raw_sequences' not in st.session_state:
    st.session_state.raw_sequences = None


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
        
        # Template download section
        st.subheader("📋 Download Template")
        st.info("Download template CSV/Excel file sebagai contoh format data. File hanya perlu kolom 'sequence' (dan optional 'sample_id').")
        
        # Try to load template file if exists, otherwise create default
        template_path = Path('template_inference_sequences.csv')
        if template_path.exists():
            template_data = pd.read_csv(template_path)
        else:
            # Create template data with realistic sequences
            template_data = pd.DataFrame({
                'sample_id': ['SAMPLE_001', 'SAMPLE_002', 'SAMPLE_003'],
                'sequence': [
                    'AGTTGTTAGTCTGTGTGGACCGACAAGGACAGTTCCAAATCGGAAGCTTGCTTAACACAGTTCTAACAGTTTGTTTAAATAGAGAGCAGATCTCTGGAAAAATGAACCAACGAAAAAAGGTGGTCAGACCACCTTTCAATATGCTGAAACGCGAGAGAAACCGCGTATCAACCCCTCAAGGGTTGGTGAAGAGATTCTCAACCGGACTTTTCTCCGGGAAAGGACCTTTGCGGATGGTGCTAGCATTCATCACGTTTTTGCGGGTCCTTTCCATCCCACCAACAGCAGGGATTCTGAAAAGATGGGGACAGTTGAAAAAGAACAAGGCCGTCAAAATACTGATTGGATTCAGGAAGGAGATAGGTCGCATGTTAAACATCTTGAATAGGAGAAGAAGGTCAACAATGACATTGCTGTGTTTGATTCCCACCGTAATGGCGTTTCACCTGTCAACAAGAGATGGCGAACCCCTCATGATAGTGGCAAAACACGAAAGGGGGAGACCTCTCTTGTTTAAGACAACAGAAGGGATCAACAAATGTACCCTCATTGCTATGGACCTGGGTGAAATGTGCGAAGACACTGTCACGTACAAGTGTCCTCTACTGGTTAACACCGAACCTGAAGACATTGACTGCTGGTGCAATCTCACGTCTACTTGGGTCATGTACGGGACATGCACCCAGAACGGAGAACGGAGACGAGAGAAGCGCTCAGTAGCTTTAACACCACATTCAGGAATGGGATTGGAAACAAGAGCTGAGACATGGATGTCATCGGAAGGGGCTTGGAAACATGCTCAGAGAGTAGAAAGCTGGATACTCAGAAACCCAGGATTCGCGCTCTTGGCAGGATTTATGGCTTATATGATTGGGCAAACAGGAATCCAGCGAATTGTTTTCTTTGTCCTGATGATGCTAGTCGCCCCATCCTACGGAATGCGATGCGTAGGGGTAGGGAACAGAGACTTCGTGGAAGGAGTCTCGGGT',
                    'AGTTGTTAGTCTGTGTGGACCGACAAGGACAGTTCCAAATCGGAAGCTTGCTTAACACAGTTCTAACAGTTTGTTTAAATAGAGAGCAGATCTCTGGAAAAATGAACCAACGAAAAAAGGTGGTCAGACCACCTTTCAATATGCTGAAACGCGAGAGAAACCGCGTATCAACCCCTCAAGGGTTGGTGAAGAGATTCTCAACCGGACTCTTCTCCGGGAAAGGACCTTTGCGGATGGTGCTTGCATTCATTACGTTTTTGCGGGTCCTTTCCATCCCACCAACAGCAGGGATTCTGAAAAGATGGGGACAGTTGAAAAAGAACAAGGCCGTCAGAATACTGATTGGATTCAGGAAGGAGATAGGTCGCATGTTAAACATCTTGAATAGGAGAAGAAGGTCAACAATGACATTGCTGTGTTTGATTCCCACCGTAATGGCGTTTCACCTGTCAACAAGAGATGGCGAACCCCTCATGATAGTGGCAAAACACGAAAGGGGGAGACCTCTCTTGTTTAAGACAACAGAAGGGATCAACAAATGTACCCTTATTGCTATGGACCTGGGTGAAATGTGCGAAGACACCGTTACGTATAAGTGTCCTCTACTGGTTAACACCGAACCTGAAGACATTGACTGCTGGTGCAACCTCACGTCCACCTGGGTCATGTACGGGACATGCACTCAGAACGGAGAACGGAGGCGAGAGAAGCGCTCAGTAGCTTTAACACCACATTCAGGAATGGGATTGGAAACAAGAGCTGAGACATGGATGTCATCGGAAGGGGCTTGGAAACATGCTCAGAGAGTAGAAAGCTGGATACTCAGAAACCCAGGATTCGCGCTCTTGGCAGGATTCATGGCTTATATGATTGGGCAAACAGGAATCCAGCGAATTGTTTTCTTTGTCCTGATGATGCTAGTCGCCCCATCCTACGGAATGCGATGCGTAGGAGTAGGGAACAGAGACTTCGTGGAAGGAGTCTCGGGT',
                    'AGTTGTTAGTCTGTGTGGACCGACAAGGACAGTTCCAAATCGGAAGCTTGCTTAACACAGTTCTAACAGTTTGTTTAAATAGAGAGCAGATCTCTGGAAAAATGAACCAACGAAAAAAGGTGGTCAAACCACCTTTCAATATGCTGAAACGCGAGAGAAACCGCGTATCAACCCCCCAAGGGTTGGTGAAGAGATTCTCAACCGGACTTTTCTCCGGGAAAGGACCTTTGCGGATGGTGCTAGCATTCATCACGTTTTTGCGGGTCCTTTCCATCCCACCAACAGCAGGGATTCTGAAAAGATGGGGACAGTTGAAAAAGAACAAGGCCGTCAAAATACTGATTGGATTCAGGAAGGAAATAGGTCGTATGTTAAACATCTTGAATAGGAGAAGAAGGTCAACAATGACATTGCTGTGTTTGATTCCCACCGTAATGGCGTTTCACCTGTCAACAAGAGATGGCGAACCCCTCATGATAGTAGCAAAACACGAAAGGGGGAGACCTCTCTTGTTTAAGACAACAGAAGGGATCAACAAATGTACCCTCATTGCTATGGACCTGGGTGAAATGTGCGAAGACACTGTCACGTACAAGTGTCCTCTACTGGTTAACACCGAACCTGAAGACATTGATTGCTGGTGCAATCTCACGTCCACCTGGGTCATGTACGGGACATGCACCCAGAATGGAGAACGGAGACGAGAGAAGCGCTCAGTAGCTTTAACACCACATTCAGGAATGGGATTGGAAACAAGAGCTGAGACATGGATGTCATCGGAAGGGGCTTGGAAACATGCTCAAAGAGTAGAAAGCTGGATACTTAGAAACCCAGGGTTCGCGCTCTTGGCAGGATTTATGGCTTATATGATTGGGCAAACAGGAATCCAGCGAACTGTTTTCTTTGTCCTGATGATGCTAGTCGCCCCATCCTACGGAATGCGATGCGTAGGGGTAGGGAACAGAGACTTTGTGGAAGGAGTCTCGGGT'
                ]
            })
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_template = template_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Template CSV",
                data=csv_template,
                file_name="template_inference_sequences.csv",
                mime="text/csv",
                help="Download template CSV dengan kolom sequence"
            )
        
        with col2:
            # Excel template
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                template_data.to_excel(writer, index=False, sheet_name='Sequences')
            excel_template = output.getvalue()
            st.download_button(
                label="📥 Download Template Excel",
                data=excel_template,
                file_name="template_inference_sequences.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download template Excel dengan kolom sequence"
            )
        
        st.markdown("---")
        
        # Input method selection
        input_method = st.radio(
            "Select input method:",
            ["Input Raw Sequence", "Upload File (CSV/Excel)", "Use Sample Dataset"],
            horizontal=True
        )
        
        uploaded_file = None
        # Use session state to persist data
        input_data = st.session_state.input_data if 'input_data' in st.session_state else None
        raw_sequences = st.session_state.raw_sequences if 'raw_sequences' in st.session_state else None
        
        if input_method == "Input Raw Sequence":
            st.subheader("🧬 Input Raw Genomic Sequence")
            
            # Clear button
            if st.session_state.raw_sequences is not None:
                if st.button("🗑️ Clear Current Sequence", help="Clear the currently loaded sequence"):
                    st.session_state.raw_sequences = None
                    st.session_state.input_data = None
                    st.rerun()
            
            # Input options
            input_mode = st.radio(
                "Input mode:",
                ["Single Sequence", "Multiple Sequences"],
                horizontal=True,
                help="Choose to input one sequence or multiple sequences"
            )
            
            if input_mode == "Single Sequence":
                sequence_input = st.text_area(
                    "Enter genomic sequence (DNA):",
                    height=200,
                    help="Paste or type the DNA sequence (ATCG characters). Ambiguous nucleotides will be ignored.",
                    placeholder="Example:\nAGTTGTTAGTCTGTGTGGACCGACAAGGACAGTTCCAAATCGGAAGCTTGCTTAACACAGTTCTAACAGTTTGTTTAAATAGAGAGCAGATCTCTGGAAAAATGAACCAACGAAAAAAGGTGGTCAGACCACCTTTCAATATGCTGAAACGCGAGAGAAACCGCGTATCAACCCCTCAAGGGTTGGTGAAGAGATTCTCAACCGGACTTTTCTCCGGGAAAGGACCTTTGCGGATGGTGCTAGCATTCATCACGTTTTTGCGGGTCCTTTCCATCCCACCAACAGCAGGGATTCTGAAAAGATGGGGACAGTTGAAAAAGAACAAGGCCGTCAAAATACTGATTGGATTCAGGAAGGAGATAGGTCGCATGTTAAACATCTTGAATAGGAGAAGAAGGTCAACAATGACATTGCTGTGTTTGATTCCCACCGTAATGGCGTTTCACCTGTCAACAAGAGATGGCGAACCCCTCATGATAGTGGCAAAACACGAAAGGGGGAGACCTCTCTTGTTTAAGACAACAGAAGGGATCAACAAATGTACCCTCATTGCTATGGACCTGGGTGAAATGTGCGAAGACACTGTCACGTACAAGTGTCCTCTACTGGTTAACACCGAACCTGAAGACATTGACTGCTGGTGCAATCTCACGTCTACTTGGGTCATGTACGGGACATGCACCCAGAACGGAGAACGGAGACGAGAGAAGCGCTCAGTAGCTTTAACACCACATTCAGGAATGGGATTGGAAACAAGAGCTGAGACATGGATGTCATCGGAAGGGGCTTGGAAACATGCTCAGAGAGTAGAAAGCTGGATACTCAGAAACCCAGGATTCGCGCTCTTGGCAGGATTTATGGCTTATATGATTGGGCAAACAGGAATCCAGCGAATTGTTTTCTTTGTCCTGATGATGCTAGTCGCCCCATCCTACGGAATGCGATGCGTAGGGGTAGGGAACAGAGACTTCGTGGAAGGAGTCTCGGGT"
                )
                
                sample_id_input = st.text_input(
                    "Sample ID (optional):",
                    value="SAMPLE_001",
                    help="Optional: Provide a sample ID for this sequence"
                )
                
                if st.button("✅ Process Sequence", type="primary"):
                    if sequence_input and len(sequence_input.strip()) > 0:
                        # Clean sequence (remove whitespace, newlines, convert to uppercase)
                        sequence_clean = ''.join(sequence_input.split()).upper()
                        
                        # Validate sequence (should contain only ATCG and ambiguous nucleotides)
                        valid_chars = set('ATCGNMRWSYKVHDB')
                        if not all(c in valid_chars for c in sequence_clean):
                            invalid_chars = set(sequence_clean) - valid_chars
                            st.error(f"❌ Invalid characters found in sequence: {invalid_chars}")
                            st.info("Sequence should only contain DNA nucleotides: A, T, C, G (and ambiguous nucleotides: N, M, R, W, S, Y, K, V, H, D, B)")
                        elif len(sequence_clean) < 100:
                            st.warning(f"⚠️ Sequence is very short ({len(sequence_clean)} bp). Minimum recommended length is 100 bp.")
                        else:
                            # Create DataFrame
                            raw_sequences = pd.DataFrame({
                                'sample_id': [sample_id_input if sample_id_input else 'SAMPLE_001'],
                                'sequence': [sequence_clean]
                            })
                            
                            # Save to session state
                            st.session_state.raw_sequences = raw_sequences
                            
                            st.success(f"✅ Sequence loaded: {len(sequence_clean)} bp")
                            
                            # Display sequence info
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Sequence Length", f"{len(sequence_clean):,} bp")
                            with col2:
                                gc_count = sequence_clean.count('G') + sequence_clean.count('C')
                                gc_content = gc_count / len(sequence_clean) if len(sequence_clean) > 0 else 0
                                st.metric("GC Content", f"{gc_content:.2%}")
                            with col3:
                                ambiguous_count = sum(sequence_clean.count(c) for c in 'NMRWSYKVHDB')
                                st.metric("Ambiguous Nucleotides", ambiguous_count)
                            
                            # Show sequence preview
                            st.subheader("📄 Sequence Preview")
                            st.text_area(
                                "First 200 bp:",
                                value=sequence_clean[:200] + ('...' if len(sequence_clean) > 200 else ''),
                                height=100,
                                disabled=True
                            )
                            
            else:  # Multiple Sequences
                st.info("Enter multiple sequences, one per line. Each line can be in format: 'sample_id,sequence' or just 'sequence'")
                
                sequences_input = st.text_area(
                    "Enter sequences (one per line):",
                    height=300,
                    help="Format: 'sample_id,sequence' or just 'sequence' (one per line). Example:\nSAMPLE_001,AGTTGTTAGTCTGTGTGGACCGACAAGGACAGTTCCAA...\nSAMPLE_002,GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA...",
                    placeholder="SAMPLE_001,AGTTGTTAGTCTGTGTGGACCGACAAGGACAGTTCCAAATCGGAAGCTTGCTTAACACAGTTCTAACAGTTTGTTTAAATAGAGAGCAGATCTCTGGAAAAATGAACCAACGAAAAAAGGTGGTCAGACCACCTTTCAATATGCTGAAACGCGAGAGAAACCGCGTATCAACCCCTCAAGGGTTGGTGAAGAGATTCTCAACCGGACTTTTCTCCGGGAAAGGACCTTTGCGGATGGTGCTAGCATTCATCACGTTTTTGCGGGTCCTTTCCATCCCACCAACAGCAGGGATTCTGAAAAGATGGGGACAGTTGAAAAAGAACAAGGCCGTCAAAATACTGATTGGATTCAGGAAGGAGATAGGTCGCATGTTAAACATCTTGAATAGGAGAAGAAGGTCAACAATGACATTGCTGTGTTTGATTCCCACCGTAATGGCGTTTCACCTGTCAACAAGAGATGGCGAACCCCTCATGATAGTGGCAAAACACGAAAGGGGGAGACCTCTCTTGTTTAAGACAACAGAAGGGATCAACAAATGTACCCTCATTGCTATGGACCTGGGTGAAATGTGCGAAGACACTGTCACGTACAAGTGTCCTCTACTGGTTAACACCGAACCTGAAGACATTGACTGCTGGTGCAATCTCACGTCTACTTGGGTCATGTACGGGACATGCACCCAGAACGGAGAACGGAGACGAGAGAAGCGCTCAGTAGCTTTAACACCACATTCAGGAATGGGATTGGAAACAAGAGCTGAGACATGGATGTCATCGGAAGGGGCTTGGAAACATGCTCAGAGAGTAGAAAGCTGGATACTCAGAAACCCAGGATTCGCGCTCTTGGCAGGATTTATGGCTTATATGATTGGGCAAACAGGAATCCAGCGAATTGTTTTCTTTGTCCTGATGATGCTAGTCGCCCCATCCTACGGAATGCGATGCGTAGGGGTAGGGAACAGAGACTTCGTGGAAGGAGTCTCGGGT"
                )
                
                if st.button("✅ Process Sequences", type="primary"):
                    if sequences_input and len(sequences_input.strip()) > 0:
                        lines = [line.strip() for line in sequences_input.strip().split('\n') if line.strip()]
                        
                        if len(lines) == 0:
                            st.error("❌ No sequences found!")
                        else:
                            sequences_list = []
                            sample_ids_list = []
                            
                            for idx, line in enumerate(lines, 1):
                                # Check if line contains comma (sample_id,sequence format)
                                if ',' in line:
                                    parts = line.split(',', 1)
                                    sample_id = parts[0].strip()
                                    sequence = parts[1].strip()
                                else:
                                    sample_id = f'SAMPLE_{idx:04d}'
                                    sequence = line.strip()
                                
                                # Clean sequence
                                sequence_clean = ''.join(sequence.split()).upper()
                                
                                if len(sequence_clean) > 0:
                                    sequences_list.append(sequence_clean)
                                    sample_ids_list.append(sample_id)
                            
                            if len(sequences_list) > 0:
                                raw_sequences = pd.DataFrame({
                                    'sample_id': sample_ids_list,
                                    'sequence': sequences_list
                                })
                                
                                # Save to session state
                                st.session_state.raw_sequences = raw_sequences
                                
                                st.success(f"✅ Loaded {len(sequences_list)} sequences")
                                
                                # Display summary
                                st.subheader("📊 Sequences Summary")
                                summary_df = pd.DataFrame({
                                    'Sample ID': sample_ids_list,
                                    'Length (bp)': [len(seq) for seq in sequences_list],
                                    'GC Content': [f"{(seq.count('G') + seq.count('C')) / len(seq):.2%}" if len(seq) > 0 else "0%" for seq in sequences_list]
                                })
                                st.dataframe(summary_df, use_container_width=True)
                            else:
                                st.error("❌ No valid sequences found!")
            
            # Process sequences if available
            if raw_sequences is not None and len(raw_sequences) > 0:
                # Extract features from sequences
                with st.spinner("Extracting features from sequences..."):
                    try:
                        # Get expected feature names from preprocessor if available
                        expected_feature_names = None
                        if st.session_state.models_loaded and st.session_state.inference_pipeline:
                            if st.session_state.inference_pipeline.feature_engineer.feature_names:
                                expected_feature_names = st.session_state.inference_pipeline.feature_engineer.feature_names
                        
                        input_data = extract_features_from_sequences(
                            raw_sequences,
                            sequence_column='sequence',
                            sample_id_column='sample_id' if 'sample_id' in raw_sequences.columns else None,
                            k=3,
                            expected_feature_names=expected_feature_names
                        )
                        
                        # Add mutation features (with default values)
                        input_data = add_mutation_features(input_data)
                        
                        # Add label columns if expected
                        if expected_feature_names:
                            label_cols = ['serotype_label', 'genotype_label', 'known_genotype']
                            for col in label_cols:
                                if col in expected_feature_names and col not in input_data.columns:
                                    if col == 'known_genotype':
                                        input_data[col] = False
                                    else:
                                        input_data[col] = ''
                        
                        # Save to session state
                        st.session_state.input_data = input_data
                        
                        st.success(f"✅ Features extracted: {input_data.shape[1]} features")
                        st.info(f"📊 Extracted features include: gc_content, genome_length, and {len([c for c in input_data.columns if c.startswith('kmer_')])} k-mer frequencies")
                        
                    except Exception as e:
                        st.error(f"Error extracting features: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                        input_data = None
                        st.session_state.input_data = None
        
        elif input_method == "Upload File (CSV/Excel)":
            uploaded_file = st.file_uploader(
                "Choose a CSV or Excel file with genomic sequences",
                type=['csv', 'xlsx', 'xls'],
                help="Upload CSV atau Excel file dengan kolom 'sequence' (dan optional 'sample_id')"
            )
            
            if uploaded_file is not None:
                try:
                    # Determine file type
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    
                    if file_extension == 'csv':
                        raw_sequences = pd.read_csv(uploaded_file)
                        st.success(f"✅ CSV file loaded: {len(raw_sequences)} sequences")
                    elif file_extension in ['xlsx', 'xls']:
                        # Read Excel file
                        excel_file = pd.ExcelFile(uploaded_file)
                        sheet_names = excel_file.sheet_names
                        
                        if len(sheet_names) == 1:
                            raw_sequences = pd.read_excel(uploaded_file, sheet_name=sheet_names[0])
                            st.success(f"✅ Excel file loaded from sheet '{sheet_names[0]}': {len(raw_sequences)} sequences")
                        else:
                            selected_sheet = st.selectbox(
                                "Select sheet to read:",
                                sheet_names,
                                help="Multiple sheets found. Please select which sheet to use."
                            )
                            raw_sequences = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                            st.success(f"✅ Excel file loaded from sheet '{selected_sheet}': {len(raw_sequences)} sequences")
                    
                    # Check if sequence column exists
                    if 'sequence' not in raw_sequences.columns:
                        st.error("❌ Error: 'sequence' column not found in file!")
                        st.info(f"Available columns: {list(raw_sequences.columns)}")
                        st.info("Please ensure your file has a 'sequence' column containing genomic sequences.")
                    else:
                        # Display preview
                        st.subheader("📄 Raw Sequences Preview")
                        preview_df = raw_sequences[['sample_id', 'sequence']].copy() if 'sample_id' in raw_sequences.columns else raw_sequences.copy()
                        preview_df['sequence_length'] = preview_df['sequence'].str.len()
                        preview_df['sequence_preview'] = preview_df['sequence'].str[:50] + '...'
                        st.dataframe(preview_df[['sample_id', 'sequence_length', 'sequence_preview']] if 'sample_id' in preview_df.columns else preview_df[['sequence_length', 'sequence_preview']], use_container_width=True)
                        st.info(f"📊 Data shape: {len(raw_sequences)} sequences")
                        
                        # Extract features from sequences
                        with st.spinner("Extracting features from sequences..."):
                            try:
                                # Get expected feature names from preprocessor if available
                                expected_feature_names = None
                                if st.session_state.models_loaded and st.session_state.inference_pipeline:
                                    if st.session_state.inference_pipeline.feature_engineer.feature_names:
                                        expected_feature_names = st.session_state.inference_pipeline.feature_engineer.feature_names
                                
                                input_data = extract_features_from_sequences(
                                    raw_sequences,
                                    sequence_column='sequence',
                                    sample_id_column='sample_id' if 'sample_id' in raw_sequences.columns else None,
                                    k=3,
                                    expected_feature_names=expected_feature_names  # Use expected features from training
                                )
                                
                                # Add mutation features (with default values)
                                input_data = add_mutation_features(input_data)
                                
                                # Don't add metadata columns here - let feature engineering handle it
                                # The expected_feature_names will ensure alignment after encoding
                                
                                # Add label columns if expected (will be excluded during feature engineering)
                                if expected_feature_names:
                                    label_cols = ['serotype_label', 'genotype_label', 'known_genotype']
                                    for col in label_cols:
                                        if col in expected_feature_names and col not in input_data.columns:
                                            if col == 'known_genotype':
                                                input_data[col] = False
                                            else:
                                                input_data[col] = ''
                                
                                # Save to session state
                                st.session_state.input_data = input_data
                                st.session_state.raw_sequences = raw_sequences
                                
                                st.success(f"✅ Features extracted: {input_data.shape[1]} features")
                                st.info(f"📊 Extracted features include: gc_content, genome_length, and {len([c for c in input_data.columns if c.startswith('kmer_')])} k-mer frequencies")
                                
                            except Exception as e:
                                st.error(f"Error extracting features: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                                input_data = None
                                st.session_state.input_data = None
                    
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        else:  # Use Sample Dataset
            raw_sequences_path = Path('dataset/raw_sequences.csv')
            if raw_sequences_path.exists():
                st.info("Using raw sequences from 'dataset/raw_sequences.csv'")
                try:
                    raw_sequences = pd.read_csv(raw_sequences_path)
                    st.success(f"✅ Raw sequences loaded: {len(raw_sequences)} sequences")
                    
                    # Display preview
                    if 'sequence' in raw_sequences.columns:
                        preview_df = raw_sequences[['sample_id', 'sequence']].copy() if 'sample_id' in raw_sequences.columns else raw_sequences.copy()
                        preview_df['sequence_length'] = preview_df['sequence'].str.len()
                        preview_df['sequence_preview'] = preview_df['sequence'].str[:50] + '...'
                        st.dataframe(preview_df[['sample_id', 'sequence_length', 'sequence_preview']] if 'sample_id' in preview_df.columns else preview_df[['sequence_length', 'sequence_preview']], use_container_width=True)
                        
                        # Extract features from sequences
                        with st.spinner("Extracting features from sequences..."):
                            # Get expected feature names from preprocessor if available
                            expected_feature_names = None
                            if st.session_state.models_loaded and st.session_state.inference_pipeline:
                                if st.session_state.inference_pipeline.feature_engineer.feature_names:
                                    expected_feature_names = st.session_state.inference_pipeline.feature_engineer.feature_names
                            
                            input_data = extract_features_from_sequences(
                                raw_sequences,
                                sequence_column='sequence',
                                sample_id_column='sample_id' if 'sample_id' in raw_sequences.columns else None,
                                k=3,
                                expected_feature_names=expected_feature_names  # Use expected features from training
                            )
                            
                            # Add mutation features (with default values)
                            input_data = add_mutation_features(input_data)
                            
                            # Don't add metadata columns here - let feature engineering handle it
                            # The expected_feature_names will ensure alignment after encoding
                            
                            # Add label columns if expected (will be excluded during feature engineering)
                            if expected_feature_names:
                                label_cols = ['serotype_label', 'genotype_label', 'known_genotype']
                                for col in label_cols:
                                    if col in expected_feature_names and col not in input_data.columns:
                                        if col == 'known_genotype':
                                            input_data[col] = False
                                        else:
                                            input_data[col] = ''
                            
                            # Save to session state
                            st.session_state.input_data = input_data
                            st.session_state.raw_sequences = raw_sequences
                            
                            st.success(f"✅ Features extracted: {input_data.shape[1]} features")
                    else:
                        st.error("❌ Error: 'sequence' column not found in raw_sequences.csv!")
                        
                except Exception as e:
                    st.error(f"Error loading raw sequences: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            else:
                st.warning("Raw sequences file not found at 'dataset/raw_sequences.csv'!")
        
        # Task selection
        if input_data is not None and len(input_data) > 0:
            st.markdown("---")
            st.subheader("Select Prediction Tasks")
            
            # Show extracted features info
            st.info(f"✅ Ready for inference: {len(input_data)} samples with {input_data.shape[1]} features")
            
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
                            
                            # Ensure input_data has sample_id
                            if 'sample_id' not in input_data.columns:
                                input_data['sample_id'] = [f'SAMPLE_{i+1:04d}' for i in range(len(input_data))]
                            
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
        
        Input data hanya perlu **kolom sequence** (dan optional `sample_id`):
        - `sample_id`: ID unik untuk setiap sample (optional, akan dibuat otomatis jika tidak ada)
        - `sequence`: Genomic sequence dalam format DNA (ATCG)
        
        **Supported file formats:**
        - CSV (.csv)
        - Excel (.xlsx, .xls)
        
        **Template:** Download template CSV/Excel dari tab "Upload Data" untuk melihat format yang diperlukan.
        
        **Note:** Features akan diextract otomatis dari sequence (k-mer frequencies, GC content, dll).
        
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

