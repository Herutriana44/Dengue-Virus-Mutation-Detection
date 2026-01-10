# Streamlit Dashboard Guide

Panduan penggunaan dashboard Streamlit untuk EDA dataset Dengue Virus.

## Instalasi

Install dependencies:

```bash
pip install -r requirements.txt
```

## Menjalankan Dashboard

```bash
python run_streamlit.py
```

Atau langsung menggunakan Streamlit:

```bash
streamlit run src/streamlit_dashboard.py
```

Dashboard akan terbuka di browser pada `http://localhost:8501`

## Fitur Dashboard

### 1. 📈 Overview
- Ringkasan dataset
- Metrics kualitas data
- Distribusi serotipe
- Data processing pipeline

### 2. 🔬 Raw Data Analysis
- Informasi dataset mentah
- Missing values analysis
- Preview data
- Descriptive statistics
- Distribusi genome length

### 3. ✨ Cleaned Data Analysis
- Perbandingan raw vs cleaned data
- Data quality improvements
- Feature comparison

### 4. 🧬 Sequence Features
- GC content analysis
- K-mer frequency analysis
- Top k-mers
- K-mer correlation matrix

### 5. 🔀 Mutation Analysis
- Mutation density distribution
- Genome length difference
- Mutation density by serotype

### 6. 📊 Serotype & Genotype
- Distribusi serotipe dan genotipe
- Cross-tabulation
- Heatmap serotipe-genotipe

### 7. 🌍 Geographic Analysis
- Distribusi sample per negara
- Serotipe per negara
- Distribusi regional

### 8. 📅 Temporal Analysis
- Trend koleksi sample over time
- Trend serotipe over time

### 9. 🔍 Feature Importance
- Feature importance dari model (jika model sudah di-train)
- Top features yang paling penting

### 10. 📋 Summary Report
- Comprehensive summary
- Key findings
- Export report

## Catatan

- Dashboard menggunakan caching untuk performa yang lebih baik
- Pastikan dataset files ada di folder `dataset/`
- Untuk melihat feature importance, pastikan model sudah di-train

