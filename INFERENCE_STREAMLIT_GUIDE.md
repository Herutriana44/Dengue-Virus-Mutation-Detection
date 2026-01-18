# Streamlit Inference App Guide

Panduan penggunaan aplikasi Streamlit untuk melakukan inference menggunakan model yang sudah di-train.

## Instalasi

Pastikan semua dependencies terinstall:

```bash
pip install -r requirements.txt
```

## Menjalankan Aplikasi

```bash
python run_streamlit_inference.py
```

Atau langsung menggunakan Streamlit:

```bash
streamlit run src/streamlit_inference.py --server.port 8502
```

Aplikasi akan terbuka di browser pada `http://localhost:8502`

## Fitur Aplikasi

### 1. 📤 Upload Data Tab

#### Input Methods:
- **Upload CSV File**: Upload file CSV dengan format yang sama seperti dataset training
- **Use Sample Dataset**: Gunakan dataset dari folder `dataset/` untuk testing

#### Task Selection:
- ✅ **Baseline Classification**: Prediksi serotipe/genotipe
- ✅ **Novelty Detection**: Deteksi genotipe baru
- ✅ **Open-set Detection**: Deteksi potensi serotipe baru

### 2. 📊 View Results Tab

#### Summary Statistics:
- Total samples yang diprediksi
- Unique serotypes yang terdeteksi
- Novel genotypes yang ditemukan
- Potential new serotypes yang terdeteksi

#### Detailed Results:
- Tabel lengkap dengan semua prediksi
- Download results sebagai CSV

#### Visualizations:
- **Serotype Distribution**: Bar chart distribusi prediksi serotipe
- **Novel Genotype Detection**: Pie chart dan histogram anomaly scores
- **Open-set Detection**: Pie chart dan histogram reconstruction errors

#### Alerts:
- Warning untuk novel genotypes yang terdeteksi
- Warning untuk potential new serotypes yang terdeteksi

### 3. ℹ️ About Tab

Informasi tentang aplikasi, format input/output, dan model yang digunakan.

## Workflow

1. **Load Models**: Klik tombol "🔄 Load Models" di sidebar
2. **Upload Data**: Pilih metode input (upload CSV atau gunakan sample dataset)
3. **Select Tasks**: Pilih task yang ingin dijalankan
4. **Run Inference**: Klik tombol "🚀 Run Inference"
5. **View Results**: Lihat hasil di tab "View Results"
6. **Download**: Download hasil sebagai CSV jika diperlukan

## Format Input

Input CSV harus memiliki kolom:
- `sample_id`: ID unik untuk setiap sample
- Semua features yang digunakan saat training (sequence features, mutation profile, metadata, dll)

## Format Output

Output CSV berisi:
- `sample_id`: ID sample
- `predicted_serotype`: Prediksi serotipe (jika baseline task)
- `prediction_confidence`: Confidence score (jika baseline task)
- `is_novel_genotype`: Boolean flag untuk novel genotype (jika novelty task)
- `anomaly_score`: Anomaly score untuk novelty detection
- `novelty_threshold`: Threshold yang digunakan
- `is_potential_new_serotype`: Boolean flag untuk potential new serotype (jika open-set task)
- `reconstruction_error`: Reconstruction error untuk open-set detection
- `open_set_threshold`: Threshold yang digunakan

## Catatan Penting

1. **Model Files**: Pastikan model sudah di-train dan tersimpan di `results/models/`
2. **Preprocessor**: File `results/preprocessor.pkl` harus ada untuk baseline classification
3. **Feature Consistency**: Input data harus memiliki features yang sama dengan training data
4. **Port**: Aplikasi inference menggunakan port 8502 (beda dengan EDA dashboard yang menggunakan port 8501)

## Troubleshooting

### Models tidak bisa di-load
- Pastikan file model ada di `results/models/`
- Check console untuk error messages
- Pastikan semua dependencies terinstall

### Error saat inference
- Pastikan input data memiliki format yang benar
- Check apakah semua features yang diperlukan ada
- Pastikan preprocessor sudah di-save saat training

### Results tidak muncul
- Pastikan inference sudah selesai dijalankan
- Check tab "View Results"
- Refresh halaman jika perlu

