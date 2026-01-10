# Inference Guide

Panduan penggunaan modul inference untuk melakukan prediksi menggunakan model yang sudah di-train.

## Instalasi

Pastikan semua dependencies terinstall:

```bash
pip install -r requirements.txt
```

## Menjalankan Inference

### 1. Menggunakan Script Command Line

```bash
# Inference pada dataset directory
python run_inference.py --input dataset --output predictions.csv

# Inference pada file CSV
python run_inference.py --input new_samples.csv --output predictions.csv

# Hanya baseline classification
python run_inference.py --input dataset --tasks baseline --output baseline_predictions.csv

# Semua tasks
python run_inference.py --input dataset --tasks baseline novelty open_set --output all_predictions.csv
```

### 2. Menggunakan Python API

```python
from src.inference import InferencePipeline
import pandas as pd

# Initialize pipeline
inference = InferencePipeline(models_dir='results/models')

# Load models
inference.load_models()

# Prepare input data (DataFrame atau path ke CSV)
input_data = pd.read_csv('new_samples.csv')

# Run inference
results = inference.run_full_inference(
    input_data,
    is_dataframe=True,
    tasks=['baseline', 'novelty', 'open_set']
)

# Format results
df_results = inference.format_results(results, output_format='dataframe')

# Save results
inference.save_results(results, output_path='predictions.csv')
```

## Format Input Data

Input data harus memiliki kolom yang sama dengan dataset training:
- `sample_id`: ID unik untuk setiap sample
- Kolom-kolom features yang digunakan saat training (sequence features, mutation profile, dll)

## Output Format

Output berupa CSV dengan kolom:
- `sample_id`: ID sample
- `predicted_serotype`: Prediksi serotipe (jika baseline task dijalankan)
- `prediction_confidence`: Confidence score prediksi
- `is_novel_genotype`: Apakah sample adalah novel genotype (jika novelty task dijalankan)
- `anomaly_score`: Anomaly score untuk novelty detection
- `is_potential_new_serotype`: Apakah sample berpotensi serotipe baru (jika open-set task dijalankan)
- `reconstruction_error`: Reconstruction error untuk open-set detection

## Catatan Penting

1. **Preprocessor**: Pastikan file `results/preprocessor.pkl` ada (disimpan saat training)
2. **Model Files**: Semua model harus ada di `results/models/`
3. **Feature Consistency**: Input data harus memiliki features yang sama dengan training data

