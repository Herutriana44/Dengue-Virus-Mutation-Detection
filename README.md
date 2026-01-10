# Dengue Virus Mutation Detection - ML Pipeline

Dataset didapat menggunakan kode menggunakan link [google colab ini](https://colab.research.google.com/drive/1CE1BqGXitpgmoiQsubzX_7wK7gcgz-xG?usp=sharing)

Pipeline Machine Learning lengkap untuk deteksi mutasi Dengue virus, genotipe baru, dan potensi serotipe baru berdasarkan dataset tabular.

## Struktur Pipeline

Pipeline ini mengikuti alur yang dijelaskan dalam `alur_pipeline.md`:

### STAGE 1: Dataset Preparation
- Merge semua tabel CSV (metadata, sequence features, mutation profile, labels)
- Filter missing labels
- Filter outliers berdasarkan genome_length

### STAGE 2: Feature Selection & Encoding
- Group features (sequence, mutation, protein, metadata)
- Encode categorical features (One-hot / Label encoding)
- Scale numerical features (StandardScaler)

### TASK 1: Closed-set Classification (Baseline)
- Klasifikasi serotipe/genotipe menggunakan Random Forest dan XGBoost
- Stratified k-fold cross-validation
- Evaluasi dengan accuracy, F1-score, confusion matrix

### TASK 2: Genotype Novelty Detection
- Deteksi genotipe yang tidak dikenal saat training
- Menggunakan Isolation Forest dan One-Class SVM
- Anomaly score untuk identifikasi genotipe baru

### TASK 3: Serotype Open-set Detection
- Deteksi divergensi genetika ekstrem (potensi serotipe baru)
- Menggunakan Autoencoder dan distance-based classifier
- Reconstruction error untuk identifikasi open-set samples

### STAGE 6: Model Interpretation
- Feature importance analysis
- SHAP values untuk interpretasi model
- Mapping features ke interpretasi biologis

## Instalasi

1. Clone repository atau download kode
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Penggunaan

### Menjalankan Full Pipeline

```bash
python src/main_pipeline.py --dataset-dir dataset --output-dir results
```

### Menjalankan Task Tertentu

```bash
# Hanya baseline classification
python src/main_pipeline.py --tasks baseline

# Baseline + Novelty Detection
python src/main_pipeline.py --tasks baseline novelty

# Semua tasks
python src/main_pipeline.py --tasks baseline novelty open_set interpretation
```

### Menjalankan Stage Tertentu

```bash
# Stage 1: Data Preparation
python src/main_pipeline.py --stage 1

# Stage 2: Feature Engineering
python src/main_pipeline.py --stage 2
```

### Menggunakan Modul Secara Terpisah

#### 1. Data Cleaning

```python
from src.data_cleaning import DataCleaner

cleaner = DataCleaner(dataset_dir='dataset')
cleaned_df = cleaner.run_full_cleaning()
```

#### 2. Feature Engineering

```python
from src.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
engineer.group_features(df)
X, y, feature_names = engineer.prepare_features(df, target_column='serotype')
X_scaled = engineer.scale_features(X, fit=True)
```

#### 3. Baseline Classification

```python
from src.task1_baseline_classification import BaselineClassifier

classifier = BaselineClassifier()
classifier.train_random_forest(X_train, y_train)
classifier.train_xgboost(X_train, y_train)
cv_results = classifier.cross_validate(X_train, y_train)
metrics = classifier.evaluate(X_test, y_test)
```

#### 4. Novelty Detection

```python
from src.task2_novelty_detection import NoveltyDetector

detector = NoveltyDetector()
X_train_df, X_test_df, y_test = detector.prepare_novelty_data(
    df, train_genotypes=['Genotype_A'], test_genotypes=['Genotype_B']
)
X_train, _ = detector.extract_features(X_train_df)
detector.train_isolation_forest(X_train)
metrics, scores = detector.evaluate(X_test, y_test)
```

#### 5. Open-set Detection

```python
from src.task3_open_set_detection import OpenSetDetector

detector = OpenSetDetector()
X_train_df, X_test_df, y_test = detector.prepare_open_set_data(
    df, train_serotypes=['DENV-1', 'DENV-2'], test_serotypes=['DENV-4']
)
X_train, _ = detector.extract_features(X_train_df)
detector.train_autoencoder(X_train)
metrics, scores = detector.evaluate(X_test, y_test)
```

#### 6. Model Interpretation

```python
from src.model_interpretation import ModelInterpreter

interpreter = ModelInterpreter()
importance_df = interpreter.plot_feature_importance(model, feature_names)
shap_values = interpreter.compute_shap_values(model, X, feature_names)
report = interpreter.generate_interpretation_report(model, X, y, feature_names)
```

## Struktur Dataset

Pipeline mengharapkan dataset dalam format berikut di folder `dataset/`:

- `sample_metadata.csv`: Metadata isolat (sample_id, serotype, genotype, country, year, etc.)
- `sequence_features.csv`: Fitur sekuens (k-mer frequencies, GC content, etc.)
- `mutation_profile.csv`: Profil mutasi (mutation_density, length_diff, etc.)
- `label_table.csv`: Label untuk training (sample_id, serotype, genotype, known_genotype)

## Output

Pipeline menghasilkan output di folder `results/`:

- `ml_dataset_raw.csv`: Dataset yang sudah dibersihkan dan di-merge
- `models/`: Model yang sudah di-train (baseline_classifier.pkl, novelty_detector.pkl, open_set_detector.pkl)
- `interpretation/`: Hasil interpretasi model (feature importance plots, SHAP plots, report)

## Requirements

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- xgboost >= 1.7.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- shap >= 0.41.0 (optional, untuk interpretasi)

## Catatan

- Pipeline menggunakan stratified splitting untuk menjaga distribusi kelas
- Threshold untuk novelty/open-set detection ditentukan dari 95th percentile training data
- SHAP analysis membutuhkan waktu lebih lama untuk dataset besar (menggunakan sampling otomatis)

## Lisensi

Proyek ini untuk keperluan penelitian akademik.

