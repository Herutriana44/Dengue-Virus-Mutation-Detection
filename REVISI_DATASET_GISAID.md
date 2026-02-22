# Revisi Program - Dataset GISAID

## Ringkasan Perubahan

Program telah direvisi untuk mendukung dataset GISAID:
- **Tanpa filter**: dataset diasumsikan sudah sesuai (tidak ada filter serotipe/lokasi)
- **Merge**: GISAID digabung dengan dataset lain (sample_metadata, raw_sequences, dll) menggunakan JOIN/UNION
- **Output**: 1 tabel dataset terpadu (`merged_dataset.csv`)

### Revisi Terbaru (Join Key & Envelope)

1. **Join key GISAID**: Gunakan kolom **"NCBI Accession ID"** (bukan Accession ID) untuk join dengan dataset lain
2. **Deteksi mutasi**: Gunakan **envelope_sequence** saja (bukan full genome)
   - GISAID: ekstrak region envelope (posisi 937–2423) dari full Sequence
   - Legacy: gunakan kolom `envelope_sequence` dari `raw_sequences.csv`
   - Mutasi: hanya hitung protein E_ (envelope) dari AA Substitutions

## File: `src/gisaid_preprocessor.py`

Modul preprocessing untuk data GISAID:
- Load `from_gisaid_data.csv` (tanpa filter)
- Extract fitur: k-mer, GC content, mutation count dari AA Substitutions
- **Merge dengan dataset lain**: JOIN/UNION dengan sample_metadata, sequence_features, mutation_profile, label_table
- Output: `merged_dataset.csv` + file intermediate untuk pipeline

## File yang Diubah

### `src/data_cleaning.py`
- Tambah parameter `use_gisaid`: saat True, load dari `from_gisaid_data.csv`
- Auto-detect dan jalankan GISAID preprocessor
- Target label: **genotype** (bukan serotype) karena hanya 1 serotipe

### `src/main_pipeline.py`
- Auto-detect GISAID dari keberadaan `from_gisaid_data.csv`
- Task 1 (Baseline): klasifikasi **genotype** (I, II, IV, dll)
- Task 2 (Novelty): deteksi genotype novel - tetap berjalan
- Task 3 (Open-set): **di-skip** untuk GISAID (hanya 1 serotipe)
- CLI: `--gisaid` untuk memaksa mode GISAID

### `src/inference.py`
- Support input format GISAID (kolom `Sequence`, `Accession ID`, `AA Substitutions`)
- Auto-extract fitur dari sequence jika input mentah
- Map `Accession ID` → `sample_id`

### `run_pipeline.py`
- Auto-detect mode GISAID
- Task open_set di-skip otomatis untuk dataset DENV-2

### `run_inference.py`
- Auto-detect GISAID saat load dari directory
- Default tasks: `baseline`, `novelty` (tanpa open_set)

## Cara Penggunaan

### 1. Persiapan Data
Pastikan `dataset/from_gisaid_data.csv` ada (hasil scraping dari `GetDataDengueVirusGenome.ipynb`).

### 2. Jalankan Pipeline
```bash
# Auto-detect GISAID (jika from_gisaid_data.csv ada)
python run_pipeline.py

# Atau dengan flag eksplisit
python -m src.main_pipeline --gisaid
```

### 3. Inference
```bash
# Dari folder dataset (auto-detect GISAID)
python run_inference.py --input dataset --output hasil_inference.csv

# Dari file CSV GISAID mentah
python run_inference.py --input dataset/from_gisaid_data.csv --output hasil.csv
```

## Alur Data

```
from_gisaid_data.csv (GISAID)     sample_metadata.csv, raw_sequences, dll
    ↓                                    ↓
Extract: k-mer, GC content, mutation     Load legacy
    ↓                                    ↓
    └─────────── JOIN / UNION ───────────┘
                    ↓
            merged_dataset.csv (1 tabel)
                    ↓
            [data_cleaning - preprocessing pipeline]
                    ↓
            ml_dataset_raw.csv
                    ↓
            [main_pipeline: feature engineering, tasks]
```

## Catatan

- **Tanpa filter**: tidak ada filter serotipe/lokasi - dataset diasumsikan sudah sesuai
- **Merge**: GISAID + legacy digabung jadi 1 tabel (UNION jika beda sample_id, JOIN jika ada overlap)
- Preprocessing lanjut mengikuti pipeline proyek sebelumnya
