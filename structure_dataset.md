Baik, ini arah yang **sangat solid** untuk riset jangka panjang 👍
Saya akan jelaskan **apa saja data yang dibutuhkan dari NCBI GenBank** dan **bagaimana struktur dataset tabular-nya** agar:

* bisa dipakai **ML klasik (scikit-learn, XGBoost)**
* **future-proof** untuk deteksi **genotipe / serotipe baru**
* reviewer *bioinformatika* dan *ML* sama-sama “ngeh”

Saya susun **dari konsep → struktur tabel → contoh konkret**.

---

# 1. Tujuan Model → Menentukan Jenis Data

Anda punya **2 target besar**:

### A. **Deteksi mutasi**

→ problem **feature extraction + anomaly / pattern detection**

### B. **Deteksi genotipe / serotipe baru**

→ problem **classification + open-set / novelty detection**

Artinya dataset harus memuat:

1. **informasi sekuens**
2. **informasi mutasi (per posisi)**
3. **label biologis**
4. **metadata evolusioner**

---

# 2. Data yang Perlu Diambil dari NCBI GenBank

Minimal Anda butuh **4 layer data**:

---

## LAYER 1 — Metadata Isolat (WAJIB)

Dari GenBank fields:

| Field GenBank       | Digunakan sebagai  |
| ------------------- | ------------------ |
| Accession ID        | primary key        |
| Serotype            | label utama        |
| Genotype (jika ada) | secondary label    |
| Country / Region    | fitur epidemiologi |
| Collection Year     | fitur temporal     |
| Host                | kontrol bias       |
| Genome length       | QC feature         |

---

## LAYER 2 — Sekuens Nukleotida

* Full genome (~10,700 nt) **atau**
* Targeted gene:

  * **E gene** (paling umum)
  * NS1, NS3, NS5

📌 **Saran reviewer**:

> Gunakan **E gene** untuk baseline, lalu full genome sebagai eksperimen lanjutan.

---

## LAYER 3 — Mutasi / Variasi

Hasil dari:

* Multiple Sequence Alignment (MAFFT / MUSCLE)
* Dibandingkan terhadap **reference strain per serotipe**

Jenis mutasi:

* SNP
* Insertion / deletion
* Synonymous vs non-synonymous

---

## LAYER 4 — Fitur Numerik (ML-ready)

Ekstraksi dari sekuens:

* k-mer frequency
* GC content
* Codon usage bias
* Amino acid composition
* Mutation density

---

# 3. Struktur Dataset Tabular (FINAL FORM)

## TABEL 1 — `sample_metadata.csv`

**1 baris = 1 isolat virus**

| Kolom         | Tipe        | Keterangan                   |
| ------------- | ----------- | ---------------------------- |
| sample_id     | string      | Accession ID                 |
| serotype      | categorical | DENV-1 / 2 / 3 / 4           |
| genotype      | categorical | misal: Asian I, Cosmopolitan |
| country       | categorical | Indonesia, Brazil            |
| region        | categorical | Asia, Americas               |
| year          | int         | Tahun isolasi                |
| host          | categorical | Human / Mosquito             |
| genome_length | int         | QC                           |
| is_complete   | boolean     | full genome?                 |

📌 **Dipakai untuk**

* Stratified split
* Bias analysis
* Domain adaptation

---

## TABEL 2 — `sequence_features.csv`

**Fitur numerik hasil ekstraksi sekuens**

| Kolom          | Tipe   | Keterangan |
| -------------- | ------ | ---------- |
| sample_id      | string | join key   |
| gc_content     | float  | 0–1        |
| kmer_AAA       | float  | freq       |
| kmer_AAC       | float  | freq       |
| ...            | ...    | ...        |
| codon_bias_ENC | float  |            |
| aa_entropy     | float  |            |

📌 Biasanya **1.000–10.000 kolom**

---

## TABEL 3 — `mutation_profile.csv`

**Mutasi berbasis posisi (reference-based)**

| Kolom     | Tipe   | Keterangan |
| --------- | ------ | ---------- |
| sample_id | string | join key   |
| pos_123   | binary | mutated?   |
| pos_124   | binary | mutated?   |
| pos_125   | binary | mutated?   |

Alternatif ringkas:

| Kolom           | Tipe  |
| --------------- | ----- |
| total_mutations | int   |
| nonsyn_mut      | int   |
| syn_mut         | int   |
| mut_density     | float |

📌 Untuk ML klasik → **pakai versi ringkas**

---

## TABEL 4 — `protein_level_features.csv`

Jika diterjemahkan ke protein:

| Kolom            | Tipe   |
| ---------------- | ------ |
| sample_id        | string |
| E_protein_length | int    |
| E_hydrophobicity | float  |
| E_charge         | float  |
| NS1_entropy      | float  |

---

## TABEL 5 — `label_table.csv`

Untuk fleksibilitas eksperimen:

| Kolom          | Tipe    |
| -------------- | ------- |
| sample_id      | string  |
| serotype_label | int     |
| genotype_label | int     |
| known_genotype | boolean |

📌 `known_genotype = False` → **simulasi serotipe baru**

---

# 4. Final Dataset untuk Training ML

Biasanya Anda **merge jadi satu table besar**:

```
final_dataset.csv
```

Struktur:

```
[sample_id]
[metadata]
[sequence_features]
[mutation_summary_features]
[protein_features]
[label]
```

Contoh:

| sample_id | year | gc | mut_density | kmer_AAA | E_entropy | serotype |
| --------- | ---- | -- | ----------- | -------- | --------- | -------- |

---

# 5. Strategi Deteksi Serotipe / Genotipe BARU 🧠

Ini penting untuk novelty detection:

### Strategi dataset:

* Train hanya pada **DENV-1,2,3**
* Test pada **DENV-4**
* Atau hold-out genotype tertentu

### Label tambahan:

```text
known_class = 1 / 0
```

### Model cocok:

* One-Class SVM
* Isolation Forest
* Autoencoder
* Open-set classifier

---

# 6. Tools yang Cocok (Reviewer-Friendly)

| Tahap              | Tool                  |
| ------------------ | --------------------- |
| Download           | NCBI Virus / Entrez   |
| Alignment          | MAFFT                 |
| Mutation calling   | snp-sites             |
| Feature extraction | iFeature, BioPython   |
| ML                 | scikit-learn, XGBoost |
| Visualization      | Nextstrain (validasi) |

---

# 7. Kenapa Struktur Ini Kuat untuk Paper?

✔ Bisa direproduksi
✔ Bisa diperluas ke strain baru
✔ Bisa digabung metadata epidemiologi
✔ Cocok untuk **Scopus Q2–Q1 bioinformatics / AI in biology**

---

## Langkah Selanjutnya (opsional)

Kalau Anda mau, saya bisa:

1. Membuat **contoh CSV dummy (realistic)**
2. Menyusun **pipeline preprocessing GenBank → CSV**
3. Mendesain **eksperimen deteksi serotipe baru**

Tinggal bilang:
👉 *mau fokus ke ML klasik atau deep learning dulu?*
