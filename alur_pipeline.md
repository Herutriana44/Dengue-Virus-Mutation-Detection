Baik 👍
Di bawah ini saya susun **alur pipeline machine learning (ML) yang lengkap, sistematis, dan siap ditulis ke paper** untuk **deteksi mutasi Dengue virus → genotipe baru → potensi serotipe baru**, berbasis **dataset tabular** yang sudah Anda desain.

Saya bagi menjadi **alur konseptual + alur operasional (step-by-step)**.

---

# A. ALUR KONSEPTUAL (HIGH-LEVEL)

```
GenBank
  ↓
Preprocessing & QC
  ↓
Alignment & Mutation Calling
  ↓
Feature Engineering
  ↓
Feature Fusion (Tabular)
  ↓
ML Training (Closed-set)
  ↓
Novelty / Open-set Detection
  ↓
Biological Validation
```

Ini **alur yang reviewer pahami dengan cepat**.

---

# B. ALUR PIPELINE MACHINE LEARNING (DETAIL)

## STAGE 1 — Dataset Preparation (ML View)

### Input tables:

* `sample_metadata.csv`
* `sequence_features.csv`
* `mutation_profile.csv`
* `protein_level_features.csv`
* `label_table.csv`

### Langkah:

1. **Merge tables**

```text
sample_id → left join
```

2. **Filtering ML**

* Buang missing label
* Buang outlier genome_length ekstrem

📌 Output:

```
ml_dataset_raw.csv
```

---

## STAGE 2 — Feature Selection & Encoding

### 2.1 Feature grouping

```
X_seq   = k-mer, GC, codon bias
X_mut   = total_mut, nonsyn, mut_density
X_prot  = protein-level features
X_meta  = year (scaled), region (encoded)
```

### 2.2 Encoding

* Categorical → OneHot / Target Encoding
* Numerical → StandardScaler

📌 **Label tidak boleh ikut scaling**

---

## STAGE 3 — Problem Formulation (3 TASK ML)

---

### TASK 1 — Closed-set Classification (Baseline)

**Tujuan**
Menunjukkan bahwa fitur biologis memang informatif.

**Target**

* `y = serotype` atau `genotype`

**Split**

* Stratified k-fold (by serotype)

**Model**

* Random Forest
* XGBoost

📊 **Evaluasi**

* Accuracy
* Macro-F1
* Confusion matrix

📌 *Paper narrative*:

> “The model reliably distinguishes known serotypes, validating the biological relevance of the extracted features.”

---

### TASK 2 — Genotype Novelty Detection ⭐

**Tujuan**
Deteksi **genotipe yang tidak dikenal saat training**.

#### Dataset setup:

* Train: genotype A, B, C
* Test: genotype D (held-out)

#### Feature:

```
X = X_seq + X_mut
```

#### Model:

* Isolation Forest
* One-Class SVM
* Autoencoder (optional)

📊 **Output**

| sample_id | anomaly_score | novel_genotype |
| --------- | ------------- | -------------- |

📌 *Interpretasi*:

* anomaly_score tinggi → pola mutasi berbeda

---

### TASK 3 — Potensi Serotipe Baru (Open-set)

**Tujuan**
Deteksi **divergensi genetika ekstrem**.

#### Dataset setup:

* Train: DENV-1,2,3
* Test: DENV-4 + synthetic mutants

#### Model:

* OpenMax
* Autoencoder + threshold
* Distance-based classifier

📊 **Output**

```
Known serotype? → No
Genetic divergence → High
```

⚠️ **Klaim ilmiah aman**:

> “Genetically divergent lineage inconsistent with known serotypes.”

---

## STAGE 4 — Model Training Flow (OPERATIONAL)

```
for each task:
    load dataset
    preprocess features
    train model
    validate
    save model
```

---

## STAGE 5 — Thresholding & Decision Logic

### Contoh:

```text
if anomaly_score > T:
    classify as novel
else:
    known class
```

Threshold ditentukan dari:

* 95th percentile training data
* ROC-based cutoff

---

## STAGE 6 — Model Interpretation (WAJIB)

### 6.1 Feature importance

* Random Forest importance
* SHAP values

### 6.2 Biological mapping

* High-importance k-mers → genome regions
* Mutation hotspots → E / NS1

📌 Ini **membuat reviewer biologi puas**.

---

## STAGE 7 — Biological Validation Loop

Untuk sampel “novel”:

1. Phylogenetic tree
2. Protein impact analysis
3. Literature cross-check

📌 ML **tidak berdiri sendiri**.

---

# C. PIPELINE MACHINE LEARNING (FLOWCHART RINGKAS)

```
Input CSVs
   ↓
Feature Merge
   ↓
Scaling & Encoding
   ↓
Baseline Classifier
   ↓
Novelty Detector
   ↓
Threshold Decision
   ↓
Interpretation & Validation
```

---

# D. RINGKASAN MODEL YANG DIREKOMENDASIKAN

| Task              | Model            | Kenapa                |
| ----------------- | ---------------- | --------------------- |
| Baseline          | RF / XGBoost     | Stabil, interpretable |
| Genotype novelty  | Isolation Forest | Cocok high-dim        |
| Serotype open-set | Autoencoder      | Deteksi divergensi    |
| Interpretability  | SHAP             | Reviewer-friendly     |

---

# E. KENAPA PIPELINE INI KUAT UNTUK PUBLIKASI?

✔ Mengikuti standar bioinformatika
✔ ML bukan klaim berlebihan
✔ Bisa direplikasi
✔ Bisa dikembangkan ke surveillance real-time
✔ Aman dari kritik “overclaiming serotype”

---

## NEXT STEP (PILIH SATU)

1️⃣ Saya buatkan **diagram pipeline (siap masuk paper)**
2️⃣ Saya buatkan **kode ML (scikit-learn) end-to-end**
3️⃣ Saya bantu **menulis section “Machine Learning Pipeline” (Methods)**

👉 Anda mau lanjut ke **kode ML** atau **penulisan Methods** dulu?
