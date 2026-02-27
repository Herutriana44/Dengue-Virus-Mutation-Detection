"""
GISAID Data Preprocessor
Modul untuk memproses data dari from_gisaid_data.csv dan menggabungkan dengan dataset lain.
- Join GISAID: gunakan kolom "NCBI Accession ID" (bukan Accession ID)
- Deteksi mutasi: gunakan envelope_sequence saja (bukan full genome)
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import logging
from sequence_feature_extractor import extract_features_from_sequences

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Posisi envelope gene pada genome DENV (0-based, untuk slicing)
ENVELOPE_START = 937
ENVELOPE_END = 2423
ENVELOPE_REF_LEN = ENVELOPE_END - ENVELOPE_START  # 1486


def parse_aa_substitutions(aa_substitutions_str, envelope_only=False):
    """
    Parse AA Substitutions string dari GISAID format: (NS5_I637A,E_Q52H,...)
    
    Args:
        aa_substitutions_str: String AA Substitutions
        envelope_only: Jika True, hanya hitung mutasi protein E_ (envelope)
    
    Returns:
        int: jumlah mutasi (count)
    """
    if pd.isna(aa_substitutions_str) or aa_substitutions_str == '':
        return 0
    
    s = str(aa_substitutions_str).strip()
    if s.startswith('('):
        s = s[1:]
    if s.endswith(')'):
        s = s[:-1]
    
    if not s:
        return 0
    
    mutations = [m.strip() for m in s.split(',') if m.strip()]
    if envelope_only:
        mutations = [m for m in mutations if m.startswith('E_')]
    return len(mutations)


def extract_year(collection_date):
    """Extract year dari collection date (bisa format 2009 atau 2009-07-17)"""
    if pd.isna(collection_date):
        return None
    s = str(collection_date).strip()
    match = re.search(r'\d{4}', s)
    return int(match.group()) if match else None


def _extract_envelope_sequence(full_seq):
    """Extract envelope region dari full genome (posisi 937-2423, 0-based)."""
    if pd.isna(full_seq) or not full_seq:
        return ''
    s = str(full_seq).strip()
    if len(s) < ENVELOPE_END:
        return s[ENVELOPE_START:] if len(s) > ENVELOPE_START else ''
    return s[ENVELOPE_START:ENVELOPE_END]


def _process_gisaid_to_pipeline_format(df):
    """Convert GISAID DataFrame ke format pipeline (metadata + features + mutation + labels).
    - sample_id: NCBI Accession ID (untuk join dengan legacy)
    - Deteksi mutasi: gunakan envelope_sequence saja
    """
    df = df.copy()
    # Join key GISAID: NCBI Accession ID (bukan Accession ID)
    ncbicol = 'NCBI Accession ID' if 'NCBI Accession ID' in df.columns else None
    if ncbicol and df[ncbicol].notna().any():
        df['sample_id'] = df[ncbicol].fillna(df.get('Accession ID', '')).astype(str)
    else:
        df['sample_id'] = df['Accession ID'] if 'Accession ID' in df.columns else df.index.astype(str)
    
    # Metadata
    loc_col = df.get('Location_x', df.get('Location_y', pd.Series([''] * len(df))))
    year_series = df.get('Collection date_x', df.get('Collection date_y', df.index))
    if hasattr(year_series, 'apply'):
        year_series = year_series.apply(extract_year)
    else:
        year_series = pd.Series([extract_year(y) for y in year_series])
    
    seq_len = df.get('Sequence Length', pd.Series([0] * len(df)))
    if not isinstance(seq_len, pd.Series):
        seq_len = pd.Series([seq_len] * len(df))
    
    sample_metadata = pd.DataFrame({
        'sample_id': df['sample_id'],
        'description': df.get('Virus name', df['sample_id']),
        'serotype': df['Serotype'] if 'Serotype' in df.columns else df.get('serotype', ''),
        'genotype': df['Genotype'] if 'Genotype' in df.columns else df.get('genotype', ''),
        'lineage': df['Lineage'] if 'Lineage' in df.columns else '',
        'country': df.get('country', ''),
        'year': year_series,
        'host': df.get('Host', 'Human'),
        'genome_length': seq_len.fillna(0),
        'is_complete': seq_len.fillna(0) > 10000,
        'location': loc_col
    })
    
    # Mutation count: hanya E_ (envelope) dari AA Substitutions
    aa_col = 'AA Substitutions' if 'AA Substitutions' in df.columns else 'aa_substitutions'
    mutation_counts = df.get(aa_col, pd.Series([''] * len(df))).apply(
        lambda x: parse_aa_substitutions(x, envelope_only=True)
    )
    
    # Sequence features: gunakan envelope_sequence (bukan full genome)
    seq_col = 'Sequence' if 'Sequence' in df.columns else 'sequence'
    if seq_col not in df.columns:
        envelope_seqs = [''] * len(df)
    else:
        envelope_seqs = df[seq_col].fillna('').apply(_extract_envelope_sequence)
    
    seq_df = pd.DataFrame({'sample_id': df['sample_id'], 'sequence': envelope_seqs})
    sequence_features = extract_features_from_sequences(
        seq_df, sequence_column='sequence', sample_id_column='sample_id', k=3
    )
    
    # mutation_density & length_diff berdasarkan envelope
    envelope_lens = envelope_seqs.apply(len)
    mutation_profile = pd.DataFrame({
        'sample_id': df['sample_id'],
        'total_mutations': mutation_counts,
        'mutation_density': mutation_counts / ENVELOPE_REF_LEN,
        'length_diff': envelope_lens - ENVELOPE_REF_LEN
    })
    
    sequence_features = sequence_features.merge(
        mutation_profile[['sample_id', 'total_mutations', 'mutation_density', 'length_diff']],
        on='sample_id', how='left'
    )
    sequence_features['total_mutations'] = sequence_features['total_mutations'].fillna(0)
    sequence_features['mutation_density'] = sequence_features['mutation_density'].fillna(0)
    sequence_features['length_diff'] = sequence_features['length_diff'].fillna(0)
    
    label_table = pd.DataFrame({
        'sample_id': df['sample_id'],
        'serotype': df['Serotype'] if 'Serotype' in df.columns else df.get('serotype', ''),
        'genotype': df['Genotype'] if 'Genotype' in df.columns else df.get('genotype', ''),
        'known_genotype': True
    })
    
    merged = sample_metadata.merge(sequence_features, on='sample_id', how='left')
    merged = merged.merge(label_table[['sample_id', 'known_genotype']], on='sample_id', how='left')
    return merged


def _load_and_merge_legacy_datasets(dataset_dir):
    """
    Load dataset legacy (sample_metadata, sequence_features, mutation_profile, label_table)
    dan merge jadi 1 tabel.
    Deteksi mutasi: gunakan envelope_sequence dari raw_sequences.csv (bukan full genome).
    """
    dataset_path = Path(dataset_dir)
    
    # Load metadata
    meta_path = dataset_path / 'sample_metadata.csv'
    if not meta_path.exists():
        return None
    
    merged = pd.read_csv(meta_path)
    logger.info(f"Loaded sample_metadata: {len(merged)} rows")
    
    # JOIN sequence_features: gunakan envelope_sequence dari raw_sequences jika ada
    raw_seq_path = dataset_path / 'raw_sequences.csv'
    if raw_seq_path.exists() and 'envelope_sequence' in pd.read_csv(raw_seq_path, nrows=1).columns:
        raw_df = pd.read_csv(raw_seq_path)
        logger.info(f"Extracting features from {len(raw_df)} legacy envelope_sequences (bisa memakan waktu)...")
        seq_df = extract_features_from_sequences(
            raw_df[['sample_id', 'envelope_sequence']].rename(columns={'envelope_sequence': 'sequence'}),
            sequence_column='sequence', sample_id_column='sample_id', k=3
        )
        # length_diff berdasarkan envelope
        env_lens = raw_df['envelope_sequence'].fillna('').apply(len)
        seq_df['length_diff'] = env_lens - ENVELOPE_REF_LEN
        seq_cols = [c for c in seq_df.columns if c not in merged.columns or c == 'sample_id']
        if len(seq_cols) > 1:
            merged = merged.merge(seq_df[seq_cols], on='sample_id', how='left', suffixes=('', '_dup'))
            merged = merged.loc[:, ~merged.columns.duplicated()]
        logger.info("Joined sequence_features (from envelope_sequence)")
    else:
        seq_path = dataset_path / 'sequence_features.csv'
        if seq_path.exists():
            seq_df = pd.read_csv(seq_path)
            seq_cols = [c for c in seq_df.columns if c not in merged.columns or c == 'sample_id']
            if len(seq_cols) > 1:
                merged = merged.merge(seq_df[seq_cols], on='sample_id', how='left', suffixes=('', '_dup'))
                merged = merged.loc[:, ~merged.columns.duplicated()]
            logger.info("Joined sequence_features")
    
    # JOIN mutation_profile
    mut_path = dataset_path / 'mutation_profile.csv'
    if mut_path.exists():
        mut_df = pd.read_csv(mut_path)
        mut_cols = [c for c in mut_df.columns if c not in merged.columns or c == 'sample_id']
        if len(mut_cols) > 1:
            merged = merged.merge(mut_df[mut_cols], on='sample_id', how='left')
            merged = merged.loc[:, ~merged.columns.duplicated()]
        logger.info("Joined mutation_profile")
    
    # JOIN label_table
    label_path = dataset_path / 'label_table.csv'
    if label_path.exists():
        label_df = pd.read_csv(label_path)
        label_cols = [c for c in label_df.columns if c not in merged.columns or c == 'sample_id']
        if len(label_cols) > 1:
            merged = merged.merge(label_df[label_cols], on='sample_id', how='left')
            merged = merged.loc[:, ~merged.columns.duplicated()]
        logger.info("Joined label_table")
    
    return merged


def preprocess_gisaid_data(
    input_path='dataset/from_gisaid_data.csv',
    output_dir='dataset',
    merge_with_legacy=True,
    save_intermediate=True
):
    """
    Preprocess data GISAID dan gabung dengan dataset lain (jika ada).
    
    Args:
        input_path: Path ke from_gisaid_data.csv
        output_dir: Directory untuk output
        merge_with_legacy: Gabung dengan sample_metadata, raw_sequences, dll (JOIN/UNION)
        save_intermediate: Simpan file intermediate
        
    Returns:
        DataFrame merged - 1 tabel dataset terpadu
    """
    logger.info("=" * 60)
    logger.info("GISAID Data Preprocessing + Merge dengan Dataset Lain")
    logger.info("=" * 60)
    
    # Load GISAID (tanpa filter - dataset sudah sesuai)
    df_gisaid = pd.read_csv(input_path)
    logger.info(f"Loaded GISAID: {len(df_gisaid)} rows")
    
    # Process GISAID ke format pipeline
    gisaid_merged = _process_gisaid_to_pipeline_format(df_gisaid)
    gisaid_merged['data_source'] = 'gisaid'
    logger.info(f"GISAID processing done: {len(gisaid_merged)} samples")
    
    # Merge dengan dataset lain
    output_path = Path(output_dir)
    legacy_merged = None
    
    if merge_with_legacy:
        logger.info("Loading legacy datasets (sample_metadata, raw_sequences, ...)...")
        legacy_merged = _load_and_merge_legacy_datasets(output_dir)
        if legacy_merged is not None:
            legacy_merged['data_source'] = 'legacy'
            
            # Join key: GISAID pakai NCBI Accession ID, legacy pakai Accession ID (sample_id).
            # Jika ada NCBI Accession ID di GISAID yang match dengan legacy sample_id -> overlap
            # Jika tidak overlap -> UNION (concat)
            ncbicol = 'NCBI Accession ID' if 'NCBI Accession ID' in df_gisaid.columns else None
            
            if ncbicol and df_gisaid[ncbicol].notna().any():
                # Coba JOIN: GISAID sample_id (Accession ID) vs legacy.
                # Atau: buat mapping GISAID NCBI ID = legacy sample_id
                gisaid_ncbi = df_gisaid[['Accession ID', ncbicol]].dropna(subset=[ncbicol])
                gisaid_ncbi = gisaid_ncbi.rename(columns={'Accession ID': 'gisaid_id', ncbicol: 'sample_id'})
                
                # Records di legacy yang match NCBI ID -> bisa di-enrich dari GISAID
                common_ids = set(gisaid_ncbi['sample_id']) & set(legacy_merged['sample_id'])
                if common_ids:
                    logger.info(f"Found {len(common_ids)} samples in both GISAID and legacy - will merge")
            
            # UNION: gabung semua rows (align columns)
            all_cols = list(set(gisaid_merged.columns) | set(legacy_merged.columns))
            for c in all_cols:
                if c not in gisaid_merged.columns:
                    gisaid_merged[c] = np.nan
                if c not in legacy_merged.columns:
                    legacy_merged[c] = np.nan
            
            merged = pd.concat([
                gisaid_merged[all_cols],
                legacy_merged[all_cols]
            ], ignore_index=True)
            logger.info(f"Merged: {len(gisaid_merged)} GISAID + {len(legacy_merged)} legacy = {len(merged)} total")
        else:
            merged = gisaid_merged
            logger.info("No legacy dataset found, using GISAID only")
    else:
        merged = gisaid_merged
    
    # Drop duplicates by sample_id (keep first)
    if merged['sample_id'].duplicated().any():
        before = len(merged)
        merged = merged.drop_duplicates(subset=['sample_id'], keep='first')
        logger.info(f"Removed {before - len(merged)} duplicate sample_ids")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    if save_intermediate:
        # Simpan tabel terpisah untuk kompatibilitas pipeline
        meta_cols = ['sample_id', 'description', 'serotype', 'genotype', 'country', 'year', 'host', 
                     'genome_length', 'is_complete', 'location', 'lineage', 'data_source']
        meta_cols = [c for c in meta_cols if c in merged.columns]
        merged[meta_cols].to_csv(output_path / 'sample_metadata.csv', index=False)
        
        seq_cols = ['sample_id'] + [c for c in merged.columns if c.startswith('kmer_') or c in 
                    ['gc_content', 'genome_length', 'total_mutations', 'mutation_density', 'length_diff']]
        seq_cols = [c for c in seq_cols if c in merged.columns]
        merged[seq_cols].to_csv(output_path / 'sequence_features.csv', index=False)
        
        mut_cols = ['sample_id', 'total_mutations', 'mutation_density', 'length_diff']
        mut_cols = [c for c in mut_cols if c in merged.columns]
        merged[mut_cols].to_csv(output_path / 'mutation_profile.csv', index=False)
        
        label_cols = ['sample_id', 'serotype', 'genotype', 'known_genotype']
        label_cols = [c for c in label_cols if c in merged.columns]
        merged[label_cols].to_csv(output_path / 'label_table.csv', index=False)
        
        logger.info(f"Saved intermediate files to {output_dir}/")
    
    merged.to_csv(output_path / 'merged_dataset.csv', index=False)
    logger.info(f"Saved merged dataset: {len(merged)} samples -> merged_dataset.csv")
    
    logger.info("\nDataset Summary:")
    logger.info(f"  Total samples: {len(merged)}")
    if 'genotype' in merged.columns:
        logger.info(f"  Genotypes: {merged['genotype'].value_counts().to_dict()}")
    if 'serotype' in merged.columns:
        logger.info(f"  Serotypes: {merged['serotype'].value_counts().to_dict()}")
    if 'data_source' in merged.columns:
        logger.info(f"  By source: {merged['data_source'].value_counts().to_dict()}")
    logger.info("=" * 60)
    
    return merged


if __name__ == "__main__":
    preprocess_gisaid_data(
        input_path='dataset/from_gisaid_data.csv',
        output_dir='dataset',
        merge_with_legacy=True,
        save_intermediate=True
    )
