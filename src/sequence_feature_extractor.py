"""
Sequence Feature Extractor
Modul untuk extract features dari raw genomic sequences
"""

import pandas as pd
import numpy as np
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def gc_content(seq):
    """
    Calculate GC content dari sequence
    
    Args:
        seq: DNA sequence string
        
    Returns:
        GC content (float between 0 and 1)
    """
    if pd.isna(seq) or seq == '':
        return 0.0
    
    seq = str(seq).upper()
    if len(seq) == 0:
        return 0.0
    
    gc_count = seq.count("G") + seq.count("C")
    return gc_count / len(seq)


def kmer_features(seq, k=3):
    """
    Extract k-mer frequencies dari sequence
    
    Args:
        seq: DNA sequence string
        k: k-mer size (default 3)
        
    Returns:
        Dictionary dengan k-mer frequencies
    """
    if pd.isna(seq) or seq == '':
        return {}
    
    seq = str(seq).upper()
    
    # Remove ambiguous nucleotides
    seq_clean = ''.join([n for n in seq if n in 'ATCG'])
    
    if len(seq_clean) < k:
        return {}
    
    # Count k-mers
    counts = Counter(
        seq_clean[i:i+k] for i in range(len(seq_clean)-k+1)
    )
    
    total = sum(counts.values())
    if total == 0:
        return {}
    
    # Normalize to frequencies
    return {f"kmer_{kmer}": v/total for kmer, v in counts.items()}


def extract_features_from_sequences(df, sequence_column='sequence', sample_id_column='sample_id', k=3, 
                                   expected_feature_names=None):
    """
    Extract features dari DataFrame dengan kolom sequence
    
    Args:
        df: DataFrame dengan kolom sequence
        sequence_column: Nama kolom yang berisi sequence
        sample_id_column: Nama kolom untuk sample_id (jika tidak ada, akan dibuat)
        k: k-mer size (default 3)
        expected_feature_names: List feature names yang diharapkan dari training (untuk alignment)
        
    Returns:
        DataFrame dengan extracted features
    """
    logger.info(f"Extracting features from {len(df)} sequences...")
    
    # Ensure sample_id exists
    if sample_id_column and sample_id_column not in df.columns:
        df[sample_id_column] = [f'SAMPLE_{i+1:04d}' for i in range(len(df))]
        logger.info(f"Created {sample_id_column} column")
    
    # Check if sequence column exists
    if sequence_column not in df.columns:
        raise ValueError(f"Sequence column '{sequence_column}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    rows = []
    
    for idx, row in df.iterrows():
        seq = row[sequence_column]
        
        # Extract features
        feats = {}
        if sample_id_column:
            feats[sample_id_column] = row[sample_id_column]
        
        # GC content
        feats['gc_content'] = gc_content(seq)
        
        # K-mer features (only ATCG k-mers)
        kmer_feats = kmer_features(seq, k=k)
        feats.update(kmer_feats)
        
        # Genome length
        if pd.notna(seq):
            feats['genome_length'] = len(str(seq))
        else:
            feats['genome_length'] = 0
        
        rows.append(feats)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(rows)
    
    # If expected_feature_names provided, align features
    if expected_feature_names is not None:
        logger.info(f"Aligning features with {len(expected_feature_names)} expected feature names from training")
        
        # Create DataFrame with expected feature names
        aligned_df = pd.DataFrame(0, index=features_df.index, columns=expected_feature_names)
        
        # Copy values from extracted features
        common_cols = set(features_df.columns) & set(expected_feature_names)
        for col in common_cols:
            aligned_df[col] = features_df[col].values
        
        # Log missing and extra columns
        missing_cols = set(expected_feature_names) - set(features_df.columns)
        extra_cols = set(features_df.columns) - set(expected_feature_names)
        
        if missing_cols:
            logger.info(f"Added {len(missing_cols)} missing columns (filled with 0): {list(missing_cols)[:10]}...")
        if extra_cols:
            logger.warning(f"Removed {len(extra_cols)} extra columns not in training: {list(extra_cols)[:10]}...")
        
        features_df = aligned_df
    else:
        # Fill missing k-mer columns with 0 (only standard ATCG k-mers)
        if k == 3:
            nucleotides = ['A', 'T', 'C', 'G']
            all_kmers = [f"kmer_{n1}{n2}{n3}" for n1 in nucleotides 
                        for n2 in nucleotides for n3 in nucleotides]
            
            for kmer in all_kmers:
                if kmer not in features_df.columns:
                    features_df[kmer] = 0.0
    
    # Fill NaN with 0
    features_df = features_df.fillna(0)
    
    logger.info(f"Extracted features: {features_df.shape[1]} columns")
    logger.info(f"Features include: gc_content, genome_length, and {len([c for c in features_df.columns if c.startswith('kmer_')])} k-mer frequencies")
    
    return features_df


def add_mutation_features(features_df, reference_length=None):
    """
    Add mutation-related features (dummy values for inference)
    Since we don't have reference sequence, we'll add default values
    
    Args:
        features_df: DataFrame dengan sequence features
        reference_length: Reference genome length (optional)
        
    Returns:
        DataFrame dengan mutation features added
    """
    # Add default mutation features
    if 'mutation_density' not in features_df.columns:
        features_df['mutation_density'] = 0.0
    
    if 'total_mutations' not in features_df.columns:
        features_df['total_mutations'] = 0
    
    if 'length_diff' not in features_df.columns:
        if reference_length is not None:
            features_df['length_diff'] = features_df['genome_length'] - reference_length
        else:
            features_df['length_diff'] = 0
    
    return features_df


if __name__ == "__main__":
    # Test feature extraction
    test_data = pd.DataFrame({
        'sample_id': ['TEST_001', 'TEST_002'],
        'sequence': [
            'ATCGATCGATCGATCG',
            'GCTAGCTAGCTAGCTA'
        ]
    })
    
    features = extract_features_from_sequences(test_data)
    print(f"\nExtracted features shape: {features.shape}")
    print(f"\nFirst few columns: {list(features.columns[:10])}")
    print(f"\nFeatures:\n{features.head()}")

