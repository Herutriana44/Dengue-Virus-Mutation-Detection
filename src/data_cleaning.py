"""
STAGE 1 - Dataset Preparation (ML View)
Modul untuk cleaning dan merging dataset
Mendukung: format legacy (sample_metadata, dll) dan GISAID (from_gisaid_data.csv)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Class untuk cleaning dan merging dataset"""
    
    def __init__(self, dataset_dir='dataset', use_gisaid=False):
        """
        Initialize DataCleaner
        
        Args:
            dataset_dir: Path ke folder dataset
            use_gisaid: Jika True, gunakan from_gisaid_data.csv (DENV-2 Indonesia only)
        """
        self.dataset_dir = Path(dataset_dir)
        self.use_gisaid = use_gisaid
        self.raw_data = {}
        self.cleaned_data = None
        
    def load_datasets(self):
        """Load dataset - GISAID format atau legacy format"""
        if self.use_gisaid:
            return self._load_gisaid_data()
        return self._load_legacy_datasets()
    
    def _load_gisaid_data(self):
        """Load dan preprocess data dari from_gisaid_data.csv, gabung dengan dataset lain (JOIN)"""
        logger.info("Loading GISAID data + merge dengan dataset lain...")
        
        gisaid_path = self.dataset_dir / 'from_gisaid_data.csv'
        merged_path = self.dataset_dir / 'merged_dataset.csv'
        
        if not gisaid_path.exists():
            raise FileNotFoundError(
                f"from_gisaid_data.csv not found at {gisaid_path}. "
                "Jalankan scraping dari GetDataDengueVirusGenome.ipynb terlebih dahulu."
            )
        
        from gisaid_preprocessor import preprocess_gisaid_data
        
        # Preprocess + merge dengan legacy (tanpa filter - dataset sudah sesuai)
        preprocess_gisaid_data(
            input_path=str(gisaid_path),
            output_dir=str(self.dataset_dir),
            merge_with_legacy=True,
            save_intermediate=True
        )
        
        self.cleaned_data = pd.read_csv(merged_path)
        self.raw_data = {
            'metadata': self.cleaned_data,
            'sequence_features': self.cleaned_data,
            'mutation_profile': self.cleaned_data,
            'labels': self.cleaned_data
        }
        logger.info(f"Loaded merged dataset: {len(self.cleaned_data)} samples")
        return self.raw_data
    
    def _load_legacy_datasets(self):
        """Load semua CSV files dari dataset directory (format legacy)"""
        logger.info("Loading datasets (legacy format)...")
        
        files = {
            'metadata': 'sample_metadata.csv',
            'sequence_features': 'sequence_features.csv',
            'mutation_profile': 'mutation_profile.csv',
            'labels': 'label_table.csv'
        }
        
        for key, filename in files.items():
            filepath = self.dataset_dir / filename
            if filepath.exists():
                self.raw_data[key] = pd.read_csv(filepath)
                logger.info(f"Loaded {filename}: {len(self.raw_data[key])} rows")
            else:
                logger.warning(f"File {filename} not found!")
                
        return self.raw_data
    
    def merge_tables(self):
        """
        Merge semua tabel menggunakan sample_id sebagai key
        Left join untuk memastikan semua sample dari metadata tetap ada
        Untuk GISAID: data sudah merged, skip merge
        """
        logger.info("Merging tables...")
        
        # GISAID: data sudah merged di load_datasets
        if self.use_gisaid and self.cleaned_data is not None:
            logger.info("GISAID data already merged, skipping merge step")
            return self.cleaned_data
        
        if 'metadata' not in self.raw_data:
            raise ValueError("Metadata table is required!")
        
        # Start dengan metadata sebagai base
        merged = self.raw_data['metadata'].copy()
        
        # Merge dengan sequence features
        if 'sequence_features' in self.raw_data:
            seq_df = self.raw_data['sequence_features']
            # Avoid duplicate columns
            seq_cols = [c for c in seq_df.columns if c not in merged.columns or c == 'sample_id']
            if len(seq_cols) > 1:
                merged = merged.merge(
                    seq_df[seq_cols],
                    on='sample_id',
                    how='left',
                    suffixes=('', '_seq')
                )
            logger.info("Merged sequence_features")
        
        # Merge dengan mutation profile
        if 'mutation_profile' in self.raw_data:
            mut_df = self.raw_data['mutation_profile']
            mut_cols = [c for c in mut_df.columns if c not in merged.columns or c == 'sample_id']
            if len(mut_cols) > 1:
                merged = merged.merge(
                    mut_df[mut_cols],
                    on='sample_id',
                    how='left',
                    suffixes=('', '_mut')
                )
            logger.info("Merged mutation_profile")
        
        # Merge dengan labels
        if 'labels' in self.raw_data:
            label_df = self.raw_data['labels']
            label_cols = [c for c in label_df.columns if c not in merged.columns or c == 'sample_id']
            if len(label_cols) > 1:
                merged = merged.merge(
                    label_df[label_cols],
                    on='sample_id',
                    how='left',
                    suffixes=('', '_label')
                )
            logger.info("Merged label_table")
        
        # Remove duplicate columns
        merged = merged.loc[:, ~merged.columns.duplicated()]
        self.cleaned_data = merged
        logger.info(f"Merged dataset shape: {merged.shape}")
        
        return merged
    
    def filter_missing_labels(self, label_column='serotype'):
        """
        Filter samples dengan missing labels
        
        Args:
            label_column: Kolom label yang digunakan untuk filtering
        """
        if self.cleaned_data is None:
            raise ValueError("Please merge tables first!")
        
        initial_count = len(self.cleaned_data)
        
        # Filter missing labels (termasuk empty string untuk genotype)
        if label_column in self.cleaned_data.columns:
            valid_mask = self.cleaned_data[label_column].notna()
            valid_mask &= self.cleaned_data[label_column].astype(str).str.strip() != ''
            self.cleaned_data = self.cleaned_data[valid_mask].copy()
        elif f'{label_column}_label' in self.cleaned_data.columns:
            valid_mask = self.cleaned_data[f'{label_column}_label'].notna()
            valid_mask &= self.cleaned_data[f'{label_column}_label'].astype(str).str.strip() != ''
            self.cleaned_data = self.cleaned_data[valid_mask].copy()
        
        filtered_count = len(self.cleaned_data)
        removed = initial_count - filtered_count
        
        logger.info(f"Removed {removed} samples with missing {label_column}")
        logger.info(f"Remaining samples: {filtered_count}")
        
        return self.cleaned_data
    
    def filter_outliers(self, column='genome_length', lower_percentile=0.01, upper_percentile=0.99):
        """
        Filter outliers berdasarkan kolom tertentu
        
        Args:
            column: Kolom untuk outlier detection
            lower_percentile: Percentile bawah untuk filtering
            upper_percentile: Percentile atas untuk filtering
        """
        if self.cleaned_data is None:
            raise ValueError("Please merge tables first!")
        
        if column not in self.cleaned_data.columns:
            logger.warning(f"Column {column} not found, skipping outlier filtering")
            return self.cleaned_data
        
        # Skip jika kolom semua NaN/0
        valid_vals = self.cleaned_data[column].dropna()
        if len(valid_vals) < 2 or valid_vals.nunique() < 2:
            logger.warning(f"Column {column} has insufficient variation, skipping outlier filtering")
            return self.cleaned_data
        
        initial_count = len(self.cleaned_data)
        
        # Calculate percentiles
        lower_bound = self.cleaned_data[column].quantile(lower_percentile)
        upper_bound = self.cleaned_data[column].quantile(upper_percentile)
        
        # Filter outliers
        self.cleaned_data = self.cleaned_data[
            (self.cleaned_data[column] >= lower_bound) &
            (self.cleaned_data[column] <= upper_bound)
        ].copy()
        
        filtered_count = len(self.cleaned_data)
        removed = initial_count - filtered_count
        
        logger.info(f"Removed {removed} outliers based on {column}")
        logger.info(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        logger.info(f"Remaining samples: {filtered_count}")
        
        return self.cleaned_data
    
    def get_summary(self):
        """Get summary statistics dari cleaned dataset"""
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available!")
        
        summary = {
            'total_samples': len(self.cleaned_data),
            'total_features': len(self.cleaned_data.columns),
            'missing_values': self.cleaned_data.isnull().sum().sum(),
            'duplicate_samples': self.cleaned_data['sample_id'].duplicated().sum()
        }
        
        # Label distribution
        if 'serotype' in self.cleaned_data.columns:
            summary['serotype_distribution'] = self.cleaned_data['serotype'].value_counts().to_dict()
        
        if 'genotype' in self.cleaned_data.columns:
            summary['genotype_distribution'] = self.cleaned_data['genotype'].value_counts().to_dict()
        
        return summary
    
    def save_cleaned_data(self, output_path='ml_dataset_raw.csv'):
        """Save cleaned dataset ke CSV"""
        if self.cleaned_data is None:
            raise ValueError("No cleaned data to save!")
        
        self.cleaned_data.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned dataset to {output_path}")
        
        return output_path
    
    def run_full_cleaning(self, label_column=None, save_output=True):
        """
        Run full cleaning pipeline
        
        Args:
            label_column: Kolom label untuk filtering. 
                Default: 'genotype' untuk GISAID (DENV-2), 'serotype' untuk legacy
            save_output: Apakah menyimpan output ke CSV
        """
        logger.info("=" * 50)
        logger.info("Starting Data Cleaning Pipeline")
        logger.info("=" * 50)
        
        # Default label: genotype jika ada dan punya nilai, else serotype
        if label_column is None:
            if self.use_gisaid:
                # Cek setelah load - genotype untuk data GISAID, serotype untuk mixed/legacy
                label_column = 'genotype'
            else:
                label_column = 'serotype'
        logger.info(f"Using label column: {label_column}")
        
        # Load datasets
        self.load_datasets()
        
        # Merge tables
        self.merge_tables()
        
        # Filter missing labels
        self.filter_missing_labels(label_column=label_column)
        
        # Filter outliers (skip jika genome_length tidak ada atau semua 0)
        if 'genome_length' in self.cleaned_data.columns and self.cleaned_data['genome_length'].notna().any():
            self.filter_outliers(column='genome_length')
        
        # Get summary
        summary = self.get_summary()
        logger.info("\nDataset Summary:")
        logger.info(f"Total samples: {summary['total_samples']}")
        logger.info(f"Total features: {summary['total_features']}")
        logger.info(f"Missing values: {summary['missing_values']}")
        
        if 'serotype_distribution' in summary:
            logger.info("\nSerotype distribution:")
            for sero, count in summary['serotype_distribution'].items():
                logger.info(f"  {sero}: {count}")
        
        # Save if requested
        if save_output:
            self.save_cleaned_data()
        
        logger.info("=" * 50)
        logger.info("Data Cleaning Pipeline Completed!")
        logger.info("=" * 50)
        
        return self.cleaned_data


if __name__ == "__main__":
    # Test the cleaning pipeline
    cleaner = DataCleaner(dataset_dir='dataset')
    cleaned_df = cleaner.run_full_cleaning()
    print(f"\nCleaned dataset shape: {cleaned_df.shape}")
    print(f"\nFirst few columns: {list(cleaned_df.columns[:10])}")

