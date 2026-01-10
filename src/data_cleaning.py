"""
STAGE 1 - Dataset Preparation (ML View)
Modul untuk cleaning dan merging dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Class untuk cleaning dan merging dataset"""
    
    def __init__(self, dataset_dir='dataset'):
        """
        Initialize DataCleaner
        
        Args:
            dataset_dir: Path ke folder dataset
        """
        self.dataset_dir = Path(dataset_dir)
        self.raw_data = {}
        self.cleaned_data = None
        
    def load_datasets(self):
        """Load semua CSV files dari dataset directory"""
        logger.info("Loading datasets...")
        
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
        """
        logger.info("Merging tables...")
        
        if 'metadata' not in self.raw_data:
            raise ValueError("Metadata table is required!")
        
        # Start dengan metadata sebagai base
        merged = self.raw_data['metadata'].copy()
        
        # Merge dengan sequence features
        if 'sequence_features' in self.raw_data:
            merged = merged.merge(
                self.raw_data['sequence_features'],
                on='sample_id',
                how='left',
                suffixes=('', '_seq')
            )
            logger.info("Merged sequence_features")
        
        # Merge dengan mutation profile
        if 'mutation_profile' in self.raw_data:
            merged = merged.merge(
                self.raw_data['mutation_profile'],
                on='sample_id',
                how='left',
                suffixes=('', '_mut')
            )
            logger.info("Merged mutation_profile")
        
        # Merge dengan labels
        if 'labels' in self.raw_data:
            merged = merged.merge(
                self.raw_data['labels'],
                on='sample_id',
                how='left',
                suffixes=('', '_label')
            )
            logger.info("Merged label_table")
        
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
        
        # Filter missing labels
        # Cek di kolom metadata atau label_table
        if label_column in self.cleaned_data.columns:
            self.cleaned_data = self.cleaned_data[
                self.cleaned_data[label_column].notna()
            ].copy()
        elif f'{label_column}_label' in self.cleaned_data.columns:
            self.cleaned_data = self.cleaned_data[
                self.cleaned_data[f'{label_column}_label'].notna()
            ].copy()
        
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
    
    def run_full_cleaning(self, label_column='serotype', save_output=True):
        """
        Run full cleaning pipeline
        
        Args:
            label_column: Kolom label untuk filtering
            save_output: Apakah menyimpan output ke CSV
        """
        logger.info("=" * 50)
        logger.info("Starting Data Cleaning Pipeline")
        logger.info("=" * 50)
        
        # Load datasets
        self.load_datasets()
        
        # Merge tables
        self.merge_tables()
        
        # Filter missing labels
        self.filter_missing_labels(label_column=label_column)
        
        # Filter outliers
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

