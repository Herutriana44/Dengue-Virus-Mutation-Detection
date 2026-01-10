"""
STAGE 2 - Feature Selection & Encoding
Modul untuk feature engineering, encoding, dan scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Class untuk feature engineering dan preprocessing"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_groups = {}
        self.preprocessor = None
        self.feature_names = []
        
    def group_features(self, df):
        """
        Group features sesuai dengan kategori:
        - X_seq: k-mer, GC content, codon bias
        - X_mut: mutation features
        - X_prot: protein-level features (jika ada)
        - X_meta: metadata (year, region, etc.)
        
        Args:
            df: DataFrame dengan semua features
            
        Returns:
            dict dengan grouped features
        """
        logger.info("Grouping features...")
        
        # Sequence features (k-mer, GC content)
        seq_features = [col for col in df.columns 
                       if col.startswith('kmer_') or col == 'gc_content']
        
        # Mutation features
        mut_features = [col for col in df.columns 
                       if 'mutation' in col.lower() or 'mut' in col.lower() 
                       or 'length_diff' in col.lower()]
        
        # Protein features (jika ada)
        prot_features = [col for col in df.columns 
                        if any(x in col.lower() for x in ['protein', 'e_', 'ns1', 'ns3', 'ns5'])]
        
        # Metadata features
        meta_features = [col for col in df.columns 
                        if col in ['year', 'country', 'region', 'host', 'genome_length']]
        
        self.feature_groups = {
            'X_seq': seq_features,
            'X_mut': mut_features,
            'X_prot': prot_features if prot_features else [],
            'X_meta': meta_features
        }
        
        logger.info(f"Sequence features: {len(seq_features)}")
        logger.info(f"Mutation features: {len(mut_features)}")
        logger.info(f"Protein features: {len(prot_features)}")
        logger.info(f"Metadata features: {len(meta_features)}")
        
        return self.feature_groups
    
    def encode_categorical(self, df, columns, method='onehot'):
        """
        Encode categorical features
        
        Args:
            df: DataFrame
            columns: List kolom kategorikal
            method: 'onehot' atau 'label'
            
        Returns:
            DataFrame dengan encoded features
        """
        df_encoded = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
                logger.info(f"One-hot encoded: {col} -> {len(dummies.columns)} features")
                
            elif method == 'label':
                # Label encoding
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(
                        df[col].astype(str).fillna('unknown')
                    )
                else:
                    df_encoded[col] = self.label_encoders[col].transform(
                        df[col].astype(str).fillna('unknown')
                    )
                logger.info(f"Label encoded: {col}")
        
        return df_encoded
    
    def prepare_features(self, df, target_column='serotype', exclude_columns=None):
        """
        Prepare features untuk ML
        
        Args:
            df: DataFrame dengan semua data
            target_column: Kolom target (tidak akan di-scale)
            exclude_columns: Kolom yang harus di-exclude
            
        Returns:
            X: Feature matrix
            y: Target vector
            feature_names: List nama features
        """
        logger.info("Preparing features for ML...")
        
        if exclude_columns is None:
            exclude_columns = ['sample_id', 'description']
        
        # Exclude columns
        exclude_columns = exclude_columns + [target_column]
        feature_cols = [col for col in df.columns if col not in exclude_columns]
        
        # Separate numerical and categorical
        numerical_cols = []
        categorical_cols = []
        
        for col in feature_cols:
            if df[col].dtype in ['int64', 'float64']:
                # Check if it's actually categorical (low cardinality)
                if df[col].nunique() < 20 and df[col].nunique() < len(df) * 0.1:
                    categorical_cols.append(col)
                else:
                    numerical_cols.append(col)
            else:
                categorical_cols.append(col)
        
        logger.info(f"Numerical features: {len(numerical_cols)}")
        logger.info(f"Categorical features: {len(categorical_cols)}")
        
        # Encode categorical
        df_processed = self.encode_categorical(df, categorical_cols, method='onehot')
        
        # Get final feature columns (exclude target and sample_id)
        final_feature_cols = [col for col in df_processed.columns 
                             if col not in exclude_columns]
        
        # Extract X and y
        X = df_processed[final_feature_cols].copy()
        y = df[target_column].copy() if target_column in df.columns else None
        
        # Handle missing values
        X = X.fillna(0)
        
        # Convert to numeric (handle any remaining non-numeric)
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        self.feature_names = list(X.columns)
        
        logger.info(f"Final feature matrix shape: {X.shape}")
        if y is not None:
            logger.info(f"Target distribution:\n{y.value_counts()}")
        
        return X, y, self.feature_names
    
    def scale_features(self, X_train, X_test=None, fit=True):
        """
        Scale features menggunakan StandardScaler
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            fit: Apakah fit scaler pada X_train
            
        Returns:
            X_train_scaled, X_test_scaled (jika provided)
        """
        logger.info("Scaling features...")
        
        if fit:
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
        else:
            X_train_scaled = pd.DataFrame(
                self.scaler.transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
        
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def get_feature_groups_for_task(self, task='all'):
        """
        Get feature groups untuk task tertentu
        
        Args:
            task: 'baseline', 'novelty', 'open_set', atau 'all'
            
        Returns:
            List feature names sesuai task
        """
        if task == 'baseline':
            # Semua features untuk baseline classification
            return self.feature_names
        
        elif task == 'novelty':
            # Sequence + Mutation features untuk novelty detection
            seq_mut = self.feature_groups.get('X_seq', []) + \
                     self.feature_groups.get('X_mut', [])
            return [f for f in self.feature_names if any(x in f for x in seq_mut)]
        
        elif task == 'open_set':
            # Semua features untuk open-set detection
            return self.feature_names
        
        else:
            return self.feature_names
    
    def save_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Save preprocessor untuk future use"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_groups': self.feature_groups,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        logger.info(f"Saved preprocessor to {filepath}")
    
    def load_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Load preprocessor"""
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.scaler = preprocessor_data['scaler']
        self.label_encoders = preprocessor_data['label_encoders']
        self.feature_groups = preprocessor_data['feature_groups']
        self.feature_names = preprocessor_data['feature_names']
        
        logger.info(f"Loaded preprocessor from {filepath}")


if __name__ == "__main__":
    # Test feature engineering
    from data_cleaning import DataCleaner
    
    cleaner = DataCleaner(dataset_dir='dataset')
    cleaner.load_datasets()
    cleaner.merge_tables()
    df = cleaner.cleaned_data
    
    engineer = FeatureEngineer()
    engineer.group_features(df)
    X, y, feature_names = engineer.prepare_features(df, target_column='serotype')
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"\nFirst 10 features: {feature_names[:10]}")

