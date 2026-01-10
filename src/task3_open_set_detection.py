"""
TASK 3 - Potensi Serotipe Baru (Open-set Detection)
Deteksi divergensi genetika ekstrem yang mungkin mengindikasikan serotipe baru
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score, precision_recall_curve
import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Autoencoder:
    """Simple Autoencoder untuk open-set detection"""
    
    def __init__(self, encoding_dim=32, hidden_layers=[64, 32]):
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.encoder = None
        self.decoder = None
        self.reconstruction_error_threshold = None
        
    def build_model(self, input_dim):
        """Build encoder-decoder architecture"""
        from sklearn.neural_network import MLPRegressor
        
        # Encoder: input -> hidden -> encoding
        encoder_layers = self.hidden_layers + [self.encoding_dim]
        self.encoder = MLPRegressor(
            hidden_layer_sizes=tuple(encoder_layers),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
        
        # Decoder: encoding -> hidden -> output
        decoder_layers = self.hidden_layers[::-1] + [input_dim]
        self.decoder = MLPRegressor(
            hidden_layer_sizes=tuple(decoder_layers),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
        
    def fit(self, X):
        """Train autoencoder"""
        input_dim = X.shape[1]
        self.build_model(input_dim)
        
        logger.info("Training encoder...")
        encoded = self.encoder.fit(X, X).predict(X)
        
        logger.info("Training decoder...")
        self.decoder.fit(encoded, X)
        
        logger.info("Autoencoder training completed")
        
    def predict(self, X):
        """Reconstruct input"""
        encoded = self.encoder.predict(X)
        reconstructed = self.decoder.predict(encoded)
        return reconstructed
    
    def reconstruction_error(self, X):
        """Calculate reconstruction error (MSE)"""
        reconstructed = self.predict(X)
        mse = np.mean((X - reconstructed) ** 2, axis=1)
        return mse
    
    def set_threshold(self, X_train, percentile=95):
        """Set threshold berdasarkan training data"""
        errors = self.reconstruction_error(X_train)
        self.reconstruction_error_threshold = np.percentile(errors, percentile)
        logger.info(f"Reconstruction error threshold ({percentile}th percentile): {self.reconstruction_error_threshold:.4f}")
        return self.reconstruction_error_threshold


class OpenSetDetector:
    """Class untuk open-set detection (potensi serotipe baru)"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.threshold = None
        self.best_model = None
        self.best_model_name = None
        
    def prepare_open_set_data(self, df, train_serotypes, test_serotypes=None):
        """
        Prepare data untuk open-set detection
        
        Args:
            df: DataFrame dengan semua data
            train_serotypes: List serotipe yang digunakan untuk training
            test_serotypes: List serotipe untuk testing (novel serotypes)
            
        Returns:
            X_train, X_test, y_test (y adalah binary: known=0, novel=1)
        """
        logger.info("Preparing data for open-set detection...")
        
        # Filter training data (known serotypes)
        train_mask = df['serotype'].isin(train_serotypes)
        X_train = df[train_mask].copy()
        
        logger.info(f"Training samples (known serotypes): {len(X_train)}")
        logger.info(f"Known serotypes: {train_serotypes}")
        
        # Filter test data
        if test_serotypes:
            test_mask = df['serotype'].isin(test_serotypes)
            X_test_novel = df[test_mask].copy()
            
            # Also include some known serotypes for evaluation
            X_test_known = df[train_mask].sample(
                min(len(X_test_novel), len(X_train) // 4), 
                random_state=42
            ).copy()
            
            X_test = pd.concat([X_test_known, X_test_novel], ignore_index=True)
            y_test = np.array([0] * len(X_test_known) + [1] * len(X_test_novel))
        else:
            # Use all non-training data as test
            X_test = df[~train_mask].copy()
            y_test = np.array([1] * len(X_test))
        
        logger.info(f"Test samples: {len(X_test)}")
        logger.info(f"  - Known: {np.sum(y_test == 0)}")
        logger.info(f"  - Novel: {np.sum(y_test == 1)}")
        
        return X_train, X_test, y_test
    
    def extract_features(self, df, feature_cols=None):
        """
        Extract features untuk open-set detection
        Menggunakan semua features yang relevan
        
        Args:
            df: DataFrame
            feature_cols: List kolom features (jika None, auto-detect)
            
        Returns:
            Feature matrix
        """
        if feature_cols is None:
            # Auto-detect: semua features kecuali metadata
            exclude_cols = ['sample_id', 'description', 'serotype', 'genotype', 
                           'country', 'year', 'host', 'is_complete']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        X = X.fillna(0)
        
        # Convert to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        logger.info(f"Extracted {len(feature_cols)} features")
        
        return X, feature_cols
    
    def train_autoencoder(self, X_train, encoding_dim=32, hidden_layers=[64, 32]):
        """
        Train Autoencoder untuk open-set detection
        
        Args:
            X_train: Training features
            encoding_dim: Dimension of encoding layer
            hidden_layers: List of hidden layer sizes
        """
        logger.info("Training Autoencoder...")
        
        autoencoder = Autoencoder(
            encoding_dim=encoding_dim,
            hidden_layers=hidden_layers
        )
        
        autoencoder.fit(X_train)
        self.models['autoencoder'] = autoencoder
        
        # Set threshold
        autoencoder.set_threshold(X_train, percentile=95)
        self.threshold = autoencoder.reconstruction_error_threshold
        
        logger.info("Autoencoder training completed")
        return autoencoder
    
    def train_distance_based(self, X_train, method='euclidean'):
        """
        Train distance-based classifier untuk open-set detection
        
        Args:
            X_train: Training features
            method: Distance method ('euclidean', 'cosine', 'manhattan')
        """
        logger.info(f"Training distance-based detector ({method})...")
        
        from sklearn.neighbors import NearestNeighbors
        
        # Calculate centroid of training data
        centroid = np.mean(X_train, axis=0)
        
        # Store for prediction
        self.models['distance_based'] = {
            'centroid': centroid,
            'method': method,
            'distances': []
        }
        
        # Calculate distances from centroid for training data
        if method == 'euclidean':
            distances = np.sqrt(np.sum((X_train - centroid) ** 2, axis=1))
        elif method == 'cosine':
            from sklearn.metrics.pairwise import cosine_distances
            distances = cosine_distances(X_train, centroid.reshape(1, -1)).flatten()
        elif method == 'manhattan':
            distances = np.sum(np.abs(X_train - centroid), axis=1)
        else:
            raise ValueError(f"Unknown distance method: {method}")
        
        self.models['distance_based']['distances'] = distances
        
        # Set threshold (95th percentile)
        self.threshold = np.percentile(distances, 95)
        
        logger.info(f"Distance-based detector training completed")
        logger.info(f"Threshold ({method}): {self.threshold:.4f}")
        
        return self.models['distance_based']
    
    def predict_open_set(self, X, model_name=None, threshold=None):
        """
        Predict apakah sample adalah open-set (potensi serotipe baru)
        
        Args:
            X: Feature matrix
            model_name: Model to use
            threshold: Threshold untuk prediction
            
        Returns:
            Binary predictions (1 = open-set/novel, 0 = known), scores
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if threshold is None:
            threshold = self.threshold
        
        if threshold is None:
            raise ValueError("Threshold not set!")
        
        if model_name == 'autoencoder':
            model = self.models['autoencoder']
            scores = model.reconstruction_error(X)
            predictions = (scores > threshold).astype(int)
            
        elif model_name == 'distance_based':
            model = self.models['distance_based']
            centroid = model['centroid']
            method = model['method']
            
            if method == 'euclidean':
                scores = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
            elif method == 'cosine':
                from sklearn.metrics.pairwise import cosine_distances
                scores = cosine_distances(X, centroid.reshape(1, -1)).flatten()
            elif method == 'manhattan':
                scores = np.sum(np.abs(X - centroid), axis=1)
            
            predictions = (scores > threshold).astype(int)
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return predictions, scores
    
    def evaluate(self, X_test, y_test, model_name=None, threshold=None):
        """
        Evaluate open-set detection performance
        
        Args:
            X_test: Test features
            y_test: True labels (1 = novel/open-set, 0 = known)
            model_name: Model to use
            threshold: Threshold untuk prediction
            
        Returns:
            Dictionary dengan metrics
        """
        logger.info("Evaluating open-set detection...")
        
        predictions, scores = self.predict_open_set(X_test, model_name, threshold)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(y_test, scores)
        except:
            roc_auc = 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'threshold': threshold if threshold else self.threshold
        }
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"ROC-AUC: {roc_auc:.4f}")
        
        return metrics, scores
    
    def save_model(self, filepath='models/open_set_detector.pkl'):
        """Save trained model"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'models': self.models,
            'best_model_name': self.best_model_name,
            'threshold': self.threshold,
            'scaler': self.scaler
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved model to {filepath}")
    
    def load_model(self, filepath='models/open_set_detector.pkl'):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.best_model_name = model_data['best_model_name']
        self.threshold = model_data['threshold']
        self.scaler = model_data.get('scaler', StandardScaler())
        
        logger.info(f"Loaded model from {filepath}")


if __name__ == "__main__":
    # Test open-set detection
    from data_cleaning import DataCleaner
    from feature_engineering import FeatureEngineer
    
    # Load and clean data
    cleaner = DataCleaner(dataset_dir='dataset')
    cleaner.load_datasets()
    cleaner.merge_tables()
    df = cleaner.filter_missing_labels('serotype')
    
    # Get unique serotypes
    unique_serotypes = df['serotype'].dropna().unique()
    logger.info(f"Available serotypes: {unique_serotypes}")
    
    if len(unique_serotypes) >= 2:
        # Split: train on first 2 serotypes, test on others
        train_serotypes = list(unique_serotypes[:2])
        test_serotypes = list(unique_serotypes[2:]) if len(unique_serotypes) > 2 else None
        
        # Prepare data
        detector = OpenSetDetector()
        X_train_df, X_test_df, y_test = detector.prepare_open_set_data(
            df, train_serotypes, test_serotypes
        )
        
        # Extract features
        X_train, feature_cols = detector.extract_features(X_train_df)
        X_test, _ = detector.extract_features(X_test_df, feature_cols)
        
        # Scale features
        X_train_scaled = detector.scaler.fit_transform(X_train)
        X_test_scaled = detector.scaler.transform(X_test)
        
        # Train autoencoder
        detector.train_autoencoder(X_train_scaled)
        detector.best_model_name = 'autoencoder'
        
        # Evaluate
        metrics, scores = detector.evaluate(X_test_scaled, y_test)
        
        logger.info("\nOpen-set detection pipeline completed!")

