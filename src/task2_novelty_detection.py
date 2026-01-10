"""
TASK 2 - Genotype Novelty Detection
Deteksi genotipe yang tidak dikenal saat training menggunakan anomaly detection
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoveltyDetector:
    """Class untuk genotype novelty detection menggunakan anomaly detection"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.threshold = None
        self.best_model = None
        self.best_model_name = None
        
    def prepare_novelty_data(self, df, train_genotypes, test_genotypes=None):
        """
        Prepare data untuk novelty detection
        
        Args:
            df: DataFrame dengan semua data
            train_genotypes: List genotype yang digunakan untuk training
            test_genotypes: List genotype untuk testing (novel genotypes)
            
        Returns:
            X_train, X_test, y_train, y_test (y adalah binary: known=0, novel=1)
        """
        logger.info("Preparing data for novelty detection...")
        
        # Filter training data (known genotypes)
        train_mask = df['genotype'].isin(train_genotypes)
        X_train = df[train_mask].copy()
        
        logger.info(f"Training samples (known genotypes): {len(X_train)}")
        logger.info(f"Known genotypes: {train_genotypes}")
        
        # Filter test data
        if test_genotypes:
            test_mask = df['genotype'].isin(test_genotypes)
            X_test_novel = df[test_mask].copy()
            
            # Also include some known genotypes for evaluation
            X_test_known = df[~train_mask & ~test_mask].head(len(X_test_novel)).copy()
            
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
        Extract features untuk novelty detection
        Menggunakan sequence + mutation features
        
        Args:
            df: DataFrame
            feature_cols: List kolom features (jika None, auto-detect)
            
        Returns:
            Feature matrix
        """
        if feature_cols is None:
            # Auto-detect: k-mer, GC content, mutation features
            feature_cols = [col for col in df.columns 
                          if col.startswith('kmer_') or 
                          col == 'gc_content' or 
                          'mutation' in col.lower() or
                          'mut' in col.lower()]
        
        # Exclude non-feature columns
        exclude_cols = ['sample_id', 'description', 'serotype', 'genotype', 
                       'country', 'year', 'host', 'is_complete']
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        X = X.fillna(0)
        
        # Convert to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        logger.info(f"Extracted {len(feature_cols)} features")
        
        return X, feature_cols
    
    def train_isolation_forest(self, X_train, contamination=0.1, random_state=42):
        """
        Train Isolation Forest untuk anomaly detection
        
        Args:
            X_train: Training features (known genotypes only)
            contamination: Expected proportion of anomalies
            random_state: Random seed
        """
        logger.info("Training Isolation Forest...")
        
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            n_jobs=-1
        )
        
        iso_forest.fit(X_train)
        self.models['isolation_forest'] = iso_forest
        
        logger.info("Isolation Forest training completed")
        return iso_forest
    
    def train_oneclass_svm(self, X_train, nu=0.1, kernel='rbf', gamma='scale'):
        """
        Train One-Class SVM untuk novelty detection
        
        Args:
            X_train: Training features
            nu: Upper bound on fraction of outliers
            kernel: Kernel type
            gamma: Kernel coefficient
        """
        logger.info("Training One-Class SVM...")
        
        oc_svm = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma=gamma
        )
        
        oc_svm.fit(X_train)
        self.models['oneclass_svm'] = oc_svm
        
        logger.info("One-Class SVM training completed")
        return oc_svm
    
    def predict_anomaly_scores(self, X, model_name=None):
        """
        Predict anomaly scores
        
        Args:
            X: Feature matrix
            model_name: Model to use (if None, use best model)
            
        Returns:
            Anomaly scores (higher = more anomalous/novel)
        """
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.models[model_name]
        
        if model is None:
            raise ValueError("No model available!")
        
        # Predict scores
        if model_name == 'isolation_forest':
            scores = -model.score_samples(X)  # Negative because lower score = more anomalous
        elif model_name == 'oneclass_svm':
            scores = -model.decision_function(X)  # Negative decision function = more anomalous
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return scores
    
    def determine_threshold(self, X_train, percentile=95):
        """
        Determine threshold untuk novelty detection berdasarkan training data
        
        Args:
            X_train: Training features
            percentile: Percentile untuk threshold (default 95th)
            
        Returns:
            Threshold value
        """
        scores = self.predict_anomaly_scores(X_train)
        self.threshold = np.percentile(scores, percentile)
        
        logger.info(f"Threshold (95th percentile): {self.threshold:.4f}")
        
        return self.threshold
    
    def predict_novel(self, X, threshold=None):
        """
        Predict apakah sample adalah novel genotype
        
        Args:
            X: Feature matrix
            threshold: Threshold untuk novelty (if None, use self.threshold)
            
        Returns:
            Binary predictions (1 = novel, 0 = known)
        """
        if threshold is None:
            threshold = self.threshold
        
        if threshold is None:
            raise ValueError("Threshold not set! Call determine_threshold() first.")
        
        scores = self.predict_anomaly_scores(X)
        predictions = (scores > threshold).astype(int)
        
        return predictions, scores
    
    def evaluate(self, X_test, y_test, threshold=None):
        """
        Evaluate novelty detection performance
        
        Args:
            X_test: Test features
            y_test: True labels (1 = novel, 0 = known)
            threshold: Threshold untuk prediction
            
        Returns:
            Dictionary dengan metrics
        """
        logger.info("Evaluating novelty detection...")
        
        predictions, scores = self.predict_novel(X_test, threshold)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        # ROC-AUC menggunakan scores sebagai probabilities
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
    
    def save_model(self, filepath='models/novelty_detector.pkl'):
        """Save trained model"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'models': self.models,
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'threshold': self.threshold,
            'scaler': self.scaler
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved model to {filepath}")
    
    def load_model(self, filepath='models/novelty_detector.pkl'):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.threshold = model_data['threshold']
        self.scaler = model_data.get('scaler', StandardScaler())
        
        logger.info(f"Loaded model from {filepath}")


if __name__ == "__main__":
    # Test novelty detection
    from data_cleaning import DataCleaner
    from feature_engineering import FeatureEngineer
    
    # Load and clean data
    cleaner = DataCleaner(dataset_dir='dataset')
    cleaner.load_datasets()
    cleaner.merge_tables()
    df = cleaner.filter_missing_labels('genotype')
    
    # Get unique genotypes
    unique_genotypes = df['genotype'].dropna().unique()
    logger.info(f"Available genotypes: {unique_genotypes}")
    
    if len(unique_genotypes) >= 2:
        # Split: train on first genotype, test on others
        train_genotypes = [unique_genotypes[0]]
        test_genotypes = list(unique_genotypes[1:])
        
        # Prepare data
        detector = NoveltyDetector()
        X_train_df, X_test_df, y_test = detector.prepare_novelty_data(
            df, train_genotypes, test_genotypes
        )
        
        # Extract features
        X_train, feature_cols = detector.extract_features(X_train_df)
        X_test, _ = detector.extract_features(X_test_df, feature_cols)
        
        # Scale features
        X_train_scaled = detector.scaler.fit_transform(X_train)
        X_test_scaled = detector.scaler.transform(X_test)
        
        # Train models
        detector.train_isolation_forest(X_train_scaled)
        detector.best_model = detector.models['isolation_forest']
        detector.best_model_name = 'isolation_forest'
        
        # Determine threshold
        detector.determine_threshold(X_train_scaled)
        
        # Evaluate
        metrics, scores = detector.evaluate(X_test_scaled, y_test)
        
        logger.info("\nNovelty detection pipeline completed!")

