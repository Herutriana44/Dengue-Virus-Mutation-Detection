"""
TASK 1 - Closed-set Classification (Baseline)
Menunjukkan bahwa fitur biologis memang informatif untuk klasifikasi serotipe/genotipe
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb
import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineClassifier:
    """Class untuk baseline classification task"""
    
    def __init__(self):
        self.models = {}
        self.cv_results = {}
        self.best_model = None
        self.best_model_name = None
        self.label_encoder = None  # For XGBoost label encoding
        
    def train_random_forest(self, X_train, y_train, n_estimators=100, max_depth=None, 
                           random_state=42, n_jobs=-1):
        """
        Train Random Forest classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        logger.info("Training Random Forest...")
        
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight='balanced'
        )
        
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        
        logger.info("Random Forest training completed")
        return rf
    
    def train_xgboost(self, X_train, y_train, n_estimators=100, max_depth=6,
                      learning_rate=0.1, random_state=42):
        """
        Train XGBoost classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            random_state: Random seed
        """
        logger.info("Training XGBoost...")
        
        # XGBoost requires numeric labels, so encode if needed
        from sklearn.preprocessing import LabelEncoder
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            y_train_encoded = self.label_encoder.fit_transform(y_train)
        else:
            y_train_encoded = self.label_encoder.transform(y_train)
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        
        xgb_model.fit(X_train, y_train_encoded)
        self.models['xgboost'] = xgb_model
        
        logger.info("XGBoost training completed")
        return xgb_model
    
    def cross_validate(self, X, y, cv_folds=5, target='serotype'):
        """
        Perform stratified k-fold cross-validation
        
        Args:
            X: Feature matrix
            y: Target labels
            cv_folds: Number of CV folds
            target: Target name for logging
        """
        logger.info(f"Performing {cv_folds}-fold stratified cross-validation...")
        
        # Check if stratified CV is possible
        from sklearn.model_selection import KFold
        value_counts = pd.Series(y).value_counts()
        min_class_size = value_counts.min()
        
        if min_class_size < cv_folds:
            logger.warning(f"Some classes have < {cv_folds} samples. Using regular KFold instead of StratifiedKFold.")
            skf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"\nEvaluating {model_name}...")
            
            # XGBoost needs encoded labels
            if model_name == 'xgboost' and self.label_encoder is not None:
                y_encoded = self.label_encoder.transform(y)
                cv_scores = cross_val_score(
                    model, X, y_encoded, 
                    cv=skf, 
                    scoring='f1_macro',
                    n_jobs=-1
                )
            else:
                cv_scores = cross_val_score(
                    model, X, y, 
                    cv=skf, 
                    scoring='f1_macro',
                    n_jobs=-1
                )
            
            results[model_name] = {
                'mean_f1': cv_scores.mean(),
                'std_f1': cv_scores.std(),
                'scores': cv_scores
            }
            
            logger.info(f"{model_name} - Mean F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.cv_results = results
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['mean_f1'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        logger.info(f"\nBest model: {best_model_name} (F1: {results[best_model_name]['mean_f1']:.4f})")
        
        return results
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary dengan metrics
        """
        if self.best_model is None:
            raise ValueError("No model trained yet!")
        
        logger.info(f"\nEvaluating {self.best_model_name} on test set...")
        
        # XGBoost needs encoded labels for prediction
        if self.best_model_name == 'xgboost' and self.label_encoder is not None:
            y_test_encoded = self.label_encoder.transform(y_test)
            y_pred_encoded = self.best_model.predict(X_test)
            y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        else:
            y_pred = self.best_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Macro F1: {f1_macro:.4f}")
        logger.info(f"Weighted F1: {f1_weighted:.4f}")
        logger.info("\nClassification Report:")
        logger.info(metrics['classification_report'])
        
        return metrics
    
    def get_feature_importance(self, top_n=20):
        """
        Get feature importance dari best model
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame dengan feature importance
        """
        if self.best_model is None:
            raise ValueError("No model trained yet!")
        
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_names = self.best_model.feature_names_in_ if hasattr(
                self.best_model, 'feature_names_in_') else None
            
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
            
            logger.info(f"\nTop {top_n} Important Features:")
            logger.info(importance_df.to_string(index=False))
            
            return importance_df
        else:
            logger.warning("Model does not support feature importance")
            return None
    
    def save_model(self, filepath='models/baseline_classifier.pkl'):
        """Save trained model"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'cv_results': self.cv_results,
            'label_encoder': self.label_encoder  # Save label encoder for XGBoost
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved model to {filepath}")
    
    def load_model(self, filepath='models/baseline_classifier.pkl'):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.cv_results = model_data.get('cv_results', {})
        
        logger.info(f"Loaded model from {filepath}")


if __name__ == "__main__":
    # Test baseline classification
    from data_cleaning import DataCleaner
    from feature_engineering import FeatureEngineer
    from sklearn.model_selection import train_test_split
    
    # Load and clean data
    cleaner = DataCleaner(dataset_dir='dataset')
    cleaner.load_datasets()
    cleaner.merge_tables()
    df = cleaner.filter_missing_labels('serotype')
    
    # Feature engineering
    engineer = FeatureEngineer()
    engineer.group_features(df)
    X, y, feature_names = engineer.prepare_features(df, target_column='serotype')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    X_train_scaled = engineer.scale_features(X_train, fit=True)
    X_test_scaled = engineer.scale_features(X_test, fit=False)
    
    # Train models
    classifier = BaselineClassifier()
    classifier.train_random_forest(X_train_scaled, y_train)
    classifier.train_xgboost(X_train_scaled, y_train)
    
    # Cross-validation
    cv_results = classifier.cross_validate(X_train_scaled, y_train)
    
    # Evaluate on test set
    metrics = classifier.evaluate(X_test_scaled, y_test)
    
    # Feature importance
    importance_df = classifier.get_feature_importance()

