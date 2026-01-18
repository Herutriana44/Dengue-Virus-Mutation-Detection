"""
Inference Module
Modul untuk melakukan prediksi menggunakan model yang sudah di-train
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple

from data_cleaning import DataCleaner
from feature_engineering import FeatureEngineer
from task1_baseline_classification import BaselineClassifier
from task2_novelty_detection import NoveltyDetector
from task3_open_set_detection import OpenSetDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferencePipeline:
    """Class untuk inference menggunakan model yang sudah di-train"""
    
    def __init__(self, models_dir='results/models'):
        """
        Initialize Inference Pipeline
        
        Args:
            models_dir: Directory tempat model disimpan
        """
        self.models_dir = Path(models_dir)
        self.baseline_model = None
        self.baseline_classifier = BaselineClassifier()  # For label encoder access
        self.novelty_detector = None
        self.open_set_detector = None
        self.preprocessor = None
        self.feature_engineer = FeatureEngineer()
        self.data_cleaner = DataCleaner()
        
    def load_models(self):
        """Load semua model yang sudah di-train"""
        logger.info("Loading models...")
        
        # Load baseline classifier
        baseline_path = self.models_dir / 'baseline_classifier.pkl'
        if baseline_path.exists():
            with open(baseline_path, 'rb') as f:
                model_data = pickle.load(f)
                self.baseline_model = model_data['model']
                # Also store in baseline_classifier for label encoder access
                self.baseline_classifier.best_model = model_data['model']
                self.baseline_classifier.best_model_name = model_data['model_name']
                # Load label encoder if available (for XGBoost)
                if 'label_encoder' in model_data:
                    self.baseline_classifier.label_encoder = model_data['label_encoder']
                    logger.info("Loaded label encoder for XGBoost")
                logger.info(f"Loaded baseline model: {model_data['model_name']}")
        else:
            logger.warning(f"Baseline model not found at {baseline_path}")
        
        # Load novelty detector
        novelty_path = self.models_dir / 'novelty_detector.pkl'
        if novelty_path.exists():
            self.novelty_detector = NoveltyDetector()
            self.novelty_detector.load_model(str(novelty_path))
            logger.info("Loaded novelty detector")
        else:
            logger.warning(f"Novelty detector not found at {novelty_path}")
        
        # Load open-set detector
        open_set_path = self.models_dir / 'open_set_detector.pkl'
        if open_set_path.exists():
            self.open_set_detector = OpenSetDetector()
            self.open_set_detector.load_model(str(open_set_path))
            logger.info("Loaded open-set detector")
        else:
            logger.warning(f"Open-set detector not found at {open_set_path}")
        
        # Load preprocessor (important for feature alignment)
        preprocessor_path = self.models_dir.parent / 'preprocessor.pkl'
        if preprocessor_path.exists():
            self.feature_engineer.load_preprocessor(str(preprocessor_path))
            logger.info("Loaded preprocessor for feature alignment")
        else:
            logger.warning(f"Preprocessor not found at {preprocessor_path}. Feature alignment may fail.")
        
        logger.info("Model loading completed")
    
    def prepare_input_data(self, input_data, is_dataframe=True):
        """
        Prepare input data untuk inference
        
        Args:
            input_data: DataFrame atau path ke CSV file
            is_dataframe: True jika input_data adalah DataFrame, False jika path
            
        Returns:
            Prepared DataFrame
        """
        if not is_dataframe:
            # Load from CSV
            df = pd.read_csv(input_data)
        else:
            df = input_data.copy()
        
        # Ensure sample_id exists
        if 'sample_id' not in df.columns:
            df['sample_id'] = range(len(df))
        
        return df
    
    def predict_baseline(self, X, return_proba=False):
        """
        Predict serotipe menggunakan baseline classifier
        
        Args:
            X: Feature matrix (DataFrame atau numpy array)
            return_proba: Apakah return probability scores
            
        Returns:
            Predictions dan probabilities (jika return_proba=True)
        """
        if self.baseline_model is None:
            raise ValueError("Baseline model not loaded! Please load models first.")
        
        # Check if model is from BaselineClassifier class (has label_encoder attribute)
        # or if it's the model itself
        model_to_use = self.baseline_model
        label_encoder = None
        model_name = None
        
        # Get model name and label encoder from BaselineClassifier if available
        if hasattr(self, 'baseline_classifier'):
            if hasattr(self.baseline_classifier, 'label_encoder'):
                label_encoder = self.baseline_classifier.label_encoder
            if hasattr(self.baseline_classifier, 'best_model_name'):
                model_name = self.baseline_classifier.best_model_name
        
        logger.info(f"Using model: {model_name}, has label_encoder: {label_encoder is not None}")
        
        # Only use label encoder for XGBoost models
        # Other models (Random Forest) predict string labels directly
        if model_name == 'xgboost' and label_encoder is not None:
            # XGBoost uses encoded labels, need to decode
            y_pred_encoded = model_to_use.predict(X)
            
            # Check if label encoder has classes
            if not hasattr(label_encoder, 'classes_') or label_encoder.classes_ is None:
                logger.warning("Label encoder has no classes. Using predictions as-is.")
                y_pred = y_pred_encoded
            else:
                # Check if predictions are within valid range
                max_class_idx = len(label_encoder.classes_) - 1
                valid_indices = (y_pred_encoded >= 0) & (y_pred_encoded <= max_class_idx)
                
                if not np.all(valid_indices):
                    invalid_indices = np.where(~valid_indices)[0]
                    logger.warning(f"Found {len(invalid_indices)} predictions outside valid range [0, {max_class_idx}]. "
                                 f"First few invalid values: {y_pred_encoded[invalid_indices][:5]}")
                    # Clip to valid range
                    y_pred_encoded = np.clip(y_pred_encoded, 0, max_class_idx)
                
                # Decode predictions
                try:
                    y_pred = label_encoder.inverse_transform(y_pred_encoded.astype(int))
                except ValueError as e:
                    logger.error(f"Error decoding predictions: {e}")
                    logger.error(f"Predicted values: {np.unique(y_pred_encoded)}")
                    logger.error(f"Label encoder classes: {label_encoder.classes_}")
                    raise
        else:
            # Random Forest or other models predict string labels directly
            y_pred = model_to_use.predict(X)
        
        if return_proba and hasattr(model_to_use, 'predict_proba'):
            proba = model_to_use.predict_proba(X)
            return y_pred, proba
        
        return y_pred
    
    def predict_novelty(self, X):
        """
        Predict apakah sample adalah novel genotype
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions (1 = novel, 0 = known), anomaly scores
        """
        if self.novelty_detector is None:
            raise ValueError("Novelty detector not loaded! Please load models first.")
        
        predictions, scores = self.novelty_detector.predict_novel(X)
        return predictions, scores
    
    def predict_open_set(self, X):
        """
        Predict apakah sample adalah open-set (potensi serotipe baru)
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions (1 = open-set/novel, 0 = known), scores
        """
        if self.open_set_detector is None:
            raise ValueError("Open-set detector not loaded! Please load models first.")
        
        predictions, scores = self.open_set_detector.predict_open_set(X)
        return predictions, scores
    
    def run_full_inference(self, input_data, is_dataframe=True, 
                          tasks=['baseline', 'novelty', 'open_set']):
        """
        Run full inference pipeline
        
        Args:
            input_data: Input data (DataFrame atau path ke CSV)
            is_dataframe: True jika input_data adalah DataFrame
            tasks: List tasks untuk dijalankan
            
        Returns:
            Dictionary dengan semua predictions
        """
        logger.info("=" * 70)
        logger.info("STARTING INFERENCE PIPELINE")
        logger.info("=" * 70)
        
        # Prepare input data
        df = self.prepare_input_data(input_data, is_dataframe)
        logger.info(f"Input data shape: {df.shape}")
        
        results = {
            'sample_id': df['sample_id'].values if 'sample_id' in df.columns else None
        }
        
        # Extract features (same as training)
        if 'novelty' in tasks or 'open_set' in tasks:
            # For novelty/open-set, extract specific features
            if 'novelty' in tasks and self.novelty_detector:
                X_novelty, feature_cols_novelty = self.novelty_detector.extract_features(df)
                X_novelty_scaled = self.novelty_detector.scaler.transform(X_novelty)
            
            if 'open_set' in tasks and self.open_set_detector:
                X_open_set, feature_cols_open_set = self.open_set_detector.extract_features(df)
                X_open_set_scaled = self.open_set_detector.scaler.transform(X_open_set)
        
        # Baseline classification
        if 'baseline' in tasks and self.baseline_model:
            logger.info("Running baseline classification...")
            
            # Use feature engineer to prepare features
            # Preprocessor should already be loaded in load_models()
            if self.feature_engineer.feature_names:
                expected_feature_names = self.feature_engineer.feature_names
                logger.info(f"Using {len(expected_feature_names)} expected feature names from preprocessor")
            else:
                logger.warning("No preprocessor found! Feature alignment may fail.")
                expected_feature_names = None
            
            self.feature_engineer.group_features(df)
            X_baseline, _, feature_names = self.feature_engineer.prepare_features(
                df, 
                target_column='serotype',  # May not exist, that's OK
                expected_feature_names=expected_feature_names  # Use expected features from training
            )
            
            logger.info(f"Prepared features: {X_baseline.shape}, feature names: {len(feature_names)}")
            if expected_feature_names:
                columns_match = list(X_baseline.columns) == expected_feature_names
                logger.info(f"Feature names match: {columns_match}")
                if not columns_match:
                    logger.warning(f"Feature names don't match! Will align before scaling.")
                    logger.info(f"Expected: {len(expected_feature_names)} features")
                    logger.info(f"Got: {len(X_baseline.columns)} features")
                    # Show first few mismatches
                    if len(X_baseline.columns) > 0 and len(expected_feature_names) > 0:
                        missing = set(expected_feature_names) - set(X_baseline.columns)
                        extra = set(X_baseline.columns) - set(expected_feature_names)
                        if missing:
                            logger.warning(f"Missing columns (first 5): {list(missing)[:5]}")
                        if extra:
                            logger.warning(f"Extra columns (first 5): {list(extra)[:5]}")
            
            # Scale features (use existing scaler from training)
            if self.feature_engineer.scaler is not None:
                # Ensure feature names are set correctly before scaling
                if expected_feature_names and not self.feature_engineer.feature_names:
                    self.feature_engineer.feature_names = expected_feature_names
                    logger.info("Set feature_names in feature_engineer for scaling")
                
                X_baseline_scaled = self.feature_engineer.scale_features(
                    X_baseline, fit=False  # Use existing scaler
                )
            else:
                # If no scaler, fit new one (not ideal but works)
                logger.warning("No preprocessor found, fitting new scaler (may cause inconsistency)")
                X_baseline_scaled = self.feature_engineer.scale_features(
                    X_baseline, fit=True
                )
            
            # Predict
            y_pred, y_proba = self.predict_baseline(X_baseline_scaled, return_proba=True)
            
            results['baseline'] = {
                'predicted_serotype': y_pred,
                'probabilities': y_proba if y_proba is not None else None
            }
            
            logger.info(f"Baseline predictions completed: {len(y_pred)} samples")
        
        # Novelty detection
        if 'novelty' in tasks and self.novelty_detector:
            logger.info("Running novelty detection...")
            
            predictions, scores = self.predict_novelty(X_novelty_scaled)
            
            results['novelty'] = {
                'is_novel_genotype': predictions,
                'anomaly_score': scores,
                'threshold': self.novelty_detector.threshold
            }
            
            logger.info(f"Novelty detection completed: {len(predictions)} samples")
            logger.info(f"Novel samples detected: {np.sum(predictions)}")
        
        # Open-set detection
        if 'open_set' in tasks and self.open_set_detector:
            logger.info("Running open-set detection...")
            
            predictions, scores = self.predict_open_set(X_open_set_scaled)
            
            results['open_set'] = {
                'is_potential_new_serotype': predictions,
                'reconstruction_error': scores,
                'threshold': self.open_set_detector.threshold
            }
            
            logger.info(f"Open-set detection completed: {len(predictions)} samples")
            logger.info(f"Potential new serotypes detected: {np.sum(predictions)}")
        
        logger.info("=" * 70)
        logger.info("INFERENCE PIPELINE COMPLETED")
        logger.info("=" * 70)
        
        return results
    
    def format_results(self, results, output_format='dataframe'):
        """
        Format results ke format yang mudah dibaca
        
        Args:
            results: Results dari run_full_inference
            output_format: 'dataframe' atau 'dict'
            
        Returns:
            Formatted results
        """
        if output_format == 'dataframe':
            # Combine semua results ke satu DataFrame
            output_data = []
            
            n_samples = len(results.get('sample_id', []))
            if n_samples == 0:
                n_samples = len(results.get('baseline', {}).get('predicted_serotype', []))
            
            for i in range(n_samples):
                row = {}
                
                if results.get('sample_id') is not None:
                    row['sample_id'] = results['sample_id'][i]
                
                if 'baseline' in results:
                    row['predicted_serotype'] = results['baseline']['predicted_serotype'][i]
                    if results['baseline']['probabilities'] is not None:
                        row['prediction_confidence'] = np.max(results['baseline']['probabilities'][i])
                
                if 'novelty' in results:
                    row['is_novel_genotype'] = bool(results['novelty']['is_novel_genotype'][i])
                    row['anomaly_score'] = results['novelty']['anomaly_score'][i]
                    row['novelty_threshold'] = results['novelty']['threshold']
                
                if 'open_set' in results:
                    row['is_potential_new_serotype'] = bool(results['open_set']['is_potential_new_serotype'][i])
                    row['reconstruction_error'] = results['open_set']['reconstruction_error'][i]
                    row['open_set_threshold'] = results['open_set']['threshold']
                
                output_data.append(row)
            
            return pd.DataFrame(output_data)
        
        return results
    
    def save_results(self, results, output_path='inference_results.csv'):
        """Save inference results ke CSV"""
        df_results = self.format_results(results, output_format='dataframe')
        df_results.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        return output_path


if __name__ == "__main__":
    # Example usage
    inference = InferencePipeline(models_dir='results/models')
    inference.load_models()
    
    # Example: predict on new data
    # results = inference.run_full_inference('new_data.csv', is_dataframe=False)
    # df_results = inference.format_results(results)
    # inference.save_results(results, 'predictions.csv')
    
    print("Inference pipeline ready!")

