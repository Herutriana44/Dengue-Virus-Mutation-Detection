"""
Main Pipeline Script
End-to-end pipeline untuk semua tasks sesuai alur_pipeline.md
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from data_cleaning import DataCleaner
from feature_engineering import FeatureEngineer
from task1_baseline_classification import BaselineClassifier
from task2_novelty_detection import NoveltyDetector
from task3_open_set_detection import OpenSetDetector
from model_interpretation import ModelInterpreter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLPipeline:
    """Main pipeline class untuk semua ML tasks"""
    
    def __init__(self, dataset_dir='dataset', output_dir='results'):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cleaner = DataCleaner(dataset_dir=dataset_dir)
        self.engineer = FeatureEngineer()
        self.baseline_classifier = BaselineClassifier()
        self.novelty_detector = NoveltyDetector()
        self.open_set_detector = OpenSetDetector()
        self.interpreter = ModelInterpreter(output_dir=str(self.output_dir / 'interpretation'))
        
        self.cleaned_data = None
        self.X = None
        self.y = None
        self.feature_names = []
        
    def run_stage1_data_preparation(self):
        """STAGE 1: Dataset Preparation"""
        logger.info("=" * 70)
        logger.info("STAGE 1: Dataset Preparation")
        logger.info("=" * 70)
        
        # Load and clean data
        self.cleaned_data = self.cleaner.run_full_cleaning(
            label_column='serotype',
            save_output=True
        )
        
        logger.info("Stage 1 completed!\n")
        return self.cleaned_data
    
    def run_stage2_feature_engineering(self):
        """STAGE 2: Feature Selection & Encoding"""
        logger.info("=" * 70)
        logger.info("STAGE 2: Feature Selection & Encoding")
        logger.info("=" * 70)
        
        if self.cleaned_data is None:
            raise ValueError("Please run Stage 1 first!")
        
        # Group features
        self.engineer.group_features(self.cleaned_data)
        
        # Prepare features
        self.X, self.y, self.feature_names = self.engineer.prepare_features(
            self.cleaned_data,
            target_column='serotype'
        )
        
        logger.info(f"Feature matrix shape: {self.X.shape}")
        logger.info(f"Number of features: {len(self.feature_names)}")
        logger.info("Stage 2 completed!\n")
        
        return self.X, self.y
    
    def run_task1_baseline_classification(self, test_size=0.2, min_samples_per_class=2, cv_folds=5):
        """TASK 1: Closed-set Classification"""
        logger.info("=" * 70)
        logger.info("TASK 1: Closed-set Classification (Baseline)")
        logger.info("=" * 70)
        
        if self.X is None:
            raise ValueError("Please run Stage 2 first!")
        
        from sklearn.model_selection import train_test_split
        
        # Filter classes with too few samples for stratified split and CV
        # Need at least cv_folds samples per class for stratified CV
        # But for train_test_split, we need at least 2 samples per class
        value_counts = self.y.value_counts()
        
        # For stratified CV, we need at least cv_folds samples per class in training set
        # After train_test_split with test_size, training will have (1-test_size) of data
        # So we need: class_size * (1-test_size) >= cv_folds
        # Which means: class_size >= cv_folds / (1-test_size)
        min_for_cv = int(np.ceil(cv_folds / (1 - test_size)))
        min_required = max(min_samples_per_class, min_for_cv)
        
        valid_classes = value_counts[value_counts >= min_required].index.tolist()
        
        if len(valid_classes) < len(value_counts):
            removed_classes = set(value_counts.index) - set(valid_classes)
            logger.warning(f"Removing {len(removed_classes)} classes with < {min_required} samples: {removed_classes}")
            logger.info(f"Minimum required: {min_required} samples per class (for {cv_folds}-fold CV)")
            
            # Filter data to only include valid classes
            mask = self.y.isin(valid_classes)
            X_filtered = self.X[mask].copy()
            y_filtered = self.y[mask].copy()
            
            logger.info(f"Filtered dataset: {len(X_filtered)} samples (removed {len(self.X) - len(X_filtered)} samples)")
            logger.info(f"Remaining classes: {len(valid_classes)}")
        else:
            X_filtered = self.X.copy()
            y_filtered = self.y.copy()
        
        # Check if we can do stratified split
        value_counts_filtered = y_filtered.value_counts()
        min_class_size = value_counts_filtered.min()
        
        if min_class_size < 2:
            logger.warning("Cannot use stratified split (some classes have < 2 samples). Using random split.")
            stratify = None
        else:
            stratify = y_filtered
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_filtered,
            test_size=test_size,
            random_state=42,
            stratify=stratify
        )
        
        # Scale features
        X_train_scaled = self.engineer.scale_features(X_train, fit=True)
        X_test_scaled = self.engineer.scale_features(X_test, fit=False)
        
        # Train models
        logger.info("Training Random Forest...")
        self.baseline_classifier.train_random_forest(X_train_scaled, y_train)
        
        logger.info("Training XGBoost...")
        self.baseline_classifier.train_xgboost(X_train_scaled, y_train)
        
        # Cross-validation
        logger.info("\nPerforming cross-validation...")
        cv_results = self.baseline_classifier.cross_validate(
            X_train_scaled, y_train, cv_folds=cv_folds
        )
        
        # Evaluate on test set
        logger.info("\nEvaluating on test set...")
        metrics = self.baseline_classifier.evaluate(X_test_scaled, y_test)
        
        # Feature importance
        importance_df = self.baseline_classifier.get_feature_importance(top_n=20)
        
        # Save model
        self.baseline_classifier.save_model(
            filepath=str(self.output_dir / 'models' / 'baseline_classifier.pkl')
        )
        
        # Save preprocessor for inference
        self.engineer.save_preprocessor(
            filepath=str(self.output_dir / 'preprocessor.pkl')
        )
        
        # Save results
        results = {
            'cv_results': cv_results,
            'test_metrics': metrics,
            'feature_importance': importance_df.to_dict() if importance_df is not None else None
        }
        
        logger.info("Task 1 completed!\n")
        return results
    
    def run_task2_novelty_detection(self):
        """TASK 2: Genotype Novelty Detection"""
        logger.info("=" * 70)
        logger.info("TASK 2: Genotype Novelty Detection")
        logger.info("=" * 70)
        
        if self.cleaned_data is None:
            raise ValueError("Please run Stage 1 first!")
        
        # Check if genotype data available
        if 'genotype' not in self.cleaned_data.columns:
            logger.warning("Genotype column not found. Skipping Task 2.")
            return None
        
        # Get unique genotypes
        unique_genotypes = self.cleaned_data['genotype'].dropna().unique()
        logger.info(f"Available genotypes: {unique_genotypes}")
        
        if len(unique_genotypes) < 2:
            logger.warning("Need at least 2 genotypes for novelty detection. Skipping Task 2.")
            return None
        
        # Split: train on first genotype, test on others
        train_genotypes = [unique_genotypes[0]]
        test_genotypes = list(unique_genotypes[1:])
        
        # Prepare data
        X_train_df, X_test_df, y_test = self.novelty_detector.prepare_novelty_data(
            self.cleaned_data, train_genotypes, test_genotypes
        )
        
        # Extract features
        X_train, feature_cols = self.novelty_detector.extract_features(X_train_df)
        X_test, _ = self.novelty_detector.extract_features(X_test_df, feature_cols)
        
        # Scale features
        X_train_scaled = self.novelty_detector.scaler.fit_transform(X_train)
        X_test_scaled = self.novelty_detector.scaler.transform(X_test)
        
        # Train models
        logger.info("Training Isolation Forest...")
        self.novelty_detector.train_isolation_forest(X_train_scaled)
        self.novelty_detector.best_model = self.novelty_detector.models['isolation_forest']
        self.novelty_detector.best_model_name = 'isolation_forest'
        
        # Determine threshold
        self.novelty_detector.determine_threshold(X_train_scaled)
        
        # Evaluate
        metrics, scores = self.novelty_detector.evaluate(X_test_scaled, y_test)
        
        # Save model
        self.novelty_detector.save_model(
            filepath=str(self.output_dir / 'models' / 'novelty_detector.pkl')
        )
        
        logger.info("Task 2 completed!\n")
        return {'metrics': metrics, 'scores': scores}
    
    def run_task3_open_set_detection(self):
        """TASK 3: Serotype Open-set Detection"""
        logger.info("=" * 70)
        logger.info("TASK 3: Serotype Open-set Detection")
        logger.info("=" * 70)
        
        if self.cleaned_data is None:
            raise ValueError("Please run Stage 1 first!")
        
        # Get unique serotypes
        unique_serotypes = self.cleaned_data['serotype'].dropna().unique()
        logger.info(f"Available serotypes: {unique_serotypes}")
        
        if len(unique_serotypes) < 2:
            logger.warning("Need at least 2 serotypes for open-set detection. Skipping Task 3.")
            return None
        
        # Split: train on first 2 serotypes, test on others
        train_serotypes = list(unique_serotypes[:2])
        test_serotypes = list(unique_serotypes[2:]) if len(unique_serotypes) > 2 else None
        
        # Prepare data
        X_train_df, X_test_df, y_test = self.open_set_detector.prepare_open_set_data(
            self.cleaned_data, train_serotypes, test_serotypes
        )
        
        # Extract features
        X_train, feature_cols = self.open_set_detector.extract_features(X_train_df)
        X_test, _ = self.open_set_detector.extract_features(X_test_df, feature_cols)
        
        # Scale features
        X_train_scaled = self.open_set_detector.scaler.fit_transform(X_train)
        X_test_scaled = self.open_set_detector.scaler.transform(X_test)
        
        # Train autoencoder
        logger.info("Training Autoencoder...")
        self.open_set_detector.train_autoencoder(X_train_scaled)
        self.open_set_detector.best_model_name = 'autoencoder'
        
        # Evaluate
        metrics, scores = self.open_set_detector.evaluate(X_test_scaled, y_test)
        
        # Save model
        self.open_set_detector.save_model(
            filepath=str(self.output_dir / 'models' / 'open_set_detector.pkl')
        )
        
        logger.info("Task 3 completed!\n")
        return {'metrics': metrics, 'scores': scores}
    
    def run_stage6_interpretation(self):
        """STAGE 6: Model Interpretation"""
        logger.info("=" * 70)
        logger.info("STAGE 6: Model Interpretation")
        logger.info("=" * 70)
        
        if self.baseline_classifier.best_model is None:
            logger.warning("No baseline model trained. Skipping interpretation.")
            return None
        
        # Get training data for interpretation
        from sklearn.model_selection import train_test_split
        
        # Filter classes with too few samples (same as Task 1)
        value_counts = self.y.value_counts()
        min_samples_per_class = 2  # Minimum for stratified split
        valid_classes = value_counts[value_counts >= min_samples_per_class].index.tolist()
        
        if len(valid_classes) < len(value_counts):
            removed_classes = set(value_counts.index) - set(valid_classes)
            logger.warning(f"Removing {len(removed_classes)} classes with < {min_samples_per_class} samples for interpretation: {removed_classes}")
            
            mask = self.y.isin(valid_classes)
            X_filtered = self.X[mask].copy()
            y_filtered = self.y[mask].copy()
        else:
            X_filtered = self.X.copy()
            y_filtered = self.y.copy()
        
        # Check if we can do stratified split
        value_counts_filtered = y_filtered.value_counts()
        min_class_size = value_counts_filtered.min()
        
        if min_class_size < 2:
            logger.warning("Cannot use stratified split for interpretation. Using random split.")
            stratify = None
        else:
            stratify = y_filtered
        
        X_train, _, y_train, _ = train_test_split(
            X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=stratify
        )
        X_train_scaled = self.engineer.scale_features(X_train, fit=True)
        
        # Generate interpretation report
        report = self.interpreter.generate_interpretation_report(
            self.baseline_classifier.best_model,
            X_train_scaled,
            y_train,
            self.feature_names
        )
        
        logger.info("Stage 6 completed!\n")
        return report
    
    def run_full_pipeline(self, tasks=['baseline', 'novelty', 'open_set', 'interpretation']):
        """
        Run full ML pipeline
        
        Args:
            tasks: List tasks to run ['baseline', 'novelty', 'open_set', 'interpretation']
        """
        logger.info("=" * 70)
        logger.info("STARTING FULL ML PIPELINE")
        logger.info("=" * 70)
        
        results = {}
        
        # Stage 1: Data Preparation
        self.run_stage1_data_preparation()
        
        # Stage 2: Feature Engineering
        self.run_stage2_feature_engineering()
        
        # Task 1: Baseline Classification
        if 'baseline' in tasks:
            results['task1'] = self.run_task1_baseline_classification()
        
        # Task 2: Novelty Detection
        if 'novelty' in tasks:
            results['task2'] = self.run_task2_novelty_detection()
        
        # Task 3: Open-set Detection
        if 'open_set' in tasks:
            results['task3'] = self.run_task3_open_set_detection()
        
        # Stage 6: Interpretation
        if 'interpretation' in tasks and 'baseline' in tasks:
            results['interpretation'] = self.run_stage6_interpretation()
        
        logger.info("=" * 70)
        logger.info("FULL ML PIPELINE COMPLETED!")
        logger.info("=" * 70)
        
        return results


def main():
    """Main function untuk command-line interface"""
    parser = argparse.ArgumentParser(
        description='Dengue Virus Mutation Detection ML Pipeline'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='dataset',
        help='Directory containing dataset CSV files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results and models'
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=['baseline', 'novelty', 'open_set', 'interpretation'],
        choices=['baseline', 'novelty', 'open_set', 'interpretation'],
        help='Tasks to run'
    )
    parser.add_argument(
        '--stage',
        type=int,
        choices=[1, 2],
        help='Run specific stage only'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MLPipeline(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir
    )
    
    # Run pipeline
    if args.stage == 1:
        pipeline.run_stage1_data_preparation()
    elif args.stage == 2:
        pipeline.run_stage1_data_preparation()
        pipeline.run_stage2_feature_engineering()
    else:
        results = pipeline.run_full_pipeline(tasks=args.tasks)
        logger.info("\nPipeline execution completed!")
        logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

