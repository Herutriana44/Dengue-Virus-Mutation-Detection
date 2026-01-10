"""
STAGE 6 - Model Interpretation
Modul untuk interpretasi model menggunakan feature importance dan SHAP values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")


class ModelInterpreter:
    """Class untuk interpretasi model ML"""
    
    def __init__(self, output_dir='results/interpretation'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shap_explainer = None
        
    def plot_feature_importance(self, model, feature_names, top_n=20, 
                                save_path=None, title="Feature Importance"):
        """
        Plot feature importance dari model
        
        Args:
            model: Trained model dengan feature_importances_
            feature_names: List nama features
            top_n: Number of top features to plot
            save_path: Path untuk save plot
            title: Plot title
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_")
            return None
        
        importances = model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")
        else:
            save_path = self.output_dir / 'feature_importance.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")
        
        plt.close()
        
        return importance_df
    
    def compute_shap_values(self, model, X, feature_names, max_samples=100):
        """
        Compute SHAP values untuk interpretasi model
        
        Args:
            model: Trained model
            X: Feature matrix
            feature_names: List nama features
            max_samples: Maximum samples untuk SHAP (untuk performa)
            
        Returns:
            SHAP values
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Skipping SHAP analysis.")
            return None
        
        logger.info("Computing SHAP values...")
        
        # Sample data jika terlalu besar
        if len(X) > max_samples:
            X_sample = X.sample(max_samples, random_state=42)
            logger.info(f"Sampling {max_samples} samples for SHAP analysis")
        else:
            X_sample = X
        
        # Create explainer berdasarkan model type
        if hasattr(model, 'predict_proba'):
            # Tree-based models
            try:
                self.shap_explainer = shap.TreeExplainer(model)
                shap_values = self.shap_explainer.shap_values(X_sample)
            except:
                # Fallback to KernelExplainer
                logger.info("Using KernelExplainer (slower)...")
                self.shap_explainer = shap.KernelExplainer(model.predict_proba, X_sample[:50])
                shap_values = self.shap_explainer.shap_values(X_sample)
        else:
            # Other models
            logger.info("Using KernelExplainer...")
            self.shap_explainer = shap.KernelExplainer(model.predict, X_sample[:50])
            shap_values = self.shap_explainer.shap_values(X_sample)
        
        logger.info("SHAP values computed")
        
        return shap_values
    
    def plot_shap_summary(self, shap_values, X, feature_names, save_path=None, 
                         max_display=20):
        """
        Plot SHAP summary plot
        
        Args:
            shap_values: SHAP values
            X: Feature matrix
            feature_names: List nama features
            save_path: Path untuk save plot
            max_display: Maximum features to display
        """
        if not SHAP_AVAILABLE or shap_values is None:
            logger.warning("SHAP not available or values not computed")
            return
        
        logger.info("Plotting SHAP summary...")
        
        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Use first class
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names, 
                         max_display=max_display, show=False)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved SHAP summary plot to {save_path}")
        else:
            save_path = self.output_dir / 'shap_summary.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved SHAP summary plot to {save_path}")
        
        plt.close()
    
    def plot_shap_waterfall(self, shap_values, X, feature_names, sample_idx=0, 
                           save_path=None):
        """
        Plot SHAP waterfall plot untuk sample tertentu
        
        Args:
            shap_values: SHAP values
            X: Feature matrix
            feature_names: List nama features
            sample_idx: Index sample untuk diplot
            save_path: Path untuk save plot
        """
        if not SHAP_AVAILABLE or shap_values is None:
            logger.warning("SHAP not available or values not computed")
            return
        
        logger.info(f"Plotting SHAP waterfall for sample {sample_idx}...")
        
        # Handle multi-class
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[sample_idx],
                base_values=self.shap_explainer.expected_value[0] if isinstance(
                    self.shap_explainer.expected_value, list) else self.shap_explainer.expected_value,
                data=X.iloc[sample_idx] if isinstance(X, pd.DataFrame) else X[sample_idx],
                feature_names=feature_names
            ),
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved SHAP waterfall plot to {save_path}")
        else:
            save_path = self.output_dir / f'shap_waterfall_sample_{sample_idx}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved SHAP waterfall plot to {save_path}")
        
        plt.close()
    
    def map_features_to_biology(self, importance_df, feature_mapping=None):
        """
        Map features ke interpretasi biologis
        
        Args:
            importance_df: DataFrame dengan feature importance
            feature_mapping: Dictionary mapping feature names ke biological interpretation
            
        Returns:
            DataFrame dengan biological interpretation
        """
        logger.info("Mapping features to biological interpretation...")
        
        if feature_mapping is None:
            # Default mapping
            feature_mapping = {
                'gc_content': 'GC Content - Overall nucleotide composition',
                'mutation_density': 'Mutation Density - Rate of genetic variation',
                'kmer_': 'k-mer frequencies - Sequence patterns',
            }
        
        importance_df = importance_df.copy()
        importance_df['biological_interpretation'] = ''
        
        for idx, row in importance_df.iterrows():
            feature = row['feature']
            
            # Check mapping
            for key, interpretation in feature_mapping.items():
                if key in feature.lower():
                    importance_df.at[idx, 'biological_interpretation'] = interpretation
                    break
            
            # Default interpretation
            if importance_df.at[idx, 'biological_interpretation'] == '':
                if 'kmer' in feature.lower():
                    importance_df.at[idx, 'biological_interpretation'] = 'k-mer pattern'
                elif 'mutation' in feature.lower():
                    importance_df.at[idx, 'biological_interpretation'] = 'Mutation profile'
                elif 'gc' in feature.lower():
                    importance_df.at[idx, 'biological_interpretation'] = 'GC content'
                else:
                    importance_df.at[idx, 'biological_interpretation'] = 'Other feature'
        
        return importance_df
    
    def generate_interpretation_report(self, model, X, y, feature_names, 
                                      output_path=None):
        """
        Generate comprehensive interpretation report
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target labels
            feature_names: List nama features
            output_path: Path untuk save report
        """
        logger.info("Generating interpretation report...")
        
        report = []
        
        # 1. Feature Importance
        if hasattr(model, 'feature_importances_'):
            importance_df = self.plot_feature_importance(
                model, feature_names, 
                save_path=self.output_dir / 'feature_importance.png'
            )
            
            if importance_df is not None:
                report.append("=" * 50)
                report.append("FEATURE IMPORTANCE ANALYSIS")
                report.append("=" * 50)
                report.append(importance_df.to_string(index=False))
                report.append("")
        
        # 2. SHAP Analysis
        if SHAP_AVAILABLE:
            try:
                shap_values = self.compute_shap_values(model, X, feature_names)
                
                if shap_values is not None:
                    self.plot_shap_summary(
                        shap_values, X, feature_names,
                        save_path=self.output_dir / 'shap_summary.png'
                    )
                    
                    report.append("=" * 50)
                    report.append("SHAP ANALYSIS")
                    report.append("=" * 50)
                    report.append("SHAP values computed and visualized.")
                    report.append("See shap_summary.png for details.")
                    report.append("")
            except Exception as e:
                logger.warning(f"SHAP analysis failed: {e}")
        
        # 3. Biological Mapping
        if importance_df is not None:
            bio_df = self.map_features_to_biology(importance_df)
            report.append("=" * 50)
            report.append("BIOLOGICAL INTERPRETATION")
            report.append("=" * 50)
            report.append(bio_df.to_string(index=False))
        
        # Save report
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
        else:
            output_path = self.output_dir / 'interpretation_report.txt'
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        logger.info(f"Interpretation report saved to {output_path}")
        
        return report_text


if __name__ == "__main__":
    # Test interpretation
    from task1_baseline_classification import BaselineClassifier
    from data_cleaning import DataCleaner
    from feature_engineering import FeatureEngineer
    from sklearn.model_selection import train_test_split
    
    # Load data
    cleaner = DataCleaner(dataset_dir='dataset')
    cleaner.load_datasets()
    cleaner.merge_tables()
    df = cleaner.filter_missing_labels('serotype')
    
    # Feature engineering
    engineer = FeatureEngineer()
    engineer.group_features(df)
    X, y, feature_names = engineer.prepare_features(df, target_column='serotype')
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale
    X_train_scaled = engineer.scale_features(X_train, fit=True)
    X_test_scaled = engineer.scale_features(X_test, fit=False)
    
    # Train model
    classifier = BaselineClassifier()
    classifier.train_random_forest(X_train_scaled, y_train)
    classifier.best_model = classifier.models['random_forest']
    
    # Interpretation
    interpreter = ModelInterpreter()
    report = interpreter.generate_interpretation_report(
        classifier.best_model, X_train_scaled, y_train, feature_names
    )
    
    print("\nInterpretation report generated!")

