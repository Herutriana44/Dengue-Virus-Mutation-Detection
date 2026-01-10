"""
Quick start script untuk menjalankan ML pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from main_pipeline import MLPipeline
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    print("=" * 70)
    print("Dengue Virus Mutation Detection - ML Pipeline")
    print("=" * 70)
    print()
    
    # Initialize pipeline
    pipeline = MLPipeline(
        dataset_dir='dataset',
        output_dir='results'
    )
    
    # Run full pipeline
    print("Running full pipeline with all tasks...")
    print("Tasks: baseline classification, novelty detection, open-set detection, interpretation")
    print()
    
    try:
        results = pipeline.run_full_pipeline(
            tasks=['baseline', 'novelty', 'open_set', 'interpretation']
        )
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nResults saved to: results/")
        print("  - Models: results/models/")
        print("  - Interpretation: results/interpretation/")
        print("  - Cleaned dataset: ml_dataset_raw.csv")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

