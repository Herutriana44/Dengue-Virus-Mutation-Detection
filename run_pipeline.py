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
    
    # Auto-detect: gunakan GISAID jika from_gisaid_data.csv ada
    dataset_dir = Path(__file__).parent / 'dataset'
    use_gisaid = (dataset_dir / 'from_gisaid_data.csv').exists()
    if use_gisaid:
        print("Mode: GISAID (DENV-2 Indonesia) - from_gisaid_data.csv detected")
        print("Tasks: baseline (genotype), novelty detection, interpretation")
    else:
        print("Mode: Legacy - sample_metadata.csv, dll")
        print("Tasks: baseline (serotype), novelty, open-set, interpretation")
    print()
    
    # Initialize pipeline
    pipeline = MLPipeline(
        dataset_dir='dataset',
        output_dir='results',
        use_gisaid=use_gisaid
    )
    
    # Run full pipeline (open_set di-skip otomatis untuk GISAID)
    print("Running full pipeline...")
    print()
    
    try:
        results = pipeline.run_full_pipeline()
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nResults saved to: results/")
        print("  - Models: results/models/")
        print("  - Interpretation: results/interpretation/")
        print("  - Cleaned dataset: ml_dataset_raw.csv")
        
    except KeyboardInterrupt:
        print("\n\nPipeline dihentikan (Ctrl+C). Jika Anda tidak menekan Ctrl+C, "
              "mungkin ada timeout dari IDE/terminal.")
        sys.exit(130)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

