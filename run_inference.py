"""
Script untuk menjalankan inference menggunakan model yang sudah di-train
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from inference import InferencePipeline
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    parser = argparse.ArgumentParser(
        description='Dengue Virus Mutation Detection - Inference'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV file atau path ke dataset directory'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='results/models',
        help='Directory tempat model disimpan'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='inference_results.csv',
        help='Output CSV file untuk results'
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=['baseline', 'novelty', 'open_set'],
        choices=['baseline', 'novelty', 'open_set'],
        help='Tasks untuk dijalankan'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Dengue Virus Mutation Detection - Inference")
    print("=" * 70)
    print()
    
    # Initialize inference pipeline
    inference = InferencePipeline(models_dir=args.models_dir)
    
    # Load models
    print("Loading models...")
    inference.load_models()
    
    # Check if input is file or directory
    input_path = Path(args.input)
    if input_path.is_dir():
        # Load from dataset directory
        from data_cleaning import DataCleaner
        cleaner = DataCleaner(dataset_dir=str(input_path))
        cleaner.load_datasets()
        cleaner.merge_tables()
        input_data = cleaner.cleaned_data
        is_dataframe = True
    elif input_path.is_file():
        # Load from CSV
        input_data = str(input_path)
        is_dataframe = False
    else:
        print(f"Error: Input path {args.input} not found!")
        sys.exit(1)
    
    # Run inference
    print(f"\nRunning inference on {len(input_data) if isinstance(input_data, pd.DataFrame) else 'file'} samples...")
    print(f"Tasks: {', '.join(args.tasks)}")
    print()
    
    try:
        results = inference.run_full_inference(
            input_data,
            is_dataframe=is_dataframe,
            tasks=args.tasks
        )
        
        # Format and save results
        df_results = inference.format_results(results, output_format='dataframe')
        inference.save_results(results, output_path=args.output)
        
        print("\n" + "=" * 70)
        print("INFERENCE COMPLETED!")
        print("=" * 70)
        print(f"\nResults saved to: {args.output}")
        print(f"\nSummary:")
        print(f"  Total samples: {len(df_results)}")
        
        if 'baseline' in results:
            print(f"  Baseline predictions: {len(results['baseline']['predicted_serotype'])}")
        
        if 'novelty' in results:
            novel_count = sum(results['novelty']['is_novel_genotype'])
            print(f"  Novel genotypes detected: {novel_count}")
        
        if 'open_set' in results:
            open_set_count = sum(results['open_set']['is_potential_new_serotype'])
            print(f"  Potential new serotypes detected: {open_set_count}")
        
        print("\nFirst few predictions:")
        print(df_results.head(10).to_string(index=False))
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

