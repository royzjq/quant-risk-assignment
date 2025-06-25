#!/usr/bin/env python3
"""
Quantitative Risk Analysis - Main Pipeline


This script coordinates the execution of all analysis modules:
1. Data validation (exec_validator)
2. Data preparation (data_processor) 
3. Market impact analysis (market_impact_analyzer)
4. Results generation and visualization (future modules)


"""

import sys
import time
from pathlib import Path

# Add module paths
sys.path.append('validation')
sys.path.append('data_prep')
sys.path.append('analysis')
sys.path.append('visualization')

from validation.exec_validator import validate_exec_csv
from data_prep.data_processor import prepare_btcusdt_data, save_to_parquet
from analysis.market_impact_analyzer import analyze_market_impact_violations, load_processed_data
from visualization.spread_visualization import main as run_spread_visualization

def run_pipeline(exec_file='exec.csv', market_data_dir='data', force_reprocess=False, enable_visualization=True):
    """
    Run the complete quantitative risk analysis pipeline
    
    Args:
        exec_file: Path to execution data CSV file
        market_data_dir: Directory containing market data files
        force_reprocess: Whether to force reprocessing even if processed data exists
        enable_visualization: Whether to run the visualization step
        
    Returns:
        dict: Pipeline execution results
    """
    print("="*60)
    print("QUANTITATIVE RISK ANALYSIS PIPELINE")

    pipeline_start_time = time.time()
    results = {
        "pipeline_status": "RUNNING",
        "validation": None,
        "data_preparation": None,
        "market_impact_analysis": None,
        "visualization": None,
        "execution_time": None
    }
    
    try:
        # Step 1: Data Validation
        print("\n" + "="*40)
        print("STEP 1: DATA VALIDATION")

        
        validation_results = validate_exec_csv(exec_file)
        results["validation"] = {
            "status": "COMPLETE" if validation_results["summary"] == "Validation Report for exec.csv" else "ERROR",
            "missing_values_count": len(validation_results["missing_values"]),
            "data_type_issues_count": len(validation_results["data_type_issues"]),
            "value_range_issues_count": len(validation_results["value_range_issues"]),
            "anomalies_count": len(validation_results["anomalies"])
        }
        
        # Step 2: Data Preparation
        print("\n" + "="*40)
        print("STEP 2: DATA PREPARATION")

        
        processed_data_path = "results/btcusdt_processed_data.parquet"
        
        # Check if reprocessing is forced or if processed data does not exist
        if force_reprocess or not Path(processed_data_path).exists():
            print("Preparing BTCUSDT data...")
            merged_df, coverage_status = prepare_btcusdt_data(exec_file=exec_file, market_data_dir=market_data_dir)
            if merged_df is not None:
                save_to_parquet(merged_df, processed_data_path)
                print(f"Processed data saved to {processed_data_path}")
            else:
                coverage_status = "SKIPPED_NO_RAW_DATA"
                print("Skipping data preparation due to no raw data found.")
        else:
            print(f"Processed data found at {processed_data_path}")
            print("Loading existing processed data for analysis...")
            merged_df = load_processed_data(processed_data_path)
            coverage_status = "LOADED_FOR_ANALYSIS"

        results["data_preparation"] = {
            "status": coverage_status,
            "record_count": len(merged_df) if merged_df is not None else 0,
            "output_file": processed_data_path
        }
        
        # Step 3: Market Impact Analysis
        print("\n" + "="*40)
        print("STEP 3: MARKET IMPACT ANALYSIS")

        if merged_df is not None:
            market_impact_results = analyze_market_impact_violations(merged_df, save_timestamps=True)
            results["market_impact_analysis"] = market_impact_results
        else:
            results["market_impact_analysis"] = {
                "status": "SKIPPED_NO_DATA_FOR_ANALYSIS",
                "violations": {}
            }
            print("Skipping market impact analysis due to no processed data available.")
        
        # Step 4: Visualization
        if enable_visualization:
            print("\n" + "="*40)
            print("STEP 4: SPREAD VISUALIZATION")
            
            if merged_df is not None:
                try:
                    run_spread_visualization()
                    results["visualization"] = {
                        "status": "COMPLETE",
                        "output_dir": "results/images",
                        "files_generated": ["btcusdt_spread_visualization.png", "btcusdt_spread_visualization.pdf"]
                    }
                except Exception as e:
                    print(f"Visualization failed: {str(e)}")
                    results["visualization"] = {
                        "status": "ERROR",
                        "error": str(e)
                    }
            else:
                results["visualization"] = {
                    "status": "SKIPPED_NO_DATA",
                    "message": "No processed data available for visualization"
                }
                print("Skipping visualization due to no processed data available.")
        else:
            results["visualization"] = {
                "status": "DISABLED",
                "message": "Visualization step disabled"
            }
            print("Visualization step disabled.")
        
        # Pipeline completion
        pipeline_time = time.time() - pipeline_start_time
        results["execution_time"] = pipeline_time
        results["pipeline_status"] = "COMPLETE"
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION SUMMARY")

        print(f"Total execution time: {pipeline_time:.2f} seconds")
        print(f"Data validation: {results['validation']['status']}")
        print(f"Data preparation: {results['data_preparation']['status']}")
        print(f"Market impact analysis: {results['market_impact_analysis']['status']}")
        print(f"Visualization: {results['visualization']['status']}")
        
        # Violations summary
        if results['market_impact_analysis']['status'] == 'COMPLETE':
            violations = results['market_impact_analysis']['violations']
            total_violations = sum([v.get('count', 0) for v in violations.values()])
            print(f"Total market impact violations found: {total_violations}")
            
            for violation_type, details in violations.items():
                if details.get('count', 0) > 0:
                    print(f"  - {violation_type}: {details['count']} violations")
        
        print("="*60)
        
        return results
        
    except Exception as e:
        print(f"\nPipeline execution failed: {str(e)}")
        results["pipeline_status"] = "ERROR"
        results["error_message"] = str(e)
        results["execution_time"] = time.time() - pipeline_start_time
        return results

def main():
    """
    Main entry point

    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Quantitative Risk Analysis Pipeline"
    )
    parser.add_argument(
        '--exec-file', 
        default='exec.csv',
        help='Path to execution data CSV file (default: exec.csv)'
    )
    parser.add_argument(
        '--market-data-dir',
        default='data', 
        help='Directory containing market data files (default: data)'
    )
    parser.add_argument(
        '--force-reprocess',
        action='store_true',
        help='Force reprocessing even if processed data exists'
    )
    parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='Disable visualization step'
    )
    
    args = parser.parse_args()
    
    # Run the pipeline
    results = run_pipeline(
        exec_file=args.exec_file,
        market_data_dir=args.market_data_dir,
        force_reprocess=args.force_reprocess,
        enable_visualization=not args.no_visualization
    )
    
    # Exit with appropriate code
    if results["pipeline_status"] == "COMPLETE":
        print("Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 