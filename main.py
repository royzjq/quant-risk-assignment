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
from analysis.slippage_analyzer import run_comprehensive_slippage_analysis
from analysis.latency_analyzer import run_comprehensive_latency_analysis
from analysis.fee_analyzer import run_comprehensive_fee_analysis
from analysis.spread_analyzer import run_comprehensive_spread_analysis

def run_pipeline(exec_file='exec.csv', market_data_dir='data', force_reprocess=False, enable_spread_analysis=True):
    """
    Run the complete quantitative risk analysis pipeline
    
    Args:
        exec_file: Path to execution data CSV file
        market_data_dir: Directory containing market data files
        force_reprocess: Whether to force reprocessing even if processed data exists
        enable_spread_analysis: Whether to run the spread analysis step
        
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
        "slippage_analysis": None,
        "latency_analysis": None,
        "fee_analysis": None,
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
        
        # Step 4: Slippage Analysis
        print("\n" + "="*40)
        print("STEP 4: SLIPPAGE ANALYSIS")

        if merged_df is not None:
            try:
                slippage_results = run_comprehensive_slippage_analysis(
                    parquet_path=processed_data_path,
                    output_dir="results/images"
                )
                results["slippage_analysis"] = slippage_results
            except Exception as e:
                print(f"Slippage analysis failed: {str(e)}")
                results["slippage_analysis"] = {
                    "status": "ERROR",
                    "error": str(e)
                }
        else:
            results["slippage_analysis"] = {
                "status": "SKIPPED_NO_DATA_FOR_ANALYSIS",
                "message": "No processed data available for slippage analysis"
            }
            print("Skipping slippage analysis due to no processed data available.")
        
        # Step 5: Latency Analysis
        print("\n" + "="*40)
        print("STEP 5: LATENCY ANALYSIS")

        if merged_df is not None:
            try:
                latency_results = run_comprehensive_latency_analysis(
                    parquet_path=processed_data_path,
                    output_dir="results",
                    outlier_threshold=2.5
                )
                results["latency_analysis"] = latency_results
            except Exception as e:
                print(f"Latency analysis failed: {str(e)}")
                results["latency_analysis"] = {
                    "status": "ERROR",
                    "error": str(e)
                }
        else:
            results["latency_analysis"] = {
                "status": "SKIPPED_NO_DATA_FOR_ANALYSIS",
                "message": "No processed data available for latency analysis"
            }
            print("Skipping latency analysis due to no processed data available.")
        
        # Step 6: Fee Analysis
        print("\n" + "="*40)
        print("STEP 6: FEE ANALYSIS")

        if merged_df is not None:
            try:
                fee_results = run_comprehensive_fee_analysis(
                    parquet_path=processed_data_path,
                    output_dir="results",
                    outlier_threshold=2.0
                )
                results["fee_analysis"] = fee_results
            except Exception as e:
                print(f"Fee analysis failed: {str(e)}")
                results["fee_analysis"] = {
                    "status": "ERROR",
                    "error": str(e)
                }
        else:
            results["fee_analysis"] = {
                "status": "SKIPPED_NO_DATA_FOR_ANALYSIS",
                "message": "No processed data available for fee analysis"
            }
            print("Skipping fee analysis due to no processed data available.")
        
        # Step 7: Spread Analysis
        if enable_spread_analysis:
            print("\n" + "="*40)
            print("STEP 7: SPREAD ANALYSIS")
            
            if merged_df is not None:
                try:
                    spread_results = run_comprehensive_spread_analysis(
                        parquet_path=processed_data_path,
                        output_dir="results",
                        downsample_interval=100
                    )
                    results["visualization"] = spread_results
                except Exception as e:
                    print(f"Spread analysis failed: {str(e)}")
                    results["visualization"] = {
                        "status": "ERROR",
                        "error": str(e)
                    }
            else:
                results["visualization"] = {
                    "status": "SKIPPED_NO_DATA",
                    "message": "No processed data available for spread analysis"
                }
                print("Skipping spread analysis due to no processed data available.")
        else:
            results["visualization"] = {
                "status": "DISABLED",
                "message": "Spread analysis step disabled"
            }
            print("Spread analysis step disabled.")
        
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
        print(f"Slippage analysis: {results['slippage_analysis']['status']}")
        print(f"Latency analysis: {results['latency_analysis']['status']}")
        print(f"Fee analysis: {results['fee_analysis']['status']}")
        print(f"Spread analysis: {results['visualization']['status']}")
        
        # Violations summary
        if results['market_impact_analysis']['status'] == 'COMPLETE':
            violations = results['market_impact_analysis']['violations']
            total_violations = sum([v.get('count', 0) for v in violations.values()])
            print(f"Total market impact violations found: {total_violations}")
            
            for violation_type, details in violations.items():
                if details.get('count', 0) > 0:
                    print(f"  - {violation_type}: {details['count']} violations")
        
        # Slippage analysis summary
        if results['slippage_analysis']['status'] == 'COMPLETE':
            slippage_stats = results['slippage_analysis']['summary_statistics']
            correlations = results['slippage_analysis']['correlations']
            print(f"Slippage analysis results:")
            print(f"  - Total execution records analyzed: {slippage_stats.get('count', 0)}")
            print(f"  - Mean slippage: {slippage_stats.get('mean_bps', 0):.2f} bps")
            print(f"  - Median slippage: {slippage_stats.get('median_bps', 0):.2f} bps")
            
            # Show correlations
            for analysis_type, details in correlations.items():
                if details.get('correlation') is not None:
                    print(f"  - {analysis_type.replace('_', ' ').title()}: r={details['correlation']:.3f}")
        
        # Latency analysis summary
        if results['latency_analysis']['status'] == 'COMPLETE':
            latency_stats = results['latency_analysis']['summary_statistics']
            outlier_counts = results['latency_analysis']['outlier_analysis']['outlier_counts']
            print(f"Latency analysis results:")
            
            for latency_type, stats in latency_stats.items():
                if stats:
                    print(f"  - {latency_type.replace('_', ' ').title()}:")
                    print(f"    Mean: {stats.get('mean_us', 0):.0f} μs, P95: {stats.get('p95_us', 0):.0f} μs")
                    outlier_count = outlier_counts.get(latency_type, 0)
                    if outlier_count > 0:
                        total_count = stats.get('count', 0)
                        outlier_pct = (outlier_count / total_count) * 100 if total_count > 0 else 0
                        print(f"    Outliers: {outlier_count} ({outlier_pct:.1f}%)")
        
        # Fee analysis summary
        if results['fee_analysis']['status'] == 'COMPLETE':
            fee_stats = results['fee_analysis']['summary_statistics']
            fee_correlations = results['fee_analysis']['correlations']
            outlier_analysis = results['fee_analysis']['outlier_analysis']
            print(f"Fee analysis results:")
            print(f"  - Total execution records analyzed: {fee_stats.get('count', 0)}")
            print(f"  - Mean fee: {fee_stats.get('fee_stats', {}).get('mean', 0):.6f}")
            print(f"  - Fee outliers: {outlier_analysis.get('outlier_count', 0)} ({outlier_analysis.get('outlier_percentage', 0):.1f}%)")
            
            # Show correlations
            for analysis_type, details in fee_correlations.items():
                if details.get('correlation') is not None:
                    print(f"  - {analysis_type.replace('_', ' ').title()}: r={details['correlation']:.3f}")
        
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
        '--no-spread-analysis',
        action='store_true',
        help='Disable spread analysis step'
    )
    
    args = parser.parse_args()
    
    # Run the pipeline
    results = run_pipeline(
        exec_file=args.exec_file,
        market_data_dir=args.market_data_dir,
        force_reprocess=args.force_reprocess,
        enable_spread_analysis=not args.no_spread_analysis
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