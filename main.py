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

from validation.exec_validator import validate_exec_csv
from data_prep.data_processor import prepare_symbol_data, save_to_parquet
from analysis.market_impact_analyzer import analyze_market_impact_violations, load_processed_data
from analysis.slippage_analyzer import run_comprehensive_slippage_analysis_with_data
from analysis.latency_analyzer import run_comprehensive_latency_analysis_with_data
from analysis.fee_analyzer import run_comprehensive_fee_analysis_with_data
from analysis.spread_analyzer import run_comprehensive_spread_analysis_with_data

def run_symbol_pipeline(symbol, exec_file='exec.csv', market_data_dir='data', force_reprocess=False, enable_spread_analysis=True):
    """
    Run the analysis pipeline for a single symbol
    
    Args:
        symbol: Trading symbol to process (e.g., 'BTCUSDT', 'ETHUSDT')
        exec_file: Path to execution data CSV file
        market_data_dir: Directory containing market data files
        force_reprocess: Whether to force reprocessing even if processed data exists
        enable_spread_analysis: Whether to run the spread analysis step
        
    Returns:
        dict: Pipeline execution results for the symbol
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING {symbol}")
    print(f"{'='*60}")

    symbol_start_time = time.time()
    results = {
        "symbol": symbol,
        "pipeline_status": "RUNNING",
        "data_preparation": None,
        "market_impact_analysis": None,
        "slippage_analysis": None,
        "latency_analysis": None,
        "fee_analysis": None,
        "visualization": None,
        "execution_time": None
    }
    
    try:
        # Step 1: Data Preparation
        print(f"\n{'='*40}")
        print(f"STEP 2: DATA PREPARATION FOR {symbol}")
        
        processed_data_path = f"results/{symbol.lower()}_processed_data.parquet"
        
        # Check if reprocessing is forced or if processed data does not exist
        if force_reprocess or not Path(processed_data_path).exists():
            print(f"Preparing {symbol} data...")
            merged_df, coverage_status = prepare_symbol_data(exec_file=exec_file, market_data_dir=market_data_dir, symbol=symbol)
            if merged_df is not None:
                save_to_parquet(merged_df, processed_data_path)
                print(f"Processed data saved to {processed_data_path}")
            else:
                coverage_status = "SKIPPED_NO_RAW_DATA"
                print(f"Skipping {symbol} data preparation due to no raw data found.")
        else:
            print(f"Processed data found at {processed_data_path}")
            print(f"Loading existing processed data for {symbol} analysis...")
            merged_df = load_processed_data(processed_data_path)
            coverage_status = "LOADED_FOR_ANALYSIS"

        results["data_preparation"] = {
            "status": coverage_status,
            "record_count": len(merged_df) if merged_df is not None else 0,
            "output_file": processed_data_path
        }
        
        # Step 2: Market Impact Analysis
        print(f"\n{'='*40}")
        print(f"STEP 3: MARKET IMPACT ANALYSIS FOR {symbol}")

        if merged_df is not None:
            market_impact_results = analyze_market_impact_violations(merged_df, save_timestamps=True, symbol=symbol)
            results["market_impact_analysis"] = market_impact_results
        else:
            results["market_impact_analysis"] = {
                "status": "SKIPPED_NO_DATA_FOR_ANALYSIS",
                "violations": {}
            }
            print(f"Skipping {symbol} market impact analysis due to no processed data available.")
        
        # Step 3: Slippage Analysis
        print(f"\n{'='*40}")
        print(f"STEP 4: SLIPPAGE ANALYSIS FOR {symbol}")

        if merged_df is not None:
            try:
                slippage_results = run_comprehensive_slippage_analysis_with_data(
                    df=merged_df,
                    output_dir="results/images",
                    symbol=symbol
                )
                results["slippage_analysis"] = slippage_results
            except Exception as e:
                print(f"{symbol} slippage analysis failed: {str(e)}")
                results["slippage_analysis"] = {
                    "status": "ERROR",
                    "error": str(e)
                }
        else:
            results["slippage_analysis"] = {
                "status": "SKIPPED_NO_DATA_FOR_ANALYSIS",
                "message": f"No processed data available for {symbol} slippage analysis"
            }
            print(f"Skipping {symbol} slippage analysis due to no processed data available.")
        
        # Step 4: Latency Analysis
        print(f"\n{'='*40}")
        print(f"STEP 5: LATENCY ANALYSIS FOR {symbol}")

        if merged_df is not None:
            try:
                latency_results = run_comprehensive_latency_analysis_with_data(
                    df=merged_df,
                    output_dir="results",
                    outlier_threshold=2.5,
                    symbol=symbol
                )
                results["latency_analysis"] = latency_results
            except Exception as e:
                print(f"{symbol} latency analysis failed: {str(e)}")
                results["latency_analysis"] = {
                    "status": "ERROR",
                    "error": str(e)
                }
        else:
            results["latency_analysis"] = {
                "status": "SKIPPED_NO_DATA_FOR_ANALYSIS",
                "message": f"No processed data available for {symbol} latency analysis"
            }
            print(f"Skipping {symbol} latency analysis due to no processed data available.")
        
        # Step 5: Fee Analysis
        print(f"\n{'='*40}")
        print(f"STEP 6: FEE ANALYSIS FOR {symbol}")

        if merged_df is not None:
            try:
                fee_results = run_comprehensive_fee_analysis_with_data(
                    df=merged_df,
                    output_dir="results",
                    outlier_threshold=2.0,
                    symbol=symbol
                )
                results["fee_analysis"] = fee_results
            except Exception as e:
                print(f"{symbol} fee analysis failed: {str(e)}")
                results["fee_analysis"] = {
                    "status": "ERROR",
                    "error": str(e)
                }
        else:
            results["fee_analysis"] = {
                "status": "SKIPPED_NO_DATA_FOR_ANALYSIS",
                "message": f"No processed data available for {symbol} fee analysis"
            }
            print(f"Skipping {symbol} fee analysis due to no processed data available.")
        
        # Step 6: Spread Analysis
        if enable_spread_analysis:
            print(f"\n{'='*40}")
            print(f"STEP 7: SPREAD ANALYSIS FOR {symbol}")
            
            if merged_df is not None:
                try:
                    spread_results = run_comprehensive_spread_analysis_with_data(
                        df=merged_df,
                        output_dir="results",
                        downsample_interval=100,
                        symbol=symbol
                    )
                    results["visualization"] = spread_results
                except Exception as e:
                    print(f"{symbol} spread analysis failed: {str(e)}")
                    results["visualization"] = {
                        "status": "ERROR",
                        "error": str(e)
                    }
            else:
                results["visualization"] = {
                    "status": "SKIPPED_NO_DATA",
                    "message": f"No processed data available for {symbol} spread analysis"
                }
                print(f"Skipping {symbol} spread analysis due to no processed data available.")
        else:
            results["visualization"] = {
                "status": "DISABLED",
                "message": "Spread analysis step disabled"
            }
            print(f"{symbol} spread analysis step disabled.")
        
        # Symbol pipeline completion
        symbol_time = time.time() - symbol_start_time
        results["execution_time"] = symbol_time
        results["pipeline_status"] = "COMPLETE"
        
        print(f"\n{'='*60}")
        print(f"{symbol} PIPELINE EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Execution time: {symbol_time:.2f} seconds")
        print(f"Data preparation: {results['data_preparation']['status']}")
        print(f"Market impact analysis: {results['market_impact_analysis']['status']}")
        print(f"Slippage analysis: {results['slippage_analysis']['status']}")
        print(f"Latency analysis: {results['latency_analysis']['status']}")
        print(f"Fee analysis: {results['fee_analysis']['status']}")
        print(f"Spread analysis: {results['visualization']['status']}")
        
        return results
        
    except Exception as e:
        print(f"\n{symbol} pipeline execution failed: {str(e)}")
        results["pipeline_status"] = "ERROR"
        results["error_message"] = str(e)
        results["execution_time"] = time.time() - symbol_start_time
        return results

def run_pipeline(exec_file='exec.csv', market_data_dir='data', force_reprocess=False, enable_spread_analysis=True, symbols=None):
    """
    Run the complete quantitative risk analysis pipeline
    
    Args:
        exec_file: Path to execution data CSV file
        market_data_dir: Directory containing market data files
        force_reprocess: Whether to force reprocessing even if processed data exists
        enable_spread_analysis: Whether to run the spread analysis step
        symbols: List of symbols to process (if None, defaults to ['BTCUSDT'])
        
    Returns:
        dict: Pipeline execution results
    """
    print("="*60)
    print("QUANTITATIVE RISK ANALYSIS PIPELINE")
    print("="*60)

    pipeline_start_time = time.time()
    
    # Default to BTCUSDT if no symbols specified
    if symbols is None:
        symbols = ['BTCUSDT']
    
    print(f"Processing symbols: {', '.join(symbols)}")
    
    results = {
        "pipeline_status": "RUNNING",
        "validation": None,
        "symbols": {},
        "execution_time": None,
        "total_symbols": len(symbols),
        "successful_symbols": 0,
        "failed_symbols": 0
    }
    
    try:
        # Step 1: Data Validation (run once for all symbols)
        print("\n" + "="*40)
        print("STEP 1: DATA VALIDATION")

        
        validation_results = validate_exec_csv(Path(market_data_dir) / exec_file)
        results["validation"] = {
            "status": "COMPLETE" if validation_results["summary"] == "Validation Report for exec.csv" else "ERROR",
            "missing_values_count": len(validation_results["missing_values"]),
            "data_type_issues_count": len(validation_results["data_type_issues"]),
            "value_range_issues_count": len(validation_results["value_range_issues"]),
            "anomalies_count": len(validation_results["anomalies"])
        }
        
        # Process each symbol
        for symbol in symbols:
            symbol_results = run_symbol_pipeline(
                symbol=symbol,
                exec_file=exec_file,
                market_data_dir=market_data_dir,
                force_reprocess=force_reprocess,
                enable_spread_analysis=enable_spread_analysis
            )
            
            results["symbols"][symbol] = symbol_results
            
            if symbol_results["pipeline_status"] == "COMPLETE":
                results["successful_symbols"] += 1
            else:
                results["failed_symbols"] += 1
        
        # Pipeline completion
        pipeline_time = time.time() - pipeline_start_time
        results["execution_time"] = pipeline_time
        results["pipeline_status"] = "COMPLETE"
        
        print("\n" + "="*60)
        print("OVERALL PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"Total execution time: {pipeline_time:.2f} seconds")
        print(f"Symbols processed: {results['total_symbols']}")
        print(f"Successful: {results['successful_symbols']}")
        print(f"Failed: {results['failed_symbols']}")
        print(f"Data validation: {results['validation']['status']}")
        
        # Print summary for each symbol
        for symbol, symbol_results in results["symbols"].items():
            print(f"\n{symbol}:")
            print(f"  Status: {symbol_results['pipeline_status']}")
            print(f"  Execution time: {symbol_results.get('execution_time', 0):.2f} seconds")
            if symbol_results['pipeline_status'] == 'COMPLETE':
                # Print analysis summaries for successful symbols
                if symbol_results.get('market_impact_analysis', {}).get('status') == 'COMPLETE':
                    violations = symbol_results['market_impact_analysis']['violations']
                    total_violations = sum([v.get('count', 0) for v in violations.values()])
                    print(f"  Market impact violations: {total_violations}")
                
                if symbol_results.get('slippage_analysis', {}).get('status') == 'COMPLETE':
                    slippage_stats = symbol_results['slippage_analysis']['summary_statistics']
                    print(f"  Slippage (mean): {slippage_stats.get('mean_bps', 0):.2f} bps")
        
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
    parser.add_argument(
        '--symbols',
        nargs='*',
        default=['BTCUSDT', 'DOTUSDT', 'EOSUSDT', 'ETHUSDT', 'XRPUSDT'],
        help='Trading symbols to process (default: all symbols). Available: BTCUSDT DOTUSDT EOSUSDT ETHUSDT XRPUSDT. Use --symbols BTCUSDT to process only BTCUSDT.'
    )
    
    args = parser.parse_args()
    
    # Run the pipeline
    results = run_pipeline(
        exec_file=args.exec_file,
        market_data_dir=args.market_data_dir,
        force_reprocess=args.force_reprocess,
        enable_spread_analysis=not args.no_spread_analysis,
        symbols=args.symbols
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