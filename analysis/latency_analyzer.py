import polars as pl
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from pathlib import Path
import time
import json
from scipy import stats

def load_processed_data(parquet_path):
    """
    Load processed data from Parquet file
    
    Args:
        parquet_path: Path to the Parquet file
        
    Returns:
        Polars DataFrame
    """
    print(f"Loading processed data from {parquet_path}...")
    start_time = time.time()
    df = pl.read_parquet(parquet_path)
    load_time = time.time() - start_time
    print(f"Loaded {len(df)} records in {load_time:.2f} seconds")
    return df

def calculate_latencies(df):
    """
    Calculate three types of latencies:
    1. Exchange internal processing latency: receiveTs - timestamp
    2. Network latency: flashOutTs - receiveTs  
    3. Strategy processing latency: eTm_microseconds - timestamp
    
    Args:
        df: Polars DataFrame with execution and market data
        
    Returns:
        DataFrame with latency calculations
    """
    print("Calculating latencies...")
    
    # Filter out records without execution data and required timestamp fields
    exec_df = df.filter(
        (pl.col('eTm').is_not_null()) &
        (pl.col('receiveTs').is_not_null()) &
        (pl.col('flashOutTs').is_not_null()) &
        (pl.col('timestamp').is_not_null()) &
        (pl.col('eTm_microseconds').is_not_null())
    )
    
    if len(exec_df) == 0:
        print("No execution records found with complete timestamp data for latency calculation")
        return None
    
    # Calculate latencies (all in microseconds)
    latency_df = exec_df.with_columns([
        # Exchange internal processing latency
        (pl.col('receiveTs') - pl.col('timestamp')).alias('exchange_processing_latency'),
        
        # Network latency
        (pl.col('flashOutTs') - pl.col('receiveTs')).alias('network_latency'),
        
        # Strategy processing latency
        (pl.col('eTm_microseconds') - pl.col('timestamp')).alias('strategy_processing_latency')
    ])
    
    print(f"Calculated latencies for {len(latency_df)} execution records")
    return latency_df

def detect_outliers(values, threshold=2.5):
    """
    Detect outliers using z-score threshold
    
    Args:
        values: numpy array of values
        threshold: z-score threshold (default: 2.5 sigma)
        
    Returns:
        tuple: (outlier_indices, outlier_values, z_scores)
    """
    mean_val = np.mean(values)
    std_val = np.std(values)
    z_scores = np.abs((values - mean_val) / std_val)
    outlier_indices = np.where(z_scores > threshold)[0]
    outlier_values = values[outlier_indices]
    return outlier_indices, outlier_values, z_scores

def create_latency_distribution_plots(latency_df, output_dir="results/images"):
    """
    Create distribution plots for all three latency types
    """
    print("Creating latency distribution plots...")
    
    # Define latency types and their properties
    latency_types = {
        'exchange_processing_latency': {
            'title': 'Exchange Internal Processing Latency Distribution',
            'color': 'lightblue',
            'unit': 'μs'
        },
        'network_latency': {
            'title': 'Network Latency Distribution',
            'color': 'lightgreen',
            'unit': 'μs'
        },
        'strategy_processing_latency': {
            'title': 'Strategy Processing Latency Distribution',
            'color': 'orange',
            'unit': 'μs'
        }
    }
    
    # Create subplot figure
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[info['title'] for info in latency_types.values()],
        vertical_spacing=0.08
    )
    
    row = 1
    for latency_type, info in latency_types.items():
        # Get valid data
        valid_data = latency_df.filter(pl.col(latency_type).is_not_null())[latency_type].to_numpy()
        
        if len(valid_data) == 0:
            print(f"No valid data for {latency_type}")
            continue
        
        # Calculate statistics
        mean_val = np.mean(valid_data)
        median_val = np.median(valid_data)
        std_val = np.std(valid_data)
        p95_val = np.percentile(valid_data, 95)
        p99_val = np.percentile(valid_data, 99)
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=valid_data,
                nbinsx=50,
                name=f'{latency_type.replace("_", " ").title()}',
                opacity=0.7,
                marker_color=info['color']
            ),
            row=row, col=1
        )
        
        # Add statistics annotation
        fig.add_annotation(
            text=f"Mean: {mean_val:.0f} {info['unit']}<br>"
                 f"Median: {median_val:.0f} {info['unit']}<br>"
                 f"Std: {std_val:.0f} {info['unit']}<br>"
                 f"P95: {p95_val:.0f} {info['unit']}<br>"
                 f"P99: {p99_val:.0f} {info['unit']}<br>"
                 f"Count: {len(valid_data)}",
            showarrow=False,
            font=dict(size=10),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            xref="paper", yref="paper",
            x=0.98, y=0.95 - (row-1)*0.33,
            align="right",
            xanchor="right"
        )
        
        row += 1
    
    fig.update_layout(
        title="Latency Analysis - Distribution Plots",
        showlegend=False,
        template="plotly_white",
        height=1200,
        width=800
    )
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Exchange Processing Latency (μs)", row=1, col=1)
    fig.update_xaxes(title_text="Network Latency (μs)", row=2, col=1)
    fig.update_xaxes(title_text="Strategy Processing Latency (μs)", row=3, col=1)
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=3, col=1)
    
    # Save plot
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.write_html(f"{output_dir}/latency_distributions.html")
    try:
        fig.write_image(f"{output_dir}/latency_distributions.png", width=800, height=1200, scale=2)
    except:
        print("Warning: Could not save PNG image. HTML version saved successfully.")
    
    print(f"Latency distribution plots saved to {output_dir}/")
    return fig

def detect_and_export_latency_outliers(latency_df, output_dir="results", threshold=2.5):
    """
    Detect outliers (> 2.5 sigma) for all latency types and export to JSON
    """
    print(f"Detecting latency outliers with {threshold} sigma threshold...")
    
    latency_types = ['exchange_processing_latency', 'network_latency', 'strategy_processing_latency']
    outliers_data = {}
    
    for latency_type in latency_types:
        print(f"Processing {latency_type}...")
        
        # Get valid data with record indices
        valid_records = latency_df.filter(pl.col(latency_type).is_not_null())
        
        if len(valid_records) == 0:
            print(f"No valid data for {latency_type}")
            outliers_data[latency_type] = {
                "outliers_count": 0,
                "total_records": 0,
                "outlier_percentage": 0.0,
                "threshold_sigma": threshold,
                "outlier_records": []
            }
            continue
        
        values = valid_records[latency_type].to_numpy()
        
        # Detect outliers
        outlier_indices, outlier_values, z_scores = detect_outliers(values, threshold)
        
        # Get outlier records
        outlier_records = []
        if len(outlier_indices) > 0:
            outlier_df = valid_records[outlier_indices]
            
            for i, row in enumerate(outlier_df.iter_rows(named=True)):
                outlier_record = {
                    "timestamp": int(row['timestamp']) if row['timestamp'] is not None else None,
                    "receiveTs": int(row['receiveTs']) if row['receiveTs'] is not None else None,
                    "flashOutTs": int(row['flashOutTs']) if row['flashOutTs'] is not None else None,
                    "eTm": row['eTm'] if row['eTm'] is not None else None,
                    "eTm_microseconds": int(row['eTm_microseconds']) if row['eTm_microseconds'] is not None else None,
                    "latency_value": float(outlier_values[i]),
                    "z_score": float(z_scores[outlier_indices[i]]),
                    "sym": row['sym'] if row['sym'] is not None else None,
                    "sde": int(row['sde']) if row['sde'] is not None else None,
                    "eqty": float(row['eqty']) if row['eqty'] is not None else None,
                    "vwap": float(row['vwap']) if row['vwap'] is not None else None
                }
                outlier_records.append(outlier_record)
        
        # Calculate statistics
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        outlier_percentage = (len(outlier_indices) / len(values)) * 100
        
        outliers_data[latency_type] = {
            "outliers_count": len(outlier_indices),
            "total_records": len(values),
            "outlier_percentage": outlier_percentage,
            "mean_latency": mean_val,
            "std_latency": std_val,
            "threshold_sigma": threshold,
            "threshold_value": mean_val + threshold * std_val,
            "outlier_records": outlier_records
        }
        
        print(f"  - Found {len(outlier_indices)} outliers out of {len(values)} records ({outlier_percentage:.2f}%)")
    
    # Export to JSON
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/latency_outliers_{timestamp}.json"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(outliers_data, f, indent=2, ensure_ascii=False)
    
    print(f"Latency outliers exported to {output_file}")
    return outliers_data, output_file

def generate_latency_summary_stats(latency_df):
    """
    Generate comprehensive latency summary statistics
    """
    print("Generating latency summary statistics...")
    
    latency_types = ['exchange_processing_latency', 'network_latency', 'strategy_processing_latency']
    stats_dict = {}
    
    for latency_type in latency_types:
        # Filter valid latency data
        valid_latency = latency_df.filter(pl.col(latency_type).is_not_null())
        
        if len(valid_latency) == 0:
            print(f"No valid latency data found for {latency_type}")
            stats_dict[latency_type] = {}
            continue
        
        latency_values = valid_latency[latency_type].to_numpy()
        
        stats_dict[latency_type] = {
            'count': len(latency_values),
            'mean_us': float(np.mean(latency_values)),
            'median_us': float(np.median(latency_values)),
            'std_us': float(np.std(latency_values)),
            'min_us': float(np.min(latency_values)),
            'max_us': float(np.max(latency_values)),
            'p25_us': float(np.percentile(latency_values, 25)),
            'p75_us': float(np.percentile(latency_values, 75)),
            'p95_us': float(np.percentile(latency_values, 95)),
            'p99_us': float(np.percentile(latency_values, 99)),
        }
    
    return stats_dict

def run_comprehensive_latency_analysis(parquet_path="results/btcusdt_processed_data.parquet", 
                                     output_dir="results", outlier_threshold=2.5):
    """
    Run comprehensive latency analysis
    
    Args:
        parquet_path: Path to processed data parquet file
        output_dir: Directory to save outputs
        outlier_threshold: Z-score threshold for outlier detection (default: 2.5 sigma)
        
    Returns:
        dict: Analysis results and statistics
    """
    print("="*60)
    print("COMPREHENSIVE LATENCY ANALYSIS")
    print("="*60)
    
    start_time = time.time()
    
    # 1. Load processed data
    df = load_processed_data(parquet_path)
    
    # 2. Calculate latencies
    latency_df = calculate_latencies(df)
    if latency_df is None:
        return {"status": "ERROR", "message": "No execution data found with complete timestamps"}
    
    # 3. Generate summary statistics
    stats = generate_latency_summary_stats(latency_df)
    
    # 4. Create visualizations
    print("\nGenerating visualizations...")
    images_dir = f"{output_dir}/images"
    dist_fig = create_latency_distribution_plots(latency_df, images_dir)
    
    # 5. Detect and export outliers
    outliers_data, outliers_file = detect_and_export_latency_outliers(
        latency_df, output_dir, threshold=outlier_threshold
    )
    
    # 6. Compile results
    results = {
        "status": "COMPLETE",
        "execution_time": time.time() - start_time,
        "summary_statistics": stats,
        "outlier_analysis": {
            "threshold_sigma": outlier_threshold,
            "outliers_file": outliers_file,
            "outlier_counts": {
                latency_type: data.get("outliers_count", 0) 
                for latency_type, data in outliers_data.items()
            }
        },
        "visualizations": {
            "latency_distributions": f"{images_dir}/latency_distributions.html"
        }
    }
    
    # Print summary
    print("\n" + "="*60)
    print("LATENCY ANALYSIS SUMMARY")
    print("="*60)
    print(f"Analysis completed in {results['execution_time']:.2f} seconds")
    
    for latency_type, latency_stats in stats.items():
        if latency_stats:
            print(f"\n{latency_type.replace('_', ' ').title()}:")
            print(f"  - Count: {latency_stats.get('count', 0)}")
            print(f"  - Mean: {latency_stats.get('mean_us', 0):.0f} μs")
            print(f"  - Median: {latency_stats.get('median_us', 0):.0f} μs")
            print(f"  - P95: {latency_stats.get('p95_us', 0):.0f} μs")
            print(f"  - P99: {latency_stats.get('p99_us', 0):.0f} μs")
            
            # Show outlier info
            outlier_count = results['outlier_analysis']['outlier_counts'].get(latency_type, 0)
            if outlier_count > 0:
                total_count = latency_stats.get('count', 0)
                outlier_pct = (outlier_count / total_count) * 100 if total_count > 0 else 0
                print(f"  - Outliers (>{outlier_threshold}σ): {outlier_count} ({outlier_pct:.2f}%)")
    
    print(f"\nOutliers exported to: {outliers_file}")
    print(f"Visualizations saved to: {images_dir}/")
    print("="*60)
    
    return results

def main():
    """
    Main function to run latency analysis
    """
    try:
        results = run_comprehensive_latency_analysis()
        return results
    except Exception as e:
        print(f"Latency analysis failed: {str(e)}")
        return {"status": "ERROR", "error": str(e)}

if __name__ == "__main__":
    main() 