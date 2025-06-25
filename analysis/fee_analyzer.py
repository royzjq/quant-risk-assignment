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
from sklearn.linear_model import LinearRegression

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

def prepare_fee_analysis_data(df):
    """
    Prepare data for fee analysis by filtering execution records with valid fee, qty, and vwap
    
    Args:
        df: Polars DataFrame with execution and market data
        
    Returns:
        DataFrame with fee analysis data
    """
    print("Preparing fee analysis data...")
    
    # Filter out records without execution data and required fields
    fee_df = df.filter(
        (pl.col('eTm').is_not_null()) &
        (pl.col('fee').is_not_null()) &
        (pl.col('eqty').is_not_null()) &
        (pl.col('vwap').is_not_null()) &
        (pl.col('fee') != 0)  # Remove zero fees
    )
    
    if len(fee_df) == 0:
        print("No execution records found with valid fee, qty, and vwap data")
        return None
    
    # Calculate additional metrics
    analysis_df = fee_df.with_columns([
        # Fee rate (fee per unit quantity)
        (pl.col('fee') / pl.col('eqty')).alias('fee_rate'),
        
        # Notional value (qty * vwap)
        (pl.col('eqty') * pl.col('vwap')).alias('notional_value'),
        
        # Fee as percentage of notional value
        (pl.col('fee') / (pl.col('eqty') * pl.col('vwap')) * 100).alias('fee_percentage')
    ])
    
    print(f"Prepared fee analysis data for {len(analysis_df)} execution records")
    return analysis_df

def detect_fee_outliers(fee_df, threshold=2.0):
    """
    Detect fee outliers using z-score threshold
    
    Args:
        fee_df: DataFrame with fee data
        threshold: z-score threshold (default: 2.0 sigma)
        
    Returns:
        DataFrame with outlier flags
    """
    print(f"Detecting fee outliers with {threshold} sigma threshold...")
    
    fee_values = fee_df['fee'].to_numpy()
    
    # Calculate z-scores
    mean_fee = np.mean(fee_values)
    std_fee = np.std(fee_values)
    z_scores = np.abs((fee_values - mean_fee) / std_fee)
    
    # Add outlier flags
    outlier_df = fee_df.with_columns([
        pl.lit(z_scores).alias('fee_z_score'),
        (pl.lit(z_scores) > threshold).alias('is_outlier')
    ])
    
    outlier_count = len(outlier_df.filter(pl.col('is_outlier')))
    outlier_percentage = (outlier_count / len(outlier_df)) * 100
    
    print(f"Found {outlier_count} fee outliers out of {len(outlier_df)} records ({outlier_percentage:.2f}%)")
    
    return outlier_df

def create_fee_vs_qty_plot(fee_df, output_dir="results/images"):
    """
    Create scatter plot of fee vs quantity with correlation and regression line
    """
    print("Creating fee vs quantity plot...")
    
    # Prepare data
    fee_values = fee_df['fee'].to_numpy()
    qty_values = fee_df['eqty'].to_numpy()
    is_outlier = fee_df['is_outlier'].to_numpy()
    
    # Calculate correlation
    correlation, p_value = stats.pearsonr(qty_values, fee_values)
    
    # Fit linear regression
    X = qty_values.reshape(-1, 1)
    y = fee_values
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add normal points
    normal_mask = ~is_outlier
    if np.any(normal_mask):
        fig.add_trace(go.Scatter(
            x=qty_values[normal_mask],
            y=fee_values[normal_mask],
            mode='markers',
            name='Normal',
            marker=dict(
                size=4,
                color='lightblue',
                opacity=0.6
            )
        ))
    
    # Add outlier points
    outlier_mask = is_outlier
    if np.any(outlier_mask):
        fig.add_trace(go.Scatter(
            x=qty_values[outlier_mask],
            y=fee_values[outlier_mask],
            mode='markers',
            name='Outliers (>2σ)',
            marker=dict(
                size=6,
                color='red',
                opacity=0.8,
                symbol='diamond'
            )
        ))
    
    # Add regression line
    fig.add_trace(go.Scatter(
        x=qty_values,
        y=y_pred,
        mode='lines',
        name=f'Linear Fit (R²={reg.score(X, y):.3f})',
        line=dict(color='black', width=2)
    ))
    
    # Add correlation and function info
    function_text = f"y = {reg.coef_[0]:.6f}x + {reg.intercept_:.4f}"
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.98, y=0.5,
        text=f"Correlation: {correlation:.3f}<br>"
             f"P-value: {p_value:.4f}<br>"
             f"Function: {function_text}<br>"
             f"R²: {reg.score(X, y):.3f}<br>"
             f"Outliers: {np.sum(outlier_mask)} ({np.sum(outlier_mask)/len(outlier_mask)*100:.1f}%)",
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        align="right",
        xanchor="right"
    )
    
    fig.update_layout(
        title="Fee vs Quantity Analysis",
        xaxis_title="Quantity (eqty)",
        yaxis_title="Fee",
        template="plotly_white",
        width=800,
        height=500
    )
    
    # Save plot
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.write_html(f"{output_dir}/fee_vs_qty.html")
    try:
        fig.write_image(f"{output_dir}/fee_vs_qty.png", width=800, height=500, scale=2)
    except:
        print("Warning: Could not save PNG image. HTML version saved successfully.")
    
    print(f"Fee vs quantity plot saved to {output_dir}/")
    return fig, correlation, function_text

def create_fee_vs_vwap_plot(fee_df, output_dir="results/images"):
    """
    Create scatter plot of fee vs VWAP with correlation and regression line
    """
    print("Creating fee vs VWAP plot...")
    
    # Prepare data
    fee_values = fee_df['fee'].to_numpy()
    vwap_values = fee_df['vwap'].to_numpy()
    is_outlier = fee_df['is_outlier'].to_numpy()
    
    # Calculate correlation
    correlation, p_value = stats.pearsonr(vwap_values, fee_values)
    
    # Fit linear regression
    X = vwap_values.reshape(-1, 1)
    y = fee_values
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add normal points
    normal_mask = ~is_outlier
    if np.any(normal_mask):
        fig.add_trace(go.Scatter(
            x=vwap_values[normal_mask],
            y=fee_values[normal_mask],
            mode='markers',
            name='Normal',
            marker=dict(
                size=4,
                color='lightgreen',
                opacity=0.6
            )
        ))
    
    # Add outlier points
    outlier_mask = is_outlier
    if np.any(outlier_mask):
        fig.add_trace(go.Scatter(
            x=vwap_values[outlier_mask],
            y=fee_values[outlier_mask],
            mode='markers',
            name='Outliers (>2σ)',
            marker=dict(
                size=6,
                color='red',
                opacity=0.8,
                symbol='diamond'
            )
        ))
    
    # Add regression line
    fig.add_trace(go.Scatter(
        x=vwap_values,
        y=y_pred,
        mode='lines',
        name=f'Linear Fit (R²={reg.score(X, y):.3f})',
        line=dict(color='black', width=2)
    ))
    
    # Add correlation and function info
    function_text = f"y = {reg.coef_[0]:.6f}x + {reg.intercept_:.4f}"
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.98, y=0.5,
        text=f"Correlation: {correlation:.3f}<br>"
             f"P-value: {p_value:.4f}<br>"
             f"Function: {function_text}<br>"
             f"R²: {reg.score(X, y):.3f}<br>"
             f"Outliers: {np.sum(outlier_mask)} ({np.sum(outlier_mask)/len(outlier_mask)*100:.1f}%)",
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        align="right",
        xanchor="right"
    )
    
    fig.update_layout(
        title="Fee vs VWAP Analysis",
        xaxis_title="VWAP",
        yaxis_title="Fee",
        template="plotly_white",
        width=800,
        height=500
    )
    
    # Save plot
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.write_html(f"{output_dir}/fee_vs_vwap.html")
    try:
        fig.write_image(f"{output_dir}/fee_vs_vwap.png", width=800, height=500, scale=2)
    except:
        print("Warning: Could not save PNG image. HTML version saved successfully.")
    
    print(f"Fee vs VWAP plot saved to {output_dir}/")
    return fig, correlation, function_text

def create_fee_vs_notional_plot(fee_df, output_dir="results/images"):
    """
    Create scatter plot of fee vs notional value with correlation and regression line
    """
    print("Creating fee vs notional value plot...")
    
    # Prepare data
    fee_values = fee_df['fee'].to_numpy()
    notional_values = fee_df['notional_value'].to_numpy()
    is_outlier = fee_df['is_outlier'].to_numpy()
    
    # Calculate correlation
    correlation, p_value = stats.pearsonr(notional_values, fee_values)
    
    # Fit linear regression
    X = notional_values.reshape(-1, 1)
    y = fee_values
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add normal points
    normal_mask = ~is_outlier
    if np.any(normal_mask):
        fig.add_trace(go.Scatter(
            x=notional_values[normal_mask],
            y=fee_values[normal_mask],
            mode='markers',
            name='Normal',
            marker=dict(
                size=4,
                color='orange',
                opacity=0.6
            )
        ))
    
    # Add outlier points
    outlier_mask = is_outlier
    if np.any(outlier_mask):
        fig.add_trace(go.Scatter(
            x=notional_values[outlier_mask],
            y=fee_values[outlier_mask],
            mode='markers',
            name='Outliers (>2σ)',
            marker=dict(
                size=6,
                color='red',
                opacity=0.8,
                symbol='diamond'
            )
        ))
    
    # Add regression line
    fig.add_trace(go.Scatter(
        x=notional_values,
        y=y_pred,
        mode='lines',
        name=f'Linear Fit (R²={reg.score(X, y):.3f})',
        line=dict(color='black', width=2)
    ))
    
    # Add correlation and function info
    function_text = f"y = {reg.coef_[0]:.6f}x + {reg.intercept_:.4f}"
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.98, y=0.5,
        text=f"Correlation: {correlation:.3f}<br>"
             f"P-value: {p_value:.4f}<br>"
             f"Function: {function_text}<br>"
             f"R²: {reg.score(X, y):.3f}<br>"
             f"Outliers: {np.sum(outlier_mask)} ({np.sum(outlier_mask)/len(outlier_mask)*100:.1f}%)",
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        align="right",
        xanchor="right"
    )
    
    fig.update_layout(
        title="Fee vs Notional Value Analysis",
        xaxis_title="Notional Value (qty × vwap)",
        yaxis_title="Fee",
        template="plotly_white",
        width=800,
        height=500
    )
    
    # Save plot
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.write_html(f"{output_dir}/fee_vs_notional.html")
    try:
        fig.write_image(f"{output_dir}/fee_vs_notional.png", width=800, height=500, scale=2)
    except:
        print("Warning: Could not save PNG image. HTML version saved successfully.")
    
    print(f"Fee vs notional value plot saved to {output_dir}/")
    return fig, correlation, function_text

def create_fee_distribution_plot(fee_df, output_dir="results/images"):
    """
    Create fee distribution plot with outliers highlighted
    """
    print("Creating fee distribution plot...")
    
    fee_values = fee_df['fee'].to_numpy()
    is_outlier = fee_df['is_outlier'].to_numpy()
    
    # Create distribution plot
    fig = go.Figure()
    
    # Add histogram for all data
    fig.add_trace(go.Histogram(
        x=fee_values,
        nbinsx=50,
        name='All Fees',
        opacity=0.7,
        marker_color='lightblue'
    ))
    
    # Add histogram for outliers
    outlier_fees = fee_values[is_outlier]
    if len(outlier_fees) > 0:
        fig.add_trace(go.Histogram(
            x=outlier_fees,
            nbinsx=50,
            name='Outliers (>2σ)',
            opacity=0.8,
            marker_color='red'
        ))
    
    # Add statistics text
    mean_fee = np.mean(fee_values)
    median_fee = np.median(fee_values)
    std_fee = np.std(fee_values)
    outlier_count = np.sum(is_outlier)
    outlier_pct = (outlier_count / len(fee_values)) * 100
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.98, y=0.9,
        text=f"Mean: {mean_fee:.6f}<br>"
             f"Median: {median_fee:.6f}<br>"
             f"Std: {std_fee:.6f}<br>"
             f"Count: {len(fee_values)}<br>"
             f"Outliers: {outlier_count} ({outlier_pct:.1f}%)",
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        align="right",
        xanchor="right"
    )
    
    fig.update_layout(
        title="Fee Distribution with Outliers",
        xaxis_title="Fee",
        yaxis_title="Frequency",
        template="plotly_white",
        width=800,
        height=500
    )
    
    # Save plot
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.write_html(f"{output_dir}/fee_distribution.html")
    try:
        fig.write_image(f"{output_dir}/fee_distribution.png", width=800, height=500, scale=2)
    except:
        print("Warning: Could not save PNG image. HTML version saved successfully.")
    
    print(f"Fee distribution plot saved to {output_dir}/")
    return fig

def analyze_fee_outliers(fee_df, output_dir="results"):
    """
    Analyze fee outliers to understand potential causes
    """
    print("Analyzing fee outliers...")
    
    outliers = fee_df.filter(pl.col('is_outlier'))
    normal = fee_df.filter(~pl.col('is_outlier'))
    
    if len(outliers) == 0:
        print("No fee outliers found")
        return {}
    
    analysis = {
        "outlier_count": len(outliers),
        "total_count": len(fee_df),
        "outlier_percentage": (len(outliers) / len(fee_df)) * 100,
        "outlier_stats": {
            "mean_fee": float(outliers['fee'].mean()),
            "median_fee": float(outliers['fee'].median()),
            "min_fee": float(outliers['fee'].min()),
            "max_fee": float(outliers['fee'].max()),
            "mean_qty": float(outliers['eqty'].mean()),
            "mean_vwap": float(outliers['vwap'].mean()),
            "mean_notional": float(outliers['notional_value'].mean()),
            "mean_fee_rate": float(outliers['fee_rate'].mean()),
            "mean_fee_percentage": float(outliers['fee_percentage'].mean())
        },
        "normal_stats": {
            "mean_fee": float(normal['fee'].mean()),
            "median_fee": float(normal['fee'].median()),
            "mean_qty": float(normal['eqty'].mean()),
            "mean_vwap": float(normal['vwap'].mean()),
            "mean_notional": float(normal['notional_value'].mean()),
            "mean_fee_rate": float(normal['fee_rate'].mean()),
            "mean_fee_percentage": float(normal['fee_percentage'].mean())
        }
    }
    
    # Export detailed outlier records
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outliers_file = f"{output_dir}/fee_outliers_{timestamp}.json"
    
    outlier_records = []
    for row in outliers.iter_rows(named=True):
        outlier_record = {
            "timestamp": int(row['timestamp']) if row['timestamp'] is not None else None,
            "eTm": row['eTm'] if row['eTm'] is not None else None,
            "sym": row['sym'] if row['sym'] is not None else None,
            "sde": int(row['sde']) if row['sde'] is not None else None,
            "fee": float(row['fee']),
            "eqty": float(row['eqty']),
            "vwap": float(row['vwap']),
            "notional_value": float(row['notional_value']),
            "fee_rate": float(row['fee_rate']),
            "fee_percentage": float(row['fee_percentage']),
            "fee_z_score": float(row['fee_z_score'])
        }
        outlier_records.append(outlier_record)
    
    analysis["outlier_records"] = outlier_records
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(outliers_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"Fee outlier analysis exported to {outliers_file}")
    return analysis

def generate_fee_summary_stats(fee_df):
    """
    Generate comprehensive fee summary statistics
    """
    print("Generating fee summary statistics...")
    
    fee_values = fee_df['fee'].to_numpy()
    qty_values = fee_df['eqty'].to_numpy()
    vwap_values = fee_df['vwap'].to_numpy()
    notional_values = fee_df['notional_value'].to_numpy()
    fee_rate_values = fee_df['fee_rate'].to_numpy()
    fee_percentage_values = fee_df['fee_percentage'].to_numpy()
    
    stats_dict = {
        'count': len(fee_values),
        'fee_stats': {
            'mean': float(np.mean(fee_values)),
            'median': float(np.median(fee_values)),
            'std': float(np.std(fee_values)),
            'min': float(np.min(fee_values)),
            'max': float(np.max(fee_values)),
            'p25': float(np.percentile(fee_values, 25)),
            'p75': float(np.percentile(fee_values, 75)),
            'p95': float(np.percentile(fee_values, 95)),
            'p99': float(np.percentile(fee_values, 99))
        },
        'fee_rate_stats': {
            'mean': float(np.mean(fee_rate_values)),
            'median': float(np.median(fee_rate_values)),
            'std': float(np.std(fee_rate_values))
        },
        'fee_percentage_stats': {
            'mean': float(np.mean(fee_percentage_values)),
            'median': float(np.median(fee_percentage_values)),
            'std': float(np.std(fee_percentage_values))
        },
        'correlations': {
            'fee_vs_qty': float(stats.pearsonr(qty_values, fee_values)[0]),
            'fee_vs_vwap': float(stats.pearsonr(vwap_values, fee_values)[0]),
            'fee_vs_notional': float(stats.pearsonr(notional_values, fee_values)[0])
        }
    }
    
    return stats_dict

def run_comprehensive_fee_analysis(parquet_path="results/btcusdt_processed_data.parquet", 
                                 output_dir="results", outlier_threshold=2.0):
    """
    Run comprehensive fee analysis
    
    Args:
        parquet_path: Path to processed data parquet file
        output_dir: Directory to save outputs
        outlier_threshold: Z-score threshold for outlier detection (default: 2.0 sigma)
        
    Returns:
        dict: Analysis results and statistics
    """
    print("="*60)
    print("COMPREHENSIVE FEE ANALYSIS")
    print("="*60)
    
    start_time = time.time()
    
    # 1. Load processed data
    df = load_processed_data(parquet_path)
    
    # 2. Prepare fee analysis data
    fee_df = prepare_fee_analysis_data(df)
    if fee_df is None:
        return {"status": "ERROR", "message": "No execution data found with valid fee data"}
    
    # 3. Detect fee outliers
    fee_df = detect_fee_outliers(fee_df, threshold=outlier_threshold)
    
    # 4. Generate summary statistics
    stats = generate_fee_summary_stats(fee_df)
    
    # 5. Create visualizations
    print("\nGenerating visualizations...")
    images_dir = f"{output_dir}/images"
    
    # Fee distribution
    dist_fig = create_fee_distribution_plot(fee_df, images_dir)
    
    # Fee vs quantity
    qty_result = create_fee_vs_qty_plot(fee_df, images_dir)
    qty_correlation = qty_result[1] if qty_result else None
    qty_function = qty_result[2] if qty_result else None
    
    # Fee vs VWAP
    vwap_result = create_fee_vs_vwap_plot(fee_df, images_dir)
    vwap_correlation = vwap_result[1] if vwap_result else None
    vwap_function = vwap_result[2] if vwap_result else None
    
    # Fee vs notional value
    notional_result = create_fee_vs_notional_plot(fee_df, images_dir)
    notional_correlation = notional_result[1] if notional_result else None
    notional_function = notional_result[2] if notional_result else None
    
    # 6. Analyze outliers
    outlier_analysis = analyze_fee_outliers(fee_df, output_dir)
    
    # 7. Compile results
    results = {
        "status": "COMPLETE",
        "execution_time": time.time() - start_time,
        "summary_statistics": stats,
        "correlations": {
            "fee_vs_qty": {
                "correlation": qty_correlation,
                "function": qty_function
            },
            "fee_vs_vwap": {
                "correlation": vwap_correlation,
                "function": vwap_function
            },
            "fee_vs_notional": {
                "correlation": notional_correlation,
                "function": notional_function
            }
        },
        "outlier_analysis": outlier_analysis,
        "visualizations": {
            "fee_distribution": f"{images_dir}/fee_distribution.html",
            "fee_vs_qty": f"{images_dir}/fee_vs_qty.html",
            "fee_vs_vwap": f"{images_dir}/fee_vs_vwap.html",
            "fee_vs_notional": f"{images_dir}/fee_vs_notional.html"
        }
    }
    
    # Print summary
    print("\n" + "="*60)
    print("FEE ANALYSIS SUMMARY")
    print("="*60)
    print(f"Analysis completed in {results['execution_time']:.2f} seconds")
    print(f"Total execution records analyzed: {stats.get('count', 0)}")
    
    fee_stats = stats.get('fee_stats', {})
    print(f"Fee statistics:")
    print(f"  - Mean: {fee_stats.get('mean', 0):.6f}")
    print(f"  - Median: {fee_stats.get('median', 0):.6f}")
    print(f"  - Std: {fee_stats.get('std', 0):.6f}")
    
    print(f"Correlations:")
    if qty_correlation is not None:
        print(f"  - Fee vs Quantity: {qty_correlation:.3f}")
    if vwap_correlation is not None:
        print(f"  - Fee vs VWAP: {vwap_correlation:.3f}")
    if notional_correlation is not None:
        print(f"  - Fee vs Notional: {notional_correlation:.3f}")
    
    if outlier_analysis:
        print(f"Outlier analysis:")
        print(f"  - Outliers found: {outlier_analysis.get('outlier_count', 0)} ({outlier_analysis.get('outlier_percentage', 0):.1f}%)")
        print(f"  - Mean outlier fee: {outlier_analysis.get('outlier_stats', {}).get('mean_fee', 0):.6f}")
        print(f"  - Mean normal fee: {outlier_analysis.get('normal_stats', {}).get('mean_fee', 0):.6f}")
    
    print(f"Visualizations saved to: {images_dir}/")
    print("="*60)
    
    return results

def main():
    """
    Main function to run fee analysis
    """
    try:
        results = run_comprehensive_fee_analysis()
        return results
    except Exception as e:
        print(f"Fee analysis failed: {str(e)}")
        return {"status": "ERROR", "error": str(e)}

if __name__ == "__main__":
    main() 