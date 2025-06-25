import polars as pl
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from pathlib import Path
import time
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

def calculate_slippage(df):
    """
    Calculate slippage according to the framework definition:
    - Buy orders: slippage = execution_price - mid_price
    - Sell orders: slippage = mid_price - execution_price
    
    Args:
        df: Polars DataFrame with execution and market data
        
    Returns:
        DataFrame with slippage calculations
    """
    print("Calculating slippage...")
    
    # Filter out records without execution data
    exec_df = df.filter(pl.col('eTm').is_not_null())
    
    if len(exec_df) == 0:
        print("No execution records found for slippage calculation")
        return None
    
    # Calculate slippage based on side (sde)
    # sde = 1: buy order, sde = 2: sell order
    slippage_df = exec_df.with_columns([
        # Buy orders: slippage = vwap - mid_price
        # Sell orders: slippage = mid_price - vwap
        pl.when(pl.col('sde') == 1)
        .then(pl.col('vwap') - pl.col('mid_price'))
        .when(pl.col('sde') == 2)
        .then(pl.col('mid_price') - pl.col('vwap'))
        .otherwise(None)
        .alias('slippage'),
        
        # Convert slippage to basis points (bps)
        pl.when(pl.col('sde') == 1)
        .then((pl.col('vwap') - pl.col('mid_price')) / pl.col('mid_price') * 10000)
        .when(pl.col('sde') == 2)
        .then((pl.col('mid_price') - pl.col('vwap')) / pl.col('mid_price') * 10000)
        .otherwise(None)
        .alias('slippage_bps')
    ])
    
    print(f"Calculated slippage for {len(slippage_df)} execution records")
    return slippage_df

def calculate_market_volatility(df, window=100):
    """
    Calculate market volatility using rolling window of mid price
    
    Args:
        df: DataFrame with market data
        window: Number of periods for volatility calculation (default: 100)
        
    Returns:
        DataFrame with volatility column
    """
    print(f"Calculating market volatility with {window}-period window...")
    
    # Calculate rolling volatility of mid price
    df_with_vol = df.with_columns([
        pl.col('mid_price').log().diff().rolling_std(window_size=window).alias('mid_price_volatility')
    ])
    
    return df_with_vol

def calculate_liquidity_measure(df, depth=10):
    """
    Calculate average volume for bid-1 and ask-1 as liquidity measure
    
    Args:
        df: DataFrame with market data
        depth: Number of periods to average (default: 10)
        
    Returns:
        DataFrame with liquidity measure
    """
    print(f"Calculating liquidity measure with {depth}-period average...")
    
    # Calculate rolling average of bid-1 and ask-1 volumes
    df_with_liquidity = df.with_columns([
        ((pl.col('bidVol-1').rolling_mean(window_size=depth) + 
          pl.col('askVol-1').rolling_mean(window_size=depth)) / 2).alias('avg_liquidity')
    ])
    
    return df_with_liquidity

def create_slippage_distribution_plot(slippage_df, output_dir="results/images", symbol="BTCUSDT"):
    """
    Create slippage distribution plot
    """
    print("Creating slippage distribution plot...")
    
    # Prepare data
    slippage_values = slippage_df.filter(pl.col('slippage_bps').is_not_null())['slippage_bps'].to_numpy()
    
    # Create distribution plot
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=slippage_values,
        nbinsx=50,
        name='Slippage Distribution',
        opacity=0.7,
        marker_color='lightblue'
    ))
    
    # Add statistics text
    mean_slippage = np.mean(slippage_values)
    median_slippage = np.median(slippage_values)
    std_slippage = np.std(slippage_values)
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.7, y=0.9,
        text=f"Mean: {mean_slippage:.2f} bps<br>"
             f"Median: {median_slippage:.2f} bps<br>"
             f"Std: {std_slippage:.2f} bps<br>"
             f"Count: {len(slippage_values)}",
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    fig.update_layout(
        title=f"{symbol} Slippage Distribution (Basis Points)",
        xaxis_title="Slippage (bps)",
        yaxis_title="Frequency",
        showlegend=False,
        template="plotly_white",
        width=800,
        height=500
    )
    
    # Save plot with symbol prefix
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    symbol_lower = symbol.lower()
    fig.write_html(f"{output_dir}/{symbol_lower}_slippage_distribution.html")
    try:
        fig.write_image(f"{output_dir}/{symbol_lower}_slippage_distribution.png", width=800, height=500, scale=2)
    except:
        print("Warning: Could not save PNG image. HTML version saved successfully.")
    
    print(f"Slippage distribution plot saved to {output_dir}/")
    return fig

def create_slippage_vs_order_size_plot(slippage_df, output_dir="results/images", symbol="BTCUSDT"):
    """
    Create scatter plot of slippage vs order size with correlation and regression line
    """
    print("Creating slippage vs order size plot...")
    
    # Prepare data
    analysis_df = slippage_df.filter(
        (pl.col('slippage_bps').is_not_null()) & 
        (pl.col('eqty').is_not_null())
    )
    
    if len(analysis_df) == 0:
        print("No data available for slippage vs order size analysis")
        return None
    
    slippage_values = analysis_df['slippage_bps'].to_numpy()
    order_sizes = analysis_df['eqty'].to_numpy()
    
    # Calculate correlation
    correlation, p_value = stats.pearsonr(order_sizes, slippage_values)
    
    # Fit linear regression
    X = order_sizes.reshape(-1, 1)
    y = slippage_values
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=order_sizes,
        y=slippage_values,
        mode='markers',
        name='Data Points',
        marker=dict(
            size=4,
            color='lightblue',
            opacity=0.6
        )
    ))
    
    # Add regression line
    fig.add_trace(go.Scatter(
        x=order_sizes,
        y=y_pred,
        mode='lines',
        name=f'Linear Fit (R²={reg.score(X, y):.3f})',
        line=dict(color='red', width=2)
    ))
    
    # Add correlation and function info
    function_text = f"y = {reg.coef_[0]:.4f}x + {reg.intercept_:.2f}"
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.98, y=0.5,
        text=f"Correlation: {correlation:.3f}<br>"
             f"P-value: {p_value:.4f}<br>"
             f"Function: {function_text}<br>"
             f"R²: {reg.score(X, y):.3f}",
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        align="right",
        xanchor="right"
    )
    
    fig.update_layout(
        title=f"{symbol} Slippage vs Order Size",
        xaxis_title="Order Size (eqty)",
        yaxis_title="Slippage (bps)",
        template="plotly_white",
        width=800,
        height=500
    )
    
    # Save plot with symbol prefix
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    symbol_lower = symbol.lower()
    fig.write_html(f"{output_dir}/{symbol_lower}_slippage_vs_order_size.html")
    try:
        fig.write_image(f"{output_dir}/{symbol_lower}_slippage_vs_order_size.png", width=800, height=500, scale=2)
    except:
        print("Warning: Could not save PNG image. HTML version saved successfully.")
    
    print(f"Slippage vs order size plot saved to {output_dir}/")
    return fig, correlation, function_text

def create_slippage_vs_volatility_plot(analysis_df, output_dir="results/images", symbol="BTCUSDT"):
    """
    Create scatter plot of slippage vs market volatility
    """
    print("Creating slippage vs volatility plot...")
    
    # Filter valid data
    valid_df = analysis_df.filter(
        (pl.col('slippage_bps').is_not_null()) & 
        (pl.col('mid_price_volatility').is_not_null())
    )
    
    if len(valid_df) == 0:
        print("No data available for slippage vs volatility analysis")
        return None
    
    slippage_values = valid_df['slippage_bps'].to_numpy()
    volatility_values = valid_df['mid_price_volatility'].to_numpy()
    
    # Calculate correlation
    correlation, p_value = stats.pearsonr(volatility_values, slippage_values)
    
    # Fit linear regression
    X = volatility_values.reshape(-1, 1)
    y = slippage_values
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=volatility_values,
        y=slippage_values,
        mode='markers',
        name='Data Points',
        marker=dict(
            size=4,
            color='lightgreen',
            opacity=0.6
        )
    ))
    
    # Add regression line
    fig.add_trace(go.Scatter(
        x=volatility_values,
        y=y_pred,
        mode='lines',
        name=f'Linear Fit (R²={reg.score(X, y):.3f})',
        line=dict(color='red', width=2)
    ))
    
    # Add correlation and function info
    function_text = f"y = {reg.coef_[0]:.4f}x + {reg.intercept_:.2f}"
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.98, y=0.5,
        text=f"Correlation: {correlation:.3f}<br>"
             f"P-value: {p_value:.4f}<br>"
             f"Function: {function_text}<br>"
             f"R²: {reg.score(X, y):.3f}",
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        align="right",
        xanchor="right"
    )
    
    fig.update_layout(
        title=f"{symbol} Slippage vs Market Volatility",
        xaxis_title="Mid Price Volatility",
        yaxis_title="Slippage (bps)",
        template="plotly_white",
        width=800,
        height=500
    )
    
    # Save plot with symbol prefix
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    symbol_lower = symbol.lower()
    fig.write_html(f"{output_dir}/{symbol_lower}_slippage_vs_volatility.html")
    try:
        fig.write_image(f"{output_dir}/{symbol_lower}_slippage_vs_volatility.png", width=800, height=500, scale=2)
    except:
        print("Warning: Could not save PNG image. HTML version saved successfully.")
    
    print(f"Slippage vs volatility plot saved to {output_dir}/")
    return fig, correlation, function_text

def create_slippage_vs_liquidity_plot(analysis_df, output_dir="results/images", symbol="BTCUSDT"):
    """
    Create scatter plot of slippage vs liquidity
    """
    print("Creating slippage vs liquidity plot...")
    
    # Filter valid data
    valid_df = analysis_df.filter(
        (pl.col('slippage_bps').is_not_null()) & 
        (pl.col('avg_liquidity').is_not_null())
    )
    
    if len(valid_df) == 0:
        print("No data available for slippage vs liquidity analysis")
        return None
    
    slippage_values = valid_df['slippage_bps'].to_numpy()
    liquidity_values = valid_df['avg_liquidity'].to_numpy()
    
    # Calculate correlation
    correlation, p_value = stats.pearsonr(liquidity_values, slippage_values)
    
    # Fit linear regression
    X = liquidity_values.reshape(-1, 1)
    y = slippage_values
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=liquidity_values,
        y=slippage_values,
        mode='markers',
        name='Data Points',
        marker=dict(
            size=4,
            color='orange',
            opacity=0.6
        )
    ))
    
    # Add regression line
    fig.add_trace(go.Scatter(
        x=liquidity_values,
        y=y_pred,
        mode='lines',
        name=f'Linear Fit (R²={reg.score(X, y):.3f})',
        line=dict(color='red', width=2)
    ))
    
    # Add correlation and function info
    function_text = f"y = {reg.coef_[0]:.4f}x + {reg.intercept_:.2f}"
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.98, y=0.5,
        text=f"Correlation: {correlation:.3f}<br>"
             f"P-value: {p_value:.4f}<br>"
             f"Function: {function_text}<br>"
             f"R²: {reg.score(X, y):.3f}",
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        align="right",
        xanchor="right"
    )
    
    fig.update_layout(
        title=f"{symbol} Slippage vs Liquidity",
        xaxis_title="Average Liquidity (bid-1 + ask-1 volumes)",
        yaxis_title="Slippage (bps)",
        template="plotly_white",
        width=800,
        height=500
    )
    
    # Save plot with symbol prefix
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    symbol_lower = symbol.lower()
    fig.write_html(f"{output_dir}/{symbol_lower}_slippage_vs_liquidity.html")
    try:
        fig.write_image(f"{output_dir}/{symbol_lower}_slippage_vs_liquidity.png", width=800, height=500, scale=2)
    except:
        print("Warning: Could not save PNG image. HTML version saved successfully.")
    
    print(f"Slippage vs liquidity plot saved to {output_dir}/")
    return fig, correlation, function_text

def generate_slippage_summary_stats(slippage_df):
    """
    Generate comprehensive slippage summary statistics
    """
    print("Generating slippage summary statistics...")
    
    # Filter valid slippage data
    valid_slippage = slippage_df.filter(pl.col('slippage_bps').is_not_null())
    
    if len(valid_slippage) == 0:
        print("No valid slippage data found")
        return {}
    
    slippage_values = valid_slippage['slippage_bps'].to_numpy()
    
    stats_dict = {
        'count': len(slippage_values),
        'mean_bps': np.mean(slippage_values),
        'median_bps': np.median(slippage_values),
        'std_bps': np.std(slippage_values),
        'min_bps': np.min(slippage_values),
        'max_bps': np.max(slippage_values),
        'p25_bps': np.percentile(slippage_values, 25),
        'p75_bps': np.percentile(slippage_values, 75),
        'p95_bps': np.percentile(slippage_values, 95),
        'p99_bps': np.percentile(slippage_values, 99),
    }
    
    # Analysis by side (buy vs sell)
    buy_orders = valid_slippage.filter(pl.col('sde') == 1)
    sell_orders = valid_slippage.filter(pl.col('sde') == 2)
    
    if len(buy_orders) > 0:
        buy_slippage = buy_orders['slippage_bps'].to_numpy()
        stats_dict['buy_orders_count'] = len(buy_slippage)
        stats_dict['buy_mean_bps'] = np.mean(buy_slippage)
        stats_dict['buy_median_bps'] = np.median(buy_slippage)
        stats_dict['buy_std_bps'] = np.std(buy_slippage)
    
    if len(sell_orders) > 0:
        sell_slippage = sell_orders['slippage_bps'].to_numpy()
        stats_dict['sell_orders_count'] = len(sell_slippage)
        stats_dict['sell_mean_bps'] = np.mean(sell_slippage)
        stats_dict['sell_median_bps'] = np.median(sell_slippage)
        stats_dict['sell_std_bps'] = np.std(sell_slippage)
    
    return stats_dict

def run_comprehensive_slippage_analysis(parquet_path="results/btcusdt_processed_data.parquet", 
                                      output_dir="results/images"):
    """
    Run comprehensive slippage analysis according to the risk analysis framework
    
    Args:
        parquet_path: Path to processed data parquet file
        output_dir: Directory to save visualization outputs
        
    Returns:
        dict: Analysis results and statistics
    """
    print("="*60)
    print("COMPREHENSIVE SLIPPAGE ANALYSIS")
    print("="*60)
    
    start_time = time.time()
    
    # 1. Load processed data
    df = load_processed_data(parquet_path)
    
    return run_comprehensive_slippage_analysis_with_data(df, output_dir, start_time)

def run_comprehensive_slippage_analysis_with_data(df, output_dir="results/images", start_time=None, symbol="BTCUSDT"):
    """
    Run comprehensive slippage analysis with pre-loaded data
    
    Args:
        df: Pre-loaded Polars DataFrame with processed data
        output_dir: Directory to save visualization outputs
        start_time: Optional start time for timing calculations
        symbol: Trading symbol for consistency (not used in file naming for slippage analysis)
        
    Returns:
        dict: Analysis results and statistics
    """
    print("="*60)
    print("COMPREHENSIVE SLIPPAGE ANALYSIS")
    print("="*60)
    
    if start_time is None:
        start_time = time.time()
    
    print(f"Using processed data with {len(df)} records.")
    
    # 2. Calculate slippage
    slippage_df = calculate_slippage(df)
    if slippage_df is None:
        return {"status": "ERROR", "message": "No execution data found"}
    
    # 3. Calculate market volatility (100-period window)
    df_with_vol = calculate_market_volatility(df, window=100)
    
    # 4. Calculate liquidity measure (10-period average)
    df_with_liquidity = calculate_liquidity_measure(df_with_vol, depth=10)
    
    # 5. Merge slippage with volatility and liquidity data
    analysis_df = slippage_df.join(
        df_with_liquidity.select(['timestamp', 'mid_price_volatility', 'avg_liquidity']),
        on='timestamp',
        how='left'
    )
    
    # 6. Generate summary statistics
    stats = generate_slippage_summary_stats(slippage_df)
    
    # 7. Create visualizations
    print("\nGenerating visualizations...")
    
    # Slippage distribution
    dist_fig = create_slippage_distribution_plot(slippage_df, output_dir, symbol)
    
    # Slippage vs order size
    size_result = create_slippage_vs_order_size_plot(slippage_df, output_dir, symbol)
    size_correlation = size_result[1] if size_result else None
    size_function = size_result[2] if size_result else None
    
    # Slippage vs volatility
    vol_result = create_slippage_vs_volatility_plot(analysis_df, output_dir, symbol)
    vol_correlation = vol_result[1] if vol_result else None
    vol_function = vol_result[2] if vol_result else None
    
    # Slippage vs liquidity
    liq_result = create_slippage_vs_liquidity_plot(analysis_df, output_dir, symbol)
    liq_correlation = liq_result[1] if liq_result else None
    liq_function = liq_result[2] if liq_result else None
    
    # 8. Compile results
    results = {
        "status": "COMPLETE",
        "execution_time": time.time() - start_time,
        "summary_statistics": stats,
        "correlations": {
            "slippage_vs_order_size": {
                "correlation": size_correlation,
                "function": size_function
            },
            "slippage_vs_volatility": {
                "correlation": vol_correlation,
                "function": vol_function
            },
            "slippage_vs_liquidity": {
                "correlation": liq_correlation,
                "function": liq_function
            }
        },
        "visualizations": {
            "slippage_distribution": f"{output_dir}/{symbol.lower()}_slippage_distribution.html",
            "slippage_vs_order_size": f"{output_dir}/{symbol.lower()}_slippage_vs_order_size.html",
            "slippage_vs_volatility": f"{output_dir}/{symbol.lower()}_slippage_vs_volatility.html",
            "slippage_vs_liquidity": f"{output_dir}/{symbol.lower()}_slippage_vs_liquidity.html"
        }
    }
    
    # Print summary
    print("\n" + "="*60)
    print("SLIPPAGE ANALYSIS SUMMARY")
    print("="*60)
    print(f"Analysis completed in {results['execution_time']:.2f} seconds")
    print(f"Total execution records analyzed: {stats.get('count', 0)}")
    print(f"Mean slippage: {stats.get('mean_bps', 0):.2f} bps")
    print(f"Median slippage: {stats.get('median_bps', 0):.2f} bps")
    print(f"Standard deviation: {stats.get('std_bps', 0):.2f} bps")
    
    if size_correlation is not None:
        print(f"Slippage vs Order Size correlation: {size_correlation:.3f}")
    if vol_correlation is not None:
        print(f"Slippage vs Volatility correlation: {vol_correlation:.3f}")
    if liq_correlation is not None:
        print(f"Slippage vs Liquidity correlation: {liq_correlation:.3f}")
    
    print(f"Visualizations saved to: {output_dir}/")
    print("="*60)
    
    return results

def main():
    """
    Main function to run slippage analysis
    """
    try:
        results = run_comprehensive_slippage_analysis()
        return results
    except Exception as e:
        print(f"Slippage analysis failed: {str(e)}")
        return {"status": "ERROR", "error": str(e)}

if __name__ == "__main__":
    main() 