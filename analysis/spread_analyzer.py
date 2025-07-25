#!/usr/bin/env python3
"""
Spread Analysis Module

This module analyzes bid spread and ask spread over time, with execution points marked.
Uses the preprocessed parquet file from the data preparation step.
"""

import polars as pl
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np
import json
import time

pio.templates.default = "plotly_white" # Set a clean default template

def load_processed_data(parquet_file='results/btcusdt_processed_data.parquet'):
    """
    Load processed data from parquet file
    
    Args:
        parquet_file: Path to the processed parquet file
        
    Returns:
        pl.DataFrame: Loaded data
    """
    print(f"Loading processed data from {parquet_file}...")
    
    if not Path(parquet_file).exists():
        raise FileNotFoundError(f"Processed data file not found: {parquet_file}")
    
    df = pl.read_parquet(parquet_file)
    print(f"Loaded {len(df)} records from processed data")
    
    return df


def calculate_spreads(df):
    """
    Calculate bid spread and ask spread
    
    Args:
        df: DataFrame with market data
        
    Returns:
        pl.DataFrame: DataFrame with spread calculations added
    """
    print("Calculating bid and ask spreads...")
    
    df_with_spreads = df.with_columns([
        ((pl.col('bidPrice-1') + pl.col('askPrice-1')) / 2).alias('mid_price'),
        # Recalculate bid spread and ask spread relative to mid_price
        (pl.col('bidPrice-1') - pl.col('mid_price')).alias('bid_spread'),
        (pl.col('askPrice-1') - pl.col('mid_price')).alias('ask_spread'),
    ])
    
    print("Spread calculations completed")
    return df_with_spreads


def prepare_plot_data(df):
    """
    Prepare data for plotting by converting timestamps and filtering
    
    Args:
        df: DataFrame with spread data
        
    Returns:
        tuple: (market_data, execution_data) as Polars DataFrames for plotting
    """
    print("Preparing data for plotting...")

    # Ensure 'timestamp' is in microseconds (already done by data_processor)
    market_data = df.with_columns([
        (pl.from_epoch(pl.col('timestamp'), time_unit='us')).alias('datetime')
    ]).select(['datetime', 'bid_spread', 'ask_spread'])
    
    # Filter execution records (where eTm is not null)
    # Convert eTm_microseconds to datetime for execution points
    execution_data = df.filter(pl.col('eTm').is_not_null()).with_columns([
        (pl.from_epoch(pl.col('eTm_microseconds'), time_unit='us')).alias('exec_datetime')
    ]).select([
        'exec_datetime', 'px', 'qty', 'sde' # Keep relevant execution columns
    ])
    
    print(f"Market data records: {len(market_data)}")
    print(f"Execution records: {len(execution_data)}")
    
    return market_data, execution_data


def create_spread_visualization(market_data, execution_data, output_dir='results/images', downsample_interval=None, symbol="BTCUSDT"):
    """
    Create and save the spread visualization using Plotly
    
    Args:
        market_data: Polars DataFrame with market data and spreads
        execution_data: Polars DataFrame with execution records
        output_dir: Directory to save the plot
        downsample_interval: If not None, take every Nth market data point for plotting to speed up.
        symbol: Trading symbol for file naming and title
    """
    print("Creating spread visualization...")
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig = go.Figure()

    # Apply downsampling for market data if interval is provided
    if downsample_interval and isinstance(downsample_interval, int) and downsample_interval > 1:
        print(f"Downsampling market data for visualization (interval: {downsample_interval})...")
        # Select every Nth row for downsampling
        market_data_sampled = market_data.with_row_index().filter(pl.col("index") % downsample_interval == 0).drop("index")
        print(f"Original market data points: {len(market_data)}, Sampled points: {len(market_data_sampled)}")
    else:
        market_data_sampled = market_data
    
    # Plot bid and ask spreads
    fig.add_trace(go.Scatter(x=market_data_sampled['datetime'], y=market_data_sampled['bid_spread'],
                             mode='lines', name='Bid Spread', line=dict(color='blue', width=1)))
    fig.add_trace(go.Scatter(x=market_data_sampled['datetime'], y=market_data_sampled['ask_spread'],
                             mode='lines', name='Ask Spread', line=dict(color='green', width=1)))

    # Add execution points at y=0 (spread of 0 relative to mid price)
    if len(execution_data) > 0:
        fig.add_trace(go.Scatter(x=execution_data['exec_datetime'], y=[0] * len(execution_data),
                                 mode='markers', name='Executions',
                                 marker=dict(color='red', size=8, symbol='circle', opacity=0.8),
                                 hoverinfo='text',
                                 hovertext=[
                                     f"Time: {t}<br>Exec Price: {p:.6f}<br>Qty: {q}<br>Side: {s}"
                                     for t, p, q, s in zip(execution_data['exec_datetime'], execution_data['px'], execution_data['qty'], execution_data['sde'])
                                 ]))
                                 
    # Update layout
    fig.update_layout(
        title_text=f'{symbol} Bid/Ask Spreads Over Time with Executions',
        xaxis_title_text='Time',
        yaxis_title_text='Spread (USDT) relative to Mid Price',
        hovermode='x unified', # Show hover info for all traces at a given x-coordinate
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.5)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1),
        xaxis_rangeslider_visible=True, # Add a rangeslider for easy navigation
        yaxis_zeroline=True, yaxis_zerolinecolor='lightgray', yaxis_zerolinewidth=1 # Show zero line for clarity
    )

    # Save the plot with symbol prefix
    symbol_lower = symbol.lower()
    output_file_png = Path(output_dir) / f'{symbol_lower}_spread_execution_visualization.png'
    output_file_html = Path(output_dir) / f'{symbol_lower}_spread_execution_visualization.html' # Plotly can save interactive HTML

    try:
        fig.write_image(str(output_file_png), scale=2) # scale for higher resolution
    except:
        print("Warning: Could not save PNG image. HTML version saved successfully.")
    fig.write_html(str(output_file_html))
    
    print(f"Visualization saved to {output_file_png}")
    print(f"Interactive HTML saved to {output_file_html}")
    
    return output_file_png


def create_bidask_spread_distribution(market_data, execution_data, output_dir='results/images', df=None, symbol="BTCUSDT"):
    """
    Create bid-ask spread distribution chart with execution count overlay
    
    Args:
        market_data: Polars DataFrame with market data and spreads
        execution_data: Polars DataFrame with execution records
        output_dir: Directory to save the plot
        df: Optional pre-loaded DataFrame with processed data
        symbol: Trading symbol for file naming and title
    """
    print("Creating bid-ask spread distribution chart...")
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load original data to calculate bid-ask spread if not provided
    if df is None:
        df = load_processed_data()
    
    # Calculate bid-ask spread (positive values)
    df_with_bidask_spread = df.with_columns([
        (pl.col('askPrice-1') - pl.col('bidPrice-1')).alias('bidask_spread')
    ]).filter(pl.col('bidask_spread') > 0)
    
    # Separate market data and execution data
    market_data_spreads = df_with_bidask_spread.filter(pl.col('eTm').is_null())
    execution_data_spreads = df_with_bidask_spread.filter(pl.col('eTm').is_not_null())
    
    # Convert to numpy for histogram
    market_bidask_spread_data = market_data_spreads['bidask_spread'].to_numpy()
    exec_bidask_spread_data = execution_data_spreads['bidask_spread'].to_numpy()
    
    # Use all data to determine the percentile range
    all_bidask_spread_data = df_with_bidask_spread['bidask_spread'].to_numpy()
    
    # Create custom bins using 99% percentile as upper bound for visualization
    num_fine_bins = 30
    min_spread = np.min(all_bidask_spread_data)
    percentile_99 = np.percentile(all_bidask_spread_data, 99)
    
    # Create bin edges from min to 99% percentile
    fine_bin_edges = np.linspace(min_spread, percentile_99, num_fine_bins + 1)
    
    # For visualization, cap values at 99% percentile
    market_spread_for_plot = np.clip(market_bidask_spread_data, min_spread, percentile_99)
    exec_spread_for_plot = np.clip(exec_bidask_spread_data, min_spread, percentile_99)
    
    # Calculate histogram counts using the capped data for plotting
    market_counts, _ = np.histogram(market_spread_for_plot, bins=fine_bin_edges)
    exec_counts, _ = np.histogram(exec_spread_for_plot, bins=fine_bin_edges)
    
    # Bin centers for plotting
    bin_centers = (fine_bin_edges[:-1] + fine_bin_edges[1:]) / 2
    
    # Count how many values were capped at the 99% percentile
    market_capped_count = np.sum(market_bidask_spread_data > percentile_99)
    exec_capped_count = np.sum(exec_bidask_spread_data > percentile_99)
    
    print(f"Market data: {len(market_bidask_spread_data)} records, {market_capped_count} ({market_capped_count/len(market_bidask_spread_data)*100:.1f}%) exceed 99% percentile")
    print(f"Execution data: {len(exec_bidask_spread_data)} records, {exec_capped_count} ({exec_capped_count/len(exec_bidask_spread_data)*100:.1f}%) exceed 99% percentile")
    print(f"99% percentile value: {percentile_99:.6f}")
    
    # Create histogram plot with dual y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add market data distribution (bar chart)
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=market_counts,
            name='Market Data Count',
            marker_color='lightblue',
            opacity=0.7,
            yaxis='y'
        ),
        secondary_y=False,
    )
    
    # Add execution data distribution (line chart)
    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=exec_counts,
            mode='lines+markers',
            name='Execution Count',
            line=dict(color='red', width=3),
            marker=dict(size=6, color='red'),
            yaxis='y2'
        ),
        secondary_y=True,
    )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Bid-Ask Spread Distribution: Market vs Execution',
        xaxis_title='Bid-Ask Spread (USDT)',
        template='plotly_white',
        legend=dict(x=0.7, y=0.95)
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Market Data Frequency", secondary_y=False)
    fig.update_yaxes(title_text="Execution Count", secondary_y=True)
    
    # Save the plot with symbol prefix
    symbol_lower = symbol.lower()
    output_file_png = Path(output_dir) / f'{symbol_lower}_bidask_spread_distribution.png'
    output_file_html = Path(output_dir) / f'{symbol_lower}_bidask_spread_distribution.html'
    
    try:
        fig.write_image(str(output_file_png), scale=2)
    except:
        print("Warning: Could not save PNG image. HTML version saved successfully.")
    fig.write_html(str(output_file_html))
    
    print(f"Bid-ask spread distribution saved to {output_file_png}")
    print(f"Interactive HTML saved to {output_file_html}")


def export_large_spread_trades(threshold=0.2, output_dir='results', df=None, symbol="BTCUSDT"):
    """
    Export trades that occurred when bid-ask spread was larger than the threshold
    
    Args:
        threshold: Spread threshold in USDT
        output_dir: Directory to save the output file
        df: Optional pre-loaded DataFrame with processed data
        symbol: Trading symbol for file naming
    """
    print(f"Exporting trades with bid-ask spread > {threshold} USDT...")
    
    # Load processed data if not provided
    if df is None:
        df = load_processed_data()
    
    # Calculate bid-ask spread and filter for large spreads
    df_with_bidask_spread = df.with_columns([
        (pl.col('askPrice-1') - pl.col('bidPrice-1')).alias('bidask_spread')
    ])
    
    # Filter for executions with large spreads
    large_spread_trades = df_with_bidask_spread.filter(
        (pl.col('eTm').is_not_null()) &  # Has execution
        (pl.col('bidask_spread') > threshold)  # Large spread
    ).select([
        'eTm', 'timestamp', 'sym', 'sde', 'px', 'qty', 'vwap', 'fee',
        'bidPrice-1', 'askPrice-1', 'bidask_spread'
    ])
    
    if len(large_spread_trades) == 0:
        print(f"No trades found with bid-ask spread > {threshold} USDT")
        return
    
    # Convert to dict for JSON export
    trades_data = {
        "threshold_usdt": threshold,
        "trade_count": len(large_spread_trades),
        "trades": []
    }
    
    for row in large_spread_trades.iter_rows(named=True):
        trade = {
            "execution_time": row['eTm'],
            "market_timestamp": int(row['timestamp']) if row['timestamp'] is not None else None,
            "symbol": row['sym'],
            "side": int(row['sde']) if row['sde'] is not None else None,
            "execution_price": float(row['px']) if row['px'] is not None else None,
            "quantity": float(row['qty']) if row['qty'] is not None else None,
            "vwap": float(row['vwap']) if row['vwap'] is not None else None,
            "fee": float(row['fee']) if row['fee'] is not None else None,
            "bid_price": float(row['bidPrice-1']) if row['bidPrice-1'] is not None else None,
            "ask_price": float(row['askPrice-1']) if row['askPrice-1'] is not None else None,
            "bidask_spread": float(row['bidask_spread']) if row['bidask_spread'] is not None else None
        }
        trades_data["trades"].append(trade)
    
    # Export to JSON with symbol prefix
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f'{symbol.lower()}_large_spread_trades_{threshold}usdt_{timestamp}.json'
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(trades_data, f, indent=2, ensure_ascii=False)
    
    print(f"Exported {len(large_spread_trades)} trades with large spreads to {output_file}")


def generate_summary_stats(market_data, execution_data, df_with_spreads=None):
    """
    Generate summary statistics for spreads and executions
    
    Args:
        market_data: Polars DataFrame with market data
        execution_data: Polars DataFrame with execution data
        df_with_spreads: Optional pre-computed DataFrame with spreads
        
    Returns:
        dict: Summary statistics
    """
    print("Generating summary statistics...")
    
    # Use provided data or load full data for complete statistics
    if df_with_spreads is None:
        df = load_processed_data()
        df_with_spreads = calculate_spreads(df)
    
    # Calculate bid-ask spread
    df_with_bidask = df_with_spreads.with_columns([
        (pl.col('askPrice-1') - pl.col('bidPrice-1')).alias('bidask_spread')
    ])
    
    # Market data statistics
    bid_spreads = df_with_spreads['bid_spread'].to_numpy()
    ask_spreads = df_with_spreads['ask_spread'].to_numpy()
    bidask_spreads = df_with_bidask['bidask_spread'].to_numpy()
    
    stats = {
        "market_data": {
            "total_records": len(df_with_spreads),
            "bid_spread_stats": {
                "mean": float(np.mean(bid_spreads)),
                "median": float(np.median(bid_spreads)),
                "std": float(np.std(bid_spreads)),
                "min": float(np.min(bid_spreads)),
                "max": float(np.max(bid_spreads))
            },
            "ask_spread_stats": {
                "mean": float(np.mean(ask_spreads)),
                "median": float(np.median(ask_spreads)),
                "std": float(np.std(ask_spreads)),
                "min": float(np.min(ask_spreads)),
                "max": float(np.max(ask_spreads))
            },
            "bidask_spread_stats": {
                "mean": float(np.mean(bidask_spreads)),
                "median": float(np.median(bidask_spreads)),
                "std": float(np.std(bidask_spreads)),
                "min": float(np.min(bidask_spreads)),
                "max": float(np.max(bidask_spreads))
            }
        },
        "execution_data": {
            "total_executions": len(execution_data),
            "buy_orders": len(execution_data.filter(pl.col('sde') == 1)) if len(execution_data) > 0 else 0,
            "sell_orders": len(execution_data.filter(pl.col('sde') == 2)) if len(execution_data) > 0 else 0
        }
    }
    
    return stats


def run_comprehensive_spread_analysis(parquet_path="results/btcusdt_processed_data.parquet", 
                                     output_dir="results", downsample_interval=100):
    """
    Run comprehensive spread analysis
    
    Args:
        parquet_path: Path to processed data parquet file
        output_dir: Directory to save outputs
        downsample_interval: Downsampling interval for visualization
        
    Returns:
        dict: Analysis results and statistics
    """
    print("="*60)
    print("COMPREHENSIVE SPREAD ANALYSIS")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # 1. Load processed data
        df = load_processed_data(parquet_path)
        
        return run_comprehensive_spread_analysis_with_data(df, output_dir, downsample_interval, start_time)
        
    except Exception as e:
        print(f"Spread analysis failed: {str(e)}")
        return {"status": "ERROR", "error": str(e)}

def run_comprehensive_spread_analysis_with_data(df, output_dir="results", downsample_interval=100, start_time=None, symbol="BTCUSDT"):
    """
    Run comprehensive spread analysis with pre-loaded data
    
    Args:
        df: Pre-loaded Polars DataFrame with processed data
        output_dir: Directory to save outputs
        downsample_interval: Downsampling interval for visualization
        start_time: Optional start time for timing calculations
        symbol: Trading symbol for file naming
        
    Returns:
        dict: Analysis results and statistics
    """
    print("="*60)
    print("COMPREHENSIVE SPREAD ANALYSIS")
    print("="*60)
    
    if start_time is None:
        start_time = time.time()
    
    try:
        print(f"Using processed data with {len(df)} records.")
        
        # 2. Calculate spreads
        df_with_spreads = calculate_spreads(df)
        
        # 3. Prepare plot data
        market_data, execution_data = prepare_plot_data(df_with_spreads)
        
        # 4. Generate summary statistics
        stats = generate_summary_stats(market_data, execution_data, df_with_spreads)
        
        # 5. Create visualizations
        print("\nGenerating visualizations...")
        images_dir = f"{output_dir}/images"
        
        # Spread time series
        create_spread_visualization(market_data, execution_data, images_dir, downsample_interval, symbol)
        
        # Spread distribution
        create_bidask_spread_distribution(market_data, execution_data, images_dir, df, symbol)
        
        # 6. Export large spread trades
        export_large_spread_trades(threshold=0.2, output_dir=output_dir, df=df, symbol=symbol)
        
        # 7. Compile results
        results = {
            "status": "COMPLETE",
            "execution_time": time.time() - start_time,
            "summary_statistics": stats,
            "visualizations": {
                "spread_time_series": f"{images_dir}/{symbol.lower()}_spread_execution_visualization.html",
                "spread_distribution": f"{images_dir}/{symbol.lower()}_bidask_spread_distribution.html"
            }
        }
        
        # Print summary
        print("\n" + "="*60)
        print("SPREAD ANALYSIS SUMMARY")
        print("="*60)
        print(f"Analysis completed in {results['execution_time']:.2f} seconds")
        
        market_stats = stats['market_data']
        exec_stats = stats['execution_data']
        
        print(f"Market data records: {market_stats['total_records']}")
        print(f"Execution records: {exec_stats['total_executions']}")
        print(f"Bid-ask spread (mean/median): {market_stats['bidask_spread_stats']['mean']:.6f} / {market_stats['bidask_spread_stats']['median']:.6f} USDT")
        print(f"Buy orders: {exec_stats['buy_orders']}, Sell orders: {exec_stats['sell_orders']}")
        
        print(f"Visualizations saved to: {images_dir}/")
        print("="*60)
        
        return results
        
    except Exception as e:
        print(f"Spread analysis failed: {str(e)}")
        return {"status": "ERROR", "error": str(e)}


def main():
    """
    Main function to run spread analysis
    """
    try:
        results = run_comprehensive_spread_analysis()
        return results
    except Exception as e:
        print(f"Spread analysis failed: {str(e)}")
        return {"status": "ERROR", "error": str(e)}


if __name__ == "__main__":
    main() 