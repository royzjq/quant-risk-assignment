#!/usr/bin/env python3
"""
Spread Visualization Script

This script visualizes bid spread and ask spread over time, with execution points marked.
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


def create_spread_visualization(market_data, execution_data, output_dir='results/images', downsample_interval=None):
    """
    Create and save the spread visualization using Plotly
    
    Args:
        market_data: Polars DataFrame with market data and spreads
        execution_data: Polars DataFrame with execution records
        output_dir: Directory to save the plot
        downsample_interval: If not None, take every Nth market data point for plotting to speed up.
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
    
    # No need for mid price line in spread visualization
    # fig.add_trace(go.Scatter(x=market_data_sampled['datetime'], y=market_data_sampled['mid_price'],
    #                          mode='lines', name='Mid Price', line=dict(color='purple', dash='dot', width=1)))

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
        title_text='BTCUSDT Bid/Ask Spreads Over Time with Executions',
        xaxis_title_text='Time',
        yaxis_title_text='Spread (USDT) relative to Mid Price',
        hovermode='x unified', # Show hover info for all traces at a given x-coordinate
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.5)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1),
        xaxis_rangeslider_visible=True, # Add a rangeslider for easy navigation
        yaxis_zeroline=True, yaxis_zerolinecolor='lightgray', yaxis_zerolinewidth=1 # Show zero line for clarity
    )

    # Save the plot
    output_file_png = Path(output_dir) / 'btcusdt_spread_execution_visualization.png'
    output_file_html = Path(output_dir) / 'btcusdt_spread_execution_visualization.html' # Plotly can save interactive HTML

    fig.write_image(str(output_file_png), scale=2) # scale for higher resolution
    fig.write_html(str(output_file_html))
    
    print(f"Visualization saved to {output_file_png}")
    print(f"Interactive HTML saved to {output_file_html}")
    
    return output_file_png


def create_bidask_spread_distribution(market_data, execution_data, output_dir='results/images'):
    """
    Create bid-ask spread distribution chart with execution count overlay
    
    Args:
        market_data: Polars DataFrame with market data and spreads
        execution_data: Polars DataFrame with execution records
        output_dir: Directory to save the plot
    """
    print("Creating bid-ask spread distribution chart...")
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load original data to calculate bid-ask spread
    df = load_processed_data()
    
    # Calculate bid-ask spread (positive values)
    df_with_bidask_spread = df.with_columns([
        (pl.col('askPrice-1') - pl.col('bidPrice-1')).alias('bidask_spread')
    ]).filter(pl.col('bidask_spread') > 0)
    
    # Convert to numpy for histogram
    bidask_spread_data = df_with_bidask_spread['bidask_spread'].to_numpy()
    
    # Create custom bins: 30 bins from 0 to 0.3, plus one bin for >0.3
    num_fine_bins = 30
    fine_bin_upper_bound = 0.3
    fine_bin_edges = np.linspace(0, fine_bin_upper_bound, num_fine_bins + 1)
    
    # Get the maximum spread from the full dataset
    max_actual_spread = np.max(bidask_spread_data)
    
    # Create the complete bin edges for data analysis
    if max_actual_spread > fine_bin_upper_bound:
        # Add the final bin edge for values > 0.3
        bin_edges = np.concatenate((fine_bin_edges, [max_actual_spread]))
    else:
        # If all data is within the fine range, just use fine_bin_edges
        bin_edges = fine_bin_edges
    
    # Calculate bin centers for plotting - the last bin center is set to 0.3
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    if max_actual_spread > fine_bin_upper_bound:
        # Set the last bin center to 0.3 for display purposes
        bin_centers[-1] = fine_bin_upper_bound
    
    # Calculate histogram for bid-ask spread (all market data)
    spread_counts, _ = np.histogram(bidask_spread_data, bins=bin_edges)
    
    # Calculate execution distribution
    exec_records = df_with_bidask_spread.filter(pl.col('eTm').is_not_null())
    exec_counts = np.zeros(len(bin_centers))  # Initialize with zeros
    
    if len(exec_records) > 0:
        exec_spread_data = exec_records['bidask_spread'].to_numpy()
        exec_counts, _ = np.histogram(exec_spread_data, bins=bin_edges)
    
    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Calculate bin widths for bar chart - all bins have the same width
    regular_bin_width = (fine_bin_upper_bound / num_fine_bins) * 0.9  # Width of regular bins
    bin_widths = [regular_bin_width] * len(bin_centers)  # All bins have the same width
    
    # Add bid-ask spread distribution (bar chart)
    fig.add_trace(
        go.Bar(x=bin_centers, y=spread_counts,
               name='Market Data Frequency',
               marker_color='lightblue', opacity=0.7,
               width=bin_widths),
        secondary_y=False
    )
    
    # Add execution count distribution (line chart)
    fig.add_trace(
        go.Scatter(x=bin_centers, y=exec_counts,
                   mode='lines+markers',
                   name='Execution Count',
                   line=dict(color='red', width=3),
                   marker=dict(color='red', size=6)),
        secondary_y=True
    )
    
    # Prepare custom tick labels
    tick_vals = bin_centers.tolist()
    tick_texts = [f'{val:.3f}' for val in bin_centers]
    
    # Special label for the last bin if it represents >0.3
    if max_actual_spread > fine_bin_upper_bound and len(bin_centers) > num_fine_bins:
        tick_texts[-1] = f'>0.3'
    
    # Update layout
    fig.update_layout(
        title_text='BTCUSDT Bid-Ask Spread Distribution with Execution Count',
        xaxis_title_text='Bid-Ask Spread (USDT)',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1),
        showlegend=True,
        xaxis=dict(
            type='linear', 
            range=[0, bin_centers[-1] + (bin_centers[-1] - bin_centers[-2])/2],
            tickmode='array',
            tickvals=tick_vals[::2],  # Show every 2nd tick to avoid crowding
            ticktext=[tick_texts[i] for i in range(0, len(tick_texts), 2)]
        )
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Market Data Frequency", secondary_y=False)
    fig.update_yaxes(title_text="Execution Count", secondary_y=True)
    
    # Save the plot
    output_file_png = Path(output_dir) / 'btcusdt_bidask_spread_execution_distribution.png'
    output_file_html = Path(output_dir) / 'btcusdt_bidask_spread_execution_distribution.html'
    
    fig.write_image(str(output_file_png), scale=2)
    fig.write_html(str(output_file_html))
    
    print(f"Bid-Ask spread distribution chart saved to {output_file_png}")
    print(f"Interactive HTML saved to {output_file_html}")
    
    return output_file_png


def export_large_spread_trades(threshold=0.2, output_dir='results'):
    """
    Export trades with bid-ask spread larger than threshold to JSON file
    
    Args:
        threshold: Minimum bid-ask spread threshold (default: 0.2)
        output_dir: Directory to save the JSON file
        
    Returns:
        str: Path to the exported JSON file
    """
    print(f"Exporting trades with bid-ask spread > {threshold} USDT...")
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load original data
    df = load_processed_data()
    
    # Calculate bid-ask spread and filter for executions only
    df_with_trades = df.filter(pl.col('eTm').is_not_null()).with_columns([
        (pl.col('askPrice-1') - pl.col('bidPrice-1')).alias('bidask_spread')
    ])
    
    # Filter trades with large spread
    large_spread_trades = df_with_trades.filter(
        pl.col('bidask_spread') > threshold
    ).select([
        'eTm',
        'eTm_microseconds', 
        'timestamp',
        'px',
        'qty', 
        'sde',
        'bidPrice-1',
        'askPrice-1',
        'bidask_spread'
    ]).with_columns([
        # Convert timestamps to readable format
        (pl.from_epoch(pl.col('eTm_microseconds'), time_unit='us')).alias('execution_datetime'),
        (pl.from_epoch(pl.col('timestamp'), time_unit='us')).alias('market_data_datetime')
    ])
    
    # Convert to Python dict for JSON export
    trades_data = {
        'metadata': {
            'threshold_usdt': threshold,
            'total_large_spread_trades': len(large_spread_trades),
            'description': f'Trades with bid-ask spread > {threshold} USDT'
        },
        'trades': []
    }
    
    # Add trade records
    for row in large_spread_trades.rows(named=True):
        trade_record = {
            'execution_time': row['eTm'],
            'execution_datetime': str(row['execution_datetime']),
            'market_data_datetime': str(row['market_data_datetime']),
            'execution_price': row['px'],
            'quantity': row['qty'],
            'side': row['sde'],  # 1=Buy, 2=Sell
            'bid_price': row['bidPrice-1'],
            'ask_price': row['askPrice-1'],
            'bidask_spread_usdt': row['bidask_spread']
        }
        trades_data['trades'].append(trade_record)
    
    # Sort trades by execution time
    trades_data['trades'].sort(key=lambda x: x['execution_time'])
    
    # Save to JSON file
    output_file = Path(output_dir) / 'trade-with-large-spread.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(trades_data, f, indent=2, ensure_ascii=False)
    
    print(f"Exported {len(large_spread_trades)} large spread trades to {output_file}")
    
    # Print summary statistics
    if len(large_spread_trades) > 0:
        spread_stats = large_spread_trades['bidask_spread']
        print(f"Large spread trades summary:")
        print(f"  Min spread: {spread_stats.min():.6f} USDT")
        print(f"  Max spread: {spread_stats.max():.6f} USDT") 
        print(f"  Mean spread: {spread_stats.mean():.6f} USDT")
        print(f"  Median spread: {spread_stats.median():.6f} USDT")
        
        # Count by side
        side_counts = large_spread_trades.group_by('sde').len().sort('sde')
        for row in side_counts.rows():
            side_name = "Buy" if row[0] == 1 else "Sell" if row[0] == 2 else f"Side {row[0]}"
            print(f"  {side_name} trades: {row[1]}")
    
    return str(output_file)


def generate_summary_stats(market_data, execution_data):
    """
    Generate and print summary statistics using Polars
    
    Args:
        market_data: Polars DataFrame with market data and spreads
        execution_data: Polars DataFrame with execution records
    """
    print("\n" + "="*50)
    print("SPREAD VISUALIZATION SUMMARY")
    print("="*50)
    
    # Market data stats
    print(f"Time range: {market_data['datetime'].min()} to {market_data['datetime'].max()}")
    print(f"Total market data points: {len(market_data)}")
    
    # Spread statistics
    print(f"\nBid Spread Statistics:")
    print(f"  Mean: {market_data['bid_spread'].mean():.6f} USDT")
    print(f"  Std:  {market_data['bid_spread'].std():.6f} USDT")
    print(f"  Min:  {market_data['bid_spread'].min():.6f} USDT")
    print(f"  Max:  {market_data['bid_spread'].max():.6f} USDT")
    
    print(f"\nAsk Spread Statistics:")
    print(f"  Mean: {market_data['ask_spread'].mean():.6f} USDT")
    print(f"  Std:  {market_data['ask_spread'].std():.6f} USDT")
    print(f"  Min:  {market_data['ask_spread'].min():.6f} USDT")
    print(f"  Max:  {market_data['ask_spread'].max():.6f} USDT")
    
    # Execution stats
    if len(execution_data) > 0:
        print(f"\nExecution Statistics:")
        print(f"  Total executions: {len(execution_data)}")
        
        # Group by side if available
        if 'sde' in execution_data.columns:
            side_counts = execution_data.group_by('sde').len().sort('sde') # Use len() instead of count() for row count
            for row in side_counts.rows():
                print(f"  {row[0]} executions: {row[1]}") # row[0] is sde, row[1] is count
        
        # Execution time range
        if 'exec_datetime' in execution_data.columns:
            print(f"  First execution: {execution_data['exec_datetime'].min()}")
            print(f"  Last execution:  {execution_data['exec_datetime'].max()}")
    else:
        print(f"\nNo execution records found in the dataset")
    
    print("="*50)


def main():
    """
    Main function to run the spread visualization
    """
    try:
        # Load processed data
        df = load_processed_data()
        
        # Calculate spreads (now also calculates full_spread)
        df_with_spreads = calculate_spreads(df)
        
        # Prepare data for plotting (returns Polars DataFrames)
        market_data, execution_data = prepare_plot_data(df_with_spreads)
        
        # Generate summary statistics
        generate_summary_stats(market_data, execution_data)
        
        # Create visualization
        output_file = create_spread_visualization(market_data, execution_data, downsample_interval=100)
        
        # Create bid-ask spread distribution chart
        create_bidask_spread_distribution(market_data, execution_data)
        
        # Export large spread trades
        export_large_spread_trades()
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main() 