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
    
    fig.show()
    
    return output_file_png # Return one path for consistency


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


def create_spread_distribution_chart(market_data, execution_data, output_dir='results/images', n_bins=100):
    """
    Create bid-ask spread distribution chart with overlaid execution distribution
    
    Args:
        market_data: Polars DataFrame with market data and spreads
        execution_data: Polars DataFrame with execution records
        output_dir: Directory to save the plot
        n_bins: Number of bins for the histogram
    """
    print("Creating bid-ask spread distribution chart...")
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create subplots with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Calculate bid-ask spread (positive values)
    # We need to go back to the original data to calculate the actual bid-ask spread
    # Since market_data only has bid_spread and ask_spread relative to mid price
    # We need to recalculate from the original processed data
    
    # Load the original processed data again to get bidPrice-1 and askPrice-1
    df = load_processed_data()
    
    # Calculate bid-ask spread (positive values)
    df_with_bidask_spread = df.with_columns([
        (pl.col('askPrice-1') - pl.col('bidPrice-1')).alias('bidask_spread')
    ])
    
    # Filter out any invalid spreads
    df_with_bidask_spread = df_with_bidask_spread.filter(
        pl.col('bidask_spread') > 0
    )
    
    # Convert to numpy for histogram
    bidask_spread_data = df_with_bidask_spread['bidask_spread'].to_numpy()
    
    # Determine bin range: custom bins for 0-0.1, and one 'others' bin
    # Define fine bins from 0 to 0.1 (e.g., 20 bins of 0.005 width)
    fine_bin_upper_bound = 0.1
    num_fine_bins = 20 # 20 bins between 0 and 0.1 (step of 0.005)
    fine_bin_edges = np.linspace(0, fine_bin_upper_bound, num_fine_bins + 1)

    # Get the maximum spread from the full dataset
    max_actual_spread = np.max(bidask_spread_data)

    # Create the 'others' bin edge
    if max_actual_spread > fine_bin_upper_bound:
        # Concatenate fine_bin_edges with max_actual_spread to form the last bin
        bin_edges = np.concatenate((fine_bin_edges, [max_actual_spread]))
    else:
        # If all data is within the fine range, just use fine_bin_edges
        bin_edges = fine_bin_edges
    
    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate histogram for bid-ask spread
    spread_counts, _ = np.histogram(bidask_spread_data, bins=bin_edges)
    
    # Add bid-ask spread distribution
    fig.add_trace(
        go.Bar(x=bin_centers, y=spread_counts, 
               name='Bid-Ask Spread Distribution',
               marker_color='blue', opacity=0.7,
               width=[(bin_edges[i+1] - bin_edges[i]) * 0.9 for i in range(len(bin_centers))]
              ),
        secondary_y=False
    )
    
    # Calculate execution distribution if executions exist
    if len(execution_data) > 0:
        # Filter the combined data for execution records
        exec_records = df_with_bidask_spread.filter(pl.col('eTm').is_not_null())
        
        if len(exec_records) > 0:
            exec_spread_data = exec_records['bidask_spread'].to_numpy()
            exec_counts, _ = np.histogram(exec_spread_data, bins=bin_edges)
            
            # Filter out bins with zero execution counts for plotting
            non_zero_exec_indices = exec_counts > 0
            filtered_bin_centers = bin_centers[non_zero_exec_indices]
            filtered_exec_counts = exec_counts[non_zero_exec_indices]
            
            if len(filtered_exec_counts) > 0:
                # Add execution distribution on secondary y-axis
                fig.add_trace(
                    go.Scatter(x=filtered_bin_centers, y=filtered_exec_counts,
                              mode='lines+markers',
                              name='Execution Count Distribution',
                              line=dict(color='red', width=3),
                              marker=dict(color='red', size=6)),
                    secondary_y=True
                )
    
    # Prepare custom tick text for x-axis
    tick_vals = bin_edges
    tick_texts = [f'{val:.3f}' for val in bin_edges]
    if max_actual_spread > fine_bin_upper_bound and len(bin_edges) > 1:
        tick_texts[-1] = f'>{fine_bin_upper_bound:.1f} (Others)' # Label for the last bin

    # Update layout
    fig.update_layout(
        title_text='BTCUSDT Bid-Ask Spread Distribution with Execution Distribution (Custom Fine Bins)',
        xaxis_title_text='Bid-Ask Spread (USDT)',
        barmode='overlay',
        legend=dict(x=0.7, y=0.99, bgcolor='rgba(255,255,255,0.8)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1),
        showlegend=True,
        xaxis=dict(type='linear', range=[0, bin_edges[-1]], 
                   tickmode='array', 
                   tickvals=tick_vals,
                   ticktext=tick_texts
                  )
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Market Data Frequency", secondary_y=False)
    fig.update_yaxes(title_text="Execution Count", secondary_y=True)
    
    # Save the plot
    output_file_png = Path(output_dir) / 'btcusdt_bidask_spread_distribution_custom_fine_bins.png'
    output_file_html = Path(output_dir) / 'btcusdt_bidask_spread_distribution_custom_fine_bins.html'

    fig.write_image(str(output_file_png), scale=2)
    fig.write_html(str(output_file_html))
    
    print(f"Bid-Ask spread distribution chart (custom fine bins) saved to {output_file_png}")
    print(f"Interactive HTML saved to {output_file_html}")
    
    fig.show()
    
    return output_file_png


def generate_distribution_summary_stats(market_data, execution_data):
    """
    Generate summary statistics for bid-ask spread distribution analysis
    """
    print("\n" + "="*60)
    print("BID-ASK SPREAD DISTRIBUTION ANALYSIS SUMMARY")
    print("="*60)
    
    # Load original data to calculate bid-ask spread
    df = load_processed_data()
    df_with_bidask_spread = df.with_columns([
        (pl.col('askPrice-1') - pl.col('bidPrice-1')).alias('bidask_spread')
    ]).filter(pl.col('bidask_spread') > 0)
    
    # Basic bid-ask spread statistics
    bidask_spread_stats = {
        'mean': df_with_bidask_spread['bidask_spread'].mean(),
        'std': df_with_bidask_spread['bidask_spread'].std(),
        'min': df_with_bidask_spread['bidask_spread'].min(),
        'max': df_with_bidask_spread['bidask_spread'].max(),
        'q25': df_with_bidask_spread['bidask_spread'].quantile(0.25),
        'q50': df_with_bidask_spread['bidask_spread'].quantile(0.50),
        'q75': df_with_bidask_spread['bidask_spread'].quantile(0.75),
        'q95': df_with_bidask_spread['bidask_spread'].quantile(0.95),
        'q99': df_with_bidask_spread['bidask_spread'].quantile(0.99)
    }
    
    print(f"Bid-Ask Spread Distribution (askPrice-1 - bidPrice-1):")
    print(f"  Count: {len(df_with_bidask_spread):,} records")
    print(f"  Mean:  {bidask_spread_stats['mean']:.6f} USDT")
    print(f"  Std:   {bidask_spread_stats['std']:.6f} USDT")
    print(f"  Min:   {bidask_spread_stats['min']:.6f} USDT")
    print(f"  25%:   {bidask_spread_stats['q25']:.6f} USDT")
    print(f"  50%:   {bidask_spread_stats['q50']:.6f} USDT")
    print(f"  75%:   {bidask_spread_stats['q75']:.6f} USDT")
    print(f"  95%:   {bidask_spread_stats['q95']:.6f} USDT")
    print(f"  99%:   {bidask_spread_stats['q99']:.6f} USDT")
    print(f"  Max:   {bidask_spread_stats['max']:.6f} USDT")
    
    # Execution analysis if available
    if len(execution_data) > 0:
        # Get execution records with their spreads
        exec_records = df_with_bidask_spread.filter(pl.col('eTm').is_not_null())
        
        if len(exec_records) > 0:
            exec_spread_stats = {
                'mean': exec_records['bidask_spread'].mean(),
                'std': exec_records['bidask_spread'].std(),
                'min': exec_records['bidask_spread'].min(),
                'max': exec_records['bidask_spread'].max(),
                'q50': exec_records['bidask_spread'].quantile(0.50)
            }
            
            print(f"\nExecution Analysis:")
            print(f"  Total executions: {len(exec_records):,}")
            print(f"  Avg execution spread: {exec_spread_stats['mean']:.6f} USDT")
            print(f"  Median execution spread: {exec_spread_stats['q50']:.6f} USDT")
            print(f"  Min execution spread: {exec_spread_stats['min']:.6f} USDT")
            print(f"  Max execution spread: {exec_spread_stats['max']:.6f} USDT")
            
            # Count executions by side
            if 'sde' in exec_records.columns:
                side_counts = exec_records.group_by('sde').len().sort('sde')
                for row in side_counts.rows():
                    side_name = "Buy" if row[0] == 1 else "Sell" if row[0] == 2 else f"Side {row[0]}"
                    print(f"  {side_name} executions: {row[1]:,}")
            
            # Compare execution spread vs market spread
            overall_median = bidask_spread_stats['q50']
            exec_median = exec_spread_stats['q50']
            spread_premium = ((exec_median - overall_median) / overall_median) * 100
            print(f"  Execution spread premium vs market median: {spread_premium:.2f}%")
    
    print("="*60)


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
        
        # Create spread distribution chart
        create_spread_distribution_chart(market_data, execution_data, n_bins=100)
        
        # Generate distribution summary statistics
        generate_distribution_summary_stats(market_data, execution_data)
        
        print(f"\nMarket data visualization completed successfully!")
        print(f"Output saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main() 