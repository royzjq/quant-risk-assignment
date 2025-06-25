import polars as pl
import numpy as np
from pathlib import Path
import time

def load_exec_data(file_path: Path = Path('exec.csv')):
    """
    Load execution data using Polars
    """
    print(f"Loading execution data from {file_path}...")
    start_time = time.time()
    df = pl.read_csv(file_path)
    load_time = time.time() - start_time
    print(f"Loaded {len(df)} execution records in {load_time:.2f} seconds")
    return df

def load_market_data(file_path):
    """
    Load market depth data using Polars
    """
    print(f"Loading market data from {file_path}...")
    start_time = time.time()
    df = pl.read_csv(file_path)
    load_time = time.time() - start_time
    print(f"Loaded {len(df)} market data records in {load_time:.2f} seconds")
    return df

def filter_symbol_exec(exec_df, symbol):
    """
    Filter execution records containing specified symbol
    
    Args:
        exec_df: DataFrame with execution data
        symbol: Symbol to filter (e.g., 'BTCUSDT', 'ETHUSDT')
    """
    symbol_exec = exec_df.filter(pl.col('sym').str.contains(symbol))
    print(f"Found {len(symbol_exec)} {symbol} execution records")
    return symbol_exec

def find_market_data_file(market_data_dir, symbol):
    """
    Find market data file for the specified symbol
    
    Args:
        market_data_dir: Directory containing market data files
        symbol: Symbol to find (e.g., 'BTCUSDT', 'ETHUSDT')
        
    Returns:
        Path to the market data file or None if not found
    """
    data_dir = Path(market_data_dir)
    symbol_files = list(data_dir.glob(f'*{symbol}*.csv'))
    
    if not symbol_files:
        print(f"No {symbol} market data file found in data directory")
        return None
    
    symbol_file = symbol_files[0]  # Use the first file found
    print(f"Using market data file: {symbol_file}")
    return symbol_file

def convert_etm_to_microseconds(exec_df):
    """
    Convert eTm from milliseconds to microseconds
    """
    exec_df = exec_df.with_columns(
        (pl.col('eTm') * 1000).alias('eTm_microseconds')
    )
    print("Converted eTm from milliseconds to microseconds")
    return exec_df

def preprocess_exec_data(exec_df):
    """
    Preprocess execution data by handling duplicate eTm records based on specified rules.
    If eTm, ech, sym, sde, px, epx, vwap are the same:
    - sde, px, qty, epx, vwap are averaged.
    - eqty, fee, ceqty are summed.
    - rqty is assigned 0.
    """
    print("\nPreprocessing execution data...")
    
    # Identify columns to group by and columns for aggregation
    group_cols = ['eTm', 'eTm_microseconds', 'ech', 'sym', 'sde', 'px', 'epx', 'vwap']
    avg_cols = ['sde', 'px', 'qty', 'epx', 'vwap']
    sum_cols = ['eqty', 'fee', 'ceqty']
    
    # Check if all group_cols exist in the DataFrame
    available_cols = exec_df.columns
    missing_group_cols = [col for col in group_cols if col not in available_cols]
    if missing_group_cols:
        print(f"Warning: Missing group columns in exec_df: {missing_group_cols}. Skipping preprocessing for these.")
        group_cols = [col for col in group_cols if col in available_cols]

    # Check if all avg_cols and sum_cols exist in the DataFrame
    available_avg_cols = [col for col in avg_cols if col in available_cols]
    available_sum_cols = [col for col in sum_cols if col in available_cols]

    if not group_cols:
        print("No valid columns to group by. Skipping preprocessing.")
        return exec_df

    # Prepare aggregation expressions
    agg_exprs = []
    
    # Add mean expressions for avg_cols (excluding those already in group_cols)
    for col in available_avg_cols:
        if col not in group_cols:  # Only aggregate columns that are not in group_cols
            agg_exprs.append(pl.col(col).mean().alias(col))
    
    # Add sum expressions for sum_cols (excluding those already in group_cols)
    for col in available_sum_cols:
        if col not in group_cols:  # Only aggregate columns that are not in group_cols
            agg_exprs.append(pl.col(col).sum().alias(col))
    
    # Group and aggregate
    if agg_exprs:  # Only perform aggregation if there are expressions to aggregate
        aggregated_df = exec_df.group_by(group_cols).agg(agg_exprs)
    else:
        # If no aggregation needed, just get unique combinations
        aggregated_df = exec_df.select(group_cols).unique()
    
    # Assign rqty to 0 for all preprocessed records
    if 'rqty' in aggregated_df.columns:
        aggregated_df = aggregated_df.with_columns(pl.lit(0).alias('rqty'))
    
    print(f"Original exec records: {len(exec_df)}")
    print(f"Preprocessed exec records: {len(aggregated_df)}")
    print("Preprocessing complete.")
    
    return aggregated_df

def calculate_mid_price(market_df):
    """
    Calculate market mid price
    """
    market_df = market_df.with_columns(
        ((pl.col('askPrice-1') + pl.col('bidPrice-1')) / 2).alias('mid_price')
    )
    print("Calculated mid price from best bid and ask")
    return market_df

def filter_exec_by_market_timerange(exec_df, market_df):
    """
    Filter execution records to only include those within market data time range
    """
    market_min_time = market_df['timestamp'].min()
    market_max_time = market_df['timestamp'].max()
    
    print(f"Market data time range: {market_min_time} to {market_max_time}")
    print(f"Original exec records: {len(exec_df)}")
    
    # Filter exec records within market time range
    filtered_exec_df = exec_df.filter(
        (pl.col('eTm_microseconds') >= market_min_time) & 
        (pl.col('eTm_microseconds') <= market_max_time)
    )
    
    excluded_count = len(exec_df) - len(filtered_exec_df)
    print(f"Exec records within market time range: {len(filtered_exec_df)}")
    print(f"Exec records excluded (outside time range): {excluded_count}")
    
    if excluded_count > 0:
        early_records = exec_df.filter(pl.col('eTm_microseconds') < market_min_time)
        late_records = exec_df.filter(pl.col('eTm_microseconds') > market_max_time)
        print(f"  - Records too early: {len(early_records)}")
        print(f"  - Records too late: {len(late_records)}")
    
    return filtered_exec_df

def merge_market_with_exec(market_df, exec_df):
    """
    Merge market data with execution records using join_asof
    Market data as the base (left), exec data as the right
    For each execution record, keep execution data only in the LAST matching market record,
    while preserving all market records in the timeline with execution fields as null for non-last matches.
    """
    print("\nMerging market data with execution records using join_asof...")
    
    # Ensure data is sorted by time
    market_df_sorted = market_df.sort('timestamp')
    exec_df_sorted = exec_df.sort('eTm_microseconds')
    
    # Use join_asof with market data as base (forward search - find earliest exec record not earlier than market time)
    merged_df = market_df_sorted.join_asof(
        exec_df_sorted,
        left_on='timestamp',
        right_on='eTm_microseconds',
        strategy='forward'  # Find earliest exec record not earlier than market timestamp
    )
    
    # Find exec records that were not matched
    # Use anti-join to find unmatched records efficiently in Polars
    unmatched_exec_df = exec_df_sorted.join(
        merged_df.filter(pl.col('eTm').is_not_null()).select(pl.col('eTm').alias('eTm_matched')),
        left_on='eTm',
        right_on='eTm_matched',
        how='anti'
    )
    
    if len(unmatched_exec_df) > 0:
        print(f"Found {len(unmatched_exec_df)} exec records not matched")
        print("Creating new rows for unmatched exec records...")
        
        # Create new rows with only exec data by adding NaN columns for market data
        market_cols = market_df_sorted.columns
        # Add market columns as null to unmatched exec records, and set timestamp = eTm_microseconds
        unmatched_with_market_cols = unmatched_exec_df.with_columns([
            pl.col('eTm_microseconds').alias('timestamp')
        ])
        
        # Add missing market columns as null
        for col in market_cols:
            if col not in unmatched_with_market_cols.columns:
                unmatched_with_market_cols = unmatched_with_market_cols.with_columns(
                    pl.lit(None).alias(col)
                )
        
        # Select columns in the same order as merged_df
        unmatched_with_market_cols = unmatched_with_market_cols.select(merged_df.columns)
        
        # Concatenate with merged data
        merged_df = pl.concat([merged_df, unmatched_with_market_cols])
        print(f"Added {len(unmatched_with_market_cols)} new rows for unmatched exec records")
    
    # Now we need to keep only the LAST market record for each execution, but preserve all market records
    print("Processing execution data to keep only the last match per execution...")
    
    # Separate records with and without exec data
    exec_records = merged_df.filter(pl.col('eTm').is_not_null())
    market_only_records = merged_df.filter(pl.col('eTm').is_null())
    
    print(f"Records with exec data before processing: {len(exec_records)}")
    print(f"Market-only records: {len(market_only_records)}")
    
    if len(exec_records) > 0:
        # Find the last (maximum timestamp) market record for each execution
        last_exec_records = (exec_records
                           .sort(['eTm', 'timestamp'])
                           .group_by('eTm', maintain_order=True)
                           .last())
        
        print(f"Found {len(last_exec_records)} unique executions with their last matches")
        
        # Identify non-last execution records more efficiently
        # Create a unique identifier for (eTm, timestamp) in last_exec_records
        last_matches_identifier = last_exec_records.with_columns(
            (pl.col('eTm').cast(pl.Utf8) + '_' + pl.col('timestamp').cast(pl.Utf8)).alias('join_id')
        ).select('join_id')

        # Create the same identifier for exec_records
        exec_records_with_id = exec_records.with_columns(
            (pl.col('eTm').cast(pl.Utf8) + '_' + pl.col('timestamp').cast(pl.Utf8)).alias('join_id')
        )

        # Use anti-join to find non-last execution records
        non_last_exec_records = exec_records_with_id.join(
            last_matches_identifier,
            on='join_id',
            how='anti'
        ).drop('join_id') # Drop the temporary join_id column
        
        print(f"Converting {len(non_last_exec_records)} non-last execution matches to market-only records")
        
        # Set execution columns to null for non-last records
        if len(non_last_exec_records) > 0:
            # Identify execution columns
            # Get a list of columns from exec_df_sorted that are not in market_df_sorted (pure exec cols)
            # We need to make sure 'eTm_microseconds' is considered, even if it's derived.
            # Also, 'timestamp' from market data should not be nulled out.
            # Let's rebuild the exec_cols list more robustly
            exec_base_cols = set(exec_df_sorted.columns) - set(market_df_sorted.columns)
            # Ensure eTm and eTm_microseconds are always in exec_base_cols if they exist
            if 'eTm' in exec_df_sorted.columns: exec_base_cols.add('eTm')
            if 'eTm_microseconds' in exec_df_sorted.columns: exec_base_cols.add('eTm_microseconds')

            # Filter exec_cols to only include those present in non_last_exec_records
            exec_cols_to_null = [col for col in exec_base_cols if col in non_last_exec_records.columns]

            # Create expressions to set execution columns to null
            null_exprs = [pl.lit(None).alias(col) for col in exec_cols_to_null]

            # Apply the transformations
            non_last_market_only = non_last_exec_records.with_columns(null_exprs)
            
            # Ensure column order matches the expected schema
            target_columns = merged_df.columns
            non_last_market_only = non_last_market_only.select(target_columns)
            market_only_records = market_only_records.select(target_columns)
            last_exec_records = last_exec_records.select(target_columns)
            
            # Combine all market-only records
            all_market_only = pl.concat([market_only_records, non_last_market_only])
        else:
            all_market_only = market_only_records.select(merged_df.columns)
            last_exec_records = last_exec_records.select(merged_df.columns)
        
        # Combine last execution records with all market-only records
        merged_df = pl.concat([last_exec_records, all_market_only])
        
        print(f"Final result: {len(last_exec_records)} execution records + {len(all_market_only)} market-only records")
    else:
        merged_df = market_only_records
    
    # Sort by timestamp
    merged_df = merged_df.sort('timestamp')
    
    # Forward fill market columns for NaN values
    print("Forward filling market columns for NaN values...")
    
    # Identify market columns (columns that exist in market_df but not in exec_df)
    market_cols = set(market_df_sorted.columns)
    exec_cols = set(exec_df_sorted.columns)
    market_only_cols = market_cols - exec_cols
    
    # Forward fill these market columns
    if market_only_cols:
        fill_exprs = []
        for col in market_only_cols:
            if col in merged_df.columns:
                fill_exprs.append(pl.col(col).forward_fill().alias(col))
        
        if fill_exprs:
            # Apply forward fill to market columns while keeping other columns unchanged
            other_cols = [pl.col(col) for col in merged_df.columns if col not in market_only_cols]
            merged_df = merged_df.with_columns(fill_exprs + other_cols)
            print(f"Forward filled {len(fill_exprs)} market columns")
    
    print(f"Successfully merged data. Result contains {len(merged_df)} records")
    return merged_df

def analyze_exec_coverage(merged_df, original_exec_df):
    """
    Analyze if all execution records can find corresponding market data points
    """
    print("\n=== Execution Coverage Analysis ===")
    
    # Count non-null exec records in merged data
    exec_matched = merged_df.filter(pl.col('eTm').is_not_null())
    
    print(f"Total merged records: {len(merged_df)}")
    print(f"Records with exec data: {len(exec_matched)}")
    print(f"Records without exec data (market only): {len(merged_df) - len(exec_matched)}")
    
    # Check if all original exec records are covered
    original_exec_count = len(original_exec_df)
    unique_exec_in_merged = merged_df.filter(pl.col('eTm').is_not_null())['eTm'].n_unique()
    
    print(f"\nOriginal BTCUSDT exec records: {original_exec_count}")
    print(f"Unique exec records found in merged data: {unique_exec_in_merged}")
    
    # Find missing exec records - use eTm_microseconds for precise comparison
    merged_exec_micro_times = set(merged_df.filter(pl.col('eTm').is_not_null())['eTm_microseconds'].to_list())
    original_exec_micro_times = set(original_exec_df['eTm_microseconds'].to_list())
    missing_exec_micro_times = original_exec_micro_times - merged_exec_micro_times
    
    print(f"DEBUG: Original unique eTm_microseconds count: {len(original_exec_micro_times)}")
    print(f"DEBUG: Merged unique eTm_microseconds count: {len(merged_exec_micro_times)}")
    print(f"DEBUG: Missing unique eTm_microseconds count: {len(missing_exec_micro_times)}")
    
    # The primary check for coverage should be based on missing_exec_micro_times
    if len(missing_exec_micro_times) == 0:
        print("✓ All execution records found corresponding data points")
        coverage_status = "COMPLETE"
    else:
        missing_count = len(missing_exec_micro_times)
        print(f"✗ {missing_count} execution records could NOT find corresponding market data")
        coverage_status = "INCOMPLETE"
    
    # Time range analysis
    if len(exec_matched) > 0:
        market_time_range = (merged_df['timestamp'].min(), merged_df['timestamp'].max())
        exec_time_range = (exec_matched['eTm_microseconds'].min(), exec_matched['eTm_microseconds'].max())
        
        print(f"\nTime Range Analysis:")
        print(f"Merged data time range: {market_time_range[0]} to {market_time_range[1]}")
        print(f"Exec data time range: {exec_time_range[0]} to {exec_time_range[1]}")
        
        # Check if exec time range is within market time range
        if (exec_time_range[0] >= market_time_range[0] and 
            exec_time_range[1] <= market_time_range[1]):
            print("✓ All execution times are within merged data time range")
        else:
            print("✗ Some execution times are outside merged data time range")
    
    return coverage_status, exec_matched

def prepare_symbol_data(exec_file='exec.csv', market_data_dir='data', symbol='BTCUSDT'):
    """
    Complete data preparation pipeline for specified symbol data
    
    Args:
        exec_file: Path to execution data CSV file
        market_data_dir: Directory containing market data files
        symbol: Symbol to process (e.g., 'BTCUSDT', 'ETHUSDT')
    
    Returns:
        merged_df: Polars DataFrame with merged market and execution data
        coverage_status: String indicating coverage completeness
    """
    total_start_time = time.time()
    
    try:
        # Define paths for JSON and CSV
        exec_json_file_path = Path(market_data_dir) / 'exec.json'
        exec_csv_file_path = Path(market_data_dir) / exec_file

        # Convert exec.json to exec.csv if exec.csv does not exist
        convert_json_to_csv_if_not_exists(exec_json_file_path, exec_csv_file_path)

        # 1. Load data
        exec_df = load_exec_data(exec_csv_file_path)
        
        # Find symbol market data file
        market_file = find_market_data_file(market_data_dir, symbol)
        if market_file is None:
            raise FileNotFoundError(f"No {symbol} market data file found in data directory")
        
        market_df = load_market_data(market_file)
        
        # 2. Filter symbol execution records
        symbol_exec_df = filter_symbol_exec(exec_df, symbol)
        
        if len(symbol_exec_df) == 0:
            raise ValueError(f"No {symbol} execution records found")
        
        # 3. Convert timestamps
        symbol_exec_df = convert_etm_to_microseconds(symbol_exec_df)
        
        # 4. Preprocess execution data
        symbol_exec_df = preprocess_exec_data(symbol_exec_df)
        
        # 5. Calculate market mid price
        market_df = calculate_mid_price(market_df)
        
        # 6. Filter exec records by market time range
        filtered_exec_df = filter_exec_by_market_timerange(symbol_exec_df, market_df)
        
        if len(filtered_exec_df) == 0:
            raise ValueError(f"No {symbol} execution records remain after time range filtering")
        
        # 7. Merge data (market as base)
        merge_start_time = time.time()
        merged_df = merge_market_with_exec(market_df, filtered_exec_df)
        merge_time = time.time() - merge_start_time
        print(f"Merge operation completed in {merge_time:.2f} seconds")
        
        # 8. Analyze coverage
        coverage_status, exec_matched = analyze_exec_coverage(merged_df, filtered_exec_df)
        
        total_time = time.time() - total_start_time
        print(f"\nData preparation for {symbol} completed in {total_time:.2f} seconds")
        
        return merged_df, coverage_status
        
    except Exception as e:
        print(f"Error during {symbol} data preparation: {str(e)}")
        return None, "ERROR"

def prepare_btcusdt_data(exec_file='exec.csv', market_data_dir='data'):
    """
    Complete data preparation pipeline for BTCUSDT data (backward compatibility)
    
    Returns:
        merged_df: Polars DataFrame with merged market and execution data
        coverage_status: String indicating coverage completeness
    """
    return prepare_symbol_data(exec_file, market_data_dir, 'BTCUSDT')

def save_to_parquet(df, output_path):
    """
    Save DataFrame to Parquet format for faster loading
    """
    print(f"Saving data to {output_path}...")
    start_time = time.time()
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    df.write_parquet(output_path)
    save_time = time.time() - start_time
    print(f"Data saved to {output_path} in {save_time:.2f} seconds")

def convert_json_to_csv_if_not_exists(json_file_path: Path, csv_file_path: Path):
    """
    Converts a CSV-like JSON file to CSV format if the CSV file does not already exist.
    This function is specifically designed for cases where the 'JSON' file is actually
    a CSV file with a .json extension or a malformed JSON that resembles CSV.

    Args:
        json_file_path: Path to the input CSV-like JSON file.
        csv_file_path: Path to the output CSV file.
    """
    print(f"Checking for existing CSV file at: {csv_file_path}")
    if csv_file_path.exists():
        print(f"CSV file already exists at {csv_file_path}. Skipping conversion.")
        return

    print(f"CSV file not found. Attempting to convert {json_file_path} (treating as CSV) to {csv_file_path}...")
    start_time = time.time()
    try:
        print(f"Reading CSV-like data from {json_file_path}...")
        # Load CSV data using Polars, assuming it's a CSV with a .json extension
        df = pl.read_csv(json_file_path)
        print(f"Successfully read {len(df)} records from {json_file_path}. Ensuring output directory exists...")
        
        # Ensure output directory exists for the CSV file
        csv_file_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving DataFrame to CSV at {csv_file_path}...")
        
        # Save as CSV
        df.write_csv(csv_file_path)
        
        conversion_time = time.time() - start_time
        print(f"Successfully converted {json_file_path} to {csv_file_path} in {conversion_time:.2f} seconds.")
    except Exception as e:
        print(f"Error converting {json_file_path} to CSV: {e}")
        print("Conversion failed. Please ensure the JSON file is correctly formatted or is a CSV-like file.") 