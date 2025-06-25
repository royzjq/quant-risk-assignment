import polars as pl
import numpy as np
from pathlib import Path
import time
import json

def load_exec_data(file_path='exec.csv'):
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

def filter_btcusdt_exec(exec_df):
    """
    Filter execution records containing BTCUSDT
    """
    btc_exec = exec_df.filter(pl.col('sym').str.contains('BTCUSDT'))
    print(f"Found {len(btc_exec)} BTCUSDT execution records")
    return btc_exec

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
    
    # For columns that are in both group_cols and aggregation lists, we keep the group value
    # (no need to aggregate them since they're the same within each group)
    
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
    For duplicate exec records that can't be matched, create new rows with only exec data
    Keep only the closest (last) market record for each exec record
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
    matched_exec_times = set(merged_df.filter(pl.col('eTm').is_not_null())['eTm'].to_list())
    all_exec_times = set(exec_df_sorted['eTm'].to_list())
    missing_exec_times = all_exec_times - matched_exec_times
    
    if len(missing_exec_times) > 0:
        print(f"Found {len(missing_exec_times)} exec records not matched")
        print("Creating new rows for unmatched exec records...")
        
        # Get unmatched exec records
        unmatched_exec = exec_df_sorted.filter(pl.col('eTm').is_in(list(missing_exec_times)))
        
        # Create new rows with only exec data by adding NaN columns for market data
        market_cols = market_df_sorted.columns
        exec_cols = exec_df_sorted.columns
        
        # Add market columns as null to unmatched exec records, and set timestamp = eTm_microseconds
        unmatched_with_market_cols = unmatched_exec.with_columns([
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
    
    # Keep only the last (closest) market record for each exec record
    print("Filtering to keep only the closest market record for each exec...")
    
    # Separate records with and without exec data
    exec_records = merged_df.filter(pl.col('eTm').is_not_null())
    market_only_records = merged_df.filter(pl.col('eTm').is_null())
    
    print(f"Records with exec data before filtering: {len(exec_records)}")
    print(f"Market-only records: {len(market_only_records)}")
    
    if len(exec_records) > 0:
        # For exec records, keep only the one with maximum timestamp for each eTm
        # Use a simpler approach to avoid schema issues
        closest_exec_records = (exec_records
                              .with_row_index()
                              .sort(['eTm', 'timestamp'])
                              .group_by('eTm', maintain_order=True)
                              .last()
                              .drop('index'))
        
        print(f"Reduced exec-matched records from {len(exec_records)} to {len(closest_exec_records)}")
        
        # Ensure column order consistency before concatenating
        target_columns = merged_df.columns  # Use original merged_df columns
        closest_exec_records = closest_exec_records.select(target_columns)
        market_only_records = market_only_records.select(target_columns)
        
        # Combine closest exec records with ALL market-only records
        merged_df = pl.concat([closest_exec_records, market_only_records])
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

def debug_missing_exec_records(missing_exec_times, original_exec_df):
    """
    Debug and print details of execution records that are missing after merging.
    """
    print("\n=== Debugging Missing Execution Records ===")
    
    if len(missing_exec_times) > 0:
        print(f"Found {len(missing_exec_times)} unique eTm_microseconds values missing from merged data:")
        
        # Get the missing records details from the original (preprocessed) exec_df
        missing_records = original_exec_df.filter(pl.col('eTm_microseconds').is_in(list(missing_exec_times)))
        missing_records = missing_records.sort('eTm_microseconds')
        
        print("\nMissing execution records details:")
        # Include Unnamed: 0 column if it exists
        columns_to_show = ['Unnamed: 0', 'eTm', 'eTm_microseconds', 'px', 'qty', 'sde', 'rqty', 'eqty', 'fee', 'ceqty']
        available_columns = [col for col in columns_to_show if col in missing_records.columns]
        print(missing_records.select(available_columns))
        
        print(f"\nTotal missing records: {len(missing_records)}")
    else:
        print("No unique eTm_microseconds values are missing from merged data. Check other potential issues if records count still differs.")

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
        
        # Always call debug function to see what's happening
        debug_missing_exec_records(missing_exec_micro_times, original_exec_df)
    
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

def validate_exec_market_data(merged_df):
    """
    Validate execution and market data relationships based on specified rules.
    sde = 1 (Buy), sde = 2 (Sell).
    """
    print("\n=== Data Validation ===")
    
    # Ensure relevant columns exist
    required_cols = ['eqty', 'vwap', 'sde', 'bidVol-1', 'askVol-1', 'bidPrice-1', 'askPrice-1']
    missing_cols = [col for col in required_cols if col not in merged_df.columns]
    if missing_cols:
        print(f"Warning: Missing required columns for validation: {missing_cols}. Skipping validation.")
        return

    # Filter for records with actual exec data
    exec_records = merged_df.filter(pl.col('eTm').is_not_null())
    
    if len(exec_records) == 0:
        print("No execution records to validate.")
        return

    # --- eqty validation ---
    # sde = 2 (sell): eqty <= bidVol-1
    eqty_sell_violations = exec_records.filter(
        (pl.col('sde') == 2) & (pl.col('eqty') > pl.col('bidVol-1'))
    )
    print(f"eqty violations (sde=2, eqty > bidVol-1): {len(eqty_sell_violations)}")

    # sde = 1 (buy): eqty <= askVol-1
    eqty_buy_violations = exec_records.filter(
        (pl.col('sde') == 1) & (pl.col('eqty') > pl.col('askVol-1'))
    )
    print(f"eqty violations (sde=1, eqty > askVol-1): {len(eqty_buy_violations)}")

    # --- vwap validation ---
    # sde = 2 (sell): vwap >= bidPrice-1
    vwap_sell_violations = exec_records.filter(
        (pl.col('sde') == 2) & (pl.col('vwap') < pl.col('bidPrice-1'))
    )
    print(f"vwap violations (sde=2, vwap < bidPrice-1): {len(vwap_sell_violations)}")

    # sde = 1 (buy): vwap <= askPrice-1
    vwap_buy_violations = exec_records.filter(
        (pl.col('sde') == 1) & (pl.col('vwap') > pl.col('askPrice-1'))
    )
    print(f"vwap violations (sde=1, vwap > askPrice-1): {len(vwap_buy_violations)}")

    print("Data Validation Complete.")
    return {
        "eqty_sell_violations": eqty_sell_violations,
        "eqty_buy_violations": eqty_buy_violations,
        "vwap_sell_violations": vwap_sell_violations,
        "vwap_buy_violations": vwap_buy_violations
    }

def main():
    """
    Main function: analyze exec-market data correspondence using Polars
    """
    total_start_time = time.time()
    
    try:
        # 1. Load data
        exec_df = load_exec_data('exec.csv')
        
        # Find BTCUSDT market data file
        data_dir = Path('data')
        btc_files = list(data_dir.glob('*BTCUSDT*.csv'))
        
        if not btc_files:
            raise FileNotFoundError("No BTCUSDT market data file found in data directory")
        
        btc_market_file = btc_files[0]  # Use the first file found
        print(f"Using market data file: {btc_market_file}")
        
        market_df = load_market_data(btc_market_file)
        
        # 2. Filter BTCUSDT execution records
        btc_exec_df = filter_btcusdt_exec(exec_df)
        
        if len(btc_exec_df) == 0:
            raise ValueError("No BTCUSDT execution records found")
        
        # 3. Convert timestamps
        btc_exec_df = convert_etm_to_microseconds(btc_exec_df)
        
        # 4. Preprocess execution data
        btc_exec_df = preprocess_exec_data(btc_exec_df)
        
        # 5. Calculate market mid price
        market_df = calculate_mid_price(market_df)
        
        # 6. Filter exec records by market time range
        filtered_exec_df = filter_exec_by_market_timerange(btc_exec_df, market_df)
        
        if len(filtered_exec_df) == 0:
            raise ValueError("No execution records remain after time range filtering")
        
        # 7. Merge data (market as base)
        merge_start_time = time.time()
        merged_df = merge_market_with_exec(market_df, filtered_exec_df)
        merge_time = time.time() - merge_start_time
        print(f"Merge operation completed in {merge_time:.2f} seconds")
        
        # 8. Analyze coverage
        coverage_status, exec_matched = analyze_exec_coverage(merged_df, filtered_exec_df)
        
        # 9. Validate execution and market data relationships
        validation_results = validate_exec_market_data(merged_df)
        
        # Prepare data for JSON export
        violations_for_json = {}

        if validation_results:
            print("\n--- Detailed Validation Records ---")
            for key, df_violations in validation_results.items():
                if not df_violations.is_empty():
                    print(f"\nViolations for {key}:")

                    # Define a mapping for clearer JSON keys
                    json_key_mapping = {
                        "eqty_sell_violations": "eqty violation (sde=2, eqty > bidVol-1) - potential market impact (1)",
                        "eqty_buy_violations": "eqty violation (sde=1, eqty > askVol-1) - potential market impact (1)",
                        "vwap_sell_violations": "vwap violation (sde=2, vwap < bidPrice-1) - potential market impact (1)",
                        "vwap_buy_violations": "vwap violation (sde=1, vwap > askPrice-1) - potential market impact (1)"
                    }

                    # Use the mapped key, or original key if not found in mapping
                    json_key_name = json_key_mapping.get(key, key)

                    # Select only the required columns for display
                    cols_to_display = [col for col in ['eqty', 'vwap', 'sde', 'bidVol-1', 'askVol-1', 'bidPrice-1', 'askPrice-1'] if col in df_violations.columns]
                    if cols_to_display:
                        print(df_violations.select(cols_to_display).head()) # Print top 5 violations with selected columns
                    else:
                        print("No relevant columns to display for this violation type.")

                    # Collect timestamps for JSON using the mapped key
                    if 'eTm_microseconds' in df_violations.columns:
                        violations_for_json[json_key_name] = df_violations['eTm_microseconds'].to_list()
                        print(f"Collected {len(violations_for_json[json_key_name])} timestamps for {json_key_name}")
                    else:
                        print(f"Warning: 'eTm_microseconds' not found in {key} violations, skipping for JSON export.")

                else:
                    print(f"\nNo violations for {key}.")
        
        # Save all collected timestamps to a single JSON file
        if violations_for_json:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            json_output_file = f"violation_timestamps_{timestamp_str}.json"
            with open(json_output_file, 'w') as f:
                json.dump(violations_for_json, f, indent=4)
            print(f"All violation timestamps saved to {json_output_file}")
        else:
            print("No violation timestamps to save.")

        # 10. Save results
        save_start_time = time.time()
        output_file = 'btcusdt_market_exec_merged_polars.csv'
        merged_df.head(30000).write_csv(output_file)
        save_time = time.time() - save_start_time
        print(f"\nMerged data (top 30,000 rows) saved to {output_file} in {save_time:.2f} seconds")
        
        total_time = time.time() - total_start_time
        print(f"\nTotal processing time: {total_time:.2f} seconds")
        
        return merged_df, coverage_status, validation_results
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return None, "ERROR", None

if __name__ == "__main__":
    result, status, validation_results = main()
    print(f"\nFinal Status: {status}")
    if validation_results:
        print("\nValidation Results:")
        for key, value in validation_results.items():
            print(f"{key}: {value}") 