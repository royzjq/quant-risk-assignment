import pandas as pd
import numpy as np
from pathlib import Path

def load_exec_data(file_path='exec.csv'):
    """
    Load execution data
    """
    print(f"Loading execution data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} execution records")
    return df

def load_market_data(file_path):
    """
    Load market depth data
    """
    print(f"Loading market data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} market data records")
    return df

def filter_btcusdt_exec(exec_df):
    """
    Filter execution records containing BTCUSDT
    """
    btc_mask = exec_df['sym'].str.contains('BTCUSDT', na=False)
    btc_exec = exec_df[btc_mask].copy()
    print(f"Found {len(btc_exec)} BTCUSDT execution records")
    return btc_exec

def convert_etm_to_microseconds(exec_df):
    """
    Convert eTm from milliseconds to microseconds
    """
    exec_df = exec_df.copy()
    exec_df['eTm_microseconds'] = exec_df['eTm'] * 1000
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
    avg_cols = ['sde', 'px', 'qty', 'epx', 'vwap'] # Note: sde is discrete, averaging as requested.
    sum_cols = ['eqty', 'fee', 'ceqty']
    
    # Check if all group_cols exist in the DataFrame
    missing_group_cols = [col for col in group_cols if col not in exec_df.columns]
    if missing_group_cols:
        print(f"Warning: Missing group columns in exec_df: {missing_group_cols}. Skipping preprocessing for these.")
        group_cols = [col for col in group_cols if col in exec_df.columns]

    # Check if all avg_cols and sum_cols exist in the DataFrame
    available_avg_cols = [col for col in avg_cols if col in exec_df.columns]
    available_sum_cols = [col for col in sum_cols if col in exec_df.columns]

    if not group_cols:
        print("No valid columns to group by. Skipping preprocessing.")
        return exec_df

    # Group and aggregate
    aggregated_df = exec_df.groupby(group_cols, as_index=False).agg(
        **{f'{col}_avg': (col, 'mean') for col in available_avg_cols},
        **{f'{col}_sum': (col, 'sum') for col in available_sum_cols}
    )
    
    # Rename aggregated columns back to original names
    for col in available_avg_cols:
        aggregated_df[col] = aggregated_df[f'{col}_avg']
        aggregated_df = aggregated_df.drop(columns=[f'{col}_avg'])
    for col in available_sum_cols:
        aggregated_df[col] = aggregated_df[f'{col}_sum']
        aggregated_df = aggregated_df.drop(columns=[f'{col}_sum'])
        
    # Assign rqty to 0 for all preprocessed records
    if 'rqty' in aggregated_df.columns:
        aggregated_df['rqty'] = 0
    
    print(f"Original exec records: {len(exec_df)}")
    print(f"Preprocessed exec records: {len(aggregated_df)}")
    print("Preprocessing complete.")
    
    return aggregated_df

def calculate_mid_price(market_df):
    """
    Calculate market mid price
    """
    market_df = market_df.copy()
    market_df['mid_price'] = (market_df['askPrice-1'] + market_df['bidPrice-1']) / 2
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
    time_mask = (exec_df['eTm_microseconds'] >= market_min_time) & (exec_df['eTm_microseconds'] <= market_max_time)
    filtered_exec_df = exec_df[time_mask].copy()
    
    excluded_count = len(exec_df) - len(filtered_exec_df)
    print(f"Exec records within market time range: {len(filtered_exec_df)}")
    print(f"Exec records excluded (outside time range): {excluded_count}")
    
    if excluded_count > 0:
        early_records = exec_df[exec_df['eTm_microseconds'] < market_min_time]
        late_records = exec_df[exec_df['eTm_microseconds'] > market_max_time]
        print(f"  - Records too early: {len(early_records)}")
        print(f"  - Records too late: {len(late_records)}")
    
    return filtered_exec_df

def merge_market_with_exec(market_df, exec_df):
    """
    Merge market data with execution records using merge_asof
    Market data as the base (left), exec data as the right
    For duplicate exec records that can't be matched, create new rows with only exec data
    Keep only the closest (last) market record for each exec record
    """
    print("\nMerging market data with execution records using merge_asof...")
    
    # Ensure data is sorted by time
    market_df_sorted = market_df.sort_values('timestamp').reset_index(drop=True)
    exec_df_sorted = exec_df.sort_values('eTm_microseconds').reset_index(drop=True)
    
    # Use merge_asof with market data as base
    merged_df = pd.merge_asof(
        market_df_sorted,
        exec_df_sorted,
        left_on='timestamp',
        right_on='eTm_microseconds',
        direction='forward'  # Use the earliest exec record not earlier than market timestamp
    )
    
    # Find exec records that were not matched (duplicates)
    matched_exec_times = set(merged_df.dropna(subset=['eTm'])['eTm'].unique())
    all_exec_times = set(exec_df_sorted['eTm'].unique())
    missing_exec_times = all_exec_times - matched_exec_times
    
    if len(missing_exec_times) > 0:
        print(f"Found {len(missing_exec_times)} exec records not matched (likely duplicates)")
        print("Creating new rows for unmatched exec records...")
        
        # Get unmatched exec records
        unmatched_exec = exec_df_sorted[exec_df_sorted['eTm'].isin(missing_exec_times)].copy()
        
        # Create new rows with only exec data
        # Set timestamp to be same as eTm_microseconds
        new_rows = []
        for _, exec_row in unmatched_exec.iterrows():
            new_row = {}
            # Initialize all market data columns with NaN
            for col in market_df_sorted.columns:
                new_row[col] = np.nan
            
            # Set timestamp to exec timestamp
            new_row['timestamp'] = exec_row['eTm_microseconds']
            
            # Add all exec data columns
            for col in exec_df_sorted.columns:
                new_row[col] = exec_row[col]
            
            new_rows.append(new_row)
        
        # Convert to DataFrame and append to merged data
        if new_rows:
            new_rows_df = pd.DataFrame(new_rows)
            merged_df = pd.concat([merged_df, new_rows_df], ignore_index=True)
            print(f"Added {len(new_rows)} new rows for unmatched exec records")
    
    # Keep only the last (closest) market record for each exec record
    print("Filtering to keep only the closest market record for each exec...")
    
    # Separate records with and without exec data
    exec_records = merged_df.dropna(subset=['eTm']).copy()
    market_only_records = merged_df[merged_df['eTm'].isna()].copy()
    
    if len(exec_records) > 0:
        # For exec records, keep only the one with maximum timestamp for each eTm
        closest_exec_records = exec_records.loc[exec_records.groupby('eTm')['timestamp'].idxmax()].reset_index(drop=True)
        print(f"Reduced exec-matched records from {len(exec_records)} to {len(closest_exec_records)}")
        
        # Combine closest exec records with market-only records
        merged_df = pd.concat([closest_exec_records, market_only_records], ignore_index=True)
    else:
        merged_df = market_only_records
    
    # Sort by timestamp
    merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Successfully merged data. Result contains {len(merged_df)} records")
    return merged_df

def debug_missing_exec_records(missing_exec_times, original_exec_df):
    """
    Debug and print details of execution records that are missing after merging.
    """
    print("\n=== Debugging Missing Execution Records ===")
    
    if len(missing_exec_times) > 0:
        print(f"Found {len(missing_exec_times)} unique eTm values missing from merged data:")
        
        # Get the missing records details from the original (preprocessed) exec_df
        missing_records = original_exec_df[original_exec_df['eTm'].isin(missing_exec_times)].copy()
        missing_records = missing_records.sort_values('eTm_microseconds')
        
        print("\nMissing execution records details:")
        # Include Unnamed: 0 column if it exists
        columns_to_show = ['Unnamed: 0', 'eTm', 'eTm_microseconds', 'px', 'qty', 'sde', 'rqty', 'eqty', 'fee', 'ceqty']
        available_columns = [col for col in columns_to_show if col in missing_records.columns]
        print(missing_records[available_columns])
        
        print(f"\nTotal missing records: {len(missing_records)}")
    else:
        print("No unique eTm values are missing from merged data. Check other potential issues if records count still differs.")

def analyze_exec_coverage(merged_df, original_exec_df):
    """
    Analyze if all execution records can find corresponding market data points
    """
    print("\n=== Execution Coverage Analysis ===")
    
    # Count non-null exec records in merged data
    exec_matched = merged_df.dropna(subset=['eTm']).copy()
    
    print(f"Total merged records: {len(merged_df)}")
    print(f"Records with exec data: {len(exec_matched)}")
    print(f"Records without exec data (market only): {len(merged_df) - len(exec_matched)}")
    
    # Check if all original exec records are covered
    original_exec_count = len(original_exec_df)
    unique_exec_in_merged = merged_df['eTm'].nunique() - (1 if merged_df['eTm'].isna().any() else 0)
    
    print(f"\nOriginal BTCUSDT exec records: {original_exec_count}")
    print(f"Unique exec records found in merged data: {unique_exec_in_merged}")
    
    # Find missing exec records - do this calculation early
    # Use eTm_microseconds for precise comparison
    merged_exec_micro_times = set(merged_df.dropna(subset=['eTm'])['eTm_microseconds'].unique())
    original_exec_micro_times = set(original_exec_df['eTm_microseconds'].unique())
    missing_exec_micro_times = original_exec_micro_times - merged_exec_micro_times
    
    print(f"DEBUG: Original unique eTm_microseconds count: {len(original_exec_micro_times)}")
    print(f"DEBUG: Merged unique eTm_microseconds count: {len(merged_exec_micro_times)}")
    print(f"DEBUG: Missing unique eTm_microseconds count: {len(missing_exec_micro_times)}")
    
    # The primary check for coverage should be based on truly_missing_exec_micro_times
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

def main():
    """
    Main function: analyze exec-market data correspondence
    """
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
        merged_df = merge_market_with_exec(market_df, filtered_exec_df)
        
        # 8. Analyze coverage
        coverage_status, exec_matched = analyze_exec_coverage(merged_df, filtered_exec_df)
        
        # 9. Save results
        output_file = 'btcusdt_market_exec_merged.csv'
        merged_df.head(30000).to_csv(output_file, index=False)
        print(f"\nMerged data (top 30,000 rows) saved to {output_file}")
        
        # Optional: Save only records with exec data for further analysis
        # if len(exec_matched) > 0:
        #     exec_output_file = 'btcusdt_market_exec_matched.csv'
        #     exec_matched.to_csv(exec_output_file, index=False)
        #     print(f"Records with exec data saved to {exec_output_file}")
        
        # Optional: Save filtered exec records for reference
        # filtered_output_file = 'btcusdt_exec_filtered.csv'
        # filtered_exec_df.to_csv(filtered_output_file, index=False)
        # print(f"Filtered exec records saved to {filtered_output_file}")
        
        return merged_df, coverage_status
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return None, "ERROR"

if __name__ == "__main__":
    result, status = main()
    print(f"\nFinal Status: {status}") 