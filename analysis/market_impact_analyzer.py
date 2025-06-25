import polars as pl
import json
import time

def validate_exec_market_data(merged_df):
    """
    Validate execution and market data relationships based on specified rules.
    sde = 1 (Buy), sde = 2 (Sell).
    
    Returns:
        dict: Dictionary containing validation results with violation DataFrames
    """
    print("\n=== Market Impact Data Validation ===")
    
    # Ensure relevant columns exist
    required_cols = ['eqty', 'vwap', 'sde', 'bidVol-1', 'askVol-1', 'bidPrice-1', 'askPrice-1']
    missing_cols = [col for col in required_cols if col not in merged_df.columns]
    if missing_cols:
        print(f"Warning: Missing required columns for validation: {missing_cols}. Skipping validation.")
        return {}

    # Filter for records with actual exec data
    exec_records = merged_df.filter(pl.col('eTm').is_not_null())
    
    if len(exec_records) == 0:
        print("No execution records to validate.")
        return {}

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

    print("Market Impact Data Validation Complete.")
    
    return {
        "eqty_sell_violations": eqty_sell_violations,
        "eqty_buy_violations": eqty_buy_violations,
        "vwap_sell_violations": vwap_sell_violations,
        "vwap_buy_violations": vwap_buy_violations
    }

def calculate_impact_metrics(merged_df):
    """
    Calculate market impact metrics and identify significant impacts based on statistical analysis.
    Only impacts exceeding 1.5 sigma are considered real market impacts.
    """
    
    exec_records = merged_df.filter(pl.col('eTm').is_not_null())
    if exec_records.is_empty():
        return None

    # Prepare market data for joining
    market_snapshots = merged_df.select(
        ["timestamp", "bidPrice-1", "askPrice-1", "bidVol-1", "askVol-1"]
    ).sort("timestamp")

    # Calculate daily volatility for bid/ask prices and spreads
    daily_spreads = market_snapshots.with_columns(
        (pl.col('askPrice-1') - pl.col('bidPrice-1')).alias('spread')
    )
    
    # Calculate standard deviations for the day
    bid_price_std = market_snapshots['bidPrice-1'].std()
    ask_price_std = market_snapshots['askPrice-1'].std()
    spread_std = daily_spreads['spread'].std()
    
    # Calculate t+1s timestamp for each execution
    exec_records_with_post_ts = exec_records.with_columns(
        (pl.col("eTm_microseconds") + 1_000_000).alias("timestamp_post1s")
    )

    # Join to get post-trade market state
    exec_with_post_trade_state = exec_records_with_post_ts.join_asof(
        market_snapshots, 
        left_on="timestamp_post1s", 
        right_on="timestamp",
        suffix="_post1s"
    )

    # Calculate metrics
    final_metrics_df = exec_with_post_trade_state.with_columns([
        # Pre-trade metrics
        (pl.col('askPrice-1') - pl.col('bidPrice-1')).alias('pre_trade_spread'),
        
        # Post-trade metrics (1s after)
        pl.when(pl.col('bidPrice-1_post1s').is_not_null() & pl.col('askPrice-1_post1s').is_not_null()).then(
            (pl.col('askPrice-1_post1s') - pl.col('bidPrice-1_post1s'))
        ).otherwise(None).alias('post_trade_spread_1s'),
        
        # Price impact calculation based on side
        # For sell orders (sde=2): impact on bid price
        pl.when(pl.col('sde') == 2).then(
            pl.when(pl.col('bidPrice-1_post1s').is_not_null()).then(
                pl.col('bidPrice-1_post1s') - pl.col('bidPrice-1')
            ).otherwise(None)
        # For buy orders (sde=1): impact on ask price  
        ).when(pl.col('sde') == 1).then(
            pl.when(pl.col('askPrice-1_post1s').is_not_null()).then(
                pl.col('askPrice-1_post1s') - pl.col('askPrice-1')
            ).otherwise(None)
        ).otherwise(None).alias('price_impact_1s'),
    ])

    # Calculate spread impact
    final_metrics_df = final_metrics_df.with_columns([
        (pl.col('post_trade_spread_1s') - pl.col('pre_trade_spread')).alias('spread_impact_1s'),
    ])

    # Filter for significant impacts (> 1.5 sigma)
    threshold_multiplier = 1.5
    
    significant_impacts = final_metrics_df.filter(
        (pl.col('price_impact_1s').is_not_null()) &
        (pl.col('spread_impact_1s').is_not_null()) &
        (
            (pl.col('price_impact_1s').abs() > threshold_multiplier * (
                pl.when(pl.col('sde') == 2).then(pl.lit(bid_price_std))
                .when(pl.col('sde') == 1).then(pl.lit(ask_price_std))
                .otherwise(pl.lit(0.0))
            )) |
            (pl.col('spread_impact_1s').abs() > threshold_multiplier * spread_std)
        )
    )

    if significant_impacts.is_empty():
        return None
    else:
        # Prepare data for JSON export
        impact_data = {}
        for row in significant_impacts.iter_rows(named=True):
            timestamp = row['eTm_microseconds']
            impact_data[str(timestamp)] = {
                "execution_time_ms": row['eTm'],
                "side": "SELL" if row['sde'] == 2 else "BUY",
                "quantity": row['eqty'],
                "vwap": row['vwap'],
                "price_impact_1s": row['price_impact_1s'],
                "spread_impact_1s": row['spread_impact_1s'],
                "price_impact_sigma": row['price_impact_1s'] / (bid_price_std if row['sde'] == 2 else ask_price_std),
                "spread_impact_sigma": row['spread_impact_1s'] / spread_std
            }
        
        return impact_data

def analyze_market_impact_violations(merged_df, save_timestamps=True):
    """
    Analyze market impact violations and optionally save timestamps to JSON
    
    Args:
        merged_df: Polars DataFrame with merged market and execution data
        save_timestamps: Boolean, whether to save violation timestamps to JSON file
        
    Returns:
        dict: Analysis results including violation counts and details
    """
    print("\n=== Market Impact Analysis ===")
    
    # Run validation
    validation_results = validate_exec_market_data(merged_df)
    
    # Calculate significant market impacts
    significant_impacts = calculate_impact_metrics(merged_df)
    
    if not validation_results:
        print("No validation results to analyze.")
        return {"status": "NO_DATA", "violations": {}}
    
    # Prepare data for JSON export and display
    violations_for_json = {}
    violation_summary = {}
    
    # Define mapping for clearer JSON keys
    json_key_mapping = {
        "eqty_sell_violations": "eqty violation (sde=2, eqty > bidVol-1) - potential market impact (1)",
        "eqty_buy_violations": "eqty violation (sde=1, eqty > askVol-1) - potential market impact (1)",
        "vwap_sell_violations": "vwap violation (sde=2, vwap < bidPrice-1) - potential market impact (1)",
        "vwap_buy_violations": "vwap violation (sde=1, vwap > askPrice-1) - potential market impact (1)"
    }
    
    # Process violations (existing logic kept for compatibility)
    for key, df_violations in validation_results.items():
        json_key_name = json_key_mapping.get(key, key)
        
        if not df_violations.is_empty():
            if 'eTm_microseconds' in df_violations.columns:
                timestamps = df_violations['eTm_microseconds'].to_list()
                violations_for_json[json_key_name] = timestamps
                violation_summary[json_key_name] = {
                    "count": len(timestamps),
                    "first_timestamp": min(timestamps),
                    "last_timestamp": max(timestamps)
                }
        else:
            violation_summary[json_key_name] = {"count": 0}
    
    # Add significant market impacts to JSON
    if significant_impacts:
        violations_for_json["significant_market_impacts_1s"] = significant_impacts
        violation_summary["significant_market_impacts_1s"] = {"count": len(significant_impacts)}
    
    # Save to JSON file if requested
    if save_timestamps and (violations_for_json or significant_impacts):
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        json_output_file = f"results/market_impact_violation_{timestamp_str}.json"
        with open(json_output_file, 'w') as f:
            json.dump(violations_for_json, f, indent=4)
    
    return {
        "status": "COMPLETE",
        "violations": violation_summary,
        "timestamps": violations_for_json if save_timestamps else None,
        "validation_results": validation_results
    }

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