import pandas as pd
import numpy as np
import json
from pathlib import Path

def validate_exec_csv(file_path: Path = Path("exec.csv")):
    """
    Validates the content of exec.csv for reasonableness and anomalies.

    Args:
        file_path (Path): The path to the exec.csv file.

    Returns:
        dict: A dictionary containing validation results and detected anomalies.
    """
    results = {
        "summary": "Validation Report for exec.csv",
        "missing_values": {},
        "data_type_issues": {},
        "value_range_issues": {},
        "anomalies": {}
    }

    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}. Total rows: {len(df)}")
    except FileNotFoundError:
        results["summary"] = f"Error: File not found at {file_path}"
        return results
    except Exception as e:
        results["summary"] = f"Error loading CSV: {e}"
        return results

    # --- 1. Check for Missing Values ---
    print("Checking for missing values...")
    for col in df.columns:
        missing_indices = df[df[col].isnull()].index.tolist()
        if len(missing_indices) > 0:
            if col not in results["missing_values"]:
                results["missing_values"][col] = {
                    "count": 0,
                    "indices": [],
                    "csv_row_numbers": [],
                    "data": []
                }
            results["missing_values"][col]["count"] = len(missing_indices)
            results["missing_values"][col]["indices"] = [int(i) for i in missing_indices]
            if 'Unnamed: 0' in df.columns:
                results["missing_values"][col]["csv_row_numbers"] = [int(df.loc[i, 'Unnamed: 0']) for i in missing_indices]
            else:
                results["missing_values"][col]["csv_row_numbers"] = [int(i) for i in missing_indices]
            results["missing_values"][col]["data"] = df.loc[missing_indices].to_dict(orient='records')
            print(f"  - Column '{col}': {len(missing_indices)} missing values")
    if not results["missing_values"]:
        print("  - No missing values found in any column.")

    # --- 2. Check Data Types and Convert if necessary ---
    print("\nChecking data types and converting...")
    numerical_cols = ['eTm', 'px', 'qty', 'epx', 'eqty', 'fee', 'ceqty', 'vwap', 'rqty']
    categorical_cols = ['EventType', 'ech', 'sym', 'sde']

    for col in numerical_cols:
        initial_missing_count = df[col].isnull().sum()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        non_numeric_after_coerce_indices = df[df[col].isnull() & (df[col].index.isin(df.index.difference(df.loc[df[col].isnull()].index)))].index.tolist()

        if len(non_numeric_after_coerce_indices) > 0:
            if col not in results["data_type_issues"]:
                results["data_type_issues"][col] = {
                    "count": 0,
                    "indices": [],
                    "csv_row_numbers": [],
                    "data": []
                }
            results["data_type_issues"][col]["count"] = len(non_numeric_after_coerce_indices)
            results["data_type_issues"][col]["indices"] = [int(i) for i in non_numeric_after_coerce_indices]
            if 'Unnamed: 0' in df.columns:
                results["data_type_issues"][col]["csv_row_numbers"] = [int(df.loc[i, 'Unnamed: 0']) for i in non_numeric_after_coerce_indices]
            else:
                results["data_type_issues"][col]["csv_row_numbers"] = [int(i) for i in non_numeric_after_coerce_indices]
            results["data_type_issues"][col]["data"] = df.loc[non_numeric_after_coerce_indices].to_dict(orient='records')
            print(f"  - Column '{col}': {len(non_numeric_after_coerce_indices)} non-numeric values found (converted to NaN).")
    if not results["data_type_issues"]:
        print("  - All numerical columns seem to have correct data types.")

    # --- 3. Value Range and Consistency Checks ---
    print("\nPerforming value range and consistency checks...")

    # EventType check
    if 'EventType' in df.columns:
        expected_event_types = [
            'send_order_ack', 'send_order_nack', 'send_cxl_ack',
            'send_cxl_nack', 'send_cxr_ack', 'send_cxr_nack',
            'exch_open_ack', 'exch_open_nack', 'exch_cxl_ack',
            'exch_cxl_nack', 'exec'
        ]
        unexpected_events_df = df[~df['EventType'].isin(expected_event_types)]
        if not unexpected_events_df.empty:
            unexpected_event_indices = unexpected_events_df.index.tolist()
            results["value_range_issues"]["EventType"] = {
                "count": len(unexpected_event_indices),
                "indices": [int(i) for i in unexpected_event_indices],
                "csv_row_numbers": [int(df.loc[i, 'Unnamed: 0']) if 'Unnamed: 0' in df.columns else int(i) for i in unexpected_event_indices],
                "data": unexpected_events_df.to_dict(orient='records'),
                "unique_values_found": list(unexpected_events_df['EventType'].unique())
            }
            print(f"  - EventType: Unexpected values found: {list(unexpected_events_df['EventType'].unique())}")
        else:
            print("  - EventType: All values are in the expected list.")

    # ech check
    if 'ech' in df.columns:
        expected_ech_types = ['Binance']
        unexpected_ech_df = df[~df['ech'].isin(expected_ech_types)]
        if not unexpected_ech_df.empty:
            unexpected_ech_indices = unexpected_ech_df.index.tolist()
            results["value_range_issues"]["ech"] = {
                "count": len(unexpected_ech_indices),
                "indices": [int(i) for i in unexpected_ech_indices],
                "csv_row_numbers": [int(df.loc[i, 'Unnamed: 0']) if 'Unnamed: 0' in df.columns else int(i) for i in unexpected_ech_indices],
                "data": unexpected_ech_df.to_dict(orient='records'),
                "unique_values_found": list(unexpected_ech_df['ech'].unique())
            }
            print(f"  - ech: Unexpected values found: {list(unexpected_ech_df['ech'].unique())}")
        else:
            print("  - ech: All values are 'Binance'.")

    # sde check
    if 'sde' in df.columns:
        df['sde'] = pd.to_numeric(df['sde'], errors='coerce')
        invalid_sde_df = df[~df['sde'].isin([1, 2]) & df['sde'].notnull()]
        if not invalid_sde_df.empty:
            invalid_sde_indices = invalid_sde_df.index.tolist()
            results["value_range_issues"]["sde"] = {
                "count": len(invalid_sde_indices),
                "indices": [int(i) for i in invalid_sde_indices],
                "csv_row_numbers": [int(df.loc[i, 'Unnamed: 0']) if 'Unnamed: 0' in df.columns else int(i) for i in invalid_sde_indices],
                "data": invalid_sde_df.to_dict(orient='records'),
                "unique_values_found": list(invalid_sde_df['sde'].unique())
            }
            print(f"  - sde: Unexpected values found: {list(invalid_sde_df['sde'].unique())}")
        else:
            print("  - sde: All values are 1 or 2 (or NaN).")

    # Price and Quantity related checks
    for col in ['px', 'qty', 'epx', 'eqty', 'vwap', 'rqty']:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            negative_values_df = df[df[col] < 0]
            if not negative_values_df.empty:
                negative_indices = negative_values_df.index.tolist()
                if col not in results["value_range_issues"]:
                    results["value_range_issues"][col] = {}
                results["value_range_issues"][col]["negative_values"] = {
                    "count": len(negative_indices),
                    "indices": [int(i) for i in negative_indices],
                    "csv_row_numbers": [int(df.loc[i, 'Unnamed: 0']) if 'Unnamed: 0' in df.columns else int(i) for i in negative_indices],
                    "data": negative_values_df.to_dict(orient='records')
                }
                print(f"  - Column '{col}': {len(negative_indices)} negative values found.")
            else:
                print(f"  - Column '{col}': All values are non-negative.")

    # Fee check
    if 'fee' in df.columns and pd.api.types.is_numeric_dtype(df['fee']):
        non_negative_fee_df = df[df['fee'] >= 0]
        if not non_negative_fee_df.empty:
            non_negative_fee_indices = non_negative_fee_df.index.tolist()
            results["value_range_issues"]["fee_positive_or_zero"] = {
                "count": len(non_negative_fee_indices),
                "indices": [int(i) for i in non_negative_fee_indices],
                "csv_row_numbers": [int(df.loc[i, 'Unnamed: 0']) if 'Unnamed: 0' in df.columns else int(i) for i in non_negative_fee_indices],
                "data": non_negative_fee_df.to_dict(orient='records')
            }
            print(f"  - Column 'fee': {len(non_negative_fee_indices)} non-negative values found (expected strictly negative).")
        else:
            print(f"  - Column 'fee': All values are strictly negative.")

    # Quantity consistency check
    if all(col in df.columns for col in ['qty', 'eqty', 'ceqty', 'rqty']) and \
       all(pd.api.types.is_numeric_dtype(df[col]) for col in ['qty', 'eqty', 'ceqty', 'rqty']):
        
        inconsistent_rows = []
        inconsistency_reasons = {}
        
        # First check: eqty > qty
        eqty_gt_qty = df[df['eqty'] > df['qty']]
        if not eqty_gt_qty.empty:
            for idx in eqty_gt_qty.index:
                inconsistent_rows.append(idx)
                inconsistency_reasons[idx] = "eqty > qty"
        
        # Second check: eqty + rqty != qty
        eqty_plus_rqty_ne_qty = df[abs((df['eqty'] + df['rqty']) - df['qty']) > 1e-10]
        if not eqty_plus_rqty_ne_qty.empty:
            for idx in eqty_plus_rqty_ne_qty.index:
                if idx in inconsistent_rows:
                    continue
                    
                current_row = df.loc[idx]
                current_sym = current_row['sym']
                current_eqty_plus_rqty = current_row['eqty'] + current_row['rqty']
                
                same_sym_df = df[df['sym'] == current_sym]
                same_sym_before_current = same_sym_df[same_sym_df.index < idx]
                
                if not same_sym_before_current.empty:
                    prev_row = same_sym_before_current.iloc[-1]
                    prev_rqty = prev_row['rqty']
                    
                    if abs(current_eqty_plus_rqty - prev_rqty) > 1e-10:
                        inconsistent_rows.append(idx)
                        inconsistency_reasons[idx] = f"eqty + rqty ({current_eqty_plus_rqty}) != prev_rqty ({prev_rqty})"
                else:
                    inconsistent_rows.append(idx)
                    inconsistency_reasons[idx] = "No previous row with same sym found"
        
        inconsistent_indices = list(set(inconsistent_rows))
        inconsistent_qty_df = df.loc[inconsistent_indices] if inconsistent_indices else pd.DataFrame()
        
        if not inconsistent_qty_df.empty:
            results["anomalies"]["qty_consistency"] = {
                "count": len(inconsistent_qty_df),
                "all_indices": [int(i) for i in inconsistent_indices],
                "all_data": inconsistent_qty_df.to_dict(orient='records'),
                "all_reasons": [inconsistency_reasons[idx] for idx in inconsistent_indices],
                "all_csv_row_numbers": [int(df.loc[idx, 'Unnamed: 0']) if 'Unnamed: 0' in df.columns else int(idx) for idx in inconsistent_indices]
            }
            print(f"  - Quantity Consistency: {len(inconsistent_qty_df)} inconsistent rows found.")
        else:
            print("  - Quantity Consistency: qty, eqty, rqty values are consistent.")

    print("\nValidation complete.")
    
    # Save results to a JSON file
    output_json_file = "results/exec_validation.json"
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Validation report saved to {output_json_file}")

    return results 