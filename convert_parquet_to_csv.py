import pandas as pd

def convert_parquet_to_csv(parquet_file_path, csv_file_path, num_rows=None):
    """
    Reads a Parquet file and saves it to a CSV file.

    Args:
        parquet_file_path (str): The path to the input Parquet file.
        csv_file_path (str): The path to the output CSV file.
        num_rows (int, optional): The number of rows to convert. If None, all rows are converted.
    """
    try:
        # Read the Parquet file into a pandas DataFrame
        df = pd.read_parquet(parquet_file_path)

        # If num_rows is specified, take only the head of the DataFrame
        if num_rows is not None:
            df = df.head(num_rows)

        # Save the DataFrame to a CSV file
        df.to_csv(csv_file_path, index=False)
        print(f"Successfully converted '{parquet_file_path}' (first {num_rows if num_rows is not None else 'all'} rows) to '{csv_file_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Example usage:
    # Convert the first 30000 rows of btcusdt_processed_data.parquet to CSV
    input_parquet_file = "results/btcusdt_processed_data.parquet"
    output_csv_file = "top_30000_rows.csv"
    rows_to_convert = 30000

    convert_parquet_to_csv(input_parquet_file, output_csv_file, rows_to_convert) 