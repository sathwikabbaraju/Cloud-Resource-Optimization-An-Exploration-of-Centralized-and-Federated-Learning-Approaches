import os
import pandas as pd
from pathlib import Path

# --- Configuration ---
base_dataset_path = Path("/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/cpu")
processed_data_dir = Path("./processed_client_data") # Directory to save processed data
processed_data_dir.mkdir(parents=True, exist_ok=True) # Create if it doesn't exist

# --- Data Processing Function for a Single Client ---
def process_client_data(client_folder_path: Path):
    """
    Processes all timeseries and summary data for a single client folder.
    Combines relevant timeseries data and saves as Parquet.
    """
    client_id = client_folder_path.name
    print(f"\nProcessing data for client: {client_id}")

    all_timeseries_dfs = []
    timeseries_files_found = 0

    # 1. Load and Concatenate Timeseries Data
    for file_path in client_folder_path.iterdir():
        if "timeseries" in file_path.name and file_path.suffix == '.csv':
            timeseries_files_found += 1
            try:
                # To address UserWarning: "Could not infer format..."
                # We'll read EpochTime as a string/object initially, then convert it
                # after concatenation to ensure consistent numeric type before pd.to_datetime
                # Removed parse_dates from here to handle it manually later more robustly
                for chunk in pd.read_csv(file_path, chunksize=100000):
                    all_timeseries_dfs.append(chunk)
            except Exception as e:
                print(f"  Error reading {file_path.name}: {e}")
                continue

    if not all_timeseries_dfs:
        print(f"  No timeseries files found for client {client_id} or all failed to load. Skipping.")
        return None

    # Concatenate all timeseries data for this client
    client_timeseries_df = pd.concat(all_timeseries_dfs, ignore_index=True)
    print(f"  Loaded {timeseries_files_found} timeseries files. Total rows: {len(client_timeseries_df)}")

    # 2. Initial Cleaning and Preprocessing for Timeseries Data

    # Handle 'Step' column: coerce to numeric, fill NaNs
    # This addresses: "Conversion failed for column Step with type object" error
    if 'Step' in client_timeseries_df.columns:
        # Coerce non-numeric values to NaN, then fill with a reasonable value (e.g., 0 or the mean)
        client_timeseries_df['Step'] = pd.to_numeric(client_timeseries_df['Step'], errors='coerce')
        client_timeseries_df['Step'] = client_timeseries_df['Step'].fillna(0).astype(int) # Fill and convert to int
        print(f"  Cleaned 'Step' column for client {client_id}.")


    # Convert EpochTime (Unix timestamp) to datetime and set as index
    if 'EpochTime' in client_timeseries_df.columns:
        # To address FutureWarning: "to_datetime with 'unit' when parsing strings is deprecated."
        # Ensure EpochTime is numeric before converting to datetime
        client_timeseries_df['EpochTime'] = pd.to_numeric(client_timeseries_df['EpochTime'], errors='coerce') # Coerce non-numeric to NaN
        client_timeseries_df['EpochTime'] = pd.to_datetime(client_timeseries_df['EpochTime'], unit='s')
        client_timeseries_df = client_timeseries_df.set_index('EpochTime').sort_index()
        # Handle potential duplicate timestamps if they exist (e.g., take the mean or drop)
        client_timeseries_df = client_timeseries_df[~client_timeseries_df.index.duplicated(keep='first')]
    else:
        print(f"  Warning: 'EpochTime' column not found in timeseries data for client {client_id}. Cannot set time index.")
        # If no EpochTime, you might need to infer it or use row index
        # For now, let's assume it exists as per README

    # Handle missing values: Fill NaNs with 0 for usage metrics.
    # Resource columns identified from data_explorer.py output
    resource_columns = ['CPUUtilization', 'RSS', 'ReadMB', 'WriteMB']

    # Identify numerical columns for filling NaNs
    numerical_cols = client_timeseries_df.select_dtypes(include=['float64', 'int64']).columns
    # Filter for columns that are likely resource usage metrics and exist in the dataframe
    actual_resource_cols = [col for col in numerical_cols if col in resource_columns]

    if not actual_resource_cols:
        print(f"  Warning: No common resource columns found. Filling all numerical NaNs with 0 for client {client_id}.")
        # Use new ffill/bfill syntax here too for general numerical columns
        client_timeseries_df = client_timeseries_df.ffill().bfill().fillna(0) # Fill remaining NaNs (e.g., at ends of series)
    else:
        print(f"  Filling NaNs with 0 for resource columns: {actual_resource_cols}")
        client_timeseries_df[actual_resource_cols] = client_timeseries_df[actual_resource_cols].fillna(0)
        # For other numerical columns, use new ffill/bfill syntax
        # To address FutureWarning: DataFrame.fillna with 'method' is deprecated
        client_timeseries_df = client_timeseries_df.ffill().bfill() # Forward fill, then backward fill for other NaNs
        client_timeseries_df = client_timeseries_df.fillna(0) # Catch any remaining NaNs (e.g., at start if no prior value)


    # Convert data types to memory-efficient formats
    for col in actual_resource_cols:
        if col in client_timeseries_df.columns:
            # Downcast floats
            client_timeseries_df[col] = pd.to_numeric(client_timeseries_df[col], downcast='float')
            # Downcast integers (if any are not resource_cols but are numerical)
    for col in client_timeseries_df.select_dtypes(include=['int64']).columns:
        # Ensure it's not the index, or handle separately if index needs specific type
        if col != client_timeseries_df.index.name:
            client_timeseries_df[col] = pd.to_numeric(client_timeseries_df[col], downcast='integer')


    # 3. Optional: Integrate Summary Data (Conceptual) - No changes needed here for now.

    # 4. Save Processed Data to Parquet
    output_parquet_path = processed_data_dir / f"client_{client_id}.parquet"
    try:
        # Parquet doesn't like duplicated columns, ensure no issues
        # Drop columns that are not useful or might cause issues (e.g., original 'Step' if already handled)
        # Assuming 'Node', 'Series' are useful to keep for later grouping
        # You might want to drop 'ElapsedTime', 'CPUFrequency', 'CPUTime', 'VMSize', 'Pages' if not directly used
        # or if they are highly correlated/redundant with other chosen features.
        # For now, let's keep all columns except 'Step' if it caused problems.
        # But since we're handling 'Step' to be numeric, it should be fine.
        client_timeseries_df.to_parquet(output_parquet_path)
        print(f"  Successfully processed and saved data for client {client_id} to {output_parquet_path}")
        return output_parquet_path
    except Exception as e:
        print(f"  Error saving data for client {client_id} to Parquet: {e}")
        # Print info about problematic columns for debugging if it's a new error
        print(f"  DataFrame dtypes for client {client_id}:")
        print(client_timeseries_df.info()) # Print info for debugging
        return None

# --- Main Processing Loop ---
def run_data_preprocessing():
    print("Starting data preprocessing for all clients...")
    # Get all subfolders with 4 digits and sort them
    subfolders = sorted([f for f in base_dataset_path.iterdir() if f.is_dir() and f.name.isdigit() and len(f.name) == 4])

    if not subfolders:
        print(f"No client subfolders found in {base_dataset_path}. Exiting.")
        return

    processed_client_paths = []
    # Process only the first few for quick testing if needed, remove [0:2] for full run
    for folder in subfolders: #[0:2] # Uncomment for testing with first 2 folders
        processed_path = process_client_data(folder)
        if processed_path:
            processed_client_paths.append(processed_path)

    print(f"\nFinished preprocessing. Successfully processed {len(processed_client_paths)} clients.")
    print(f"Processed data saved to: {processed_data_dir}")

    if not processed_client_paths:
        print("No data was successfully processed. Please check errors above.")
    else:
        # Example of loading one processed file to confirm
        print(f"\nLoading a sample processed file to confirm (e.g., {processed_client_paths[0].name}):")
        sample_loaded_df = pd.read_parquet(processed_client_paths[0])
        print("Sample loaded DataFrame head:")
        print(sample_loaded_df.head())
        print("Sample loaded DataFrame info:")
        sample_loaded_df.info()

if __name__ == "__main__":
    run_data_preprocessing()


    # 3. Optional: Integrate Summary Data (Conceptual)
    # This part depends heavily on the content of your summary files and how you want to use them.
    # For now, we'll keep it simple by focusing on timeseries data for prediction.
    # If summary files contain per-job metadata that applies to segments of the timeseries,
    # you might join them. But for a first pass, concentrating on timeseries is better.
    # For example, if summary has 'job_id' and timeseries also has 'job_id', you could merge.
    # All summary files for this client are loaded here.
    # all_summary_dfs = []
    # for file_path in client_folder_path.iterdir():
    #     if "summary" in file_path.name and file_path.suffix == '.csv':
    #         try:
    #             all_summary_dfs.append(pd.read_csv(file_path))
    #         except Exception as e:
    #             print(f"  Error reading {file_path.name}: {e}")
    #             continue
    # if all_summary_dfs:
    #     client_summary_df = pd.concat(all_summary_dfs, ignore_index=True)
    #     print(f"  Loaded {len(client_summary_df)} summary rows.")
    #     # TODO: Decide how to join or use summary data with timeseries
    #     # Example: Merge client_timeseries_df with client_summary_df on a common 'job_id' or 'serial_number'
    #     # client_timeseries_df = pd.merge(client_timeseries_df, client_summary_df, on='job_id', how='left')
