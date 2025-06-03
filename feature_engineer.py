import pandas as pd
from pathlib import Path
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler # Import StandardScaler
import numpy as np
import pickle # To save the scaler for later use

# --- Configuration ---
processed_data_dir = Path("./processed_client_data")
feature_engineered_data_dir = Path("./feature_engineered_beta_client_data")
feature_engineered_data_dir.mkdir(parents=True, exist_ok=True) # Create if it doesn't exist

# Define the target variables we want to predict
# These are your core resource metrics identified earlier
TARGET_COLUMNS = ['CPUUtilization', 'RSS', 'ReadMB', 'WriteMB']

# Define the window size for lagged features and rolling statistics
# This means we will look at the past 5 time steps (50 seconds, as data is 10s intervals)
LAG_WINDOW = 5
ROLLING_WINDOW = '50s' # 50 seconds rolling window

# --- Feature Engineering Function for a Single Client ---
def engineer_features_for_client(client_parquet_path: Path):
    client_id = client_parquet_path.stem.replace("client_", "")
    print(f"\nEngineering features for client: {client_id}")

    try:
        df = pd.read_parquet(client_parquet_path)
    except Exception as e:
        print(f"  Error reading {client_parquet_path.name}: {e}. Skipping client.")
        return None

    # --- 1. Handle Multiple Time Series (Node, Series) per Client (Important!) ---
    # Each client's data (a folder) can contain data from multiple jobs/nodes/series.
    # We need to ensure that lagged features and rolling statistics are calculated
    # independently for each unique (Node, Series) combination, otherwise
    # we'll mix data from different sources.
    # Group by 'Node' and 'Series' and apply feature engineering to each group.

    # Identify columns to apply lags and rolling features to
    # These are the TARGET_COLUMNS and potentially other numerical features
    features_to_lag_roll = TARGET_COLUMNS + [
        'CPUFrequency', 'CPUTime', 'VMSize', 'Pages', 'ElapsedTime'
    ]
    # Filter to only include columns actually present in the DataFrame
    features_to_lag_roll = [col for col in features_to_lag_roll if col in df.columns]

    engineered_dfs = []

    # Iterate over unique (Node, Series) combinations within the client's data
    # This ensures features are calculated within a continuous series
    unique_node_series_combinations = df[['Node', 'Series']].drop_duplicates().values

    if len(unique_node_series_combinations) > 100: # Heuristic to avoid too many groups
        print(f"  Warning: Client {client_id} has {len(unique_node_series_combinations)} unique (Node, Series) combinations. "
              "Processing all, but consider further aggregation if performance is an issue or series are too short.")

    for node, series in unique_node_series_combinations:
        # Select the sub-dataframe for the current (Node, Series)
        sub_df = df[(df['Node'] == node) & (df['Series'] == series)].copy()

        if sub_df.empty:
            continue

        # Ensure index is sorted for time-series operations
        sub_df = sub_df.sort_index()

        # --- Lagged Features ---
        for col in features_to_lag_roll:
            for i in range(1, LAG_WINDOW + 1):
                sub_df[f'{col}_lag_{i}'] = sub_df[col].shift(i)

        # --- Rolling Statistics ---
        # Note: Rolling window requires a Resample object for time-based windows
        # If your data has irregular timestamps, a fixed number of periods (e.g., .rolling(window=LAG_WINDOW))
        # might be more appropriate. Given 10s intervals, '50s' should work.
        for col in features_to_lag_roll:
            sub_df[f'{col}_rolling_mean_{ROLLING_WINDOW}'] = sub_df[col].rolling(window=ROLLING_WINDOW).mean()
            sub_df[f'{col}_rolling_std_{ROLLING_WINDOW}'] = sub_df[col].rolling(window=ROLLING_WINDOW).std()

        # --- Time-based Features (from EpochTime index) ---
        sub_df['hour_of_day'] = sub_df.index.hour
        sub_df['day_of_week'] = sub_df.index.dayofweek # Monday=0, Sunday=6
        sub_df['month'] = sub_df.index.month
        sub_df['day_of_month'] = sub_df.index.day
        sub_df['is_weekend'] = (sub_df.index.dayofweek >= 5).astype(int)
        sub_df['minute_of_hour'] = sub_df.index.minute


        engineered_dfs.append(sub_df)

    if not engineered_dfs:
        print(f"  No valid time series found after grouping for client {client_id}. Skipping.")
        return None

    # Concatenate all engineered sub-dataframes for this client
    client_engineered_df = pd.concat(engineered_dfs)
    print(f"  Finished feature engineering for client {client_id}. Total rows: {len(client_engineered_df)}")

    # Drop rows with NaN values created by lagging/rolling (first few rows)
    # This is critical as they don't have full historical context
    client_engineered_df.dropna(inplace=True)
    print(f"  Rows after dropping NaNs: {len(client_engineered_df)}")

    if client_engineered_df.empty:
        print(f"  Warning: No data left after dropping NaNs for client {client_id}. Skipping save.")
        return None

    # --- 2. Normalization/Scaling (Min-Max Scaling) ---
    # Apply scaling to all numerical features.
    # We must save the scaler for each client if we want to reverse the scaling later for predictions
    # or apply the same scaling to new, unseen data from that client.
    # For Federated Learning, clients train locally, so local scaling is appropriate.

    # Exclude non-numeric columns like 'Node' (object dtype), and the index
    numerical_cols_for_scaling = client_engineered_df.select_dtypes(include=np.number).columns.tolist()

    # Exclude the target columns themselves if you're predicting their raw values
    # or include them if you're predicting their scaled values.
    # For now, let's scale both features and targets (as the model will learn scaled values)
    # and we'll inverse transform targets later.
    features_and_targets_to_scale = [col for col in numerical_cols_for_scaling if col not in ['Step', 'Series', 'is_weekend', 'hour_of_day', 'day_of_week', 'month', 'day_of_month', 'minute_of_hour']]


    #scaler = MinMaxScaler()
    scaler = StandardScaler() # Instantiate StandardScaler
    client_engineered_df[features_and_targets_to_scale] = scaler.fit_transform(client_engineered_df[features_and_targets_to_scale])

    # Save the scaler object for this client
    scaler_path = feature_engineered_data_dir / f"scaler_client_{client_id}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved for client {client_id} to {scaler_path}")

    # --- 3. Save Feature Engineered Data ---
    output_parquet_path = feature_engineered_data_dir / f"client_{client_id}_features.parquet"
    client_engineered_df.to_parquet(output_parquet_path)
    print(f"  Successfully saved feature engineered data for client {client_id} to {output_parquet_path}")

    return output_parquet_path

# --- Main Feature Engineering Loop ---
def run_feature_engineering():
    print("Starting feature engineering for all clients...")
    client_parquet_files = sorted(list(processed_data_dir.glob("client_*.parquet")))

    if not client_parquet_files:
        print(f"No processed client data found in {processed_data_dir}. Please run data preprocessing first.")
        return

    processed_feature_paths = []
    for file_path in client_parquet_files:
        engineered_path = engineer_features_for_client(file_path)
        if engineered_path:
            processed_feature_paths.append(engineered_path)

    print(f"\nFinished feature engineering. Successfully processed {len(processed_feature_paths)} clients.")
    print(f"Feature engineered data saved to: {feature_engineered_data_dir}")

    if not processed_feature_paths:
        print("No feature engineered data was successfully processed. Please check errors above.")
    else:
        # Example of loading one processed file to confirm
        print(f"\nLoading a sample feature engineered file to confirm (e.g., {processed_feature_paths[0].name}):")
        sample_loaded_df = pd.read_parquet(processed_feature_paths[0])
        print("Sample loaded DataFrame head:")
        print(sample_loaded_df.head())
        print("Sample loaded DataFrame info:")
        sample_loaded_df.info()
        print("\nSample columns for feature engineering:")
        print([col for col in sample_loaded_df.columns if 'lag' in col or 'rolling' in col][:10]) # Show some engineered columns


if __name__ == "__main__":
    run_feature_engineering()