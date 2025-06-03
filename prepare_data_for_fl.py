import pandas as pd
from pathlib import Path
import numpy as np
import pickle # To load the scaler
import sys # For getting object size

# --- Configuration ---
feature_engineered_data_dir = Path("./feature_engineered_client_data")
model_data_dir = Path("./model_data") # Directory to save prepared data for modeling
model_data_dir.mkdir(parents=True, exist_ok=True)

# Define the target variables we want to predict
TARGET_COLUMNS = ['CPUUtilization', 'RSS', 'ReadMB', 'WriteMB']

# --- ADJUSTMENT HERE: REDUCED SEQUENCE_LENGTH ---
# Previous: 10
# Try a much smaller value to prevent OOM errors.
# A sequence length of 3-5 is often a good starting point for time series.
SEQUENCE_LENGTH = 3 # Use past 3 time steps (30 seconds) to predict next
PREDICTION_HORIZON = 1 # Predict 1 time step into the future (10 seconds ahead)

# Split ratios (chronological split)
TRAIN_RATIO = 0.7  # 70% for training
VAL_RATIO = 0.15   # 15% for validation
TEST_RATIO = 0.15  # 15% for testing (should sum to 1 with TRAIN_RATIO)

# --- Data Preparation Function for a Single Client ---
def prepare_client_data(client_feature_path: Path):
    client_id = client_feature_path.stem.replace("_features", "").replace("client_", "")
    print(f"\nPreparing data for client: {client_id}")

    try:
        df = pd.read_parquet(client_feature_path)
    except Exception as e:
        print(f"  Error reading {client_feature_path.name}: {e}. Skipping client.")
        return None

    if df.empty:
        print(f"  DataFrame for client {client_id} is empty after feature engineering. Skipping.")
        return None

    # --- 1. Identify Input Features (X) and Target Variables (Y) ---
    # Drop 'Node' as it's a string and would require one-hot encoding, adding complexity for initial models
    if 'Node' in df.columns:
        df = df.drop(columns=['Node'])
    
    feature_cols = [col for col in df.columns if col not in TARGET_COLUMNS]

    X_df = df[feature_cols] # Keep as DataFrame for now
    Y_df = df[TARGET_COLUMNS] # Keep as DataFrame for now

    # Convert to NumPy array only when needed for sequence creation
    X_raw = X_df.values
    Y_raw = Y_df.values

    # Check for NaNs after selecting X and Y (should be handled by previous steps, but good check)
    if np.isnan(X_raw).any() or np.isinf(X_raw).any():
        print(f"  Warning: NaNs or Infs found in raw features for client {client_id}. Dropping rows.")
        nan_rows = np.isnan(X_raw).any(axis=1) | np.isinf(X_raw).any(axis=1)
        X_raw = X_raw[~nan_rows]
        Y_raw = Y_raw[~nan_rows]
        if X_raw.shape[0] == 0:
            print(f"  No valid raw data left after dropping NaNs/Infs for client {client_id}. Skipping.")
            return None

    # --- 2. Create Sequences (for Time Series Models like LSTM/GRU) ---
    data_sequences = []
    target_sequences = []

    for i in range(len(X_raw) - SEQUENCE_LENGTH - PREDICTION_HORIZON + 1):
        x_seq = X_raw[i : i + SEQUENCE_LENGTH]
        y_seq = Y_raw[i + SEQUENCE_LENGTH : i + SEQUENCE_LENGTH + PREDICTION_HORIZON]

        if not (np.isnan(x_seq).any() or np.isinf(x_seq).any() or
                np.isnan(y_seq).any() or np.isinf(y_seq).any()):
            data_sequences.append(x_seq)
            target_sequences.append(y_seq)
    
    if not data_sequences:
        print(f"  No valid sequences could be generated for client {client_id}. Skipping.")
        return None

    X_seq = np.array(data_sequences, dtype=np.float32) # Specify dtype to save memory
    Y_seq = np.array(target_sequences, dtype=np.float32) # Specify dtype to save memory

    print(f"  Created sequences. X_seq shape: {X_seq.shape}, Y_seq shape: {Y_seq.shape}")
    print(f"  X_seq size (approx): {sys.getsizeof(X_seq) / (1024**3):.2f} GB") # Monitor memory
    print(f"  Y_seq size (approx): {sys.getsizeof(Y_seq) / (1024**3):.2f} GB") # Monitor memory


    # --- 3. Chronological Train-Validation-Test Split ---
    total_samples = X_seq.shape[0]
    train_end = int(total_samples * TRAIN_RATIO)
    val_end = int(total_samples * (TRAIN_RATIO + VAL_RATIO))

    X_train, Y_train = X_seq[:train_end], Y_seq[:train_end]
    X_val, Y_val = X_seq[train_end:val_end], Y_seq[train_end:val_end]
    X_test, Y_test = X_seq[val_end:], Y_seq[val_end:]

    print(f"  Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")

    # Store prepared data for client
    client_data = {
        'X_train': X_train, 'Y_train': Y_train,
        'X_val': X_val, 'Y_val': Y_val,
        'X_test': X_test, 'Y_test': Y_test,
        'feature_names': feature_cols,
        'target_names': TARGET_COLUMNS
    }

    output_pickle_path = model_data_dir / f"client_{client_id}_prepared_data.pkl"
    try:
        with open(output_pickle_path, 'wb') as f:
            pickle.dump(client_data, f)
        print(f"  Successfully prepared and saved data for client {client_id} to {output_pickle_path}")
    except Exception as e:
        print(f"  Error saving prepared data for client {client_id}: {e}")
        return None

    return output_pickle_path

# --- Main Data Preparation Loop ---
def run_data_preparation():
    print("Starting data preparation for modeling for all clients...")
    client_feature_files = sorted(list(feature_engineered_data_dir.glob("client_*_features.parquet")))

    if not client_feature_files:
        print(f"No feature engineered data found in {feature_engineered_data_dir}. Please run feature engineering first.")
        return

    prepared_client_paths = []
    # --- OPTIONAL: LIMIT CLIENTS FOR TESTING ---
    # Uncomment the following line if you want to test with only a few clients first
    # For example, to process only the first 2 clients:
    # client_feature_files = client_feature_files[:2]

    for file_path in client_feature_files:
        prepared_path = prepare_client_data(file_path)
        if prepared_path:
            prepared_client_paths.append(prepared_path)

    print(f"\nFinished data preparation. Successfully prepared {len(prepared_client_paths)} clients.")
    print(f"Prepared data saved to: {model_data_dir}")

    if not prepared_client_paths:
        print("No data was successfully prepared. Please check errors above.")
    else:
        # Example of loading one prepared file to confirm
        print(f"\nLoading a sample prepared file to confirm (e.g., {prepared_client_paths[0].name}):")
        with open(prepared_client_paths[0], 'rb') as f:
            sample_loaded_data = pickle.load(f)
        print("Sample loaded data shapes:")
        print(f"X_train: {sample_loaded_data['X_train'].shape}")
        print(f"Y_train: {sample_loaded_data['Y_train'].shape}")
        print(f"X_val: {sample_loaded_data['X_val'].shape}")
        print(f"Y_val: {sample_loaded_data['Y_val'].shape}")
        print(f"X_test: {sample_loaded_data['X_test'].shape}")
        print(f"Y_test: {sample_loaded_data['Y_test'].shape}")
        print(f"Feature names (first 5): {sample_loaded_data['feature_names'][:5]}")
        print(f"Target names: {sample_loaded_data['target_names']}")


if __name__ == "__main__":
    run_data_preparation()