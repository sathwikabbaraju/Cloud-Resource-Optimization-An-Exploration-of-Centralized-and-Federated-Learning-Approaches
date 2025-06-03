import pandas as pd
from pathlib import Path
import numpy as np
import pickle # To load the scaler

# --- Configuration ---
#feature_engineered_data_dir = Path("./feature_engineered_client_data")
feature_engineered_data_dir = Path("./feature_engineered_beta_client_data") # CHANGED!
# Directory to save the final processed PyTorch-ready data (e.g., numpy arrays or tensors)
# This is where the .npy files will go, organized by client
pytorch_data_dir = Path("./pytorch_client_beta_data") # Renamed from model_data_dir for clarity here
pytorch_data_dir.mkdir(parents=True, exist_ok=True)

# Define the target variables we want to predict
TARGET_COLUMNS = ['CPUUtilization', 'RSS', 'ReadMB', 'WriteMB']

SEQUENCE_LENGTH = 3 
PREDICTION_HORIZON = 1 

TRAIN_RATIO = 0.7 
VAL_RATIO = 0.15 
TEST_RATIO = 0.15 

# --- Data Preparation Function for a Single Client ---
def prepare_client_data(client_feature_path: Path):
    client_id = client_feature_path.stem.replace("_features", "").replace("client_", "")
    print(f"\nPreparing data for client: {client_id}")

    try:
        df = pd.read_parquet(client_feature_path)
    except Exception as e:
        print(f"  Error reading {client_feature_path.name}: {e}. Skipping client.")
        return None, None # Return None for both output paths

    if df.empty:
        print(f"  DataFrame for client {client_id} is empty after feature engineering. Skipping.")
        return None, None

    # --- Identify Input Features (X) and Target Variables (Y) ---
    if 'Node' in df.columns:
        df = df.drop(columns=['Node'])

    # Ensure that 'Step' and 'Series' are also removed if they were included as numerical features
    # They should be removed from feature_cols unless we intend to embed them.
    # For simplicity, let's remove them from features as they're categorical/identifier.
    feature_cols = [col for col in df.columns if col not in TARGET_COLUMNS + ['Step', 'Series']]

    X_df = df[feature_cols] 
    Y_df = df[TARGET_COLUMNS] 

    X_raw = X_df.values
    Y_raw = Y_df.values

    # Check for NaNs after selecting X and Y
    if np.isnan(X_raw).any() or np.isinf(X_raw).any():
        print(f"  Warning: NaNs or Infs found in raw features for client {client_id}. Dropping rows.")
        nan_rows = np.isnan(X_raw).any(axis=1) | np.isinf(X_raw).any(axis=1)
        X_raw = X_raw[~nan_rows]
        Y_raw = Y_raw[~nan_rows]
        if X_raw.shape[0] == 0:
            print(f"  No valid raw data left after dropping NaNs/Infs for client {client_id}. Skipping.")
            return None, None

    # --- Create Sequences (for Time Series Models like LSTM/GRU) ---
    data_sequences = []
    target_sequences = []

    # Assuming df is sorted by EpochTime index and contains a single continuous series
    # If there are multiple (Node, Series) combinations within a client, you MUST group here
    # and create sequences independently for each sub_df.
    # For current purpose, since data_engineer.py groups by Node/Series, and then concatenates
    # and drops NaNs, it's possible some clients' data are not perfectly continuous.
    # A more robust sequence generation would group by (Node, Series) again and then generate sequences.
    # For now, let's proceed with the existing structure, assuming the concatenated DataFrame behaves.

    for i in range(len(X_raw) - SEQUENCE_LENGTH - PREDICTION_HORIZON + 1):
        x_seq = X_raw[i : i + SEQUENCE_LENGTH]
        y_seq = Y_raw[i + SEQUENCE_LENGTH + PREDICTION_HORIZON - 1] # Target is a single point

        # Check if sequence or target contains NaNs/Infs BEFORE appending
        if not (np.isnan(x_seq).any() or np.isinf(x_seq).any() or
                np.isnan(y_seq).any() or np.isinf(y_seq).any()):
            data_sequences.append(x_seq)
            target_sequences.append(y_seq)

    if not data_sequences:
        print(f"  No valid sequences could be generated for client {client_id}. Skipping.")
        return None, None

    X_seq = np.array(data_sequences, dtype=np.float32) 
    # Y_seq should have shape (num_samples, 1, output_size) to match model output
    Y_seq = np.array(target_sequences, dtype=np.float32).reshape(-1, 1, len(TARGET_COLUMNS)) 

    print(f"  Created sequences. X_seq shape: {X_seq.shape}, Y_seq shape: {Y_seq.shape}")

    # --- 3. Chronological Train-Validation-Test Split ---
    total_samples = X_seq.shape[0]
    train_end = int(total_samples * TRAIN_RATIO)
    val_end = int(total_samples * (TRAIN_RATIO + VAL_RATIO))

    X_train, Y_train = X_seq[:train_end], Y_seq[:train_end]
    X_val, Y_val = X_seq[train_end:val_end], Y_seq[train_end:val_end]
    X_test, Y_test = X_seq[val_end:], Y_seq[val_end:]

    print(f"  Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
    if X_train.shape[0] == 0 or X_val.shape[0] == 0 or X_test.shape[0] == 0:
        print(f"  Warning: One or more splits are empty for client {client_id}. Skipping.")
        return None, None

    # Store prepared data for client in separate .npy files
    output_client_dir = pytorch_data_dir / f"client_{client_id}"
    output_client_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_client_dir / "train_sequences.npy", X_train)
    np.save(output_client_dir / "train_targets.npy", Y_train)
    np.save(output_client_dir / "val_sequences.npy", X_val)
    np.save(output_client_dir / "val_targets.npy", Y_val)
    np.save(output_client_dir / "test_sequences.npy", X_test)
    np.save(output_client_dir / "test_targets.npy", Y_test)

    # Save metadata for this client
    with open(output_client_dir / "metadata.pkl", 'wb') as f:
        pickle.dump({
            'feature_columns': feature_cols,
            'target_columns': TARGET_COLUMNS,
            'sequence_length': SEQUENCE_LENGTH,
            'prediction_horizon': PREDICTION_HORIZON
        }, f)

    print(f"  Successfully prepared and saved PyTorch-ready data for client {client_id} to {output_client_dir}")
    return output_client_dir, output_client_dir # Returning the same path for both train/test directories for now

# --- Main Data Preparation Loop ---
def run_data_preparation():
    print("Starting data preparation for modeling for all clients...")
    feature_engineered_files = sorted(list(feature_engineered_data_dir.glob("client_*_features.parquet")))

    if not feature_engineered_files:
        print(f"No feature engineered data found in {feature_engineered_data_dir}. Please run feature engineering first.")
        return

    prepared_client_dirs = []
    for file_path in feature_engineered_files:
        client_dir, _ = prepare_client_data(file_path)
        if client_dir:
            prepared_client_dirs.append(client_dir)

    print(f"\nFinished data preparation. Successfully prepared {len(prepared_client_dirs)} clients.")
    print(f"PyTorch-ready data saved to: {pytorch_data_dir}")

    if not prepared_client_dirs:
        print("No PyTorch-ready data was successfully processed. Please check errors above.")
    else:
        # Example of loading one prepared client's data to confirm
        print(f"\nLoading a sample prepared client's data to confirm (e.g., from {processed_client_dirs[0].name}):")
        sample_client_dir = processed_client_dirs[0]
        sample_X_train = np.load(sample_client_dir / "train_sequences.npy", mmap_mode='r')
        sample_Y_train = np.load(sample_client_dir / "train_targets.npy", mmap_mode='r')

        print("Sample loaded data shapes (from .npy memmap):")
        print(f"X_train: {sample_X_train.shape}")
        print(f"Y_train: {sample_Y_train.shape}")

        with open(sample_client_dir / "metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        print("Sample metadata:", metadata)

if __name__ == "__main__":
    run_data_preparation()