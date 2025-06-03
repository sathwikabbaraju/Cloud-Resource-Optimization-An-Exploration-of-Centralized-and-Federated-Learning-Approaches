import numpy as np
import pickle
from pathlib import Path
import sys
import os

# --- Configuration ---
pytorch_data_dir = Path("/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/pytorch_client_data")
global_data_output_path = Path("./global_consolidated_beta_data.pkl")

# --- IMPORTANT: ADJUST THIS NUMBER BASED ON YOUR RAM ---
NUM_CLIENTS_TO_CONSOLIDATE = 2 # Start with 2, if it works, try 3, 4, etc.

# --- Data Consolidation Function ---
def consolidate_prepared_data(pytorch_data_directory: Path, output_path: Path):
    print(f"Starting consolidation of prepared data from {pytorch_data_directory}...")

    client_dirs = sorted([d for d in pytorch_data_directory.iterdir() if d.is_dir() and d.name.startswith('client_')])

    if not client_dirs:
        print(f"No client data subdirectories found in {pytorch_data_directory}. Please run data preparation (prepare_data_for_fl_npy.py) first.")
        sys.exit(1)

    # Limit to a subset of clients
    if len(client_dirs) > NUM_CLIENTS_TO_CONSOLIDATE:
        print(f"  Warning: Limiting consolidation to the first {NUM_CLIENTS_TO_CONSOLIDATE} clients out of {len(client_dirs)}.")
        client_dirs = client_dirs[:NUM_CLIENTS_TO_CONSOLIDATE]
    else:
        print(f"  Consolidating all {len(client_dirs)} available clients.")


    all_X_train = []
    all_Y_train = []
    all_X_val = []
    all_Y_val = []
    all_X_test = []
    all_Y_test = []

    feature_names = None
    target_names = None
    sequence_length = None
    prediction_horizon = None

    for client_dir in client_dirs:
        client_id = client_dir.name.replace("client_", "")
        print(f"  Loading data for client: {client_id}")
        
        try:
            # Load metadata
            with open(client_dir / "metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
            
            # Verify consistent metadata across clients
            if feature_names is None:
                feature_names = metadata['feature_columns']
                target_names = metadata['target_columns']
                sequence_length = metadata['sequence_length']
                prediction_horizon = metadata['prediction_horizon']
            else:
                if feature_names != metadata['feature_columns'] or target_names != metadata['target_columns']:
                    print(f"    Warning: Metadata mismatch for client {client_id}. Features/targets might not be consistent. Using first client's metadata.")

            # Load .npy files (using mmap_mode='r' to prevent loading everything into RAM immediately
            # when handling individual client files, though concatenate will load fully)
            X_train = np.load(client_dir / "train_sequences.npy", mmap_mode='r')
            Y_train = np.load(client_dir / "train_targets.npy", mmap_mode='r')
            X_val = np.load(client_dir / "val_sequences.npy", mmap_mode='r')
            Y_val = np.load(client_dir / "val_targets.npy", mmap_mode='r')
            X_test = np.load(client_dir / "test_sequences.npy", mmap_mode='r')
            Y_test = np.load(client_dir / "test_targets.npy", mmap_mode='r')

            all_X_train.append(X_train)
            all_Y_train.append(Y_train)
            all_X_val.append(X_val)
            all_Y_val.append(Y_val)
            all_X_test.append(X_test)
            all_Y_test.append(Y_test)

            print(f"    Loaded X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")

        except Exception as e:
            print(f"  Error loading data for client {client_id} from {client_dir}: {e}. Skipping this client.")
            continue

    if not all_X_train:
        print("No valid client data was loaded for consolidation. Exiting.")
        sys.exit(1)

    print("\nConcatenating all client data...")
    # Concatenate along the first axis (samples)
    X_train_global = np.concatenate(all_X_train, axis=0)
    Y_train_global = np.concatenate(all_Y_train, axis=0)
    X_val_global = np.concatenate(all_X_val, axis=0)
    Y_val_global = np.concatenate(all_Y_val, axis=0)
    X_test_global = np.concatenate(all_X_test, axis=0)
    Y_test_global = np.concatenate(all_Y_test, axis=0)

    print(f"  Global X_train shape: {X_train_global.shape}, Y_train shape: {Y_train_global.shape}")
    print(f"  Global X_val shape: {X_val_global.shape}, Y_val shape: {Y_val_global.shape}")
    print(f"  Global X_test shape: {X_test_global.shape}, Y_test shape: {Y_test_global.shape}")

    total_memory_gb = (X_train_global.nbytes + Y_train_global.nbytes +
                       X_val_global.nbytes + Y_val_global.nbytes +
                       X_test_global.nbytes + Y_test_global.nbytes) / (1024**3)
    print(f"  Total memory for global data (approx): {total_memory_gb:.2f} GB")

    global_data = {
        'X_train': X_train_global, 'Y_train': Y_train_global,
        'X_val': X_val_global, 'Y_val': Y_val_global,
        'X_test': X_test_global, 'Y_test': Y_test_global,
        'feature_names': feature_names,
        'target_names': target_names,
        'sequence_length': sequence_length,
        'prediction_horizon': prediction_horizon
    }

    try:
        with open(output_path, 'wb') as f:
            pickle.dump(global_data, f)
        print(f"\nSuccessfully consolidated all data to {output_path}")
    except Exception as e:
        print(f"Error saving global data to {output_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    consolidate_prepared_data(pytorch_data_dir, global_data_output_path)
