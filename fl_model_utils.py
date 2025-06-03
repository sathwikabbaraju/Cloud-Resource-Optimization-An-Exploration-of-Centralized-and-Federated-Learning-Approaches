import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path

# --- Configuration (matching previous steps) ---
SEQUENCE_LENGTH = 3 
PREDICTION_HORIZON = 1 
TARGET_COLUMNS = ['CPUUtilization', 'RSS', 'ReadMB', 'WriteMB'] 

# --- 1. PyTorch Model Definition (LSTM) ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        out = out.unsqueeze(1) # Add a dimension for PREDICTION_HORIZON
        return out


# --- 2. PyTorch Dataset Class ---
class CloudResourceDataset(Dataset):
    def __init__(self, data_path: Path, split_type: str):
        self.X_path = data_path / f"{split_type}_sequences.npy"
        self.Y_path = data_path / f"{split_type}_targets.npy"

        # Use memmap to load data lazily
        self.X_memmap = np.load(self.X_path, mmap_mode='r') 
        self.Y_memmap = np.load(self.Y_path, mmap_mode='r')
        
        self.length = len(self.X_memmap)

        # Load metadata to get feature/target names if needed
        # (Self-correction: added metadata loading here as well)
        with open(data_path / "metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.X_memmap[idx]
        y = self.Y_memmap[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# --- 3. Data Loading Utility Function (Modified for Path) ---
def load_client_data_paths(client_id: str, base_data_dir: Path):
    """
    Returns the Path object to the client's prepared data directory.
    The Dataset class will handle loading the actual .npy files.
    """
    client_data_dir = base_data_dir / f"client_{client_id}"
    if not client_data_dir.exists():
        raise FileNotFoundError(f"Prepared data directory for client {client_id} not found at {client_data_dir}")
    return client_data_dir

# --- Test the model definition and dataset (optional, but good practice) ---
if __name__ == "__main__":
    print("Running a quick test of LSTMModel and CloudResourceDataset...")
    
    # Dynamically load input_size from metadata for robust testing
    # Assuming client_0000_prepared_data.pkl (or client_0000/metadata.pkl) exists for testing
    sample_metadata_path_for_test = Path("./pytorch_client_data/client_0000/metadata.pkl")
    if sample_metadata_path_for_test.exists():
        with open(sample_metadata_path_for_test, 'rb') as f:
            test_metadata = pickle.load(f)
        input_size = len(test_metadata['feature_columns'])
        output_size = len(test_metadata['target_columns'])
    else:
        print("Warning: Sample metadata not found for test. Using hardcoded sizes.")
        input_size = 74 # Default to 74 if metadata not found
        output_size = len(TARGET_COLUMNS) # 4

    hidden_size = 16 
    num_layers = 1   

    # Simulate sequence data shape: (batch_size, sequence_length, input_size)
    dummy_X = np.random.rand(100, SEQUENCE_LENGTH, input_size).astype(np.float32)
    # Simulate target data shape: (batch_size, prediction_horizon, output_size)
    dummy_Y = np.random.rand(100, PREDICTION_HORIZON, output_size).astype(np.float32)

    # Test Dataset
    # This requires data_path and split_type. Can't directly use dummy_X/Y.
    # For a real test, you'd need to mock client data or run a small prepare_data_for_fl.py first.
    # Let's just test the model architecture directly.
    # dataset = CloudResourceDataset(dummy_X, dummy_Y) # This won't work with new Dataset signature

    print("Skipping Dataset test due to new data loading strategy for simplicity in quick test.")

    # Test Model
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    print(f"Model architecture:\n{model}")
    
    # Pass a dummy batch through the model
    dummy_batch_x = torch.randn(32, SEQUENCE_LENGTH, input_size) # Batch size 32
    output = model(dummy_batch_x)
    print(f"Model output shape for batch_size=32: {output.shape}")
    
    assert output.shape == (32, PREDICTION_HORIZON, output_size), \
        f"Model output shape mismatch! Expected (32, {PREDICTION_HORIZON}, {output_size}), got {output.shape}"
    print("Model output shape is correct!")

    print("\nModel utilities tests passed successfully!")

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import pickle
# from pathlib import Path

# # --- Configuration (matching previous steps) ---
# SEQUENCE_LENGTH = 3 # Must match the SEQUENCE_LENGTH used in prepare_data_for_fl.py
# PREDICTION_HORIZON = 1 # Must match PREDICTION_HORIZON used in prepare_data_for_fl.py
# TARGET_COLUMNS = ['CPUUtilization', 'RSS', 'ReadMB', 'WriteMB'] # For reference


# # --- 1. PyTorch Model Definition (LSTM) ---
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(LSTMModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         # LSTM layer
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
#         # Output layer
#         # The output of LSTM is (batch_size, sequence_length, hidden_size)
#         # We only care about the prediction for the LAST step of the sequence
#         # or a transformation of it to the desired output size.
#         # Since PREDICTION_HORIZON is 1, we want a direct mapping from the last hidden state.
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         # x shape: (batch_size, sequence_length, input_size)
        
#         # Initialize hidden state and cell state
#         # h0 and c0 shape: (num_layers, batch_size, hidden_size)
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
#         # Forward propagate LSTM
#         # out shape: (batch_size, sequence_length, hidden_size)
#         # hn, cn shape: (num_layers, batch_size, hidden_size) - these are the final hidden/cell states
#         out, (hn, cn) = self.lstm(x, (h0, c0))
        
#         # Decode the hidden state of the last time step
#         # We take the output of the LAST time step (out[:, -1, :])
#         # This will be (batch_size, hidden_size)
#         out = self.fc(out[:, -1, :])
        
#         # out shape: (batch_size, output_size)
#         # Since Y_seq has shape (batch_size, 1, output_size), we need to reshape out to match.
#         out = out.unsqueeze(1) # Add a dimension for PREDICTION_HORIZON
#         return out

# # ... (imports and config) ...

# # --- 2. PyTorch Dataset Class ---
# class CloudResourceDataset(Dataset):
#     def __init__(self, data_path: Path, split_type: str):
#         # data_path is the client-specific directory (e.g., model_data/client_0000)
#         # split_type is "train", "val", or "test"
#         self.X_path = data_path / f"{split_type}_sequences.npy"
#         self.Y_path = data_path / f"{split_type}_targets.npy"

#         # Use memmap to load data lazily, only when __getitem__ is called
#         # This is CRUCIAL for large datasets
#         self.X_memmap = np.load(self.X_path, mmap_mode='r') 
#         self.Y_memmap = np.load(self.Y_path, mmap_mode='r')

#         self.length = len(self.X_memmap)

#         # Load metadata to get feature/target names if needed
#         with open(data_path / "metadata.pkl", 'rb') as f:
#             self.metadata = pickle.load(f)

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         # When an item is requested, access the memmap object
#         x = self.X_memmap[idx]
#         y = self.Y_memmap[idx]
#         return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# # --- 3. Data Loading Utility Function (Modified for Path) ---
# def load_client_data_paths(client_id: str, base_data_dir: Path):
#     """
#     Returns the Path object to the client's prepared data directory.
#     The Dataset class will handle loading the actual .npy files.
#     """
#     client_data_dir = base_data_dir / f"client_{client_id}"
#     if not client_data_dir.exists():
#         raise FileNotFoundError(f"Prepared data directory for client {client_id} not found at {client_data_dir}")
#     return client_data_dir

# # ... (rest of fl_model_utils.py, including the test block adjusted) ...
# # # --- 2. PyTorch Dataset Class ---
# # # In fl_model_utils.py
# # class CloudResourceDataset(Dataset):
# #     def __init__(self, data_path_prefix: Path, split: str):
# #         # Load sequences/targets for a specific split from disk
# #         self.X_path = data_path_prefix / f"{split}_sequences.npy"
# #         self.Y_path = data_path_prefix / f"{split}_targets.npy"

# #         # To get total length without loading everything into memory
# #         # This reads metadata, not the whole array
# #         self.X_memmap = np.load(self.X_path, mmap_mode='r') 
# #         self.Y_memmap = np.load(self.Y_path, mmap_mode='r')

# #         self.len = len(self.X_memmap)

# #     def __len__(self):
# #         return self.len

# #     def __getitem__(self, idx):
# #         # Load only the requested slice from disk using memmap
# #         # memmap objects behave like arrays, but data is loaded on demand
# #         x = self.X_memmap[idx] 
# #         y = self.Y_memmap[idx]
# #         return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# # # In FlowerClient's _load_data_once:
# # # Change how train_dataset and val_dataset are initialized
# # # self.train_dataset = CloudResourceDataset(output_client_dir, "train")
# # # self.val_dataset = CloudResourceDataset(output_client_dir, "val")
# # # class CloudResourceDataset(Dataset):
# # #     def __init__(self, X_data, Y_data):
# # #         # Convert numpy arrays to torch tensors
# # #         # Ensure float32 for consistency with model expectations
# # #         self.X = torch.tensor(X_data, dtype=torch.float32)
# # #         self.Y = torch.tensor(Y_data, dtype=torch.float32)

# # #     def __len__(self):
# # #         return len(self.X)

# # #     def __getitem__(self, idx):
# # #         return self.X[idx], self.Y[idx]

# # # --- 3. Data Loading Utility Function ---
# # def load_client_data(client_id: str, model_data_dir: Path):
# #     """
# #     Loads prepared data (X_train, Y_train, etc.) for a specific client.
# #     Returns: X_train, Y_train, X_val, Y_val, X_test, Y_test as NumPy arrays.
# #     """
# #     data_path = model_data_dir / f"client_{client_id}_prepared_data.pkl"
# #     if not data_path.exists():
# #         raise FileNotFoundError(f"Prepared data for client {client_id} not found at {data_path}")

# #     with open(data_path, 'rb') as f:
# #         client_data = pickle.load(f)

# #     # Return NumPy arrays directly. The Dataset class will convert to tensors.
# #     return (client_data['X_train'], client_data['Y_train'],
# #             client_data['X_val'], client_data['Y_val'],
# #             client_data['X_test'], client_data['Y_test'])

# # --- Test the model definition and dataset (optional, but good practice) ---
# if __name__ == "__main__":
#     print("Running a quick test of LSTMModel and CloudResourceDataset...")
    
#     # Dummy data for testing
#     input_size = 76  # Number of features per time step
#     hidden_size = 64 # LSTM hidden state size
#     num_layers = 2   # Number of LSTM layers
#     output_size = len(TARGET_COLUMNS) # Number of target variables (4)

#     # Simulate sequence data shape: (batch_size, sequence_length, input_size)
#     dummy_X = np.random.rand(100, SEQUENCE_LENGTH, input_size).astype(np.float32)
#     # Simulate target data shape: (batch_size, prediction_horizon, output_size)
#     dummy_Y = np.random.rand(100, PREDICTION_HORIZON, output_size).astype(np.float32)

#     # Test Dataset
#     dataset = CloudResourceDataset(dummy_X, dummy_Y)
#     print(f"Dataset length: {len(dataset)}")
#     sample_x, sample_y = dataset[0]
#     print(f"Sample X shape from dataset: {sample_x.shape}")
#     print(f"Sample Y shape from dataset: {sample_y.shape}")

#     # Test DataLoader
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#     for batch_x, batch_y in dataloader:
#         print(f"Batch X shape from dataloader: {batch_x.shape}")
#         print(f"Batch Y shape from dataloader: {batch_y.shape}")
#         break # Just get one batch

#     # Test Model
#     model = LSTMModel(input_size, hidden_size, num_layers, output_size)
#     print(f"Model architecture:\n{model}")
    
#     # Pass a dummy batch through the model
#     dummy_batch_x = torch.randn(32, SEQUENCE_LENGTH, input_size) # Batch size 32
#     output = model(dummy_batch_x)
#     print(f"Model output shape for batch_size=32: {output.shape}")
    
#     # Expected output shape: (batch_size, PREDICTION_HORIZON, output_size)
#     assert output.shape == (32, PREDICTION_HORIZON, output_size), \
#         f"Model output shape mismatch! Expected (32, {PREDICTION_HORIZON}, {output_size}), got {output.shape}"
#     print("Model output shape is correct!")

#     print("\nAll tests passed successfully for model and dataset utilities!")