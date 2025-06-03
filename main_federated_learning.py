import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path
import numpy as np
import ray # Import ray
import pickle

# --- Load GLOBAL_INPUT_SIZE and GLOBAL_OUTPUT_SIZE dynamically from a sample client's metadata ---
sample_metadata_path = Path("./pytorch_client_data/client_0000/metadata.pkl")
if not sample_metadata_path.exists():
    raise FileNotFoundError(f"Sample metadata for client 0000 not found at {sample_metadata_path}. Please run prepare_data_for_fl.py first.")
with open(sample_metadata_path, 'rb') as f:
    sample_metadata = pickle.load(f)

GLOBAL_INPUT_SIZE = len(sample_metadata['feature_columns'])
GLOBAL_OUTPUT_SIZE = len(sample_metadata['target_columns'])

# Import the modified load_client_data_paths
from fl_model_utils import LSTMModel, CloudResourceDataset, load_client_data_paths, SEQUENCE_LENGTH, PREDICTION_HORIZON, TARGET_COLUMNS

# --- Configuration for Federated Learning ---
NUM_CLIENTS = 12 

# --- AGGRESSIVE ADJUSTMENTS FOR MEMORY REDUCTION ---
BATCH_SIZE = 1 # Already drastically reduced batch size
LOCAL_EPOCHS = 1 # Already reduced local epochs
LEARNING_RATE = 0.001

# Flower server strategy parameters
NUM_ROUNDS = 3 # Reduced number of rounds for quick testing
FRACTION_FIT = 0.1 # Try to select only 1-2 clients (0.1 * 12 = 1.2)
MIN_FIT_CLIENTS = 1 # CRITICAL: Only require 1 client to start training in a round
MIN_AVAILABLE_CLIENTS = NUM_CLIENTS # All clients must still be visible to the server for sampling

# Define paths
pytorch_data_dir = Path("./pytorch_client_data") 
checkpoint_dir = Path("./checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# --- 1. Flower Client Implementation (Deferred Data Loading with memmap) ---
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: str):
        self.client_id = client_id
        print(f"Client {self.client_id}: Initializing FlowerClient object.")

        # Use the dynamically loaded GLOBAL_INPUT_SIZE here
        input_size = GLOBAL_INPUT_SIZE 
        hidden_size = 16 
        num_layers = 1   
        output_size = GLOBAL_OUTPUT_SIZE # Use the dynamically loaded GLOBAL_OUTPUT_SIZE
        self.model = LSTMModel(input_size, hidden_size, num_layers, output_size)

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print(f"Client {self.client_id}: Using MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Client {self.client_id}: Using CUDA (NVIDIA GPU)")
        else:
            self.device = torch.device("cpu")
            print(f"Client {self.client_id}: Using CPU")
        self.model.to(self.device)
        print(f"Client {self.client_id}: Model initialized on {self.device}.")

        self.client_data_path = load_client_data_paths(self.client_id, pytorch_data_dir) 

        self.train_loader = None
        self.val_loader = None
        self.train_dataset_size = 0
        self.val_dataset_size = 0

    def _load_data_once(self):
        """Helper to create data loaders from memmap datasets."""
        if self.train_loader is None: 
            print(f"Client {self.client_id}: Creating data loaders from memmap.")
            # CloudResourceDataset now directly takes the Path and split_type
            self.train_dataset = CloudResourceDataset(self.client_data_path, "train")
            self.val_dataset = CloudResourceDataset(self.client_data_path, "val")

            self.train_dataset_size = len(self.train_dataset)
            self.val_dataset_size = len(self.val_dataset)

            self.train_loader = DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=0)
            self.val_loader = DataLoader(self.val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=0)
            
            print(f"Client {self.client_id}: Data loaders created (Train: {self.train_dataset_size} samples, Val: {self.val_dataset_size} samples).")
            # REMOVED: del X_train, Y_train, X_val, Y_val as they are no longer assigned here

    def get_parameters(self, config: Dict):
        """Returns the current local model parameters as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Sets the local model parameters received from the server."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v).to(self.device) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict):
        """Trains the local model on the client's training data."""
        self.set_parameters(parameters)
        self._load_data_once() # Ensure data is loaded

        print(f"Client {self.client_id}: Starting local training for {config['local_epochs']} epochs.")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(config["local_epochs"]):
            running_loss = 0.0
            for X_batch, Y_batch in self.train_loader:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * X_batch.size(0)

                del X_batch, Y_batch, outputs, loss
                if self.device.type != 'cpu':
                    if self.device.type == 'mps':
                        torch.mps.empty_cache()
                    elif torch.cuda.is_available(): 
                        torch.cuda.empty_cache()

            epoch_loss = running_loss / self.train_dataset_size
            print(f"Client {self.client_id}: Epoch {epoch+1}/{config['local_epochs']}, Loss: {epoch_loss:.4f}")
        
        print(f"Client {self.client_id}: Local training finished. Cache cleared.")
        
        return self.get_parameters(config={}), self.train_dataset_size, {"local_loss": epoch_loss}

    def evaluate(self, parameters: List[np.ndarray], config: Dict):
        """Evaluates the local model on the client's validation data."""
        self.set_parameters(parameters)
        self._load_data_once() # Ensure data is loaded

        print(f"Client {self.client_id}: Starting local evaluation.")

        criterion = nn.MSELoss()
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in self.val_loader: # Use val_loader for evaluation
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = criterion(outputs, Y_batch)
                total_loss += loss.item() * X_batch.size(0)
                
                del X_batch, Y_batch, outputs, loss
                if self.device.type != 'cpu':
                    if self.device.type == 'mps':
                        torch.mps.empty_cache()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()

        val_loss = total_loss / self.val_dataset_size
        
        print(f"Client {self.client_id}: Local evaluation finished. Val Loss: {val_loss:.4f}. Cache cleared.")

        return val_loss, self.val_dataset_size, {"val_loss": val_loss}


# --- 2. Flower Server Strategy and Client Management ---

from flwr.client import Client 
def client_fn(cid: str) -> Client: 
    """Returns a FlowerClient instance for a given client ID."""
    formatted_cid = f"{int(cid):04d}"
    return FlowerClient(formatted_cid).to_client()

def get_evaluate_fn(model: nn.Module):
    server_device = torch.device("cpu") # Server usually runs on CPU
    model.to(server_device) 

    all_test_data_dirs = [] 
    for i in range(NUM_CLIENTS):
        client_id_str = f"{i:04d}"
        path = pytorch_data_dir / f"client_{client_id_str}" 
        if path.exists():
            all_test_data_dirs.append(path)
        else:
            print(f"Warning: Test data directory for client {client_id_str} not found for server evaluation at {path}.")

    if not all_test_data_dirs:
        print("Server: No global test data directories found. Server evaluation will not run.")
        return lambda server_round, parameters, config: (0.0, {})

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Tuple[float, Dict[str, fl.common.Scalar]]:
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v).to(server_device) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        criterion = nn.MSELoss()
        model.eval()
        total_loss = 0.0
        num_samples = 0
        
        print(f"Server: Starting evaluation for round {server_round} on {len(all_test_data_dirs)} client test sets.")

        for client_data_dir_path in all_test_data_dirs: 
            try:
                test_dataset = CloudResourceDataset(client_data_dir_path, "test")
                print(f"Server Evaluation: Loaded test data from {client_data_dir_path.name} ({len(test_dataset)} samples).")

                if len(test_dataset) == 0:
                    print(f"Server Evaluation: Skipping {client_data_dir_path.name} due to no samples.")
                    continue

                test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=0)

                with torch.no_grad():
                    for X_batch, Y_batch in test_loader:
                        X_batch, Y_batch = X_batch.to(server_device), Y_batch.to(server_device)
                        outputs = model(X_batch)
                        loss = criterion(outputs, Y_batch)
                        total_loss += loss.item() * X_batch.size(0)
                        num_samples += X_batch.size(0)
                        
                        del X_batch, Y_batch, outputs, loss
                
                import gc; gc.collect() 
                
                print(f"Server Evaluation: Finished with {client_data_dir_path.name}. Cache cleared.")

            except Exception as e:
                print(f"Server: Error during evaluation of {client_data_dir_path.name}: {e}. Skipping this client's test data.")
                continue 

        if num_samples == 0:
            print("Server: No samples evaluated during server-side evaluation.")
            return 0.0, {"server_loss": 0.0}

        server_loss = total_loss / num_samples
        print(f"Server: Round {server_round} validation loss: {server_loss:.4f}")
        return server_loss, {"server_loss": server_loss}

    return evaluate

def fit_config(server_round: int) -> Dict[str, fl.common.Scalar]:
    return {"local_epochs": LOCAL_EPOCHS}

def evaluate_config(server_round: int) -> Dict[str, fl.common.Scalar]:
    return {"local_epochs": LOCAL_EPOCHS}

class FedAvgWithCheckpointing(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        print(f"Server: Aggregating fit results for round {server_round}. {len(results)} clients returned results.")
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            ndarrays_parameters = fl.common.parameters_to_ndarrays(aggregated_parameters)
            save_global_model_checkpoint(server_round, ndarrays_parameters, {}) 
        
        print(f"Server: Aggregation and checkpointing for round {server_round} completed.")
        return aggregated_parameters, metrics

def save_global_model_checkpoint(server_round: int, parameters: fl.common.NDArrays, client_metrics: Dict) -> None:
    # Use dynamic input_size and output_size
    global_model_to_save = LSTMModel(input_size=GLOBAL_INPUT_SIZE, hidden_size=16, num_layers=1, output_size=GLOBAL_OUTPUT_SIZE)
    
    params_dict = zip(global_model_to_save.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    global_model_to_save.load_state_dict(state_dict, strict=True)

    checkpoint_path = checkpoint_dir / f"global_model_round_{server_round}.pth"
    torch.save(global_model_to_save.state_dict(), checkpoint_path)
    print(f"Server: Saved global model checkpoint to {checkpoint_path}")

def main():
    ray.init(num_cpus=1, object_store_memory=2_000_000_000, ignore_reinit_error=True) 

    # Use dynamic input_size and output_size
    input_size = GLOBAL_INPUT_SIZE
    output_size = GLOBAL_OUTPUT_SIZE

    global_model = LSTMModel(input_size=input_size, hidden_size=16, num_layers=1, output_size=output_size)

    strategy = FedAvgWithCheckpointing(
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=FRACTION_FIT,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_FIT_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=get_evaluate_fn(global_model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
    )

    print("Starting Flower simulation...")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0}
    )
    print("\nFlower simulation finished.")

    ray.shutdown()


if __name__ == "__main__":
    main()

# import os
# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


# import flwr as fl
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from collections import OrderedDict
# from typing import Dict, List, Tuple, Optional
# import sys
# from pathlib import Path
# import numpy as np
# import os
# import pickle
# import ray # Import ray

# import pickle

# # Load metadata from a sample client to get dynamic input_size
# sample_metadata_path = Path("./pytorch_client_data/client_0000/metadata.pkl")
# if not sample_metadata_path.exists():
#     raise FileNotFoundError(f"Sample metadata for client 0000 not found at {sample_metadata_path}. Please run prepare_data_for_fl.py first.")
# with open(sample_metadata_path, 'rb') as f:
#     sample_metadata = pickle.load(f)

# GLOBAL_INPUT_SIZE = len(sample_metadata['feature_columns'])
# GLOBAL_OUTPUT_SIZE = len(sample_metadata['target_columns'])

# # Import the modified load_client_data_paths
# from fl_model_utils import LSTMModel, CloudResourceDataset, load_client_data_paths, SEQUENCE_LENGTH, PREDICTION_HORIZON, TARGET_COLUMNS

# # # Import the model and dataset utilities we just created
# # from fl_model_utils import LSTMModel, CloudResourceDataset, load_client_data, SEQUENCE_LENGTH, PREDICTION_HORIZON, TARGET_COLUMNS # <-- RESTORED THIS IMPORT

# # --- Configuration for Federated Learning ---
# NUM_CLIENTS = 12 

# # --- AGGRESSIVE ADJUSTMENTS FOR MEMORY REDUCTION ---
# BATCH_SIZE = 1 # DRastically reduced batch size
# LOCAL_EPOCHS = 1 # Reduced local epochs
# LEARNING_RATE = 0.001

# # Flower server strategy parameters
# NUM_ROUNDS = 3 # Reduced number of rounds for quick testing
# FRACTION_FIT = 0.1 # Try to select only 1-2 clients (0.1 * 12 = 1.2)
# MIN_FIT_CLIENTS = 1 # CRITICAL: Only require 1 client to start training in a round
# MIN_AVAILABLE_CLIENTS = NUM_CLIENTS # All clients must still be visible to the server for sampling

# # Define paths
# # Use pytorch_data_dir as defined in prepare_data_for_fl.py
# pytorch_data_dir = Path("./pytorch_client_data") 
# checkpoint_dir = Path("./checkpoints")
# checkpoint_dir.mkdir(parents=True, exist_ok=True)
# # # Define paths
# # model_data_dir = Path("./model_data")
# # checkpoint_dir = Path("./checkpoints")
# # checkpoint_dir.mkdir(parents=True, exist_ok=True)

# # --- 1. Flower Client Implementation (Deferred Data Loading with memmap) ---
# class FlowerClient(fl.client.NumPyClient):
#     def __init__(self, client_id: str):
#         self.client_id = client_id
#         print(f"Client {self.client_id}: Initializing FlowerClient object.")

#         input_size = 76 
#         hidden_size = 16 
#         num_layers = 1   
#         output_size = len(TARGET_COLUMNS) 
#         self.model = LSTMModel(input_size, hidden_size, num_layers, output_size)

#         if torch.backends.mps.is_available():
#             self.device = torch.device("mps")
#             print(f"Client {self.client_id}: Using MPS (Apple Silicon GPU)")
#         elif torch.cuda.is_available():
#             self.device = torch.device("cuda")
#             print(f"Client {self.client_id}: Using CUDA (NVIDIA GPU)")
#         else:
#             self.device = torch.device("cpu")
#             print(f"Client {self.client_id}: Using CPU")
#         self.model.to(self.device)
#         print(f"Client {self.client_id}: Model initialized on {self.device}.")

#         # Store path to client data directory
#         self.client_data_path = load_client_data_paths(self.client_id, pytorch_data_dir) # Use pytorch_data_dir here

#         self.train_loader = None
#         self.val_loader = None
#         self.train_dataset_size = 0
#         self.val_dataset_size = 0

#     def _load_data_once(self):
#         """Helper to create data loaders from memmap datasets."""
#         if self.train_loader is None: 
#             print(f"Client {self.client_id}: Creating data loaders from memmap.")
#             self.train_dataset = CloudResourceDataset(self.client_data_path, "train")
#             self.val_dataset = CloudResourceDataset(self.client_data_path, "val")

#             self.train_dataset_size = len(self.train_dataset)
#             self.val_dataset_size = len(self.val_dataset)

#             # IMPORTANT: Set num_workers=0 and pin_memory=False for DataLoaders
#             self.train_loader = DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=0)
#             self.val_loader = DataLoader(self.val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=0)

#             print(f"Client {self.client_id}: Data loaders created (Train: {self.train_dataset_size} samples, Val: {self.val_dataset_size} samples).")
#             # No need to del large NumPy arrays here as they are not fully loaded in __init__

# # # --- 1. Flower Client Implementation (Deferred Data Loading) ---
# # class FlowerClient(fl.client.NumPyClient):
# #     def __init__(self, client_id: str):
# #         self.client_id = client_id
# #         print(f"Client {self.client_id}: Initializing FlowerClient object.")
        
# #         # Initialize model immediately, but data loaders are created in fit/evaluate
# #         input_size = 76 # Assuming this is consistent
        
# #         # AGGRESSIVE: Reduce model size further if needed
# #         hidden_size = 16 # Previously 32, try 16
# #         num_layers = 1   # Previously 2
        
# #         output_size = len(TARGET_COLUMNS) # 4
# #         self.model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        
# #         if torch.backends.mps.is_available():
# #             self.device = torch.device("mps")
# #             print(f"Client {self.client_id}: Using MPS (Apple Silicon GPU)")
# #         elif torch.cuda.is_available():
# #             self.device = torch.device("cuda")
# #             print(f"Client {self.client_id}: Using CUDA (NVIDIA GPU)")
# #         else:
# #             self.device = torch.device("cpu")
# #             print(f"Client {self.client_id}: Using CPU")
# #         self.model.to(self.device)
# #         print(f"Client {self.client_id}: Model initialized on {self.device}.")

# #         self.train_loader = None
# #         self.val_loader = None
# #         self.train_dataset_size = 0
# #         self.val_dataset_size = 0

# #     def _load_data_once(self):
# #         """Helper to load data only when needed."""
# #         if self.train_loader is None: # Load data only if not already loaded
# #             print(f"Client {self.client_id}: Loading data for the first time...")
# #             X_train, Y_train, X_val, Y_val, _, _ = \
# #                 load_client_data(self.client_id, model_data_dir) # X_test, Y_test are not needed by client's fit/evaluate

# #             self.train_dataset = CloudResourceDataset(X_train, Y_train)
# #             self.val_dataset = CloudResourceDataset(X_val, Y_val)
# #             self.train_dataset_size = len(self.train_dataset)
# #             self.val_dataset_size = len(self.val_dataset)

# #             self.train_loader = DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)
# #             self.val_loader = DataLoader(self.val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False)
            
# #             print(f"Client {self.client_id}: Data loaded (Train: {self.train_dataset_size} samples, Val: {self.val_dataset_size} samples).")
# #             # Explicitly delete original NumPy arrays to free memory if DataLoader copied them
#             del X_train, Y_train, X_val, Y_val
#             # Optionally, trigger Python's garbage collector more frequently
#             import gc; gc.collect()

#     def get_parameters(self, config: Dict):
#         """Returns the current local model parameters as a list of NumPy ndarrays."""
#         return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

#     def set_parameters(self, parameters: List[np.ndarray]):
#         """Sets the local model parameters received from the server."""
#         params_dict = zip(self.model.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.tensor(v).to(self.device) for k, v in params_dict})
#         self.model.load_state_dict(state_dict, strict=True)

#     def fit(self, parameters: List[np.ndarray], config: Dict):
#         """Trains the local model on the client's training data."""
#         self.set_parameters(parameters)
#         self._load_data_once() # Ensure data is loaded

#         print(f"Client {self.client_id}: Starting local training for {config['local_epochs']} epochs.")

#         optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
#         criterion = nn.MSELoss()

#         self.model.train()
#         for epoch in range(config["local_epochs"]):
#             running_loss = 0.0
#             for X_batch, Y_batch in self.train_loader:
#                 X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)

#                 optimizer.zero_grad()
#                 outputs = self.model(X_batch)
#                 loss = criterion(outputs, Y_batch)
#                 loss.backward()
#                 optimizer.step()
#                 running_loss += loss.item() * X_batch.size(0)

#                 # AGGRESSIVE: Delete batch-specific tensors immediately
#                 del X_batch, Y_batch, outputs, loss
#                 if self.device.type != 'cpu':
#                     if self.device.type == 'mps':
#                         torch.mps.empty_cache()
#                     elif torch.cuda.is_available(): # Check for CUDA too
#                         torch.cuda.empty_cache()

#             epoch_loss = running_loss / self.train_dataset_size
#             print(f"Client {self.client_id}: Epoch {epoch+1}/{config['local_epochs']}, Loss: {epoch_loss:.4f}")
        
#         print(f"Client {self.client_id}: Local training finished. Cache cleared.")
#         # Optional: Unload data after fit if memory is *extremely* tight,
#         # but this might slow down evaluation if it's called immediately after.
#         # For now, keep it loaded between fit/evaluate calls in the same round.

#         return self.get_parameters(config={}), self.train_dataset_size, {"local_loss": epoch_loss}

#     def evaluate(self, parameters: List[np.ndarray], config: Dict):
#         """Evaluates the local model on the client's validation data."""
#         self.set_parameters(parameters)
#         self._load_data_once() # Ensure data is loaded

#         print(f"Client {self.client_id}: Starting local evaluation.")

#         criterion = nn.MSELoss()
#         self.model.eval()
#         total_loss = 0.0
#         with torch.no_grad():
#             for X_batch, Y_batch in self.val_loader:
#                 X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
#                 outputs = self.model(X_batch)
#                 loss = criterion(outputs, Y_batch)
#                 total_loss += loss.item() * X_batch.size(0)
                
#                 # AGGRESSIVE: Delete batch-specific tensors immediately
#                 del X_batch, Y_batch, outputs, loss
#                 if self.device.type != 'cpu':
#                     if self.device.type == 'mps':
#                         torch.mps.empty_cache()
#                     elif torch.cuda.is_available():
#                         torch.cuda.empty_cache()

#         val_loss = total_loss / self.val_dataset_size
        
#         print(f"Client {self.client_id}: Local evaluation finished. Val Loss: {val_loss:.4f}. Cache cleared.")

#         return val_loss, self.val_dataset_size, {"val_loss": val_loss}


# # --- 2. Flower Server Strategy and Client Management ---

# from flwr.client import Client # Ensure Client is imported
# def client_fn(cid: str) -> Client: # It should take 'cid: str' directly
#     """Returns a FlowerClient instance for a given client ID."""
#     formatted_cid = f"{int(cid):04d}"
#     return FlowerClient(formatted_cid).to_client()

# def get_evaluate_fn(model: nn.Module):
#     server_device = torch.device("cpu") # Server usually runs on CPU
#     model.to(server_device) 

#     all_test_data_dirs = [] # Changed to store directories, not direct data
#     for i in range(NUM_CLIENTS):
#         client_id_str = f"{i:04d}"
#         path = pytorch_data_dir / f"client_{client_id_str}" # Path to client's data directory
#         if path.exists():
#             all_test_data_dirs.append(path)
#         else:
#             print(f"Warning: Test data directory for client {client_id_str} not found for server evaluation at {path}.")

#     if not all_test_data_dirs:
#         print("Server: No global test data directories found. Server evaluation will not run.")
#         return lambda server_round, parameters, config: (0.0, {})

#     def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Tuple[float, Dict[str, fl.common.Scalar]]:
#         params_dict = zip(model.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.tensor(v).to(server_device) for k, v in params_dict})
#         model.load_state_dict(state_dict, strict=True)

#         criterion = nn.MSELoss()
#         model.eval()
#         total_loss = 0.0
#         num_samples = 0

#         print(f"Server: Starting evaluation for round {server_round} on {len(all_test_data_dirs)} client test sets.")

#         for client_data_dir_path in all_test_data_dirs: # Iterate through directories
#             try:
#                 # Load test dataset using the new CloudResourceDataset
#                 test_dataset = CloudResourceDataset(client_data_dir_path, "test")
#                 print(f"Server Evaluation: Loaded test data from {client_data_dir_path.name} ({len(test_dataset)} samples).")

#                 if len(test_dataset) == 0:
#                     print(f"Server Evaluation: Skipping {client_data_dir_path.name} due to no samples.")
#                     continue

#                 # IMPORTANT: Set num_workers=0 and pin_memory=False for DataLoaders
#                 test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=0)

#                 with torch.no_grad():
#                     for X_batch, Y_batch in test_loader:
#                         X_batch, Y_batch = X_batch.to(server_device), Y_batch.to(server_device)
#                         outputs = model(X_batch)
#                         loss = criterion(outputs, Y_batch)
#                         total_loss += loss.item() * X_batch.size(0)
#                         num_samples += X_batch.size(0)

#                         del X_batch, Y_batch, outputs, loss
#                         # No explicit MPS/CUDA empty_cache needed if server_device is CPU

#                 # No need to del X_test, Y_test etc. as they are loaded via memmap
#                 import gc; gc.collect() 

#                 print(f"Server Evaluation: Finished with {client_data_dir_path.name}. Cache cleared.")

#             except Exception as e:
#                 print(f"Server: Error during evaluation of {client_data_dir_path.name}: {e}. Skipping this client's test data.")
#                 continue 

#         if num_samples == 0:
#             print("Server: No samples evaluated during server-side evaluation.")
#             return 0.0, {"server_loss": 0.0}

#         server_loss = total_loss / num_samples
#         print(f"Server: Round {server_round} validation loss: {server_loss:.4f}")
#         return server_loss, {"server_loss": server_loss}

#     return evaluate
# # def get_evaluate_fn(model: nn.Module):
# #     server_device = torch.device("cpu") # Server usually runs on CPU
# #     model.to(server_device) # Move global model to CPU for server eval

# #     test_data_paths = []
# #     for i in range(NUM_CLIENTS):
# #         client_id_str = f"{i:04d}"
# #         path = model_data_dir / f"client_{client_id_str}_prepared_data.pkl"
# #         if path.exists():
# #             test_data_paths.append(path)
# #         else:
# #             print(f"Warning: Test data for client {client_id_str} not found for server evaluation at {path}.")

# #     if not test_data_paths:
# #         print("Server: No global test set could be loaded. Server evaluation will not run.")
# #         return lambda server_round, parameters, config: (0.0, {})

# #     def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Tuple[float, Dict[str, fl.common.Scalar]]:
# #         params_dict = zip(model.state_dict().keys(), parameters)
# #         state_dict = OrderedDict({k: torch.tensor(v).to(server_device) for k, v in params_dict})
# #         model.load_state_dict(state_dict, strict=True)

# #         criterion = nn.MSELoss()
# #         model.eval()
# #         total_loss = 0.0
# #         num_samples = 0
        
# #         print(f"Server: Starting evaluation for round {server_round} on {len(test_data_paths)} client test sets.")

# #         for client_path in test_data_paths:
# #             try:
# #                 with open(client_path, 'rb') as f:
# #                     client_data = pickle.load(f)
# #                 X_test, Y_test = client_data['X_test'], client_data['Y_test']
# #                 print(f"Server Evaluation: Loaded test data from {client_path.name} ({len(X_test)} samples).")

# #                 if X_test.shape[0] == 0:
# #                     print(f"Server Evaluation: Skipping {client_path.name} due to no samples.")
# #                     continue

# #                 test_dataset = CloudResourceDataset(X_test, Y_test)
# #                 test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False)

# #                 with torch.no_grad():
# #                     for X_batch, Y_batch in test_loader:
# #                         X_batch, Y_batch = X_batch.to(server_device), Y_batch.to(server_device)
# #                         outputs = model(X_batch)
# #                         loss = criterion(outputs, Y_batch)
# #                         total_loss += loss.item() * X_batch.size(0)
# #                         num_samples += X_batch.size(0)
                        
# #                         del X_batch, Y_batch, outputs, loss
# #                         if server_device.type != 'cpu':
# #                             if server_device.type == 'mps':
# #                                 torch.mps.empty_cache()
# #                             elif torch.cuda.is_available():
# #                                 torch.cuda.empty_cache()
                
# #                 del X_test, Y_test, test_dataset, test_loader, client_data
# #                 import gc; gc.collect() 
                
# #                 print(f"Server Evaluation: Finished with {client_path.name}. Cache cleared.")

# #             except Exception as e:
# #                 print(f"Server: Error during evaluation of {client_path.name}: {e}. Skipping this client's test data.")
# #                 continue 

# #         if num_samples == 0:
# #             print("Server: No samples evaluated during server-side evaluation.")
# #             return 0.0, {"server_loss": 0.0}

# #         server_loss = total_loss / num_samples
# #         print(f"Server: Round {server_round} validation loss: {server_loss:.4f}")
# #         return server_loss, {"server_loss": server_loss}

# #     return evaluate

# def fit_config(server_round: int) -> Dict[str, fl.common.Scalar]:
#     return {"local_epochs": LOCAL_EPOCHS}

# def evaluate_config(server_round: int) -> Dict[str, fl.common.Scalar]:
#     return {"local_epochs": LOCAL_EPOCHS}

# class FedAvgWithCheckpointing(fl.server.strategy.FedAvg):
#     def aggregate_fit(
#         self,
#         server_round: int,
#         results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
#         failures: List[BaseException],
#     ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
#         print(f"Server: Aggregating fit results for round {server_round}. {len(results)} clients returned results.")
#         aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

#         if aggregated_parameters is not None:
#             ndarrays_parameters = fl.common.parameters_to_ndarrays(aggregated_parameters)
#             save_global_model_checkpoint(server_round, ndarrays_parameters, {}) 
        
#         print(f"Server: Aggregation and checkpointing for round {server_round} completed.")
#         return aggregated_parameters, metrics

# def save_global_model_checkpoint(server_round: int, parameters: fl.common.NDArrays, client_metrics: Dict) -> None:
#     input_size = 76
#     hidden_size = 16 # Match model size in client
#     num_layers = 1
#     output_size = len(TARGET_COLUMNS)
#     global_model_to_save = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
    
#     params_dict = zip(global_model_to_save.state_dict().keys(), parameters)
#     state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#     global_model_to_save.load_state_dict(state_dict, strict=True)

#     checkpoint_path = checkpoint_dir / f"global_model_round_{server_round}.pth"
#     torch.save(global_model_to_save.state_dict(), checkpoint_path)
#     print(f"Server: Saved global model checkpoint to {checkpoint_path}")

# def main():
#     # --- IMPORTANT: Initialize Ray with very limited resources ---
#     # This will ensure that only 1 client actor (and thus 1 FlowerClient instance)
#     # is active/initialized at a time, preventing initial memory spikes.
#     # Adjust object_store_memory based on your total RAM. 2GB for Ray objects is a starting point.
#     ray.init(num_cpus=1, object_store_memory=2_000_000_000, ignore_reinit_error=True) 

#     input_size = 76
#     output_size = len(TARGET_COLUMNS) # <-- FIXED: TARGET_COLUMNS is now defined due to the import

#     # Initialize global model with reduced size
#     global_model = LSTMModel(input_size=input_size, hidden_size=16, num_layers=1, output_size=output_size)

#     strategy = FedAvgWithCheckpointing(
#         fraction_fit=FRACTION_FIT,
#         fraction_evaluate=FRACTION_FIT,
#         min_fit_clients=MIN_FIT_CLIENTS,
#         min_evaluate_clients=MIN_FIT_CLIENTS,
#         min_available_clients=NUM_CLIENTS,
#         evaluate_fn=get_evaluate_fn(global_model),
#         on_fit_config_fn=fit_config,
#         on_evaluate_config_fn=evaluate_config,
#     )

#     print("Starting Flower simulation...")
#     fl.simulation.start_simulation(
#         client_fn=client_fn,
#         num_clients=NUM_CLIENTS,
#         config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
#         strategy=strategy,
#         # This client_resources dict specifies resources *per client actor*.
#         # Since ray.init() already limited total CPUs to 1, this will work as intended.
#         client_resources={"num_cpus": 1, "num_gpus": 0.0}
#     )
#     print("\nFlower simulation finished.")

#     # Disconnect Ray at the end to clean up resources
#     ray.shutdown()


# if __name__ == "__main__":
#     main()