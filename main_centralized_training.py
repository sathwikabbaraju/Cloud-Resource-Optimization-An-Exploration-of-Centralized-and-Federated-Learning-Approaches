import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import os # For creating output directories

# --- Configuration ---
global_data_path = Path("./global_consolidated_data.pkl")
model_output_dir = Path("./trained_models")
model_output_dir.mkdir(parents=True, exist_ok=True) # Create directory for saving models

# Hyperparameters for training
BATCH_SIZE = 128      # Increase batch size for faster training if VRAM/RAM allows
LEARNING_RATE = 0.001
NUM_EPOCHS = 20       # Number of training epochs

# Model architecture parameters (must match what was used for data prep)
HIDDEN_SIZE = 64
NUM_LAYERS = 2

# --- 1. PyTorch Model Definition (Reusing from fl_model_utils.py) ---
# We'll put it directly here for self-contained script, or you can import if preferred
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # Take output from the last time step
        out = out.unsqueeze(1) # Reshape to (batch_size, prediction_horizon, output_size)
        return out

# --- 2. PyTorch Dataset Class (Reusing from fl_model_utils.py) ---
class GlobalResourceDataset(Dataset):
    def __init__(self, X_data, Y_data):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.Y = torch.tensor(Y_data, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# --- Main Training and Evaluation Logic ---
def main():
    print("Starting centralized model training...")

    # Set device (CPU or GPU/MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load consolidated global data
    print(f"Loading global consolidated data from {global_data_path}...")
    try:
        with open(global_data_path, 'rb') as f:
            global_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Global consolidated data not found at {global_data_path}. Please run consolidate_data.py first.")
        return
    except Exception as e:
        print(f"Error loading global data: {e}")
        return

    X_train_global = global_data['X_train']
    Y_train_global = global_data['Y_train']
    X_val_global = global_data['X_val']
    Y_val_global = global_data['Y_val']
    X_test_global = global_data['X_test']
    Y_test_global = global_data['Y_test']
    feature_names = global_data['feature_names']
    target_names = global_data['target_names']
    sequence_length = global_data['sequence_length']
    prediction_horizon = global_data['prediction_horizon']

    print(f"Global X_train shape: {X_train_global.shape}, Y_train shape: {Y_train_global.shape}")
    print(f"Global X_val shape: {X_val_global.shape}, Y_val shape: {Y_val_global.shape}")
    print(f"Global X_test shape: {X_test_global.shape}, Y_test shape: {Y_test_global.shape}")
    print(f"Input features: {feature_names}, Targets: {target_names}")
    print(f"Sequence Length: {sequence_length}, Prediction Horizon: {prediction_horizon}")

    # Create PyTorch Datasets and DataLoaders
    train_dataset = GlobalResourceDataset(X_train_global, Y_train_global)
    val_dataset = GlobalResourceDataset(X_val_global, Y_val_global)
    test_dataset = GlobalResourceDataset(X_test_global, Y_test_global)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the model
    input_size = X_train_global.shape[2]
    output_size = Y_train_global.shape[2]
    model = LSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS, output_size).to(device)

    # Define Loss function and Optimizer
    criterion = nn.MSELoss() # Mean Squared Error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    print("\nStarting model training...")
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_path = model_output_dir / "best_model.pth"

    for epoch in range(NUM_EPOCHS):
        model.train() # Set model to training mode
        running_train_loss = 0.0
        for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * X_batch.size(0)

        epoch_train_loss = running_train_loss / len(train_dataset)
        history['train_loss'].append(epoch_train_loss)

        # --- Validation Step ---
        model.eval() # Set model to evaluation mode
        running_val_loss = 0.0
        with torch.no_grad(): # Disable gradient calculation
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                running_val_loss += loss.item() * X_batch.size(0)

        epoch_val_loss = running_val_loss / len(val_dataset)
        history['val_loss'].append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        # Save best model based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  Saved best model to {best_model_path} with Val Loss: {best_val_loss:.4f}")

    print("\nTraining complete.")

    # --- Evaluation on Test Set ---
    print("\nEvaluating model on test set...")
    # Load the best model
    model.load_state_dict(torch.load(best_model_path))
    model.eval() # Set to evaluation mode
    
    total_test_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            total_test_loss += loss.item() * X_batch.size(0)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(Y_batch.cpu().numpy())

    avg_test_loss = total_test_loss / len(test_dataset)
    print(f"Final Test Loss (MSE): {avg_test_loss:.4f}")

    # Further evaluation metrics (e.g., RMSE, MAE, R-squared)
    # We'll use scikit-learn for these, so flatten predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0) # Shape (N, 1, 4)
    all_targets = np.concatenate(all_targets, axis=0)         # Shape (N, 1, 4)

    # Flatten to (N, 4) for metric calculation per target variable
    all_predictions_flat = all_predictions.reshape(-1, len(target_names))
    all_targets_flat = all_targets.reshape(-1, len(target_names))

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    print("\nDetailed Test Metrics per Target Variable:")
    for i, target_name in enumerate(target_names):
        mse = mean_squared_error(all_targets_flat[:, i], all_predictions_flat[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_targets_flat[:, i], all_predictions_flat[:, i])
        r2 = r2_score(all_targets_flat[:, i], all_predictions_flat[:, i])
        print(f"  {target_name}:")
        print(f"    MSE: {mse:.4f}")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    MAE: {mae:.4f}")
        print(f"    R-squared: {r2:.4f}")
    
    # --- Plotting Training History ---
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(model_output_dir / "training_history.png")
    plt.show()
    print(f"Training history plot saved to {model_output_dir / 'training_history.png'}")

if __name__ == "__main__":
    # Ensure scikit-learn is installed for metrics
    try:
        import sklearn
    except ImportError:
        print("Scikit-learn not found. Installing...")
        os.system("pip install scikit-learn")
        import sklearn # Try importing again

    main()