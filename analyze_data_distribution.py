import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

# --- Configuration ---
# Path to your feature engineered data (e.g., from client 0000)
sample_client_feat_path = Path("./feature_engineered_client_data/client_0000_features.parquet")
# Path to the scaler saved for that client
scaler_path = Path("./feature_engineered_client_data/scaler_client_0000.pkl")

# Define the target columns (must match what you used in feature_engineer.py)
TARGET_COLUMNS = ['CPUUtilization', 'RSS', 'ReadMB', 'WriteMB']

def analyze_distributions():
    print(f"Loading sample feature engineered data from: {sample_client_feat_path}")
    print(f"Loading scaler from: {scaler_path}")

    if not sample_client_feat_path.exists():
        print(f"Error: Sample client features not found at {sample_client_feat_path}. Please ensure it exists.")
        return
    if not scaler_path.exists():
        print(f"Error: Scaler not found at {scaler_path}. Please ensure it exists.")
        return

    try:
        sample_df_scaled = pd.read_parquet(sample_client_feat_path)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return

    try:
        with open(scaler_path, 'rb') as f:
            sample_scaler = pickle.load(f)
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return

    # Identify numerical columns that were scaled (from feature_engineer.py logic)
    # This list must precisely match the columns that were passed to scaler.fit_transform()
    numerical_cols_for_scaling = sample_df_scaled.select_dtypes(include=np.number).columns.tolist()
    
    # Exclude columns that were explicitly NOT scaled in feature_engineer.py
    # (Step, Series, and time-based features)
    features_and_targets_to_scale = [
        col for col in numerical_cols_for_scaling 
        if col not in ['Step', 'Series', 'is_weekend', 'hour_of_day', 'day_of_week', 'month', 'day_of_month', 'minute_of_hour']
    ]

    # Create a temporary DataFrame for inverse transformation, ensuring column order matches scaler's fit
    # This step is critical: the order of columns in `temp_df` MUST be the same as when the scaler was fitted.
    # The simplest way to ensure this is to explicitly select columns in the same order.
    
    # Check if all features_and_targets_to_scale are actually in sample_df_scaled
    valid_cols_for_inverse = [col for col in features_and_targets_to_scale if col in sample_df_scaled.columns]
    
    if len(valid_cols_for_inverse) != len(features_and_targets_to_scale):
        print("Warning: Some expected scaled columns not found in sample_df_scaled. This might cause inverse transform issues.")
        print(f"Missing columns: {set(features_and_targets_to_scale) - set(valid_cols_for_inverse)}")
        features_and_targets_to_scale = valid_cols_for_inverse # Adjust to only use existing ones

    temp_df_scaled = sample_df_scaled[features_and_targets_to_scale].copy()

    # Inverse transform
    print("Performing inverse transformation...")
    temp_df_original_scale_values = sample_scaler.inverse_transform(temp_df_scaled.values)
    temp_df_original_scale = pd.DataFrame(
        temp_df_original_scale_values,
        columns=features_and_targets_to_scale,
        index=sample_df_scaled.index
    )

    print("\nVisualizing distributions of original (unscaled) target variables:")
    plt.figure(figsize=(16, 12))
    for i, col in enumerate(TARGET_COLUMNS):
        if col not in temp_df_original_scale.columns:
            print(f"Warning: Target column '{col}' not found after inverse transformation. Skipping plot.")
            continue
            
        plt.subplot(2, 2, i + 1)
        # Filter out exact zeros for better visualization if the majority are zeros
        # We can analyze the zero count separately
        non_zero_values = temp_df_original_scale[col][temp_df_original_scale[col] > 0]
        
        # Count zeros
        zero_count = (temp_df_original_scale[col] == 0).sum()
        total_count = len(temp_df_original_scale[col])
        zero_percentage = (zero_count / total_count) * 100 if total_count > 0 else 0

        # Plotting histogram
        if not non_zero_values.empty:
            plt.hist(non_zero_values, bins=50, density=True, alpha=0.7, color='skyblue')
            plt.title(f'Distribution of Original {col}\n({zero_percentage:.2f}% Zeros)')
            plt.xlabel(col)
            plt.ylabel('Density (Log Scale if Applied)')
            plt.yscale('log') # Use log scale for y-axis if data is heavily skewed
            plt.grid(True, which="both", ls="--", c='0.7')
        else:
            plt.title(f'Distribution of Original {col}\n(All Zeros or Non-positive values)')
            plt.text(0.5, 0.5, 'No non-zero data to plot', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('Original (Unscaled) Target Variable Distributions', fontsize=18)
    
    output_plot_path = Path("./trained_models/original_target_distributions.png")
    plt.savefig(output_plot_path)
    print(f"\nDistribution plots saved to {output_plot_path}")
    plt.show()

if __name__ == "__main__":
    analyze_distributions()