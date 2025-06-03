import os
import pandas as pd
from pathlib import Path

# Define your base dataset path
base_dataset_path = Path("/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/cpu")

print(f"Checking dataset path: {base_dataset_path}")

if not base_dataset_path.exists():
    print(f"Error: Dataset path does not exist: {base_dataset_path}")
    print("Please double-check the path and ensure the 'cpu' folder is there.")
    exit()

subfolders = sorted([f for f in base_dataset_path.iterdir() if f.is_dir() and f.name.isdigit()])

if not subfolders:
    print(f"Error: No numbered subfolders found in {base_dataset_path}")
    print("Expected subfolders like '0000', '0001', etc.")
    exit()

print(f"Found {len(subfolders)} subfolders:")
for i, folder in enumerate(subfolders):
    print(f"- {folder.name}")
    if i >= 10: # Just show a few if there are many
        break

# Let's inspect the first subfolder
first_subfolder = subfolders[0]
print(f"\nInspecting files in the first subfolder: {first_subfolder.name}")

file_count = 0
timeseries_files = []
summary_files = []

for file in first_subfolder.iterdir():
    if file.suffix == '.csv':
        file_count += 1
        if 'timeseries' in file.name:
            timeseries_files.append(file.name)
        elif 'summary' in file.name:
            summary_files.append(file.name)

print(f"Total CSV files found: {file_count}")
print(f"Sample timeseries files (first 5): {timeseries_files[:5]}")
print(f"Sample summary files (first 5): {summary_files[:5]}")

# Try reading a small sample of a timeseries file to inspect columns
if timeseries_files:
    sample_timeseries_path = first_subfolder / timeseries_files[0]
    try:
        print(f"\nAttempting to read a sample timeseries file: {sample_timeseries_path.name}")
        # Read only a few rows to quickly inspect columns
        sample_df = pd.read_csv(sample_timeseries_path, nrows=5)
        print("Columns in sample timeseries file:")
        print(sample_df.columns.tolist())
        print("\nFirst 3 rows:")
        print(sample_df.head(3))
    except Exception as e:
        print(f"Error reading sample timeseries file: {e}")
else:
    print("\nNo timeseries files found to sample.")

# Try reading a small sample of a summary file to inspect columns
if summary_files:
    sample_summary_path = first_subfolder / summary_files[0]
    try:
        print(f"\nAttempting to read a sample summary file: {sample_summary_path.name}")
        # Read only a few rows to quickly inspect columns
        sample_summary_df = pd.read_csv(sample_summary_path, nrows=5)
        print("Columns in sample summary file:")
        print(sample_summary_df.columns.tolist())
        print("\nFirst 3 rows:")
        print(sample_summary_df.head(3))
    except Exception as e:
        print(f"Error reading sample summary file: {e}")
else:
    print("\nNo summary files found to sample.")