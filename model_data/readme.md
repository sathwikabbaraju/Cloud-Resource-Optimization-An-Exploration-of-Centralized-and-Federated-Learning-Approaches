(pythonProject) sathwik@SATHWIKs-MacBook-Air aws data % python prepare_data_for_fl.py
Starting data preparation for modeling for all clients...

Preparing data for client: 0000
  Created sequences. X_seq shape: (7558240, 3, 76), Y_seq shape: (7558240, 1, 4)
  X_seq size (approx): 6.42 GB
  Y_seq size (approx): 0.11 GB
  Train samples: 5290768, Val samples: 1133736, Test samples: 1133736
  Successfully prepared and saved data for client 0000 to model_data/client_0000_prepared_data.pkl

Preparing data for client: 0001
  Created sequences. X_seq shape: (7451377, 3, 76), Y_seq shape: (7451377, 1, 4)
  X_seq size (approx): 6.33 GB
  Y_seq size (approx): 0.11 GB
  Train samples: 5215963, Val samples: 1117707, Test samples: 1117707
  Successfully prepared and saved data for client 0001 to model_data/client_0001_prepared_data.pkl

Preparing data for client: 0002
  Created sequences. X_seq shape: (6987679, 3, 76), Y_seq shape: (6987679, 1, 4)
  X_seq size (approx): 5.94 GB
  Y_seq size (approx): 0.10 GB
  Train samples: 4891375, Val samples: 1048152, Test samples: 1048152
  Successfully prepared and saved data for client 0002 to model_data/client_0002_prepared_data.pkl

Preparing data for client: 0003
  Created sequences. X_seq shape: (7242303, 3, 76), Y_seq shape: (7242303, 1, 4)
  X_seq size (approx): 6.15 GB
  Y_seq size (approx): 0.11 GB
  Train samples: 5069612, Val samples: 1086345, Test samples: 1086346
  Successfully prepared and saved data for client 0003 to model_data/client_0003_prepared_data.pkl

Preparing data for client: 0004
  Created sequences. X_seq shape: (7137168, 3, 76), Y_seq shape: (7137168, 1, 4)
  X_seq size (approx): 6.06 GB
  Y_seq size (approx): 0.11 GB
  Train samples: 4996017, Val samples: 1070575, Test samples: 1070576
  Successfully prepared and saved data for client 0004 to model_data/client_0004_prepared_data.pkl

Preparing data for client: 0005
  Created sequences. X_seq shape: (7548990, 3, 76), Y_seq shape: (7548990, 1, 4)
  X_seq size (approx): 6.41 GB
  Y_seq size (approx): 0.11 GB
  Train samples: 5284293, Val samples: 1132348, Test samples: 1132349
  Successfully prepared and saved data for client 0005 to model_data/client_0005_prepared_data.pkl

Preparing data for client: 0006
  Created sequences. X_seq shape: (7710348, 3, 76), Y_seq shape: (7710348, 1, 4)
  X_seq size (approx): 6.55 GB
  Y_seq size (approx): 0.11 GB
  Train samples: 5397243, Val samples: 1156552, Test samples: 1156553
  Successfully prepared and saved data for client 0006 to model_data/client_0006_prepared_data.pkl

Preparing data for client: 0007
  Created sequences. X_seq shape: (7359212, 3, 76), Y_seq shape: (7359212, 1, 4)
  X_seq size (approx): 6.25 GB
  Y_seq size (approx): 0.11 GB
  Train samples: 5151448, Val samples: 1103882, Test samples: 1103882
  Successfully prepared and saved data for client 0007 to model_data/client_0007_prepared_data.pkl

Preparing data for client: 0008
  Created sequences. X_seq shape: (7458576, 3, 76), Y_seq shape: (7458576, 1, 4)
  X_seq size (approx): 6.34 GB
  Y_seq size (approx): 0.11 GB
  Train samples: 5221003, Val samples: 1118786, Test samples: 1118787
  Successfully prepared and saved data for client 0008 to model_data/client_0008_prepared_data.pkl

Preparing data for client: 0009
  Created sequences. X_seq shape: (6894571, 3, 76), Y_seq shape: (6894571, 1, 4)
  X_seq size (approx): 5.86 GB
  Y_seq size (approx): 0.10 GB
  Train samples: 4826199, Val samples: 1034186, Test samples: 1034186
  Successfully prepared and saved data for client 0009 to model_data/client_0009_prepared_data.pkl

Preparing data for client: 0010
  Created sequences. X_seq shape: (7575380, 3, 76), Y_seq shape: (7575380, 1, 4)
  X_seq size (approx): 6.43 GB
  Y_seq size (approx): 0.11 GB
  Train samples: 5302766, Val samples: 1136307, Test samples: 1136307
  Successfully prepared and saved data for client 0010 to model_data/client_0010_prepared_data.pkl

Preparing data for client: 0011
  Created sequences. X_seq shape: (4053692, 3, 76), Y_seq shape: (4053692, 1, 4)
  X_seq size (approx): 3.44 GB
  Y_seq size (approx): 0.06 GB
  Train samples: 2837584, Val samples: 608054, Test samples: 608054
  Successfully prepared and saved data for client 0011 to model_data/client_0011_prepared_data.pkl

Finished data preparation. Successfully prepared 12 clients.
Prepared data saved to: model_data

Loading a sample prepared file to confirm (e.g., client_0000_prepared_data.pkl):
Sample loaded data shapes:
X_train: (5290768, 3, 76)
Y_train: (5290768, 1, 4)
X_val: (1133736, 3, 76)
Y_val: (1133736, 1, 4)
X_test: (1133736, 3, 76)
Y_test: (1133736, 1, 4)
Feature names (first 5): ['Step', 'Series', 'ElapsedTime', 'CPUFrequency', 'CPUTime']
Target names: ['CPUUtilization', 'RSS', 'ReadMB', 'WriteMB']