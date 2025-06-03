(pythonProject) sathwik@SATHWIKs-MacBook-Air aws data % python main_centralized_training.py
Starting centralized model training...
Using MPS (Apple Silicon GPU)
Loading global consolidated data from global_consolidated_data.pkl...
Global X_train shape: (10506731, 3, 74), Y_train shape: (10506731, 1, 4)
Global X_val shape: (2251443, 3, 74), Y_val shape: (2251443, 1, 4)
Global X_test shape: (2251443, 3, 74), Y_test shape: (2251443, 1, 4)
Input features: ['ElapsedTime', 'CPUFrequency', 'CPUTime', 'VMSize', 'Pages', 'CPUUtilization_lag_1', 'CPUUtilization_lag_2', 'CPUUtilization_lag_3', 'CPUUtilization_lag_4', 'CPUUtilization_lag_5', 'RSS_lag_1', 'RSS_lag_2', 'RSS_lag_3', 'RSS_lag_4', 'RSS_lag_5', 'ReadMB_lag_1', 'ReadMB_lag_2', 'ReadMB_lag_3', 'ReadMB_lag_4', 'ReadMB_lag_5', 'WriteMB_lag_1', 'WriteMB_lag_2', 'WriteMB_lag_3', 'WriteMB_lag_4', 'WriteMB_lag_5', 'CPUFrequency_lag_1', 'CPUFrequency_lag_2', 'CPUFrequency_lag_3', 'CPUFrequency_lag_4', 'CPUFrequency_lag_5', 'CPUTime_lag_1', 'CPUTime_lag_2', 'CPUTime_lag_3', 'CPUTime_lag_4', 'CPUTime_lag_5', 'VMSize_lag_1', 'VMSize_lag_2', 'VMSize_lag_3', 'VMSize_lag_4', 'VMSize_lag_5', 'Pages_lag_1', 'Pages_lag_2', 'Pages_lag_3', 'Pages_lag_4', 'Pages_lag_5', 'ElapsedTime_lag_1', 'ElapsedTime_lag_2', 'ElapsedTime_lag_3', 'ElapsedTime_lag_4', 'ElapsedTime_lag_5', 'CPUUtilization_rolling_mean_50s', 'CPUUtilization_rolling_std_50s', 'RSS_rolling_mean_50s', 'RSS_rolling_std_50s', 'ReadMB_rolling_mean_50s', 'ReadMB_rolling_std_50s', 'WriteMB_rolling_mean_50s', 'WriteMB_rolling_std_50s', 'CPUFrequency_rolling_mean_50s', 'CPUFrequency_rolling_std_50s', 'CPUTime_rolling_mean_50s', 'CPUTime_rolling_std_50s', 'VMSize_rolling_mean_50s', 'VMSize_rolling_std_50s', 'Pages_rolling_mean_50s', 'Pages_rolling_std_50s', 'ElapsedTime_rolling_mean_50s', 'ElapsedTime_rolling_std_50s', 'hour_of_day', 'day_of_week', 'month', 'day_of_month', 'is_weekend', 'minute_of_hour'], Targets: ['CPUUtilization', 'RSS', 'ReadMB', 'WriteMB']
Sequence Length: 3, Prediction Horizon: 1

Starting model training...
Epoch 1/20 - Train Loss: 0.0000, Val Loss: 0.0000
  Saved best model to trained_models/best_model.pth with Val Loss: 0.0000
Epoch 2/20 - Train Loss: 0.0000, Val Loss: 0.0000
  Saved best model to trained_models/best_model.pth with Val Loss: 0.0000
Epoch 3/20 - Train Loss: 0.0000, Val Loss: 0.0000
  Saved best model to trained_models/best_model.pth with Val Loss: 0.0000
Epoch 4/20 - Train Loss: 0.0000, Val Loss: 0.0000
Epoch 5/20 - Train Loss: 0.0000, Val Loss: 0.0000
Epoch 6/20 - Train Loss: 0.0000, Val Loss: 0.0000
Epoch 7/20 - Train Loss: 0.0000, Val Loss: 0.0000
Epoch 8/20 - Train Loss: 0.0000, Val Loss: 0.0000
Epoch 9/20 - Train Loss: 0.0000, Val Loss: 0.0000
Epoch 10/20 - Train Loss: 0.0000, Val Loss: 0.0000
Epoch 11/20 - Train Loss: 0.0000, Val Loss: 0.0000
Epoch 12/20 - Train Loss: 0.0000, Val Loss: 0.0000
Epoch 13/20 - Train Loss: 0.0000, Val Loss: 0.0000
Epoch 14/20 - Train Loss: 0.0000, Val Loss: 0.0000
Epoch 15/20 - Train Loss: 0.0000, Val Loss: 0.0000
Epoch 16/20 - Train Loss: 0.0000, Val Loss: 0.0000
Epoch 17/20 - Train Loss: 0.0000, Val Loss: 0.0000
Epoch 18/20 - Train Loss: 0.0000, Val Loss: 0.0000
Epoch 19/20 - Train Loss: 0.0000, Val Loss: 0.0000
Epoch 20/20 - Train Loss: 0.0000, Val Loss: 0.0000

Training complete.

Evaluating model on test set...
Final Test Loss (MSE): 0.0000

Detailed Test Metrics per Target Variable:
  CPUUtilization:
    MSE: 0.0000
    RMSE: 0.0030
    MAE: 0.0001
    R-squared: -0.0002
  RSS:
    MSE: 0.0000
    RMSE: 0.0069
    MAE: 0.0027
    R-squared: 0.4348
  ReadMB:
    MSE: 0.0000
    RMSE: 0.0009
    MAE: 0.0000
    R-squared: -0.0002
  WriteMB:
    MSE: 0.0000
    RMSE: 0.0005
    MAE: 0.0000
    R-squared: -0.0041
2025-06-02 17:46:15.136 python[31286:2541992] The class 'NSSavePanel' overrides the method identifier.  This method is implemented by class 'NSWindow'