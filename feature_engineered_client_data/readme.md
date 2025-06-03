(pythonProject) sathwik@SATHWIKs-MacBook-Air aws data % python feature_engineer.py
Starting feature engineering for all clients...

Engineering features for client: 0000
  Warning: Client 0000 has 1128 unique (Node, Series) combinations. Processing all, but consider further aggregation if performance is an issue or series are too short.
  Finished feature engineering for client 0000. Total rows: 7743155
  Rows after dropping NaNs: 7558243
  Scaler saved for client 0000 to feature_engineered_client_data/scaler_client_0000.pkl
  Successfully saved feature engineered data for client 0000 to feature_engineered_client_data/client_0000_features.parquet

Engineering features for client: 0001
  Warning: Client 0001 has 1300 unique (Node, Series) combinations. Processing all, but consider further aggregation if performance is an issue or series are too short.
  Finished feature engineering for client 0001. Total rows: 7680671
  Rows after dropping NaNs: 7451380
  Scaler saved for client 0001 to feature_engineered_client_data/scaler_client_0001.pkl
  Successfully saved feature engineered data for client 0001 to feature_engineered_client_data/client_0001_features.parquet

Engineering features for client: 0002
  Warning: Client 0002 has 935 unique (Node, Series) combinations. Processing all, but consider further aggregation if performance is an issue or series are too short.
  Finished feature engineering for client 0002. Total rows: 7223658
  Rows after dropping NaNs: 6987682
  Scaler saved for client 0002 to feature_engineered_client_data/scaler_client_0002.pkl
  Successfully saved feature engineered data for client 0002 to feature_engineered_client_data/client_0002_features.parquet

Engineering features for client: 0003
  Warning: Client 0003 has 1616 unique (Node, Series) combinations. Processing all, but consider further aggregation if performance is an issue or series are too short.
  Finished feature engineering for client 0003. Total rows: 7492898
  Rows after dropping NaNs: 7242306
  Scaler saved for client 0003 to feature_engineered_client_data/scaler_client_0003.pkl
  Successfully saved feature engineered data for client 0003 to feature_engineered_client_data/client_0003_features.parquet

Engineering features for client: 0004
  Warning: Client 0004 has 1140 unique (Node, Series) combinations. Processing all, but consider further aggregation if performance is an issue or series are too short.
  Finished feature engineering for client 0004. Total rows: 7314405
  Rows after dropping NaNs: 7137171
  Scaler saved for client 0004 to feature_engineered_client_data/scaler_client_0004.pkl
  Successfully saved feature engineered data for client 0004 to feature_engineered_client_data/client_0004_features.parquet

Engineering features for client: 0005
  Warning: Client 0005 has 1135 unique (Node, Series) combinations. Processing all, but consider further aggregation if performance is an issue or series are too short.
  Finished feature engineering for client 0005. Total rows: 7772999
  Rows after dropping NaNs: 7548993
  Scaler saved for client 0005 to feature_engineered_client_data/scaler_client_0005.pkl
  Successfully saved feature engineered data for client 0005 to feature_engineered_client_data/client_0005_features.parquet

Engineering features for client: 0006
  Warning: Client 0006 has 1014 unique (Node, Series) combinations. Processing all, but consider further aggregation if performance is an issue or series are too short.
  Finished feature engineering for client 0006. Total rows: 7989751
  Rows after dropping NaNs: 7710351
  Scaler saved for client 0006 to feature_engineered_client_data/scaler_client_0006.pkl
  Successfully saved feature engineered data for client 0006 to feature_engineered_client_data/client_0006_features.parquet

Engineering features for client: 0007
  Warning: Client 0007 has 2956 unique (Node, Series) combinations. Processing all, but consider further aggregation if performance is an issue or series are too short.
  Finished feature engineering for client 0007. Total rows: 7516688
  Rows after dropping NaNs: 7359215
  Scaler saved for client 0007 to feature_engineered_client_data/scaler_client_0007.pkl
  Successfully saved feature engineered data for client 0007 to feature_engineered_client_data/client_0007_features.parquet

Engineering features for client: 0008
  Warning: Client 0008 has 1030 unique (Node, Series) combinations. Processing all, but consider further aggregation if performance is an issue or series are too short.
  Finished feature engineering for client 0008. Total rows: 7679514
  Rows after dropping NaNs: 7458579
  Scaler saved for client 0008 to feature_engineered_client_data/scaler_client_0008.pkl
  Successfully saved feature engineered data for client 0008 to feature_engineered_client_data/client_0008_features.parquet

Engineering features for client: 0009
  Warning: Client 0009 has 3596 unique (Node, Series) combinations. Processing all, but consider further aggregation if performance is an issue or series are too short.
  Finished feature engineering for client 0009. Total rows: 7245460
  Rows after dropping NaNs: 6894574
  Scaler saved for client 0009 to feature_engineered_client_data/scaler_client_0009.pkl
  Successfully saved feature engineered data for client 0009 to feature_engineered_client_data/client_0009_features.parquet

Engineering features for client: 0010
  Warning: Client 0010 has 1360 unique (Node, Series) combinations. Processing all, but consider further aggregation if performance is an issue or series are too short.
  Finished feature engineering for client 0010. Total rows: 7751659
  Rows after dropping NaNs: 7575383
  Scaler saved for client 0010 to feature_engineered_client_data/scaler_client_0010.pkl
  Successfully saved feature engineered data for client 0010 to feature_engineered_client_data/client_0010_features.parquet

Engineering features for client: 0011
  Warning: Client 0011 has 693 unique (Node, Series) combinations. Processing all, but consider further aggregation if performance is an issue or series are too short.
  Finished feature engineering for client 0011. Total rows: 4136617
  Rows after dropping NaNs: 4053695
  Scaler saved for client 0011 to feature_engineered_client_data/scaler_client_0011.pkl
  Successfully saved feature engineered data for client 0011 to feature_engineered_client_data/client_0011_features.parquet

Finished feature engineering. Successfully processed 12 clients.
Feature engineered data saved to: feature_engineered_client_data

Loading a sample feature engineered file to confirm (e.g., client_0000_features.parquet):
Sample loaded DataFrame head:
                     Step              Node  Series  ElapsedTime  CPUFrequency       CPUTime  ...  hour_of_day  day_of_week  month  day_of_month  is_weekend  minute_of_hour
EpochTime                                                                                     ...                                                                           
2021-01-05 01:12:31     0  r8333645-n386398       0     0.000023      0.765656  7.028530e-08  ...            1            1      1             5           0              12
2021-01-05 01:12:41     0  r8333645-n386398       0     0.000028      0.765656  7.654020e-08  ...            1            1      1             5           0              12
2021-01-05 01:12:51     0  r8333645-n386398       0     0.000032      0.765656  7.752782e-08  ...            1            1      1             5           0              12
2021-01-05 01:13:01     0  r8333645-n386398       0     0.000037      0.765656  7.983225e-08  ...            1            1      1             5           0              13
2021-01-05 01:13:11     0  r8333645-n386398       0     0.000041      0.765656  6.946229e-08  ...            1            1      1             5           0              13

[5 rows x 81 columns]
Sample loaded DataFrame info:
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 7558243 entries, 2021-01-05 01:12:31 to 2021-10-01 11:27:14
Data columns (total 81 columns):
 #   Column                           Dtype  
---  ------                           -----  
 0   Step                             int8   
 1   Node                             object 
 2   Series                           int16  
 3   ElapsedTime                      float64
 4   CPUFrequency                     float64
 5   CPUTime                          float64
 6   CPUUtilization                   float64
 7   RSS                              float64
 8   VMSize                           float64
 9   Pages                            float64
 10  ReadMB                           float64
 11  WriteMB                          float64
 12  CPUUtilization_lag_1             float64
 13  CPUUtilization_lag_2             float64
 14  CPUUtilization_lag_3             float64
 15  CPUUtilization_lag_4             float64
 16  CPUUtilization_lag_5             float64
 17  RSS_lag_1                        float64
 18  RSS_lag_2                        float64
 19  RSS_lag_3                        float64
 20  RSS_lag_4                        float64
 21  RSS_lag_5                        float64
 22  ReadMB_lag_1                     float64
 23  ReadMB_lag_2                     float64
 24  ReadMB_lag_3                     float64
 25  ReadMB_lag_4                     float64
 26  ReadMB_lag_5                     float64
 27  WriteMB_lag_1                    float64
 28  WriteMB_lag_2                    float64
 29  WriteMB_lag_3                    float64
 30  WriteMB_lag_4                    float64
 31  WriteMB_lag_5                    float64
 32  CPUFrequency_lag_1               float64
 33  CPUFrequency_lag_2               float64
 34  CPUFrequency_lag_3               float64
 35  CPUFrequency_lag_4               float64
 36  CPUFrequency_lag_5               float64
 37  CPUTime_lag_1                    float64
 38  CPUTime_lag_2                    float64
 39  CPUTime_lag_3                    float64
 40  CPUTime_lag_4                    float64
 41  CPUTime_lag_5                    float64
 42  VMSize_lag_1                     float64
 43  VMSize_lag_2                     float64
 44  VMSize_lag_3                     float64
 45  VMSize_lag_4                     float64
 46  VMSize_lag_5                     float64
 47  Pages_lag_1                      float64
 48  Pages_lag_2                      float64
 49  Pages_lag_3                      float64
 50  Pages_lag_4                      float64
 51  Pages_lag_5                      float64
 52  ElapsedTime_lag_1                float64
 53  ElapsedTime_lag_2                float64
 54  ElapsedTime_lag_3                float64
 55  ElapsedTime_lag_4                float64
 56  ElapsedTime_lag_5                float64
 57  CPUUtilization_rolling_mean_50s  float64
 58  CPUUtilization_rolling_std_50s   float64
 59  RSS_rolling_mean_50s             float64
 60  RSS_rolling_std_50s              float64
 61  ReadMB_rolling_mean_50s          float64
 62  ReadMB_rolling_std_50s           float64
 63  WriteMB_rolling_mean_50s         float64
 64  WriteMB_rolling_std_50s          float64
 65  CPUFrequency_rolling_mean_50s    float64
 66  CPUFrequency_rolling_std_50s     float64
 67  CPUTime_rolling_mean_50s         float64
 68  CPUTime_rolling_std_50s          float64
 69  VMSize_rolling_mean_50s          float64
 70  VMSize_rolling_std_50s           float64
 71  Pages_rolling_mean_50s           float64
 72  Pages_rolling_std_50s            float64
 73  ElapsedTime_rolling_mean_50s     float64
 74  ElapsedTime_rolling_std_50s      float64
 75  hour_of_day                      int32  
 76  day_of_week                      int32  
 77  month                            int32  
 78  day_of_month                     int32  
 79  is_weekend                       int64  
 80  minute_of_hour                   int32  
dtypes: float64(72), int16(1), int32(5), int64(1), int8(1), object(1)
memory usage: 4.4+ GB

Sample columns for feature engineering:
['CPUUtilization_lag_1', 'CPUUtilization_lag_2', 'CPUUtilization_lag_3', 'CPUUtilization_lag_4', 'CPUUtilization_lag_5', 'RSS_lag_1', 'RSS_lag_2', 'RSS_lag_3', 'RSS_lag_4', 'RSS_lag_5']
(pythonProject) sathwik@SATHWIKs-MacBook-Air aws data %