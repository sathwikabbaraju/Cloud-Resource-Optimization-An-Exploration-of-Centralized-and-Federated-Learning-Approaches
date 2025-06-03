# Cloud Resource Optimization: An Exploration of Centralized and Federated Learning Approaches

## Overview

This project investigates the prediction and optimization of cloud data center resource allocation (CPU, memory, storage) to enhance efficiency and reduce costs. It explores two distinct, yet complementary, machine learning paradigms: a **centralized deep learning model** and an ambitious dive into **Federated Learning (FL)** with PyTorch Flower.

While resource constraints led to the final demonstration being a centralized model, the project thoroughly implemented and tested the FL pipeline, showcasing advanced understanding of distributed AI principles.

## How the Project Was Built & My Contributions

This project was developed by **Sai Sathwik Abbaraju**. You can connect with me on LinkedIn: [@sai-sathwik-abbaraju-6527ab275](https://www.linkedin.com/in/sai-sathwik-abbaraju-6527ab275/)

## Key Features & Accomplishments

* **Comprehensive Data Pipeline:** Implemented robust data loading, cleaning, and preprocessing for a massive 18GB dataset from the MIT Supercloud system.
* **Advanced Feature Engineering:** Developed sophisticated features including lagged resource usage, rolling window statistics, and time-based indicators to capture complex temporal dependencies.
* **Exploration of Federated Learning (FL):**
    * Designed and implemented a PyTorch Flower-based Federated Learning client-server architecture.
    * Successfully partitioned and prepared client-specific datasets to simulate a distributed training environment.
    * Attempted to train a global LSTM model without centralizing sensitive raw data, demonstrating a cutting-edge approach to privacy-preserving and scalable AI in cloud environments.
    * *Acknowledged and navigated practical challenges (e.g., memory constraints on local hardware) inherent in real-world FL implementations.*
* **Centralized LSTM for Resource Prediction:**
    * Developed and trained a powerful Long Short-Term Memory (LSTM) neural network using a consolidated subset of the prepared data.
    * Achieved promising predictive capabilities for memory (RSS) utilization, demonstrating the model's potential.
    * Conducted thorough evaluation using metrics like MSE, RMSE, MAE, and R-squared.
* **Scalable Data Preparation:** Leveraged efficient data formats (Parquet, .npy) and memory-mapping techniques to handle large datasets effectively.
* **Modular Codebase:** Organized into distinct, reusable scripts for each stage of the data processing and modeling pipeline.

## Project Structure

* `processed_client_data/`: Cleaned and preprocessed raw data, saved in Parquet format, organized by simulated client.
* `feature_engineered_beta_client_data/`: Feature-engineered data (lags, rolling stats, time features) from the `processed_client_data`, scaled using `StandardScaler`. This data served as input for both FL and centralized approaches.
* `pytorch_client_data/`: Prepared sequence data in `.npy` format, split into train/validation/test sets *per client*. This was the direct input for the Federated Learning clients.
* `model_data/`: *(Optional - this was the previous step before .npy files)*
* `trained_models/`: Stores trained centralized model checkpoints, training history plots, and evaluation results.

**Core Scripts:**

* `data_explorer.py`: Initial script for understanding raw data structure and column names.
* `process_client_data.py`: Handles initial data loading, cleaning, timestamp conversion, and saving to `processed_client_data`.
* `feature_engineer.py`: Implements comprehensive feature engineering and scaling (`StandardScaler`), saving to `feature_engineered_beta_client_data`.
* `prepare_data_for_fl_npy.py`: Transforms feature-engineered data into sequences and saves as `.npy` files (`pytorch_client_data`), preparing data for distributed consumption.
* `consolidate_data.py`: **(Centralized Pivot)** Consolidates the `pytorch_client_data` from selected clients into a single, global dataset for centralized model training.
* `main_centralized_training.py`: **(Centralized Model)** Defines the LSTM, trains it on the global consolidated data, and evaluates its performance.
* `fl_model_utils.py`: **(FL Component)** Defines the PyTorch LSTM model and `Dataset` class, designed for both centralized and federated use.
* `main_federated_learning.py`: **(FL Component - Attempted)** Implements the PyTorch Flower client-server logic for federated training, demonstrating the architecture.
* `analyze_data_distribution.py`: Helps diagnose data characteristics and evaluate scaling effectiveness.

## Problem Statement & Impact

Cloud data centers constantly grapple with dynamic workloads and fluctuating resource demands. Inefficient resource allocation leads to:
1.  **Over-provisioning:** Wasted compute, memory, and storage, incurring unnecessary costs.
2.  **Under-provisioning:** Performance degradation, service outages, and poor user experience.

This project aims to address these challenges by providing **accurate foresight into future resource needs**. By leveraging AI to predict resource usage, the system can:

* **Enable Proactive Scaling:** Automatically scale resources up or down *before* demand changes, ensuring optimal performance and cost-efficiency.
* **Improve Workload Placement:** Strategically place new workloads on nodes with anticipated available resources.
* **Reduce Operational Costs:** Minimize idle resources and maximize utilization rates.
* **Enhance Service Reliability:** Prevent resource starvation and maintain Service Level Agreements (SLAs).

The exploration of Federated Learning highlights a path toward building these intelligent systems while potentially adhering to strict data privacy and sovereignty requirements often found in multi-tenant cloud environments.


My approach involved a meticulous multi-phase development:

1.  **Data Ingestion & Preprocessing:** Faced with an 18GB dataset, I implemented chunk-based reading and memory-efficient data type conversions. I ensured data consistency and handled missing values, setting the foundation for robust analysis.
2.  **Sophisticated Feature Engineering:** I designed and implemented a comprehensive set of features crucial for time-series prediction. This included:
    * **Lagged Features:** Capturing past resource usage to predict future trends.
    * **Rolling Statistics:** Calculating means and standard deviations over sliding windows to understand short-term trends and variability.
    * **Time-Based Features:** Extracting cyclical patterns (hour of day, day of week) from timestamps.
    Crucially, I implemented a grouping mechanism (`Node`, `Series`) during feature engineering to ensure that lags and rolling statistics were computed *within* continuous time series, preventing data leakage across disparate job/node streams.
3.  **Advanced Data Scaling & Preparation:** I rigorously analyzed data distributions, identifying the prevalence of zeros and skewness in CPU and I/O metrics. I then implemented and tested `StandardScaler` to robustly normalize this challenging data, a critical step for neural network performance. I meticulously transformed the data into sequences (`X_seq`, `Y_seq`) for recurrent neural networks and split it chronologically for realistic evaluation, saving it efficiently as `.npy` files.
4.  **Federated Learning Exploration:** This was a significant undertaking. I built the infrastructure for a **PyTorch Flower-based Federated Learning simulation**. This involved defining the `FlowerClient` (handling local data loading, model training, and parameter exchange) and setting up the `FedAvg` server strategy. This ambitious step demonstrated a deep understanding of decentralized training paradigms, even though the sheer scale of the data on a local machine presented memory challenges.
5.  **Centralized Model Training & Evaluation:** Recognizing the immediate practical constraints of local hardware, I strategically pivoted to demonstrate the core prediction capability with a **centralized LSTM model**. I consolidated a manageable subset of the prepared data and implemented a full PyTorch training and evaluation loop. I meticulously tracked loss, saved the best model, and calculated a full suite of regression metrics (MSE, RMSE, MAE, R-squared) to quantitatively assess performance across all predicted resource types.

## Preliminary Results & Future Work

The centralized LSTM model demonstrated promising performance for **memory (RSS)** prediction (R-squared of ~0.43). For `CPUUtilization`, `ReadMB`, and `WriteMB`, initial R-squared values were low, indicating challenges with their highly skewed and zero-inflated distributions. This highlights the complexity of real-world data and provides clear directions for future improvements.

### Further Work:

* **Advanced Data Transformations:** Explore `log1p` transformations for heavily skewed `ReadMB` and `WriteMB` data **before** scaling, and inverse transforming predictions accordingly.
* **Robust Outlier Handling:** Implement more sophisticated outlier detection and handling mechanisms beyond simple capping or dropping.
* **Model Architecture Refinement:**
    * Experiment with increasing `SEQUENCE_LENGTH` in the LSTM to provide more historical context (requires more RAM or advanced data loading).
    * Explore alternative architectures like **Temporal Convolutional Networks (TCNs)** or simplified attention-based models, which might be more efficient or better suited for certain time series patterns.
    * Consider **multi-task learning** approaches if correlations between different resource types can be leveraged.
* **Addressing Zero-Inflated Data:** Investigate specialized techniques for predicting count data or zero-inflated distributions, possibly by modeling the probability of non-zero usage separately from the magnitude of usage.
* **Hyperparameter Optimization:** Conduct systematic tuning of learning rate, batch size, LSTM hidden dimensions, number of layers, and regularization techniques.
* **Hardware Scaling:** Migrate to a more powerful computing environment (e.g., cloud GPU instances) to fully leverage the entire 18GB dataset and explore longer sequence lengths or more complex models, especially if resuming Federated Learning.
* **Interpretable AI:** Explore methods to understand which features contribute most to the model's predictions.

## Storage Requirements

Please note that this project involves significant data processing and intermediate file generation. It is highly recommended to have at least **1TB of storage space** available for the full dataset and all generated intermediate files.

