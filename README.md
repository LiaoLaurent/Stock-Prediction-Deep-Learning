# Deep Learning Project for Returns Predictions

This project focuses on predicting stock returns using deep learning techniques. It leverages limit order book data to forecast mid-price movements.

## Project Structure

The project is organized as follows:

- `main.ipynb`: This is the main notebook of the project. All results can be computed here.
- `feature_engineering`: The feature selection pipeline can be found here. Feature selection methods include random forest, mutual information scores, SHAP values, PCA.
- `data_preprocessing.py`: This script handles the data preprocessing steps, including feature engineering and data cleaning.
- `finance_utils.py`: This script provides utility functions for evaluating trading strategies, such as computing profit and loss (PNL) and backtesting.
- `tests/`: This directory contains test scripts for the data preprocessing module.

## Data Preprocessing

The data preprocessing pipeline involves the following steps:

1.  **Data Aggregation**: The raw data is aggregated into 2-second intervals.
2.  **Market Hours Filtering**: Data outside market hours (9:30 AM to 4:00 PM) is removed, and data from different days is concatenated.
3.  **Feature Engineering**:
    - **Technical Analysis Features**: A variety of technical indicators are calculated, including moving averages, oscillators, and trend indicators.
    - **Order Book Features**: Features derived from the limit order book are computed, such as bid-ask spread, order flow, and volume imbalance.
    - **Time Features**: Time-based features, such as time since market open and day of the week, are added.
4.  **Feature Selection**:
    - Random Forest, Mutual Information Score, and SHAP values are used for feature selection.
    - Features are selected by categories (technical analysis and order book features) and general feature selection is performed.
    - The project finds that order book indicators tend to be more useful than technical indicators.

## Models

The project explores several models for predicting mid-price movements:

- **LSTM Network**: A double-layer LSTM network with batch normalization and dropout is used as the primary deep learning model.
- **Alternative Models**: Other models, including Logistic Regression, Random Forest, and ARIMA, are also evaluated.

## Results

- The LSTM model demonstrates better performance compared to basic statistical methods.
- Models evaluation using accuracy and F1 scores.
- Backtesting is performed to evaluate the trading strategies based on the model predictions.
- The backtesting strategy involves buying at the beginning of the next period if the model predicts an upward movement and selling at the end of the period.
- The backtesting analysis does not include transaction costs or delays.

## Finance Utilities

The `finance_utils.py` script provides functions for:

- **PNL Calculation**: Computes profit and loss based on predictions and actual price changes.
- **Backtesting**: Evaluates trading strategies using historical data.
- **Visualization**: Plots backtesting results.

## How to Run the Project

1.  Install the required dependencies (e.g., pandas, scikit-learn, tensorflow).
2.  Obtain the limit order book data and place it in the specified directory.
3.  Generate the `.parquet` data file using the `process_and_combine_data` of `data_preprocessing.py`.
4.  Open the `main.ipynb` to train the models using the preprocessed data.

# DeepLOB Model Training and Evaluation (`deeplob.ipynb`)

This file (`deeplob.ipynb`) defines a Deep Learning model (DeepLOB), trains the model, and evaluates its performance for predicting short-term price movements.

## Description

### 1. DeepLOB Model Definition

- **Defines the DeepLOB architecture:** The model implemented in this code is based on the paper:

DeepLOB: Deep Convolutional Neural Networks for Limit Order Books
Zihao Zhang, Stefan Zohren, and Stephen Roberts

### 2. Model Training and Evaluation

- **Implements the training loop:**
  - Calculating loss (CrossEntropyLoss).
  - Performing backpropagation and optimization using an optimizer (Adam or SGD).
- **Hyperparameter optimization:** The script uses Optuna to optimize the model's hyperparameters (e.g., learning rate, number of layers, etc.) to improve performance.
- **Evaluation:** Generating a classification report (precision, recall, F1-score) and creating a confusion matrix.
- **Backtesting:** The script includes a basic backtesting procedure to simulate a trading strategy based on the model's predictions.
