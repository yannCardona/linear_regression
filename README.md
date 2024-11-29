# Linear Regression Model

## Project Overview
This project implements a simple **Linear Regression model** using Python. The model uses **gradient descent** for optimization and includes various utilities for data preprocessing, model training, and performance evaluation. The main goal of this project is to predict car prices based on mileage data.

## Features
- **Gradient Descent Algorithm**: Optimizes the linear model parameters (slope and intercept) to minimize the loss function.
- **Normalization**: Normalizes the input data with Z-score normalization for better training performance.
- **Loss Functions**:
  - **Mean Squared Error (MSE)**: The primary loss function used for training.
  - **Evaluation Metrics**: Includes **R² score** and **Mean Absolute Error (MAE)** to measure model performance.
- **Data Visualization**: Built-in plotting functions to visualize the dataset and the linear regression line.

## Requirements
- **Python 3.x** (Recommended: Python 3.7+)
- **pandas**: For data manipulation
- **numpy**: For numerical operations
- **matplotlib**: For plotting graphs

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yannCardona/linear-regression.git
   cd linear-regression
   ```
2. Set Up a Virtual Environment (for Mac)
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main.py script to train the model using your dataset.
   ```bash
   python main.py data.csv --learning_rate 0.01 --iterations 1000
   ```
- data.csv: Path to your CSV file containing the data.
- --learning_rate: (Optional) Learning rate for gradient descent. Default is 0.01.
- --iterations: (Optional) Number of training iterations. Default is 1000.
