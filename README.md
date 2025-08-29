# House Price Predictor 
<p> Hello! This is a Machine Learning project designed to predict real estate prices based on their characteristics. It uses a Random Forest Regressor model and is structured in a modular and easy-to-use way, with a complete pipeline that ranges from data preprocessing to forecast generation.

## Project Structure 
.
├── data/
│   ├── raw/
│   │   ├── train.csv         # Train data
│   │   └── test.csv          # Test data
│   └── processed/
│       └── submission.csv    # Submission file
├── models/
│   ├── house_price_model.pkl # Trained model saved
│   └── model_features.txt    # Features list
├── reports/
│   └── figures/
│       ├── actual_vs_predicted.png # Scatter plot
│       └── prediction_errors.png   # Errors histogram
├── src/
│   ├── __init__.py
│   ├── config.py             # All project configs
│   ├── evaluate.py           # Logic for evaluating the model and generating predictions
│   ├── preprocessing.py      # Cleanup functions and feature engineering
│   ├── train.py              # Logic for training the model
│   └── validation.py         # Functions to validate data
├── main.py                   # Entry point to execute the pipeline
└── requirements.txt          # List of required libraries

## Prerequisites

- **Python 3.8 or higher**
- **Pip (Python package manager)**

## How to run
1. First, clone the repository to your machine:
git clone https://github.com/VictorHFerreira016/house-price-predictor
cd house-price-predictor

**Create a virtual environment**

- python -m venv .venv
- source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

Now, install all the necessary libraries with a single command:

- pip install -r requirements.txt

2. Preparing the data

Place your data files (train.csv and test.csv) in the data/raw/ folder. The structure should be exactly as shown in the "Project Structure" section.

**How to use**

- python main.py train
This command will load the data from data/raw/train.csv, preprocess it, train a new RandomForestRegressor model, and save the trained model to models/house_price_model.pkl.

- python main.py evaluate
This command uses the previously trained model to make predictions on the same training data (train.csv) and calculates performance metrics. It also generates evaluation graphs and saves them in the reports/figures/ folder.

- python main.py predict