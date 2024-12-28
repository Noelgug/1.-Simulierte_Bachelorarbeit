import pandas as pd
import os
from config import (CLEANED_DATA_FILE, 
                   DATA_FILE, RAW_DATA_DIR,
                   TEST_DATA_FILE, TRAIN_DATA_FILE)

def check_raw_data_exists():
    """Check if raw data file exists and guide user if it doesn't"""
    if not os.path.exists(DATA_FILE):
        print("\nERROR: Raw data file not found!")
        print(f"Please place the UCI_Credit_Card.csv file in the following directory:")
        print(f"{RAW_DATA_DIR}")
        print("\nYou can download the dataset from:")
        print("https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients")
        return False
    return True

def load_dataset(file_path):
    """Generic function to load any CSV dataset"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    return pd.read_csv(file_path)

def load_data():
    """Load the original credit card dataset"""
    if not check_raw_data_exists():
        raise FileNotFoundError("Please add the raw data file before proceeding.")
    return load_dataset(DATA_FILE)

def load_cleaned_data():
    """Load the cleaned credit card dataset"""
    return load_dataset(CLEANED_DATA_FILE)

def load_test_data():
    """Load the test dataset"""
    return load_dataset(TEST_DATA_FILE)

def load_train_data():
    """Load the training dataset"""
    return load_dataset(TRAIN_DATA_FILE)
