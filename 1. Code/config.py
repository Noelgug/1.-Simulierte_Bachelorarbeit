import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'Raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'Processed')

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Input data file (Raw data)
DATA_FILE = os.path.join(RAW_DATA_DIR, 'UCI_Credit_Card.csv')

# Processed data files
CLEANED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'cleaned_credit_card_data.csv')
SCALED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'scaled_credit_card_data.csv')
ENCODED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'encoded_credit_card_data.csv')
FINANCIAL_RATIOS_FILE = os.path.join(PROCESSED_DATA_DIR, 'financial_ratios_data.csv')
TEST_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'test_data.csv')
TRAIN_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'train_data.csv')

# Output directories for visualizations
DIAGRAMS_DIR = 'Diagramms'
BILL_AMT_DIR_V = os.path.join(DIAGRAMS_DIR, 'BILL_AMT', 'Verteilung')
BILL_AMT_DIR_A = os.path.join(DIAGRAMS_DIR, 'BILL_AMT', 'Ausreisser')
BILL_Z_SCORE_DIR = os.path.join(DIAGRAMS_DIR, 'BILL_AMT', 'Z-Score')
PAY_AMT_DIR_V = os.path.join(DIAGRAMS_DIR, 'PAY_AMT', 'Verteilung')
PAY_AMT_DIR_A = os.path.join(DIAGRAMS_DIR, 'PAY_AMT', 'Ausreisser')
PAY_Z_SCORE_DIR = os.path.join(DIAGRAMS_DIR, 'PAY_AMT', 'Z-Score')
