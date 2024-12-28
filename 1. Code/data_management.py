import os
import shutil
from config import (
    BASE_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CLEANED_DATA_FILE,
    SCALED_DATA_FILE,
    ENCODED_DATA_FILE,
    FINANCIAL_RATIOS_FILE,
    TEST_DATA_FILE,
    TRAIN_DATA_FILE
)

def setup_data_directories():
    """Create and verify data directory structure"""
    # Create directories
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Move any existing processed files to correct directory
    files_to_check = [
        CLEANED_DATA_FILE,
        SCALED_DATA_FILE,
        ENCODED_DATA_FILE,
        FINANCIAL_RATIOS_FILE,
        TEST_DATA_FILE,
        TRAIN_DATA_FILE
    ]
    
    for file_path in files_to_check:
        filename = os.path.basename(file_path)
        old_path = os.path.join(BASE_DIR, filename)
        if os.path.exists(old_path):
            shutil.move(old_path, file_path)
            print(f"Moved {filename} to {PROCESSED_DATA_DIR}")
    
    return {
        'raw_dir': RAW_DATA_DIR,
        'processed_dir': PROCESSED_DATA_DIR,
        'files': {
            'cleaned': CLEANED_DATA_FILE,
            'scaled': SCALED_DATA_FILE,
            'encoded': ENCODED_DATA_FILE,
            'financial': FINANCIAL_RATIOS_FILE,
            'test': TEST_DATA_FILE,
            'train': TRAIN_DATA_FILE
        }
    }
