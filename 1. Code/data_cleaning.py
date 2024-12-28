from data_conversion import remove_zscore_outliers, print_removal_stats
from config import CLEANED_DATA_FILE, PROCESSED_DATA_DIR
import os

def clean_data(data):
    """Clean data by removing outliers and save to file
    
    Args:
        data: DataFrame to clean
        
    Returns:
        tuple: (cleaned DataFrame, removal statistics)
    """
    # Ensure the processed directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Clean data and save to file
    removal_stats = remove_zscore_outliers(data, CLEANED_DATA_FILE)
    print_removal_stats(removal_stats)
    
    # Return the cleaned data
    return data
