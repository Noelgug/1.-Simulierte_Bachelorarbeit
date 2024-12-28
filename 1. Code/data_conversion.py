import numpy as np
from descriptive_stats import analyze_bill_amt_outliers_z, analyze_pay_amt_outliers_z
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from config import (SCALED_DATA_FILE, 
                   ENCODED_DATA_FILE, 
                   FINANCIAL_RATIOS_FILE, 
                   TEST_DATA_FILE, 
                   TRAIN_DATA_FILE)

def remove_zscore_outliers(data, output_path):
    """
    Remove outliers identified by Z-Score method using row IDs and save cleaned data to CSV
    
    Args:
        data (pd.DataFrame): Original DataFrame
        output_path (str): Path where to save the cleaned CSV file
    
    Returns:
        dict: Statistics about removed outliers
    """
    # Get outliers information
    bill_outliers = analyze_bill_amt_outliers_z(data)
    pay_outliers = analyze_pay_amt_outliers_z(data)
    
    # Track statistics
    stats = {
        'original_rows': len(data),
        'removed_by_column': {},
        'ids_removed': set()  # Use set to avoid counting duplicates
    }
    
    # Collect all IDs to remove
    for outliers_dict in [bill_outliers, pay_outliers]:
        for column, info in outliers_dict.items():
            stats['removed_by_column'][column] = info['count']
            stats['ids_removed'].update(info['ids'])
    
    # Remove rows with collected IDs
    cleaned_data = data[~data['ID'].isin(stats['ids_removed'])]
    cleaned_data.to_csv(output_path, index=False)
    
    # Calculate final statistics
    stats['total_removed'] = len(stats['ids_removed'])
    stats['remaining_rows'] = len(cleaned_data)
    stats['removal_percentage'] = round(stats['total_removed'] / stats['original_rows'] * 100, 2)
    
    return stats

def print_removal_stats(stats):
    """
    Print formatted statistics about removed outliers
    
    Args:
        stats (dict): Statistics dictionary from remove_zscore_outliers
    """
    print("\nOutlier Removal Statistics:")
    print(f"Original rows: {stats['original_rows']}")
    print(f"Rows removed: {stats['total_removed']}")
    print(f"Rows remaining: {stats['remaining_rows']}")
    print(f"Removal percentage: {stats['removal_percentage']}%")
    
    print("\nRows removed by column:")
    for column, count in stats['removed_by_column'].items():
        print(f"  {column}: {count} rows")

def scale_features(data, output_path=SCALED_DATA_FILE):
    """
    Apply MinMax scaling to specified numeric columns and save to CSV
    
    Args:
        data (pd.DataFrame): DataFrame containing the cleaned data
        output_path (str): Path where to save the scaled CSV file
    
    Returns:
        tuple: (scaled DataFrame, dict with scaling info)
    """
    # Create copy of data
    scaled_data = data.copy()
    
    # Define columns to scale
    columns_to_scale = ['LIMIT_BAL', 'AGE']
    columns_to_scale.extend([f'BILL_AMT{i}' for i in range(1, 7)])
    columns_to_scale.extend([f'PAY_AMT{i}' for i in range(1, 7)])
    
    # Initialize scaler
    scaler = MinMaxScaler()
    
    # Store scaling information
    scaling_info = {
        'columns_scaled': columns_to_scale,
        'original_ranges': {},
        'scaled_ranges': {}
    }
    
    # Scale selected columns
    scaled_values = scaler.fit_transform(scaled_data[columns_to_scale])
    
    # Store original and scaled ranges
    for idx, column in enumerate(columns_to_scale):
        original_values = scaled_data[column]
        scaled_column_values = scaled_values[:, idx]
        
        scaling_info['original_ranges'][column] = {
            'min': float(original_values.min()),
            'max': float(original_values.max())
        }
        scaling_info['scaled_ranges'][column] = {
            'min': float(scaled_column_values.min()),
            'max': float(scaled_column_values.max())
        }
        
        # Update the data with scaled values
        scaled_data[column] = scaled_column_values
    
    # Save scaled data
    scaled_data.to_csv(output_path, index=False)
    
    return scaled_data, scaling_info

def print_scaling_info(scaling_info):
    """
    Print formatted information about the scaling process
    
    Args:
        scaling_info (dict): Dictionary containing scaling information
    """
    print("\nFeature Scaling Information:")
    print(f"Number of scaled columns: {len(scaling_info['columns_scaled'])}")
    
    print("\nScaling ranges for each column:")
    for column in scaling_info['columns_scaled']:
        orig = scaling_info['original_ranges'][column]
        scaled = scaling_info['scaled_ranges'][column]
        print(f"\n{column}:")
        print(f"  Original range: [{orig['min']:.2f}, {orig['max']:.2f}]")
        print(f"  Scaled range:   [{scaled['min']:.2f}, {scaled['max']:.2f}]")

def perform_one_hot_encoding(data, output_path=ENCODED_DATA_FILE):
    """
    Perform one-hot encoding for SEX, EDUCATION, and MARRIAGE columns and remove original columns
    
    Args:
        data (pd.DataFrame): DataFrame containing the data
        output_path (str): Path where to save the encoded CSV file
    
    Returns:
        tuple: (encoded DataFrame, dict with encoding info)
    """
    # Create copy of data
    encoded_data = data.copy()
    
    # Track encoding information
    encoding_info = {
        'original_columns': ['SEX', 'EDUCATION', 'MARRIAGE'],
        'new_columns': [],
        'removed_columns': []
    }
    
    # SEX encoding (1=male, 2=female)
    encoded_data['MALE'] = (encoded_data['SEX'] == 1).astype(int)
    encoded_data['FEMALE'] = (encoded_data['SEX'] == 2).astype(int)
    encoding_info['new_columns'].extend(['MALE', 'FEMALE'])
    
    # EDUCATION encoding
    education_mapping = {
        1: 'GRADUATE_SCHOOL',
        2: 'UNIVERSITY',
        3: 'HIGH_SCHOOL',
        4: 'OTHERS_EDUCATION',
        5: 'UNKNOWN_EDUCATION',
        6: 'UNKNOWN_EDUCATION',
        0: 'UNKNOWN_EDUCATION'  # Handle potential 0 values
    }
    
    for value, column_name in education_mapping.items():
        if column_name not in encoded_data.columns:  # Avoid duplicate columns for UNKNOWN
            encoded_data[column_name] = (encoded_data['EDUCATION'] == value).astype(int)
            encoding_info['new_columns'].append(column_name)
    
    # MARRIAGE encoding
    marriage_mapping = {
        1: 'MARRIED',
        2: 'SINGLE',
        3: 'OTHER_RELATIONSHIP'
    }
    
    for value, column_name in marriage_mapping.items():
        encoded_data[column_name] = (encoded_data['MARRIAGE'] == value).astype(int)
        encoding_info['new_columns'].append(column_name)
    
    # Remove original categorical columns
    for column in encoding_info['original_columns']:
        encoding_info['removed_columns'].append(column)
        encoded_data.drop(column, axis=1, inplace=True)
    
    # Save encoded data
    encoded_data.to_csv(output_path, index=False)
    
    # Add verification info
    encoding_info['total_new_columns'] = len(encoding_info['new_columns'])
    encoding_info['verification'] = {
        'sex_sum': encoded_data[['MALE', 'FEMALE']].sum().to_dict(),
        'education_sum': encoded_data[['GRADUATE_SCHOOL', 'UNIVERSITY', 'HIGH_SCHOOL', 
                                     'OTHERS_EDUCATION', 'UNKNOWN_EDUCATION']].sum().to_dict(),
        'marriage_sum': encoded_data[['MARRIED', 'SINGLE', 'OTHER_RELATIONSHIP']].sum().to_dict()
    }
    
    return encoded_data, encoding_info

def print_encoding_info(encoding_info):
    """
    Print formatted information about the encoding process
    
    Args:
        encoding_info (dict): Dictionary containing encoding information
    """
    print("\nOne-Hot Encoding Information:")
    print(f"Original columns removed: {', '.join(encoding_info['removed_columns'])}")
    print(f"New columns created: {encoding_info['total_new_columns']}")
    
    print("\nVerification Counts:")
    print("\nSEX encoding:")
    for col, count in encoding_info['verification']['sex_sum'].items():
        print(f"  {col}: {int(count)} rows")
    
    print("\nEDUCATION encoding:")
    for col, count in encoding_info['verification']['education_sum'].items():
        print(f"  {col}: {int(count)} rows")
    
    print("\nMARRIAGE encoding:")
    for col, count in encoding_info['verification']['marriage_sum'].items():
        print(f"  {col}: {int(count)} rows")

def calculate_financial_ratios(data, output_path=FINANCIAL_RATIOS_FILE):
    """
    Calculate and scale financial ratios from credit card data
    
    Args:
        data (pd.DataFrame): DataFrame containing the encoded credit card data
        output_path (str): Path where to save the enhanced CSV file
    
    Returns:
        tuple: (enhanced DataFrame, dict with ratio statistics)
    """
    # Create copy of data
    enhanced_data = data.copy()
    
    # Track ratio statistics
    ratio_stats = {
        'credit_utilization': {},
        'payment_to_bill': {}
    }
    
    # Calculate credit utilization ratios (BILL_AMT / LIMIT_BAL)
    for i in range(1, 7):
        column_name = f'CREDIT_UTILIZATION_RATIO_{i}'
        # Handle zero division
        enhanced_data[column_name] = np.where(
            enhanced_data['LIMIT_BAL'] != 0,
            enhanced_data[f'BILL_AMT{i}'] / enhanced_data['LIMIT_BAL'],
            0
        )
        
        ratio_stats['credit_utilization'][column_name] = {
            'mean': float(enhanced_data[column_name].mean()),
            'median': float(enhanced_data[column_name].median()),
            'min': float(enhanced_data[column_name].min()),
            'max': float(enhanced_data[column_name].max())
        }
    
    # Calculate payment to bill ratios (PAY_AMT / BILL_AMT)
    for i in range(1, 7):
        column_name = f'PAYMENT_TO_BILL_RATIO_{i}'
        # Handle zero division
        enhanced_data[column_name] = np.where(
            enhanced_data[f'BILL_AMT{i}'] != 0,
            enhanced_data[f'PAY_AMT{i}'] / enhanced_data[f'BILL_AMT{i}'],
            0
        )
        
        ratio_stats['payment_to_bill'][column_name] = {
            'mean': float(enhanced_data[column_name].mean()),
            'median': float(enhanced_data[column_name].median()),
            'min': float(enhanced_data[column_name].min()),
            'max': float(enhanced_data[column_name].max())
        }
    
    # Scale the new ratio columns
    scaler = MinMaxScaler()
    ratio_columns = (
        [f'CREDIT_UTILIZATION_RATIO_{i}' for i in range(1, 7)] +
        [f'PAYMENT_TO_BILL_RATIO_{i}' for i in range(1, 7)]
    )
    
    # Store original values for statistics
    for column in ratio_columns:
        ratio_stats[column] = {
            'original_range': {
                'min': float(enhanced_data[column].min()),
                'max': float(enhanced_data[column].max())
            }
        }
    
    # Scale values
    scaled_ratios = scaler.fit_transform(enhanced_data[ratio_columns])
    
    # Update DataFrame with scaled values and store scaled ranges
    for idx, column in enumerate(ratio_columns):
        enhanced_data[column] = scaled_ratios[:, idx]
        ratio_stats[column]['scaled_range'] = {
            'min': float(enhanced_data[column].min()),
            'max': float(enhanced_data[column].max())
        }
    
    # Save enhanced data
    enhanced_data.to_csv(output_path, index=False)
    
    return enhanced_data, ratio_stats

def print_ratio_stats(ratio_stats):
    """
    Print formatted statistics about the calculated ratios
    
    Args:
        ratio_stats (dict): Dictionary containing ratio statistics
    """
    print("\nFinancial Ratio Statistics:")
    
    print("\nCredit Utilization Ratios (BILL_AMT / LIMIT_BAL):")
    for i in range(1, 7):
        column = f'CREDIT_UTILIZATION_RATIO_{i}'
        print(f"\n{column}:")
        print("  Original range: "
              f"[{ratio_stats[column]['original_range']['min']:.2f}, "
              f"{ratio_stats[column]['original_range']['max']:.2f}]")
        print("  Scaled range: "
              f"[{ratio_stats[column]['scaled_range']['min']:.2f}, "
              f"{ratio_stats[column]['scaled_range']['max']:.2f}]")
    
    print("\nPayment to Bill Ratios (PAY_AMT / BILL_AMT):")
    for i in range(1, 7):
        column = f'PAYMENT_TO_BILL_RATIO_{i}'
        print(f"\n{column}:")
        print("  Original range: "
              f"[{ratio_stats[column]['original_range']['min']:.2f}, "
              f"{ratio_stats[column]['original_range']['max']:.2f}]")
        print("  Scaled range: "
              f"[{ratio_stats[column]['scaled_range']['min']:.2f}, "
              f"{ratio_stats[column]['scaled_range']['max']:.2f}]")

def split_data(data, train_output=TRAIN_DATA_FILE, test_output=TEST_DATA_FILE, test_size=0.2, random_state=42):
    """
    Split data into training and test sets
    
    Args:
        data (pd.DataFrame): DataFrame to split
        train_output (str): Path where to save training data
        test_output (str): Path where to save test data
        test_size (float): Proportion of data to use for testing (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
    
    Returns:
        tuple: (dict with split statistics, training DataFrame, test DataFrame)
    """
    # Create copy of data
    data_copy = data.copy()
    
    # Split the data
    train_data, test_data = train_test_split(
        data_copy,
        test_size=test_size,
        random_state=random_state,
        stratify=data_copy['default.payment.next.month']  # Ensure balanced splits
    )
    
    # Save split datasets
    train_data.to_csv(train_output, index=False)
    test_data.to_csv(test_output, index=False)
    
    # Calculate split statistics
    split_stats = {
        'total_rows': len(data),
        'train_rows': len(train_data),
        'test_rows': len(test_data),
        'train_percentage': (len(train_data) / len(data)) * 100,
        'test_percentage': (len(test_data) / len(data)) * 100,
        'train_default_ratio': train_data['default.payment.next.month'].mean() * 100,
        'test_default_ratio': test_data['default.payment.next.month'].mean() * 100
    }
    
    return split_stats, train_data, test_data

def print_split_stats(split_stats):
    """
    Print formatted information about the data split
    
    Args:
        split_stats (dict): Dictionary containing split statistics
    """
    print("\nData Split Statistics:")
    print(f"Total rows: {split_stats['total_rows']}")
    print(f"Training set: {split_stats['train_rows']} rows "
          f"({split_stats['train_percentage']:.1f}%)")
    print(f"Test set: {split_stats['test_rows']} rows "
          f"({split_stats['test_percentage']:.1f}%)")
    print("\nDefault Payment Distribution:")
    print(f"Training set: {split_stats['train_default_ratio']:.1f}% default rate")
    print(f"Test set: {split_stats['test_default_ratio']:.1f}% default rate")