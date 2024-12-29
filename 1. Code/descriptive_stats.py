import numpy as np
from scipy import stats

# Analyse der numerischen Variablen
def calculate_default_stats(data):
    """Calculate default payment statistics"""
    count_1 = data['default.payment.next.month'].sum()
    count_0 = len(data) - count_1
    percentage_1 = count_1 / len(data) * 100
    percentage_0 = count_0 / len(data) * 100
    
    return {
        'count_1': count_1,
        'count_0': count_0,
        'percentage_1': percentage_1,
        'percentage_0': percentage_0
    }

def calculate_limit_stats(data):
    """Calculate LIMIT_BAL statistics"""
    return {
        'min': data['LIMIT_BAL'].min(),
        'max': data['LIMIT_BAL'].max(),
        'median': data['LIMIT_BAL'].median()
    }

def calculate_age_stats(data):
    """Calculate AGE statistics"""
    return {
        'min': data['AGE'].min(),
        'max': data['AGE'].max(),
        'average': data['AGE'].mean()
    }

# Analyse der kategorischen Variablen
def calculate_sex_percentage(data):
    """Calculate the percentage of each gender in the SEX column
    
    Args:
        data: DataFrame containing the data
    
    Returns:
        A dictionary with the percentage of each gender
    """
    total_count = len(data)
    male_count = (data['SEX'] == 1).sum()
    female_count = (data['SEX'] == 2).sum()
    
    male_percentage = round((male_count / total_count) * 100, 2)
    female_percentage = round((female_count / total_count) * 100, 2)
    
    return {
        'male_percentage': male_percentage,
        'female_percentage': female_percentage
    }

def calculate_education_percentage(data):
    """Calculate the percentage of educated individuals in the EDUCATION column
    
    Args:
        data: DataFrame containing the data
    
    Returns:
        The percentage of educated individuals
    """
    total_count = len(data)
    educated_count = (data['EDUCATION'] == 2).sum()
    
    educated_percentage = round((educated_count / total_count) * 100, 2)
    
    return educated_percentage

def calculate_marriage_percentage(data):
    """Calculate the percentage of each marital status in the MARRIAGE column
    
    Args:
        data: DataFrame containing the data
    
    Returns:
        A dictionary with the percentage of each marital status
    """
    total_count = len(data)
    married_count = (data['MARRIAGE'] == 1).sum()
    single_count = (data['MARRIAGE'] == 2).sum()
    others_count = (data['MARRIAGE'] == 3).sum()
    
    married_percentage = round((married_count / total_count) * 100, 2)
    single_percentage = round((single_count / total_count) * 100, 2)
    others_percentage = round((others_count / total_count) * 100, 2)
    
    return {
        'married_percentage': married_percentage,
        'single_percentage': single_percentage,
        'others_percentage': others_percentage
    }

# IQR-Methode
def analyze_outliers_iqr(data, column_prefix, num_columns=6):
    """Generic function to analyze outliers using IQR method
    
    Args:
        data: DataFrame containing the data
        column_prefix: Prefix of the columns to analyze (e.g., 'BILL_AMT' or 'PAY_AMT')
        num_columns: Number of columns to analyze (default: 6)
    """
    outliers_info = {}
    
    for i in range(1, num_columns + 1):
        column = f'{column_prefix}{i}'
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Find outliers
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        
        outliers_info[column] = {
            'count': len(outliers),
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    return outliers_info

def analyze_bill_amt_outliers(data):
    """Analyze outliers in BILL_AMT columns using IQR method"""
    return analyze_outliers_iqr(data, 'BILL_AMT')

def analyze_pay_amt_outliers(data):
    """Analyze outliers in PAY_AMT columns using IQR method"""
    return analyze_outliers_iqr(data, 'PAY_AMT')

# Z-Score-Methode
def analyze_outliers_z(data, column_prefix, num_columns=6):
    """Analyze outliers in BILL_AMT columns using Z-Score method"""
    outliers_info = {}
    
    for i in range(1, 7):
        column = f'{column_prefix}{i}'
        z_scores = np.abs(stats.zscore(data[column]))
        threshold = 3
        
        # Find outliers
        outliers = data[z_scores > threshold]
        
        outliers_info[column] = {
            'count': len(outliers),
            'threshold': threshold,
            'outliers': outliers[column].values.tolist(),
            'ids': outliers['ID'].values.tolist()}
    return outliers_info

def analyze_pay_amt_outliers_z(data):
    """Analyze outliers in PAY_AMT columns using Z-Score method"""
    return analyze_outliers_z(data, 'PAY_AMT')

def analyze_bill_amt_outliers_z(data):
    """Analyze outliers in BILL_AMT columns using Z-Score method"""    
    return analyze_outliers_z(data, 'BILL_AMT')