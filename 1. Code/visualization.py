import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config import (
    BILL_AMT_DIR_V, 
    BILL_AMT_DIR_A, 
    BILL_Z_SCORE_DIR, 
    PAY_AMT_DIR_V, 
    PAY_AMT_DIR_A,
    PAY_Z_SCORE_DIR)
from descriptive_stats import analyze_bill_amt_outliers_z, analyze_pay_amt_outliers_z

def plot_distributions(data, column_prefix, output_dir):
    """Plot distributions for specified columns with logarithmic scaling"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(1, 7):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        column = f'{column_prefix}{i}'
        
        # Original distribution
        sns.histplot(data[column], kde=True, bins=30, ax=ax1)
        ax1.set_title(f'Original Verteilung von {column}')
        ax1.set_xlabel('Betrag')
        ax1.set_ylabel('Häufigkeit')
        
        # Logarithmic distribution
        sns.histplot(np.log1p(data[column]), kde=True, bins=30, ax=ax2)
        ax2.set_title(f'Logarithmische Verteilung von {column}')
        ax2.set_xlabel('Log(Betrag)')
        ax2.set_ylabel('Häufigkeit')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{column}_Verteilung.png')
        plt.savefig(output_path)
        plt.close()

def plot_boxplots(data, column_prefix, output_dir):
    """Plot boxplots for specified columns
    
    Args:
        data: DataFrame containing the data
        column_prefix: Prefix of columns to plot (e.g., 'BILL_AMT' or 'PAY_AMT')
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(1, 7):
        plt.figure()
        column = f'{column_prefix}{i}'
        sns.boxplot(x=data[column])
        plt.title(f'Boxplot von {column}')
        plt.xlabel('Betrag')
        
        output_path = os.path.join(output_dir, f'{column}_Ausreisser.png')
        plt.savefig(output_path)
        plt.close()

# Wrapper functions to maintain existing interface
def plot_bill_distributions(data):
    plot_distributions(data, 'BILL_AMT', BILL_AMT_DIR_V)

def plot_payment_distributions(data):
    plot_distributions(data, 'PAY_AMT', PAY_AMT_DIR_V)

def plot_bill_boxplots(data):
    plot_boxplots(data, 'BILL_AMT', BILL_AMT_DIR_A)

def plot_payment_boxplots(data):
    plot_boxplots(data, 'PAY_AMT', PAY_AMT_DIR_A)

# z-score-Methode visualisation

def plot_outliers_z(data, column_prefix, output_dir, analyze_func, show_extreme_outliers=True):
    """Generic function to plot outliers using Z-Score method
    
    Args:
        data: DataFrame containing the data
        column_prefix: Prefix of columns to plot (e.g., 'BILL_AMT' or 'PAY_AMT')
        output_dir: Directory to save the plots
        analyze_func: Function to analyze outliers
        show_extreme_outliers: Whether to show top 10 extreme outliers
    """
    os.makedirs(output_dir, exist_ok=True)
    outliers_info = analyze_func(data)
    
    for column, info in outliers_info.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Original distribution with outliers
        sns.histplot(data[column], kde=True, bins=30, ax=ax1)
        ax1.set_title(f'Original Verteilung von {column}\n'
                     f'Z-Score Schwelle: {info["threshold"]}, '
                     f'Ausreisser: {info["count"]}')
        ax1.set_xlabel('Betrag')
        ax1.set_ylabel('Häufigkeit')
        
        # Add extreme outliers if requested
        if show_extreme_outliers and 'outliers' in info:
            extreme_outliers = sorted(info['outliers'], reverse=True)[:10]
            for outlier in extreme_outliers:
                ax1.axvline(outlier, color='r', linestyle='--', alpha=0.5)
        
        # Logarithmic distribution
        sns.histplot(np.log1p(data[column]), kde=True, bins=30, ax=ax2)
        ax2.set_title(f'Logarithmische Verteilung von {column}\n'
                     + ('Mit Top-10 extremen Ausreissern' if show_extreme_outliers else ''))
        ax2.set_xlabel('Log(Betrag)')
        ax2.set_ylabel('Häufigkeit')
        
        # Add outliers to logarithmic plot if requested
        if show_extreme_outliers and 'outliers' in info:
            for outlier in extreme_outliers:
                ax2.axvline(np.log1p(outlier), color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{column}_Z-Score_Ausreisser.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

def plot_bill_outliers_z(data):
    """Plot BILL_AMT outliers using Z-Score method"""
    plot_outliers_z(data, 'BILL_AMT', BILL_Z_SCORE_DIR, analyze_bill_amt_outliers_z, True)

def plot_payment_outliers_z(data):
    """Plot PAY_AMT outliers using Z-Score method"""
    plot_outliers_z(data, 'PAY_AMT', PAY_Z_SCORE_DIR, analyze_pay_amt_outliers_z, True)