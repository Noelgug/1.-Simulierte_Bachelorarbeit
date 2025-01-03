import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from data_loader import (load_cleaned_data, load_data, 
                        load_test_data, 
                        load_train_data)
from descriptive_analysis import run_descriptive_analysis
from data_cleaning import clean_data
from data_conversion import (scale_features, 
                                      print_scaling_info, 
                                      perform_one_hot_encoding, 
                                      print_encoding_info,
                                      calculate_financial_ratios, 
                                      print_ratio_stats,
                                      split_data, 
                                      print_split_stats)
from xgboost_model import train_xgboost_model
from data_management import setup_data_directories
from outlier_analysis import run_outlier_analysis
from visualization_runner import (create_all_visualizations, 
                                create_model_visualizations,
                                plot_roc_curve)
from random_forest_model import train_random_forest_model
from logistic_regression_model import train_logistic_regression_model


def process_data(data):
    """Process data through the complete pipeline"""
    # Clean data
    clean_data(data) # Print removal stats
    cleaned_data = load_cleaned_data()
    
    # Scale features
    scaled_data, scaling_info = scale_features(cleaned_data)
    print_scaling_info(scaling_info)
    
    # Perform one-hot encoding
    encoded_data, encoding_info = perform_one_hot_encoding(scaled_data)
    print_encoding_info(encoding_info)
    
    # Calculate financial ratios
    enhanced_data, ratio_stats = calculate_financial_ratios(encoded_data)
    print_ratio_stats(ratio_stats)
    
    # Split data into training and test sets
    split_stats, train_data, test_data = split_data(enhanced_data)
    print_split_stats(split_stats)
    
    return train_data, test_data

def main():
    # Setup data directories
    data_paths = setup_data_directories()
    print("Data directories initialized...")
    
    try:
        # Load data
        data = load_data()
    except FileNotFoundError as e:
        print(str(e))
        return
    
    # Choose which analyses to run
    RUN_DESCRIPTIVE = False
    RUN_OUTLIER_ANALYSIS = False
    CREATE_VISUALIZATIONS = False
    CLEAN_DATA = False
    XGBOOST_MODEL = False
    RANDOM_FOREST_MODEL = False
    LOGISTIC_REGRESSION_MODEL = True
    
    if RUN_DESCRIPTIVE:
        run_descriptive_analysis(data)
    
    if RUN_OUTLIER_ANALYSIS:
        run_outlier_analysis(data)
    
    if CREATE_VISUALIZATIONS:
        create_all_visualizations(data)
    
    # Process data and get train/test sets
    if CLEAN_DATA:
        train_data, test_data = process_data(data)
    else:
        try:
            # Try to load preprocessed data
            test_data = load_test_data()
            train_data = load_train_data()
        except FileNotFoundError:
            print("\nPreprocessed data files not found. Running data processing pipeline...")
            train_data, test_data = process_data(data)
    
    if XGBOOST_MODEL:
        print("\nStarting XGBoost Model Training...")
        
        # Prepare test data for ROC curve
        X_test = test_data.drop(['ID', 'default.payment.next.month'], axis=1)
        y_test = test_data['default.payment.next.month']
        
        # Train and evaluate model
        model_results = train_xgboost_model(train_data, test_data, use_optuna=True)
        
        # Create visualizations
        create_model_visualizations(model_results, 'xgboost')
        plot_roc_curve(model_results, X_test, y_test, 'xgboost', output_dir='Diagramms/XGBoost')
        
        print("\nModel Performance:")
        print(f"Accuracy: {model_results['accuracy']:.4f}")
        print(f"AUC Score: {model_results['auc_score']:.4f}")
        print("\nBest Parameters:", model_results['best_params'])

    if RANDOM_FOREST_MODEL:
        print("\nStarting Random Forest Model Training...")
        
        # Prepare test data for ROC curve
        X_test = test_data.drop(['ID', 'default.payment.next.month'], axis=1)
        y_test = test_data['default.payment.next.month']
        
        # Train and evaluate model
        model_results = train_random_forest_model(train_data, test_data, use_optuna=True)
        
        # Create visualizations
        create_model_visualizations(model_results, 'random_forest')
        plot_roc_curve(model_results, X_test, y_test, 'random_forest', output_dir='Diagramms/RandomForest')
        
        print("\nModel Performance:")
        print(f"Accuracy: {model_results['accuracy']:.4f}")
        print(f"AUC Score: {model_results['auc_score']:.4f}")
        print("\nBest Parameters:", model_results['best_params'])
        print("\nClassification Report:")
        print(model_results['classification_report'])

    if LOGISTIC_REGRESSION_MODEL:
        print("\nStarting Logistic Regression Model Training...")
        
        # Prepare test data for ROC curve
        X_test = test_data.drop(['ID', 'default.payment.next.month'], axis=1)
        y_test = test_data['default.payment.next.month']
        
        # Train and evaluate model
        model_results = train_logistic_regression_model(train_data, test_data, use_optuna=True)
        
        # Create visualizations
        create_model_visualizations(model_results, 'logistic_regression')
        plot_roc_curve(model_results, X_test, y_test, 'logistic_regression', 
                      output_dir='Diagramms/LogisticRegression')
        
        print("\nModel Performance:")
        print(f"Accuracy: {model_results['accuracy']:.4f}")
        print(f"AUC Score: {model_results['auc_score']:.4f}")
        print("\nBest Parameters:", model_results['best_params'])
        print("\nClassification Report:")
        print(model_results['classification_report'])

if __name__ == "__main__":
    main()
