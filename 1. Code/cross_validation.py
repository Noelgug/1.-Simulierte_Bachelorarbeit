from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd

def perform_cross_validation(model, X, y, n_splits=10, random_state=42):
    """
    Perform cross-validation for any classifier model
    
    Args:
        model: The classifier model (e.g., XGBoost, Random Forest, Logistic Regression)
        X: Feature matrix
        y: Target vector
        n_splits: Number of folds for cross-validation
        random_state: Random seed for reproducibility
    
    Returns:
        dict: Dictionary containing cross-validation results and metrics
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Initialize metrics storage
    cv_scores = {
        'accuracy': [],
        'auc': [],
        'confusion_matrices': [],
        'classification_reports': []
    }
    
    fold_predictions = []
    
    # Perform k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        # Split data for this fold
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Clone and train the model
        fold_model = model.__class__(**model.get_params())
        fold_model.fit(X_train_fold, y_train_fold)
        
        # Make predictions
        y_pred = fold_model.predict(X_val_fold)
        y_pred_proba = fold_model.predict_proba(X_val_fold)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_val_fold, y_pred)
        auc = roc_auc_score(y_val_fold, y_pred_proba)
        conf_matrix = confusion_matrix(y_val_fold, y_pred)
        class_report = classification_report(y_val_fold, y_pred)
        
        # Store results
        cv_scores['accuracy'].append(accuracy)
        cv_scores['auc'].append(auc)
        cv_scores['confusion_matrices'].append(conf_matrix)
        cv_scores['classification_reports'].append(class_report)
        
        fold_predictions.append({
            'fold': fold,
            'true_values': y_val_fold,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        })
    
    # Calculate summary statistics
    cv_summary = {
        'mean_accuracy': np.mean(cv_scores['accuracy']),
        'std_accuracy': np.std(cv_scores['accuracy']),
        'mean_auc': np.mean(cv_scores['auc']),
        'std_auc': np.std(cv_scores['auc']),
        'individual_scores': cv_scores,
        'fold_predictions': fold_predictions
    }
    
    return cv_summary

def print_cv_results(cv_results, model_name="Model"):
    """
    Print formatted cross-validation results
    
    Args:
        cv_results: Dictionary containing cross-validation results
        model_name: Name of the model being evaluated
    """
    print(f"\n{model_name} Cross-Validation Results:")
    print("=" * 50)
    
    # Print accuracy scores
    print("\nAccuracy Scores:")
    for i, score in enumerate(cv_results['individual_scores']['accuracy'], 1):
        print(f"Fold {i}: {score:.4f}")
    print(f"Mean Accuracy: {cv_results['mean_accuracy']:.4f} "
          f"(+/- {cv_results['std_accuracy'] * 2:.4f})")
    
    # Print AUC scores
    print("\nAUC Scores:")
    for i, score in enumerate(cv_results['individual_scores']['auc'], 1):
        print(f"Fold {i}: {score:.4f}")
    print(f"Mean AUC: {cv_results['mean_auc']:.4f} "
          f"(+/- {cv_results['std_auc'] * 2:.4f})")
    
    # Print average confusion matrix
    print("\nAverage Confusion Matrix:")
    avg_conf_matrix = np.mean(cv_results['individual_scores']['confusion_matrices'], axis=0)
    print(avg_conf_matrix.astype(int))
    
    # Print the classification report for the last fold (as an example)
    print("\nExample Classification Report (Last Fold):")
    print(cv_results['individual_scores']['classification_reports'][-1])
