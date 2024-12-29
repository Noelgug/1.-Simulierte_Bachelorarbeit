import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report, 
                           confusion_matrix)
import optuna
import os
from adjust_class_weight import calculate_scale_pos_weight

def objective(trial, X_train, y_train):
    """Optuna objective for Logistic Regression hyperparameter tuning"""
    # First choose solver since it affects which penalties are valid
    solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga'])
    
    # Select penalty based on solver
    if solver == 'lbfgs':
        penalty = 'l2'  # lbfgs only supports l2 penalty
    else:
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    
    params = {
        'C': trial.suggest_float('C', 0.001, 10.0, log=True),
        'max_iter': trial.suggest_int('max_iter', 500, 2000),
        'solver': solver,
        'penalty': penalty
    }
    
    model = LogisticRegression(**params, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_fold_train, y_fold_train)
        y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
        scores.append(roc_auc_score(y_fold_val, y_pred_proba))
    
    return np.mean(scores)

def calculate_detailed_cv_scores(X, y, model_params, n_splits=10):
    """Calculate detailed cross-validation scores"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracy_scores = []
    auc_scores = []
    confusion_matrices = []
    last_fold_report = None
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = LogisticRegression(**model_params)
        model.fit(X_fold_train, y_fold_train)
        
        y_pred = model.predict(X_fold_val)
        y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
        
        accuracy_scores.append(accuracy_score(y_fold_val, y_pred))
        auc_scores.append(roc_auc_score(y_fold_val, y_pred_proba))
        confusion_matrices.append(confusion_matrix(y_fold_val, y_pred))
        
        if fold == n_splits:
            last_fold_report = classification_report(y_fold_val, y_pred)
    
    return {
        'accuracy_scores': accuracy_scores,
        'auc_scores': auc_scores,
        'confusion_matrices': confusion_matrices,
        'last_fold_report': last_fold_report
    }

def save_model_results(model_results, output_dir='Results/LogisticRegression'):
    """Save model results to a text file"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'Endresult_LogisticRegression.txt')
    
    with open(output_file, 'w') as f:
        f.write("Logistic Regression Results:\n")
        f.write("=" * 50 + "\n\n")
        
        if 'cv_results' in model_results:
            cv = model_results['cv_results']
            
            f.write("Cross-Validation Results:\n")
            f.write("-" * 30 + "\n\n")
            
            # Write accuracy scores
            mean_acc = np.mean(cv['accuracy_scores'])
            std_acc = np.std(cv['accuracy_scores'])
            f.write(f"Mean Accuracy: {mean_acc:.4f} (+/- {std_acc*2:.4f})\n")
            
            # Write AUC scores
            mean_auc = np.mean(cv['auc_scores'])
            std_auc = np.std(cv['auc_scores'])
            f.write(f"Mean AUC: {mean_auc:.4f} (+/- {std_auc*2:.4f})\n\n")
        
        f.write("Test Set Results:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Test Accuracy: {model_results['accuracy']:.4f}\n")
        f.write(f"Test AUC Score: {model_results['auc_score']:.4f}\n\n")
        
        f.write("Classification Report:\n")
        f.write(model_results['classification_report'] + "\n")
        
        if 'feature_importance' in model_results:
            f.write("\nFeature Coefficients:\n")
            for feature, coef in model_results['feature_importance'].items():
                f.write(f"{feature}: {coef:.4f}\n")

def train_logistic_regression_model(train_data, test_data, use_optuna):
    """Train and evaluate Logistic Regression model"""
    # Prepare data
    X_train = train_data.drop(['ID', 'default.payment.next.month'], axis=1)
    y_train = train_data['default.payment.next.month']
    X_test = test_data.drop(['ID', 'default.payment.next.month'], axis=1)
    y_test = test_data['default.payment.next.month']
    
    # Calculate class weight
    class_weight = {0: 1, 1: calculate_scale_pos_weight(y_train)}
    
    base_params = {
        'random_state': 42,
        'class_weight': class_weight
    }
    
    if use_optuna:
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train), 
                      n_trials=100)
        
        best_params = {**base_params, **study.best_params}
    else:
        study = None
        best_params = {**base_params, 'C': 1.0, 'max_iter': 200, 
                      'solver': 'lbfgs', 'penalty': 'l2'}
    
    # Train final model
    model = LogisticRegression(**best_params)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    class_report = classification_report(y_test, y_pred)
    
    # Get feature importance (coefficients)
    feature_importance = dict(zip(X_train.columns, model.coef_[0]))
    
    # Calculate detailed cross-validation scores
    cv_results = calculate_detailed_cv_scores(X_train, y_train, best_params)
    
    results = {
        'model': model,
        'accuracy': accuracy,
        'auc_score': auc_score,
        'classification_report': class_report,
        'feature_importance': feature_importance,
        'best_params': best_params,
        'y_pred_proba': y_pred_proba,
        'study': study,
        'cv_results': cv_results
    }
    
    # Save results
    save_model_results(results)
    
    return results
