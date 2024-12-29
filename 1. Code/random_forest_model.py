import optuna
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report, 
                           confusion_matrix, roc_curve, auc)
import os
from adjust_class_weight import calculate_scale_pos_weight

def objective(trial, X_train, y_train):
    """Optuna objective for Random Forest hyperparameter tuning"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
    }
    
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    return scores.mean()

def calculate_detailed_cv_scores(X, y, model_params, n_splits=10):
    """Calculate detailed cross-validation scores"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracy_scores = []
    auc_scores = []
    confusion_matrices = []
    last_fold_report = None
    
    # Use model_params directly as they should already include base parameters
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        # Split data
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model = RandomForestClassifier(**model_params)
        model.fit(X_fold_train, y_fold_train)
        
        # Predictions
        y_pred = model.predict(X_fold_val)
        y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
        
        # Calculate metrics
        accuracy_scores.append(accuracy_score(y_fold_val, y_pred))
        auc_scores.append(roc_auc_score(y_fold_val, y_pred_proba))
        confusion_matrices.append(confusion_matrix(y_fold_val, y_pred))
        
        # Store last fold's classification report
        if fold == n_splits:
            last_fold_report = classification_report(y_fold_val, y_pred)
    
    return {
        'accuracy_scores': accuracy_scores,
        'auc_scores': auc_scores,
        'confusion_matrices': confusion_matrices,
        'last_fold_report': last_fold_report
    }

def save_model_results(model_results, output_dir='Results/RandomForest'):
    """Save model results to a text file"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'Endresult_RandomForest.txt')
    
    with open(output_file, 'w') as f:
        # Cross-validation results
        if 'cv_results' in model_results:
            cv = model_results['cv_results']
            
            f.write("Random Forest Cross-Validation Results:\n")
            f.write("=" * 50 + "\n\n")
            
            # Accuracy scores
            f.write("Accuracy Scores:\n")
            for i, score in enumerate(cv['accuracy_scores'], 1):
                f.write(f"Fold {i}: {score:.4f}\n")
            mean_acc = np.mean(cv['accuracy_scores'])
            std_acc = np.std(cv['accuracy_scores'])
            f.write(f"Mean Accuracy: {mean_acc:.4f} (+/- {std_acc*2:.4f})\n\n")
            
            # AUC scores
            f.write("AUC Scores:\n")
            for i, score in enumerate(cv['auc_scores'], 1):
                f.write(f"Fold {i}: {score:.4f}\n")
            mean_auc = np.mean(cv['auc_scores'])
            std_auc = np.std(cv['auc_scores'])
            f.write(f"Mean AUC: {mean_auc:.4f} (+/- {std_auc*2:.4f})\n\n")
            
            # Average confusion matrix
            avg_conf_matrix = np.mean(cv['confusion_matrices'], axis=0)
            f.write("Average Confusion Matrix:\n")
            f.write(str(avg_conf_matrix.astype(int)) + "\n\n")
            
            # Last fold classification report
            f.write("Example Classification Report (Last Fold):\n")
            f.write(cv['last_fold_report'] + "\n\n")
        
        # Final model results
        f.write("Random Forest Model Results:\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Test Accuracy: {model_results['accuracy']:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(model_results['classification_report'] + "\n\n")
        
        # Feature importance
        f.write("Top 10 Most Important Features:\n")
        sorted_features = sorted(model_results['feature_importance'].items(), 
                               key=lambda x: x[1], reverse=True)[:10]
        for feature, importance in sorted_features:
            f.write(f" {feature}: {importance:.4f}\n")
        
        # Model performance summary
        f.write("\nModel Performance:\n")
        f.write(f"Accuracy: {model_results['accuracy']:.4f}\n")
        f.write(f"AUC Score: {model_results['auc_score']:.4f}\n")

def train_random_forest_model(train_data, test_data, use_optuna):
    """Train and evaluate Random Forest model with optional Optuna tuning"""
    # Prepare data
    X_train = train_data.drop(['ID', 'default.payment.next.month'], axis=1)
    y_train = train_data['default.payment.next.month']
    X_test = test_data.drop(['ID', 'default.payment.next.month'], axis=1)
    y_test = test_data['default.payment.next.month']

    # Calculate class weight
    scale_pos_weight = calculate_scale_pos_weight(y_train)
    class_weight = {0: 1, 1: scale_pos_weight}

    # Base parameters that will be used in all cases
    base_params = {
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': class_weight
    }

    if use_optuna:
        # Hyperparameter tuning with Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train), 
                      n_trials=17, show_progress_bar=True)
        
        # Combine Optuna's best params with base params
        best_params = {**base_params, **study.best_params}
    else:
        study = None
        best_params = base_params

    # Create and train model with parameters
    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    class_report = classification_report(y_test, y_pred)
    
    # Feature importance with proper column names
    feature_importance = dict(zip(X_train.columns, model.feature_importances_))
    
    # Calculate detailed cross-validation scores
    cv_results = calculate_detailed_cv_scores(X_train, y_train, best_params)
    
    # Add CV results to model results
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
    
    # Save results to file
    save_model_results(results)
    
    return results
