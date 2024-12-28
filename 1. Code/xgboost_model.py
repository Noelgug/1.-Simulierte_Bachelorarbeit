import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import optuna
from sklearn.metrics import accuracy_score, roc_auc_score
from cross_validation import perform_cross_validation, print_cv_results

def manual_cross_validation(X, y, model, n_splits=5):
    """Perform manual cross-validation"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model.fit(X_fold_train, y_fold_train)
        # Get predictions
        y_pred = model.predict(X_fold_val)
        # Calculate accuracy
        score = accuracy_score(y_fold_val, y_pred)
        scores.append(score)
    
    return np.array(scores)

def get_fixed_params():
    """Return the fixed hyperparameters for XGBoost model"""
    return {
        'max_depth': 7,
        'learning_rate': 0.093835,
        'n_estimators': 122,
        'min_child_weight': 7,
        'gamma': 0.6276,
        'subsample': 0.9152,
        'colsample_bytree': 0.6543,
        'reg_alpha': 0.4709,
        'reg_lambda': 0.4352,
        'objective': 'binary:logistic',
        'eval_metric': 'auc'
    }

def train_model_with_fixed_params(X_train, y_train, X_test, y_test):
    """Train XGBoost model with fixed parameters"""
    params = get_fixed_params()
    model = xgb.XGBClassifier(**params, random_state=42)
    
    # Perform cross-validation
    cv_results = perform_cross_validation(model, X_train, y_train)
    print_cv_results(cv_results, "XGBoost")
    
    # Train final model on full training data
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    return {
        'model': model,
        'best_params': params,
        'accuracy': accuracy_score(y_test, y_pred),
        'auc_score': roc_auc_score(y_test, y_pred_proba[:, 1]),
        'feature_importance': model.feature_importances_,
        'feature_names': X_train.columns.tolist(),
        'prediction_probabilities': y_pred_proba,
        'study': None,  # No study for fixed parameters
        'cv_results': cv_results  # Add cross-validation results
    }

def train_model_with_optuna(X_train, y_train, X_test, y_test, n_trials=100):
    """Train XGBoost model with Optuna optimization"""
    def objective(trial):
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'gamma': trial.suggest_float('gamma', 0, 1.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
            'objective': 'binary:logistic',
            'eval_metric': 'auc'
        }
        
        model = xgb.XGBClassifier(**param, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Train final model with best parameters
    best_params = study.best_params
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'auc'
    })
    
    model = xgb.XGBClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    return {
        'model': model,
        'best_params': best_params,
        'study': study,
        'accuracy': accuracy_score(y_test, y_pred),
        'auc_score': roc_auc_score(y_test, y_pred_proba[:, 1]),
        'feature_importance': model.feature_importances_,
        'feature_names': X_train.columns.tolist(),
        'prediction_probabilities': y_pred_proba
    }

def train_xgboost_model(train_data, test_data, use_optuna=False):
    """Train XGBoost model with either fixed parameters or Optuna optimization"""
    # Prepare data
    X_train = train_data.drop(['ID', 'default.payment.next.month'], axis=1)
    y_train = train_data['default.payment.next.month']
    X_test = test_data.drop(['ID', 'default.payment.next.month'], axis=1)
    y_test = test_data['default.payment.next.month']
    
    if use_optuna:
        results = train_model_with_optuna(X_train, y_train, X_test, y_test)
    else:
        results = train_model_with_fixed_params(X_train, y_train, X_test, y_test)
    
    # Print detailed results with all data
    print_model_results(results, X_train, y_train, X_test, y_test)
    
    return results

def calculate_cv_scores(model, X_train, y_train, n_splits=5):
    """Calculate cross-validation scores manually"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        # Split data
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train model
        fold_model = xgb.XGBClassifier(**model.get_params())
        fold_model.fit(X_fold_train, y_fold_train)
        
        # Calculate score
        y_pred = fold_model.predict(X_fold_val)
        scores.append(accuracy_score(y_fold_val, y_pred))
    
    return np.array(scores)

def print_model_results(model_results, X_train=None, y_train=None, X_test=None, y_test=None):
    """Print formatted model results and statistics"""
    print("\nXGBoost Model Results:")
    print("=" * 50)
    
    # Cross Validation Scores
    if X_train is not None and y_train is not None:
        cv_scores = calculate_cv_scores(model_results['model'], X_train, y_train)
        print(f"\nCross Validation Scores: {cv_scores}")
        print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Test Performance
    print(f"\nTest Accuracy: {model_results['accuracy']:.4f}")
    
    # Classification Report
    if X_test is not None and y_test is not None:
        y_pred = model_results['model'].predict(X_test)
        print("\nClassification report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
    # Feature Importance
    print("\nTop 10 Most Important Features:")
    feature_importance = list(zip(model_results['feature_names'], 
                                model_results['feature_importance']))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for feature, importance in feature_importance[:10]:
        print(f" {feature}: {importance:.4f}")
    
    # Prediction Probabilities (show first few)
    print("\nPrediction probabilities (first 5 samples):")
    print(model_results['prediction_probabilities'][:5])
