from visualization import (
    plot_bill_distributions,
    plot_payment_boxplots,
    plot_bill_boxplots,
    plot_payment_distributions,
    plot_bill_outliers_z,
    plot_payment_outliers_z
)

import matplotlib.pyplot as plt
import optuna.visualization as optv
import os
from sklearn.metrics import roc_curve, auc

def create_all_visualizations(data):
    """Create all visualizations"""
    plot_bill_distributions(data)
    plot_payment_boxplots(data)
    plot_bill_boxplots(data)
    plot_payment_distributions(data)
    plot_bill_outliers_z(data)
    plot_payment_outliers_z(data)

def plot_optuna_results(study, model_name, output_dir='Diagramms/Optuna'):
    """Create and save Optuna visualization plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot optimization history
    fig = optv.plot_optimization_history(study)
    fig.write_image(os.path.join(output_dir, f"optimization_history_{model_name}.png"))
    
    # Plot parameter importances
    fig = optv.plot_param_importances(study)
    fig.write_image(os.path.join(output_dir, f"param_importances_{model_name}.png"))
    
    # Plot parallel coordinate
    fig = optv.plot_parallel_coordinate(study)
    fig.write_image(os.path.join(output_dir, f"parallel_coordinate_{model_name}.png"))
    
    plt.close('all')

def plot_feature_importance(feature_importance, model_name, output_dir='Diagramms/Model'):
    """Plot feature importance from model results
    
    Args:
        feature_importance: Either a dictionary of feature importances or the raw importance values
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if isinstance(feature_importance, dict):
        features = list(feature_importance.keys())
        importance_values = list(feature_importance.values())
    else:
        features = [f"Feature {i}" for i in range(len(feature_importance))]
        importance_values = feature_importance
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importance_values)), importance_values)
    plt.xticks(range(len(features)), features, rotation=45, ha='right')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'feature_importance_{model_name}.png'))
    plt.close()

def plot_roc_curve(model_results, X_test, y_test, model_name, output_dir):
    """Create and save ROC curve plot"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions
    y_pred_proba = model_results['model'].predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve points
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create ROC curve plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Save plot with model name
    plt.savefig(os.path.join(output_dir, f'roc_curve_{model_name}.png'))
    plt.close()

def create_model_visualizations(model_results, model_name):
    """Create visualizations for model results"""
    # Plot feature importance
    if 'feature_importance' in model_results:
        plot_feature_importance(model_results['feature_importance'], model_name)
    
    # Plot optimization history if study exists
    if 'study' in model_results and model_results['study'] is not None:
        plot_optuna_results(model_results['study'], model_name)
