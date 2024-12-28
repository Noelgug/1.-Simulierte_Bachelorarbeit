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

def plot_optuna_results(study, output_dir='Diagramms/Optuna'):
    """Create and save Optuna visualization plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot optimization history
    fig = optv.plot_optimization_history(study)
    fig.write_image(os.path.join(output_dir, "optimization_history.png"))
    
    # Plot parameter importances
    fig = optv.plot_param_importances(study)
    fig.write_image(os.path.join(output_dir, "param_importances.png"))
    
    # Plot parallel coordinate
    fig = optv.plot_parallel_coordinate(study)
    fig.write_image(os.path.join(output_dir, "parallel_coordinate.png"))
    
    plt.close('all')

def plot_feature_importance(model_results, output_dir='Diagramms/XGBoost'):
    """Plot feature importance from XGBoost model"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(model_results['feature_importance'])), 
            model_results['feature_importance'])
    plt.xticks(range(len(model_results['feature_importance'])), 
               model_results['feature_names'], rotation=45, ha='right')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()

def plot_roc_curve(model_results, X_test, y_test, output_dir='Diagramms/XGBoost'):
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
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

def create_model_visualizations(model_results):
    """Create all model-related visualizations"""
    if model_results['study'] is not None:
        plot_optuna_results(model_results['study'])
    plot_feature_importance(model_results)
    
    # Note: ROC curve needs to be called separately since it requires test data
