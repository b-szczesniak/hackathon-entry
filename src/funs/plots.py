import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
)
    

def plot_groupby(df: pd.DataFrame, column: str, target: str, agg_func: str = 'mean') -> None:
    """
    Plotting function for grouped data\n
    :param df: Pandas DataFrame
    :param column: Column to group by
    :param target: Target column to plot
    :param agg_func: Aggregation function (default: 'mean')
    :return: None
    """
    
    plt.figure(figsize=(10, 6))
    data = df.groupby(column)[target].agg(agg_func).reset_index()
    ax = sns.boxplot(x=column, y=target, data=data)
    ax.set_title(f'Boxplot of {target} by {column}')
    ax.set_xlabel(column)
    ax.set_ylabel(target)
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, target: str, threshold_target: float) -> None:
    """
    Plotting function for correlation matrix\n
    :param df: Pandas DataFrame
    :param columns: List of columns to include in the correlation matrix
    :param target: Target column for correlation
    :return: None
    """

    numeric_data = df.select_dtypes(include=['number', 'bool'])
    corr_matrix = numeric_data.corr()

    threshold_general = 1       # general correlation threshold

    # Remove diagonal self-correlation
    corr_no_diag = corr_matrix.copy()
    np.fill_diagonal(corr_no_diag.values, 0)

    condition_general = (np.abs(corr_no_diag) >= threshold_general).any(axis=0)
    condition_target = np.abs(corr_matrix[target]) >= threshold_target
    keep_columns = condition_general | condition_target
    filtered_corr = corr_matrix.loc[keep_columns, keep_columns]

    plt.figure(figsize=(9, 7))
    sns.heatmap(filtered_corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
    plt.title(f'Filtered Correlation Matrix (Thresholds: General ≥ {threshold_general}, {target} ≥ {threshold_target})',
          pad=20, fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Plotting function for confusion matrix\n
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: None
    """

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def plot_roc(y_test: np.ndarray, y_pred_prob: np.ndarray) -> None:
    """
    Plotting function for ROC curve\n
    :param y_test: True labels
    :param y_pred_prob: Predicted probabilities
    :return: None
    """

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 4))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()