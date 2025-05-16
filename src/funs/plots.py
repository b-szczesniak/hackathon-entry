import numpy as np
from typing import Any
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, confusion_matrix,
    classification_report, roc_auc_score
)
from funs.utils import find_best_threshold

def plot_groupby(df: pd.DataFrame, column: str, target: str, agg_func: str = 'mean') -> None:
    """
    #### Plot a boxplot of an aggregated target by group.
    This function groups the DataFrame by a specified column and applies an aggregation function to the target column.
    It then plots a boxplot of the aggregated target values by the group.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Column to group by.
        target (str): Target column to aggregate.
        agg_func (str): Aggregation function to apply ('mean', 'median', etc.).
    Returns:
        None: Displays the plot.
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
    #### Plot a filtered correlation heatmap for numeric columns.
    Keeps columns whose absolute correlation
    with any other column ≥ 1 (general) or
    with the target ≥ threshold_target.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        target (str): Target column to filter by.
        threshold_target (float): Correlation threshold for the target column.
    Returns:
        None: Displays the plot.
    """
    numeric_data = df.select_dtypes(include=['number', 'bool'])
    corr_matrix = numeric_data.corr()

    threshold_general = 1.0  # general correlation threshold

    # zero out self-correlations to apply general threshold
    corr_no_diag = corr_matrix.copy()
    np.fill_diagonal(corr_no_diag.values, 0)

    # columns meeting either threshold
    cond_gen = (np.abs(corr_no_diag) >= threshold_general).any(axis=0)
    cond_tgt = np.abs(corr_matrix[target]) >= threshold_target
    keep = cond_gen | cond_tgt
    filtered_corr = corr_matrix.loc[keep, keep]

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        filtered_corr,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        center=0,
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.title(
        f'Filtered Correlation Matrix '
        f'(General ≥ {threshold_general}, {target} ≥ {threshold_target})',
        pad=20,
        fontsize=14
    )
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    #### Plot a confusion matrix heatmap for classification results.
    Args:
        y_true (np.ndarray): True binary labels.
        y_pred (np.ndarray): Predicted binary labels.
    Returns:
        None: Displays the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        xticklabels=['Not Fraud', 'Fraud'],
        yticklabels=['Not Fraud', 'Fraud']
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def plot_roc(y_test: np.ndarray, y_pred_prob: np.ndarray) -> None:
    """
    #### Plot the ROC curve and display AUC.
    Args:
        y_test (np.ndarray): True binary labels.
        y_pred_prob (np.ndarray): Predicted probabilities for the positive class.
    Returns:
        None: Displays the plot.
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 4))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


def plot_stat_results(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    plot_type: str,
    title: str,
    xlabel: str,
    ylabel: str,
    **kwargs
) -> None:
    """
    #### Generalized function to plot bar or scatter plots.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_col (str): Column for the x-axis.
        y_col (str): Column for the y-axis.
        plot_type (str): Type of plot ('barplot' or 'scatterplot').
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        **kwargs: Additional arguments for seaborn plotting functions.
    Returns:
        None: Displays the plot.
    """
    plt.figure(figsize=(6, 4))
    if plot_type == 'barplot':
        sns.barplot(data=df, x=x_col, y=y_col, **kwargs)
    elif plot_type == 'scatterplot':
        sns.scatterplot(data=df, x=x_col, y=y_col, **kwargs)
    else:
        raise ValueError("Unsupported plot type. Use 'barplot' or 'scatterplot'.")
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def plot_stat_results(model: Any, X_test, y_test) -> None:
    """
    #### Plot the results of a classification model.
    Args:
        model: Trained classification model.
        X_test: Test features.
        y_test: True labels for the test set.
    Returns:
        None: Displays the plot.
    """
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    best_thresh, best_f1 = find_best_threshold(y_test, y_pred_prob, metric='f1', verbose=False)
    print(f"\nBest Threshold: {best_thresh:.2f} with F1 score: {best_f1:.4f}")
    y_pred = (y_pred_prob >= best_thresh).astype(int)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_prob):.4f}")
    plot_roc(y_test, y_pred_prob)
    plot_confusion_matrix(y_test, y_pred)

def plot_feature_importance(model: Any, feature_names) -> None:
    """
    #### Plot feature importance for a given model.
    Args:
        model: Trained model with feature importances.
        feature_names: List of feature names.
    Returns:
        None: Displays the plot.
    Raises:
        ValueError: If the model does not have feature importances or coefficients.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not have feature importances or coefficients.")

    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.tight_layout()
    plt.show()