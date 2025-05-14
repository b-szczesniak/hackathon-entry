import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_groupby(df: pd.DataFrame, column: str, target: str, agg_func: str = 'mean') -> None:
    """
    Plot a boxplot of an aggregated target by group.

    :param df: DataFrame containing the data.
    :param column: Column name to group by.
    :param target: Target column to aggregate and plot.
    :param agg_func: Aggregation function name (default: 'mean').
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
    Plot a filtered correlation heatmap for numeric columns.

    Keeps columns whose absolute correlation
    with any other column ≥ 1 (general) or
    with the target ≥ threshold_target.

    :param df: DataFrame with data.
    :param target: Target column for threshold filtering.
    :param threshold_target: Minimum absolute correlation with target to keep.
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
    Plot a confusion matrix heatmap for classification results.

    :param y_true: Array of true labels.
    :param y_pred: Array of predicted labels.
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
    Plot the ROC curve and display AUC.

    :param y_test: True binary labels.
    :param y_pred_prob: Predicted probabilities for the positive class.
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
    Generalized function to plot bar or scatter plots.

    :param df: DataFrame with the data.
    :param x_col: Column for x-axis.
    :param y_col: Column for y-axis.
    :param plot_type: 'barplot' or 'scatterplot'.
    :param title: Plot title.
    :param xlabel: Label for x-axis.
    :param ylabel: Label for y-axis.
    :param kwargs: Extra keyword args for seaborn.
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