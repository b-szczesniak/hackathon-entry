import pandas as pd
import scipy.stats as stats
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score
    )
from funs.country_decoding import add_country_from_coordinates

USERS_PATH = '../../data/preprocessed/users.csv'
MERCHANTS_PATH = '../../data/preprocessed/merchants.csv'
TRANSACTIONS_PATH = '../../data/preprocessed/transactions.json'

def get_merged_dataframes() -> pd.DataFrame:
    """
    #### Load and merge dataframes
    This function loads the users, merchants, and transactions dataframes from CSV and JSON files,
    respectively. It then merges them into a single dataframe and adds country information from coordinates.\n
    The merged dataframe contains user and merchant information along with transaction details.\n
    Returns:
        pd.DataFrame: Merged dataframe containing user, merchant, and transaction information.
    """
    users = pd.read_csv(USERS_PATH) 
    users['user_country'] = users['country']
    users.drop(columns=['country'], inplace=True)

    merchants = pd.read_csv(MERCHANTS_PATH)
    merchants['merchant_country'] = merchants['country']
    merchants.drop(columns=['country'], inplace=True)

    transactions = pd.read_json(TRANSACTIONS_PATH, lines=True)

    merged = (
        transactions
        .merge(users, on='user_id', how='left')
        .merge(merchants, on='merchant_id', how='left')
    )

    # Add country information from coordinates
    merged = add_country_from_coordinates(merged)

    return merged


def find_best_threshold(y_true: np.ndarray, y_pred_prob: np.ndarray, metric: str='f1', step: float=0.01, verbose: bool=False) -> tuple[float, float]:
    """
    #### Find the best threshold for binary classification
    This function finds the best threshold for binary classification based on the specified metric.\n
    The metric can be 'f1', 'precision', 'recall', or 'balanced'.\n
    The function iterates through thresholds from 0 to 1 in steps of the specified size and calculates the score for each threshold.\n
    The best threshold is the one that maximizes the score.\n
    Args:
        y_true (array-like): True binary labels
        y_pred_prob (array-like): Predicted probabilities
        metric (str): Metric to optimize ('f1', 'precision', 'recall', 'balanced')
        step (float): Step size for threshold iteration
        verbose (bool): If True, print the scores for each threshold
    Returns:
        tuple: Best threshold and corresponding score
    """
    best_score = -1
    best_threshold = 0.5
    thresholds = np.arange(0.0, 1.01, step)

    for threshold in thresholds:
        y_pred = (y_pred_prob >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        elif metric == 'balanced':
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            score = 2 * (precision * recall) / (precision + recall + 1e-9)
        else:
            raise ValueError("Invalid metric. Choose from 'f1', 'precision', 'recall', 'balanced'.")

        if verbose:
            print(f"Threshold: {threshold:.2f} | {metric}: {score:.4f}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score

def chi2_independence(df: pd.DataFrame, factor_col: str, fraud_col: str, type: str = "description", threshold: float = 0.05) -> dict | None | pd.DataFrame:
    """
    #### Variable Statistical Test of Independence
    The null hypothesis is that there is no association between the two variables,
    and the alternative hypothesis is that there is a significant association between them.\n
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3900058/\n
    
    Args:
        df (pd.DataFrame): Pandas DataFrame
        factor_col (str): Column name by which to group the data
        fraud_col (str): Column name of the fraud indicator
        type (str): Type of output ('description' or 'table')
        threshold (float): Significance level for the test (default is 0.05)

    Returns:
        dict: Dictionary containing the results of the test if significant
        None: If the test is not significant
        pd.DataFrame: DataFrame of expected frequencies if type is 'table'
    """
    count_group = df.groupby([factor_col, fraud_col]).size().reset_index(name="n")

    table = pd.pivot_table(count_group, values="n", index=fraud_col, columns=factor_col)

    chi2, pval, dof, expected = stats.chi2_contingency(table)

    if type == "description":
        if pval <= threshold:
            return {
                "column": factor_col,
                "chi2": chi2,
                "p_value": pval,
                "dof": dof,
                "expected": pd.DataFrame(expected, index=table.index, columns=table.columns)
            }
        return
    elif type == "table":
        return stats.chi2_contingency(table)[3]

def spearman_correlation(df: pd.DataFrame, column: str, target: str) -> dict | None:
    """
    #### Spearman's rank correlation coefficient
    The null hypothesis is that there is no association between the two variables,
    and the alternative hypothesis is that there is a significant association between them.\n
    https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient\n
    Args:
        df (pd.DataFrame): Pandas DataFrame
        column (str): Column name to calculate the Spearman correlation for
        target (str): Target column
    Returns:
        dict: Dictionary containing the results of the test if significant
        None: If the test is not significant
    """
    df_valid = df.dropna(subset=[column])

    # avoid ConstantInputWarning
    if df_valid[column].nunique() < 2 or df_valid[target].nunique() < 2:
        return None

    corr, p_value = stats.spearmanr(df_valid[column], df_valid[target])
    if p_value < 0.05:
        return {
            "column": column,
            "correlation_coefficient": corr,
            "p_value": p_value,
            "significant": True
        }
    return None

def anova_test(df: pd.DataFrame, column: str, target: str) -> dict | None:
    """
    #### ANOVA test
    The null hypothesis is that there is no association between the two variables,
    and the alternative hypothesis is that there is a significant association between them.\n
    https://en.wikipedia.org/wiki/Analysis_of_variance\n
    Args:
        df (pd.DataFrame): Pandas DataFrame
        column (str): Column name to calculate the ANOVA test for
        target (str): Target column
    Returns:
        dict: Dictionary containing the results of the test if significant
        None: If the test is not significant
    """
    df_valid = df.dropna(subset=[column])

    groups = [group[target].values for name, group in df_valid.groupby(column)]
    if len(groups) < 2:
        return None
    
    f_stat, p_value = stats.f_oneway(*groups)

    if p_value < 0.05:
        return {
            "column": column,
            "f_statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < 0.05
        }
    return None
    