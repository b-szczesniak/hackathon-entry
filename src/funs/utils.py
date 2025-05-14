import pandas as pd
import scipy.stats as stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, precision_score, recall_score
    )

from country_decoding import add_country_from_coordinates

USERS_PATH = '../../data/preprocessed/users.csv'
MERCHANTS_PATH = '../../data/preprocessed/merchants.csv'
TRANSACTIONS_PATH = '../../data/preprocessed/transactions.json'

def get_merged_dataframes() -> pd.DataFrame:
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


def find_best_threshold(y_true, y_pred_prob, metric='f1', step=0.01, verbose=False):
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

def chi2_independence(df: pd.DataFrame, factor_col: str, fraud_col: str, type: str = "description", treshold: int = 0.05) -> dict | None | pd.DataFrame:
    """
    ## Variable Statistical Test of Independence
    ### The null hypothesis is that there is no association between the two variables,
    and the alternative hypothesis is that there is a significant association between them.\n
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3900058/\n
    
    :param df: Pandas DataFrame
    :param factor_col: Column name by which to group the data
    :param fraud_col: Column name of the fraud indicator
    :param type: Type of output ('description' or 'table')
    :return: None
    """
    count_group = df.groupby([factor_col, fraud_col]).size().reset_index(name="n")

    table = pd.pivot_table(count_group, values="n", index=fraud_col, columns=factor_col)

    chi2, pval, dof, expected = stats.chi2_contingency(table)

    if type == "description":
        if pval <= treshold:
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
    ## Spearman's rank correlation coefficient
    ### The null hypothesis is that there is no association between the two variables,
    and the alternative hypothesis is that there is a significant association between them.\n
    https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient\n
    :param df: Pandas DataFrame
    :param column: Column name to calculate the correlation for
    :param target: Target column
    :return: None
    """
    df_valid = df.dropna(subset=[column])

    # avoid ConstantInputWarning
    if df_valid[column].nunique() < 2 or df_valid[target].nunique() < 2:
        return

    corr, p_value = stats.spearmanr(df_valid[column], df_valid[target])
    if p_value < 0.05:
        return {
            "column": column,
            "correlation_coefficient": corr,
            "p_value": p_value,
            "significant": True
        }
    return

def anova_test(df: pd.DataFrame, column: str, target: str) -> dict | None:
    """
    ## ANOVA test
    ### The null hypothesis is that there is no association between the two variables,
    and the alternative hypothesis is that there is a significant association between them.\n
    https://en.wikipedia.org/wiki/Analysis_of_variance\n
    :param df: Pandas DataFrame
    :param column: Column name to calculate the ANOVA test for
    :param target: Target column
    :return: None
    """
    df_valid = df.dropna(subset=[column])

    groups = [group[target].values for name, group in df_valid.groupby(column)]
    if len(groups) < 2:
        return
    f_stat, p_value = stats.f_oneway(*groups)

    if p_value < 0.05:
        return {
            "column": column,
            "f_statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < 0.05
        }
    return
    