import pandas as pd
import scipy.stats as stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, precision_score, recall_score
    )

USERS_PATH = '../../data/users.csv'
MERCHANTS_PATH = '../../data/merchants.csv'
TRANSACTIONS_PATH = '../../data/transactions.json'


def get_merged_dataframes() -> pd.DataFrame:
    users = pd.read_csv(USERS_PATH) 
    users['user_country'] = users['country']
    users.drop(columns=['country'], inplace=True)
    # users = users.rename(columns={'country': 'user_country'})

    merchants = pd.read_csv(MERCHANTS_PATH)
    # merchants = merchants.rename(columns={'country': 'merchant_country'})
    merchants['merchant_country'] = merchants['country']
    merchants.drop(columns=['country'], inplace=True)

    transactions = pd.read_json(TRANSACTIONS_PATH, lines=True)

    return (
        transactions
        .merge(users, on='user_id', how='left')
        .merge(merchants, on='merchant_id', how='left')
    )


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

def chi2_independence(df: pd.DataFrame, factor_col: str, fraud_col: str, type: str = "description") -> None | pd.DataFrame:
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
        print("Chi-square statistic: ", chi2)
        print("p-value: ", pval)
        print("Degrees of freedom: ", dof)
        print("Expected frequencies: ")
        print(pd.DataFrame(expected, index=table.index, columns=table.columns))
    elif type == "table":
        return stats.chi2_contingency(table)[3]

def spearman_correlation(df: pd.DataFrame, column: str, target: str) -> None:
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

    corr, p_value = stats.spearmanr(df_valid[column], df_valid[target])
    print(f"\n{column} - Spearman correlation test:")
    print(f"Correlation coefficient: {corr:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Statistically significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    plt.figure(figsize=(10, 6))
    ax = sns.regplot(x=column, y=target, data=df_valid, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    ax.set_title(f'Relationship between {column} and Fraud Count (r={corr:.4f}, p={p_value:.4f})')
    ax.set_xlabel(column)
    ax.set_ylabel('Fraud Count')
    plt.tight_layout()
    plt.show()

def anova_test(df: pd.DataFrame, column: str, target: str) -> None:
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
    f_stat, p_value = stats.f_oneway(*groups)
    
    print(f"\n{column} - ANOVA test:")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Statistically significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x=column, y=target, data=df_valid)
    ax.set_title(f'ANOVA Test: {column} vs {target} (F={f_stat:.4f}, p={p_value:.4f})')
    ax.set_xlabel(column)
    ax.set_ylabel(target)

    medians = df_valid.groupby(column)[target].median().values
    for i, median in enumerate(medians):
        ax.text(i, median + 0.1, f'Median: {median:.1f}', ha='center')
     
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
     
    # If significant, perform post-hoc Tukey test
    if p_value < 0.05 and len(groups) > 2:
        tukey = stats.tukey_hsd(df_valid[target], df_valid[column], alpha=0.05)
        print("\nTukey HSD post-hoc test:")
        print(tukey)

