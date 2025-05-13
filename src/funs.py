import pandas as pd
import scipy.stats as stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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
    :param column: Column name to calculate the correlation for
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