import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np


def chi2_independence(df, factor_col, fraud_col, type="description"):
    # Variable Statistical Test of Independence
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3900058/
    # The null hypothesis is that there is no association between the two variables,
    # and the alternative hypothesis is that there is a significant association between them.

    # Summarize the data by the given factor and fraud flag
    count_group = df.groupby([factor_col, fraud_col]).size().reset_index(name="n")

    # Create a contingency table
    table = pd.pivot_table(count_group, values="n", index=fraud_col, columns=factor_col)

    # Perform the chi-square test of independence
    chi2, pval, dof, expected = chi2_contingency(table)

    if type == "description":
        # Print the results
        print("Chi-square statistic: ", chi2)
        print("p-value: ", pval)
        print("Degrees of freedom: ", dof)
        print("Expected frequencies: ")
        print(pd.DataFrame(expected, index=table.index, columns=table.columns))
    elif type == "table":
        return chi2_contingency(table)[3]
