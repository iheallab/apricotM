# Import libraries

import pandas as pd
import numpy as np
import os
import math
import re
from scipy.stats import t

from sklearn.metrics import roc_auc_score

from variables import HOME_DIR, OUTPUT_DIR

time_windows = [4, 24, 48]
cohorts = ["int", "ext", "prosp"]

tasks = [
    "discharge",
    "stable",
    "unstable",
    "dead",
    "unstable-stable",
    "stable-unstable",
    "no mv-mv",
    "no vp- vp",
    "no crrt-crrt",
]

table_window = pd.DataFrame()

for window in time_windows:
    
    MODEL_DIR = f"{HOME_DIR}/deepacu/main/{window}h_window/model/apricotm/results"
    
    table_auroc = pd.DataFrame(data=[], columns=cohorts, index=tasks)
    
    for cohort in cohorts:
    
        model_probs = pd.read_csv(
            f"{MODEL_DIR}/{cohort}_pred_labels.csv"
        ).iloc[:, 1:]
        model_true = pd.read_csv(
            f"{MODEL_DIR}/{cohort}_true_labels.csv"
        ).iloc[:, 1:]

        auroc = pd.DataFrame(data=[], columns=model_true.columns)

        n_iterations = 100

        for i in range(n_iterations):

            random_sample = np.random.choice(len(model_true), len(model_true), replace=True)

            sample_true = model_true.iloc[random_sample]
            sample_pred = model_probs.iloc[random_sample]

            while (sample_true == 0).all(axis=0).any():
                random_sample = np.random.choice(
                    len(model_true), len(model_true), replace=True
                )

                sample_true = model_true.iloc[random_sample]
                sample_pred = model_probs.iloc[random_sample]

            ind_auroc = []

            for column in tasks:
                score = roc_auc_score(
                    sample_true.loc[:, column], sample_pred.loc[:, column]
                )
                ind_auroc.append(score)

            ind_auroc = np.array(ind_auroc).reshape(1, -1)

            auroc = pd.concat([auroc, pd.DataFrame(data=ind_auroc, columns=tasks)], axis=0)

            print(f"Iteration {i+1}")

        auroc_sum = auroc.apply(
            lambda x: f"{x.median():.2f} ({max(0.0, x.quantile(0.025)):.2f}-{min(1, x.quantile(0.975)):.2f})",
            axis=0,
        )

        table_auroc[cohort] = auroc_sum
        
    table_auroc["time_window"] = window
    
    table_window = pd.concat([table_window, table_auroc], axis=0)

print(table_window)    

sup = ["a", "b", "c", "d"]

table_stat = table_window.copy()


def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans("".join(normal), "".join(super_s))
    return x.translate(res)


def extract_value(s):

    matches = re.findall(r"\d+\.\d+", s)

    mean = float(matches[0])
    ci = float(matches[2]) - float(matches[1])

    return mean, ci

table_stat = table_stat.reset_index().rename({"index": "task"}, axis=1)

print(table_stat)

for window in time_windows:

    count = 0
    
    for comp in time_windows:

        superscript = sup[count]

        for cohort in cohorts:

            for task in tasks:

                sample_1 = table_stat.loc[(table_stat["time_window"] == window)&(table_stat["task"] == task), cohort].values[0]
                sample_2 = table_stat.loc[(table_stat["time_window"] == comp)&(table_stat["task"] == task), cohort].values[0]

                mean_1, ci_1 = extract_value(sample_1)
                mean_2, ci_2 = extract_value(sample_2)

                std_1 = (math.sqrt(100) * ci_1) / 3.92
                std_2 = (math.sqrt(100) * ci_2) / 3.92

                dof = 100 + 100 - 2

                if std_1 > 0 or std_2 > 0:

                    t_statistic = (mean_1 - mean_2) / (
                        (std_1**2 / 100) + (std_2**2 / 100)
                    ) ** 0.5
                    p_value = 2 * (1 - t.cdf(abs(t_statistic), dof))

                elif mean_2 > mean_1:

                    p_value = 0

                else:

                    p_value = 1.00

                if p_value < 0.05:

                    table_stat.loc[(table_stat["time_window"] == window)&(table_stat["task"] == task), cohort] = (
                        table_stat.loc[(table_stat["time_window"] == window)&(table_stat["task"] == task), cohort].values[0] + "\u2E34" + get_super(superscript)
                    )

        count += 1

# Regular expression pattern to match text after closing parenthesis
pattern = r"\)([^\s])"

# Function to replace matched text with a space
def replace_text(value):
    return re.sub(pattern, r") ", value)


cohorts = ["int", "ext", "prosp"]

# Apply the replacement operation to each element in the dataframe
table_stat.loc[:,cohorts] = table_stat.loc[:,cohorts].applymap(replace_text)

table_stat["task"] = pd.Categorical(table_stat["task"], categories=tasks, ordered=True)

table_stat = table_stat.sort_values(by=["task", "time_window"])

print(table_stat)

# Write HTML string to a file
with open(f"{OUTPUT_DIR}/summary/model_performance/apricotm_windows_performance.html", 'w') as f:
    f.write(table_stat.to_html())