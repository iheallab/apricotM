#%%

# Import libraries

import pandas as pd
import numpy as np
import math
from scipy.stats import t
import re
import os

from variables import (
    time_window,
    OUTPUT_DIR,
    BASELINE_DIR,
    MODELS_DIR,
    ALL_RESULTS_DIR,
)

if not os.path.exists(f"{ALL_RESULTS_DIR}"):
    os.makedirs(f"{ALL_RESULTS_DIR}")

baselines = ["catboost", "gru", "transformer"]
models = ["apricott", "apricotm"]

cohorts = ["int", "ext", "temp", "prosp"]

tasks = [
    "Discharge",
    "Stable",
    "Unstable",
    "Deceased",
    "Unstable-Stable",
    "Stable-Unstable",
    "No MV",
    "MV",
    "No VP",
    "VP",
    "No CRRT",
    "CRRT",
]

#%%

# Combine all results

all_results = pd.DataFrame()

for baseline in baselines:

    for cohort in cohorts:

        metrics_ind = pd.read_csv(
            f"{BASELINE_DIR}/{baseline}/results/{cohort}_metrics.csv"
        )

        metrics_ind.index = tasks

        metrics_ind["cohort"] = cohort
        metrics_ind["model"] = baseline

        all_results = pd.concat([all_results, metrics_ind], axis=0)


for model in models:

    for cohort in cohorts:

        metrics_ind = pd.read_csv(f"{MODELS_DIR}/{model}/results/{cohort}_metrics.csv")

        metrics_ind.index = tasks

        metrics_ind["cohort"] = cohort
        metrics_ind["model"] = model

        all_results = pd.concat([all_results, metrics_ind], axis=0)

tasks = [
    "Discharge",
    "Stable",
    "Unstable",
    "Deceased",
    "Unstable-Stable",
    "Stable-Unstable",
    "MV",
    "VP",
    "CRRT",
]

cohorts = ["int", "ext", "temp", "prosp"]

models = ["catboost", "gru", "transformer", "apricott", "apricotm"]

all_results = all_results[all_results.index.isin(tasks)]

all_results = all_results.reset_index().rename({"index": "Task"}, axis=1)

all_results["Task"] = pd.Categorical(
    all_results["Task"], categories=tasks, ordered=True
)
all_results["cohort"] = pd.Categorical(
    all_results["cohort"], categories=cohorts, ordered=True
)
all_results["model"] = pd.Categorical(
    all_results["model"], categories=models, ordered=True
)


all_results = all_results.sort_values(by=["Task", "cohort", "model"])

all_results = all_results.reset_index(drop=True)

all_results.to_csv(f"{ALL_RESULTS_DIR}/all_results.csv")

#%%


# Perform statistical test


def extract_value(s):

    matches = re.findall(r"\d+\.\d+", s)

    mean = float(matches[0])
    ci = float(matches[2]) - float(matches[1])

    return mean, ci


metrics = all_results.columns[1:-2].tolist()


def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans("".join(normal), "".join(super_s))
    return x.translate(res)


def stat_test(group):

    sup = ["a", "b", "c", "d"]

    for i in range(len(group) - 1):

        superscript = sup[i]

        first_model = group.iloc[i, :]
        second_model = group.iloc[-1, :]

        idx = second_model.name

        for metric in metrics:

            sample_1 = first_model.loc[metric]
            sample_2 = second_model.loc[metric]

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

                p_value = 1.0000

            if p_value < 0.05:

                # group.loc[idx, metric] = group.loc[idx, metric]

                group.loc[idx, metric] = (
                    group.loc[idx, metric] + "\u2E34" + get_super(superscript)
                )

            # else:

            #     group.loc[idx, metric] = group.loc[idx, metric] + f" ({p_value:.4f})"

    return group


all_results_p = all_results.groupby(by=["Task", "cohort"]).apply(stat_test)

# Regular expression pattern to match text after closing parenthesis
pattern = r"\)([^\s])"

# Function to replace matched text with a space
def replace_text(value):
    return re.sub(pattern, r") ", value)


# Apply the replacement operation to each element in the dataframe
all_results_p.iloc[:, 1:7] = all_results_p.iloc[:, 1:7].applymap(replace_text)

from IPython.display import display, HTML

display(HTML(all_results_p.to_html()))

with open(f"{ALL_RESULTS_DIR}/results_table.html", 'w') as f:
    f.write(all_results_p.to_html())

# %%
