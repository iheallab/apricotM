#%%

# Import libraries

import pandas as pd
import numpy as np

from variables import MODEL_DIR


cohorts = [
    ["int", "Internal"],
    ["ext", "External"],
    ["temp", "Temporal"],
]

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

all_results = pd.DataFrame()

for cohort in cohorts:

    all_int = pd.read_csv(f"{MODEL_DIR}/results/{cohort[0]}_metrics.csv")

    all_int.index = tasks

    all_int["cohort"] = cohort[1]

    all_int["group"] = "All"

    all_results = pd.concat([all_results, all_int], axis=0)

    groups = ["Male", "Female", "white", "black", "other", "Young", "Old"]

    for group in groups:

        black_int = pd.read_csv(
            f"{MODEL_DIR}/subgroup/{group}_{cohort[0]}_metrics.csv"
        )

        black_int.index = tasks

        black_int["cohort"] = cohort[1]
        black_int["group"] = group

        all_results = pd.concat([all_results, black_int], axis=0)


#%%

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

cohorts = ["Internal", "External", "Temporal"]

models = ["All", "Young", "Old", "Female", "Male", "black", "other", "white"]

all_results = all_results[all_results.index.isin(tasks)]

all_results = all_results.reset_index().rename({"index": "Task"}, axis=1)

all_results["Task"] = pd.Categorical(
    all_results["Task"], categories=tasks, ordered=True
)
all_results["cohort"] = pd.Categorical(
    all_results["cohort"], categories=cohorts, ordered=True
)
all_results["group"] = pd.Categorical(
    all_results["group"], categories=models, ordered=True
)


groups = {
    "All": "None",
    "Young": "Age",
    "Old": "Age",
    "Female": "Gender",
    "Male": "Gender",
    "black": "Race",
    "other": "Race",
    "white": "Race",
}

all_results["category"] = all_results["group"].map(groups)


# %%

# Perform statistical test

import math
import re
from scipy.stats import t


def extract_value(s):

    matches = re.findall(r"\d+\.\d+", s)

    mean = float(matches[0])
    ci = float(matches[2]) - float(matches[1])

    return mean, ci


metrics = all_results.columns[1:-3].tolist()


def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans("".join(normal), "".join(super_s))
    return x.translate(res)


def stat_test(group):

    if group["category"].values[0] != "None":

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

                    group.loc[idx, metric] = group.loc[idx, metric] + "*"

                    # group.loc[idx, metric] = (
                    #     group.loc[idx, metric] + "\u2E34" + get_super(superscript)
                    # )

                # else:

                #     group.loc[idx, metric] = group.loc[idx, metric] + f" ({p_value:.4f})"

    return group


all_results_p = all_results.groupby(by=["Task", "cohort", "category"]).apply(stat_test).reset_index(drop=True)

all_results_p = all_results_p.sort_values(by=["Task", "cohort", "group"])

#%%

all_results_p.to_csv(
    "%s/subgroup/subgroup_table.csv" % MODEL_DIR, index=None
)

# %%
