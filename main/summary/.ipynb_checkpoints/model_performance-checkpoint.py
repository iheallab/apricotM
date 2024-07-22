#%%

# Import libraries

import pandas as pd
import numpy as np
import os
import math
import re
from scipy.stats import t

from sklearn.metrics import roc_auc_score

from variables import OUTPUT_DIR, MODEL_DIR

if not os.path.exists(f"{OUTPUT_DIR}/summary/model_performance"):
    os.makedirs(f"{OUTPUT_DIR}/summary/model_performance")


#%%

# Load results

cohorts = ["int", "ext", "prosp"]

models = ["apricotm"]

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

table_auroc = pd.DataFrame(data=[], columns=cohorts, index=tasks)

for model in models:

    for cohort in cohorts:

        model_probs = pd.read_csv(
            f"{MODEL_DIR}/{model}/results/{cohort}_pred_labels.csv"
        ).iloc[:, 1:]
        model_true = pd.read_csv(
            f"{MODEL_DIR}/{model}/results/{cohort}_true_labels.csv"
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

    
    # Perform statistical test


    def extract_value(s):

        matches = re.findall(r"\d+\.\d+", s)

        mean = float(matches[0])
        ci = float(matches[2]) - float(matches[1])

        return mean, ci


    cohorts = table_auroc.columns.tolist()
    tasks = table_auroc.index.tolist()

    sup = ["a", "b", "c", "d"]

    table_stat = table_auroc.copy()


    def get_super(x):
        normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
        super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
        res = x.maketrans("".join(normal), "".join(super_s))
        return x.translate(res)


    for cohort in cohorts:

        count = 0

        for comp in cohorts:

            superscript = sup[count]

            for task in tasks:

                sample_1 = table_auroc.loc[task, cohort]
                sample_2 = table_auroc.loc[task, comp]

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

                    table_stat.loc[task, cohort] = (
                        table_stat.loc[task, cohort] + "\u2E34" + get_super(superscript)
                    )

            count += 1

    # Regular expression pattern to match text after closing parenthesis
    pattern = r"\)([^\s])"

    # Function to replace matched text with a space
    def replace_text(value):
        return re.sub(pattern, r") ", value)


    # Apply the replacement operation to each element in the dataframe
    table_stat = table_stat.applymap(replace_text)

    print(table_stat)

    # Write HTML string to a file
    with open(f"{OUTPUT_DIR}/summary/model_performance/{model}_performance.html", 'w') as f:
        f.write(table_stat.to_html())

# models = ["apricotm"]

# tasks = [
#     'Discharge',
#     'Stable',
#     'Unstable',
#     'Deceased',
#     'Unstable-Stable',
#     'Stable-Unstable',
#     'No MV',
#     'MV',
#     'No VP',
#     'VP',
#     'No CRRT',
#     'CRRT',
# ]

# for model in models:
    
#     cohorts = [
#         ["int", "Development"],
#         ["ext", "External"],
#         ["prosp", "Prospective"]
#     ]
    
#     all_results = pd.DataFrame()
    
#     for cohort in cohorts:

#         metrics = pd.read_csv(f'{MODEL_DIR}/{model}/results/{cohort[0]}_metrics.csv')

#         metrics.index = tasks

#         metrics['cohort'] = cohort[0]
        
#         all_results = pd.concat([all_results, metrics], axis=0)

#     tasks = [
#         'Discharge',
#         'Stable',
#         'Unstable',
#         'Deceased',
#         'Unstable-Stable',
#         'Stable-Unstable',
#         'MV',
#         'VP',
#         'CRRT',
#     ]


#     all_results = all_results[all_results.index.isin(tasks)]

#     all_results = all_results.reset_index().rename({'index': 'Task'}, axis=1)

#     all_results['Task'] = pd.Categorical(all_results['Task'], categories=tasks, ordered=True)


#     all_results = all_results.sort_values(by=['Task'])

#     all_results = all_results.set_index(['cohort', 'Task'])

#     multi_index = all_results.index
#     data = all_results.values.flatten()
#     all_results = pd.Series(data, index=multi_index)
#     all_results = all_results.unstack('cohort')


#     all_results.to_csv(f"{OUTPUT_DIR}/summary/model_performance/")
