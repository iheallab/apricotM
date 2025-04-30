import pandas as pd
import numpy as np
import plotly.express as px
from variables import PROSP_DATA_DIR, MODEL_DIR

# Load data
adj_df = pd.read_csv('adjudication_results_final.csv').iloc[:, 17:]
cases_df = pd.read_csv('adjudication_cases_survey.csv')
admissions = pd.read_csv(f"{PROSP_DATA_DIR}/final/admissions.csv")

adjudicators = 3
questions = [f'Q{i+1}' for i in range(23)]
cases = [f'Case_{i+1}' for i in range(20)]

# Map Q# to label
q_map = {
    'Q6': 'MV Factors Relevant',
    'Q9': 'VP Factors Relevant',
    'Q12': 'CRRT Factors Relevant'
}

# Storage for results
results = []

# Loop through adjudicators
for adjudicator in range(1, adjudicators + 1):
    data = {}
    for i, case in enumerate(cases):
        data[case] = adj_df.iloc[adjudicator + 1, i * 23:(i + 1) * 23].values.flatten()

    structured_df = pd.DataFrame(data, index=questions)
    structured_df["question"] = structured_df.index.map({f'Q{i+1}': adj_df.iloc[0, i] for i in range(23)})
    
    # For each question of interest (Q6, Q9, Q12), calculate proportion of "Yes"
    for qnum, label in q_map.items():
        values = structured_df.loc[qnum].drop("question").astype(str).str.lower()
        proportion_yes = (values == "yes").mean()
        results.append({
            "Adjudicator": f"Adjudicator {adjudicator}",
            "Intervention": label,
            "Proportion": proportion_yes * 100  # convert to %
        })

# Convert to DataFrame
plot_df = pd.DataFrame(results)

# Plot using Plotly Express
fig = px.bar(
    plot_df,
    x="Intervention",
    y="Proportion",
    color="Adjudicator",
    barmode="group",
    text=plot_df["Proportion"].map("{:.0f}%".format),
    labels={"Intervention": "Clinical Intervention", "Proportion": "Proportion of 'Yes' Responses"}
)

# Layout updates
fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(size=16),
    yaxis=dict(range=[0, 100], title="Proportion of APRICOT-M alerts (%)", tickformat=".0f"),
    xaxis=dict(title=""),
    uniformtext_minsize=8,
    uniformtext_mode='hide'
)

fig.write_image("adjudication_therapies_relevance_factors.png", format="png", width=1200, height=600)

fig.show()

#%%

import pandas as pd
import numpy as np
from collections import defaultdict
from variables import PROSP_DATA_DIR, MODEL_DIR

# Load data
adj_df = pd.read_csv('adjudication_results_final.csv').iloc[:, 17:]
cases_df = pd.read_csv('adjudication_cases_survey.csv')
admissions = pd.read_csv(f"{PROSP_DATA_DIR}/final/admissions.csv")

adjudicators = 3
questions = [f'Q{i+1}' for i in range(23)]
cases = [f'Case_{i+1}' for i in range(20)]

# Map Q# to label
q_map = {
    'Q6': 'MV Factors Relevant',
    'Q9': 'VP Factors Relevant',
    'Q12': 'CRRT Factors Relevant'
}

# Storage for per-intervention proportions per adjudicator
intervention_props = defaultdict(list)

# Loop through adjudicators
for adjudicator in range(1, adjudicators + 1):
    data = {}
    for i, case in enumerate(cases):
        data[case] = adj_df.iloc[adjudicator + 1, i * 23:(i + 1) * 23].values.flatten()

    structured_df = pd.DataFrame(data, index=questions)
    structured_df["question"] = structured_df.index.map({f'Q{i+1}': adj_df.iloc[0, i] for i in range(23)})

    # For each intervention, calculate "Yes" proportion
    for qnum, label in q_map.items():
        values = structured_df.loc[qnum].drop("question").astype(str).str.lower()
        proportion_yes = (values == "yes").mean() * 100
        intervention_props[label].append(proportion_yes)

# Summarize mean and std for each intervention
summary_stats = {
    "Intervention": [],
    "Mean (%)": [],
    "Std Dev (%)": []
}

for intervention, values in intervention_props.items():
    summary_stats["Intervention"].append(intervention)
    summary_stats["Mean (%)"].append(round(np.mean(values), 2))
    summary_stats["Std Dev (%)"].append(round(np.std(values), 2))

# Create final DataFrame
summary_df = pd.DataFrame(summary_stats)

summary_df

