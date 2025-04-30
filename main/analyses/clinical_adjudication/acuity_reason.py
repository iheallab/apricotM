import pandas as pd
import numpy as np
import plotly.graph_objects as go
from variables import PROSP_DATA_DIR, MODEL_DIR

# Load data
adj_df = pd.read_csv('adjudication_results_final.csv').iloc[:, 17:]
cases_df = pd.read_csv('adjudication_cases_survey.csv')
admissions = pd.read_csv(f"{PROSP_DATA_DIR}/final/admissions.csv")

adjudicators = 3

# Reshape the dataframe to have one row per patient case and one column per question
questions = [f'Q{i+1}' for i in range(23)]
cases = [f'Case_{i+1}' for i in range(20)]

# Initialize storage
q20_data = []

for i, case in enumerate(cases):
    q20_case_answers = []
    for adj in range(1, adjudicators + 1):
        q20_response = adj_df.iloc[adj + 1, i * 23 + 19]  # Q20 is index 19 (0-based)
        q20_case_answers.append(str(q20_response).strip())
    
    q20_data.append(q20_case_answers)

# Convert to DataFrame
q20_df = pd.DataFrame(q20_data, columns=["Q20_adj1", "Q20_adj2", "Q20_adj3"])
q20_df["Case"] = cases
q20_df["patient_deiden_id"] = cases_df["patient_deiden_id"].values  # assuming same order

# Merge Q20 answers into cases_df
cases_with_q20 = cases_df.merge(q20_df, on="patient_deiden_id", how="left")


print(cases_with_q20[["Factor 1_alert", "Factor 2_alert", "Factor 3_alert", "Q20_adj1", "Q20_adj2", "Q20_adj3"]])

# %%

import pandas as pd
import plotly.express as px
from collections import Counter

# Define Q20 columns and adjudicator labels
q20_cols = ['Q20_adj1', 'Q20_adj2', 'Q20_adj3']
adjudicator_labels = ['Adjudicator 1', 'Adjudicator 2', 'Adjudicator 3']

# Desired cause order
acuity_order = [
    "Hemodynamic instability",
    "Respiratory distress",
    "Renal impairment",
    "Recent procedure",
    "Other",
    "Patient had no apparent issues"
]

# Custom soft gradient palette based on your request
custom_palette = [
    "#F8B55F",  # golden orange
    "#C95792",  # rose pink
    "#7C4585",  # muted violet
    "#3D365C",  # deep indigo
    "#88829B",  # soft gray-violet
    "#D9D5DC"   # pale lavender-gray
]

# Parse and count reasons per adjudicator
all_counts = []

for col, adj_label in zip(q20_cols, adjudicator_labels):
    reasons = []
    for val in cases_with_q20[col].dropna():
        split_reasons = [r.strip() for r in val.split(',') if r.strip()]
        reasons.extend(split_reasons)
    
    count = Counter(reasons)
    total = sum(count.values())
    
    for reason, c in count.items():
        all_counts.append({
            "Adjudicator": adj_label,
            "Reason": reason,
            "Percentage": (c / total) * 100  # Convert to percent
        })

# Create DataFrame
plot_df = pd.DataFrame(all_counts)

# Filter and order
plot_df = plot_df[plot_df["Reason"].isin(acuity_order)]
plot_df["Reason"] = pd.Categorical(plot_df["Reason"], categories=acuity_order, ordered=True)

# Plot
fig = px.bar(
    plot_df,
    y="Adjudicator",
    x="Percentage",
    color="Reason",
    orientation="h",
    text=plot_df["Percentage"].map("{:.0f}%".format),
    color_discrete_sequence=custom_palette,
    labels={"Percentage": "Percentage", "Adjudicator": "Adjudicator"},
    category_orders={"Reason": acuity_order, "Adjudicator": adjudicator_labels}
)

# Layout adjustments
fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(title="Proportion of APRICOT-M alerts (%)", tickformat=".0f"),
    yaxis=dict(title=""),
    legend_title_text="Identified Acuity Reason",
    uniformtext_minsize=8,
    uniformtext_mode='hide'
)

fig.write_image("adjudication_acuity_reason.png", format="png", width=1200, height=800)

fig.show()




# %%

import pandas as pd
from collections import Counter

# Define columns and target acuity reasons
q20_cols = ['Q20_adj1', 'Q20_adj2', 'Q20_adj3']
adjudicator_labels = ['Adjudicator 1', 'Adjudicator 2', 'Adjudicator 3']
acuity_order = [
    "Hemodynamic instability",
    "Respiratory distress",
    "Renal impairment",
    "Recent procedure",
    "Other",
    "Patient had no apparent issues"
]

# Store proportions per adjudicator
adj_proportions = {reason: [] for reason in acuity_order}

for col in q20_cols:
    reasons = []
    for val in cases_with_q20[col].dropna():
        split_reasons = [r.strip() for r in val.split(',') if r.strip()]
        reasons.extend(split_reasons)

    count = Counter(reasons)
    total = sum(count.values())

    for reason in acuity_order:
        proportion = (count[reason] / total) * 100 if reason in count else 0
        adj_proportions[reason].append(proportion)

# Compute mean and std across adjudicators
summary_stats = {
    "Acuity Reason": [],
    "Mean (%)": [],
    "Std Dev (%)": []
}

for reason in acuity_order:
    proportions = adj_proportions[reason]
    summary_stats["Acuity Reason"].append(reason)
    summary_stats["Mean (%)"].append(round(pd.Series(proportions).mean(), 2))
    summary_stats["Std Dev (%)"].append(round(pd.Series(proportions).std(), 2))

# Create final DataFrame
summary_df = pd.DataFrame(summary_stats)

summary_df
