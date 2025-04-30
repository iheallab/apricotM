#%%

import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Load data
adj_df = pd.read_csv('adjudication_results_final.csv').iloc[:, 17:]
cases_df = pd.read_csv('adjudication_cases_survey.csv')

adjudicators = 3  # Number of adjudicators
questions = [f'Q{i+1}' for i in range(23)]
cases = [f'Case_{i+1}' for i in range(20)]

# Prepare storage
all_risk_before = []
all_risk_after = []

# Loop through adjudicators and collect scores
for adjudicator in range(1, adjudicators + 1):
    data = {}
    for i, case in enumerate(cases):
        data[case] = adj_df.iloc[adjudicator + 1, i * 23:(i + 1) * 23].values.flatten()

    structured_df = pd.DataFrame(data, index=questions)
    structured_df["question"] = structured_df.index.map({f'Q{i+1}': adj_df.iloc[0, i] for i in range(23)})

    risk_score_before = structured_df.loc["Q18"].iloc[:-1].astype(float).values
    risk_score_after = structured_df.loc["Q19"].iloc[:-1].astype(float).values

    all_risk_before.append(risk_score_before)
    all_risk_after.append(risk_score_after)

# Stack and average across adjudicators (ignoring NaNs)
all_risk_before = np.vstack(all_risk_before)
all_risk_after = np.vstack(all_risk_after)

avg_risk_before = np.nanmean(all_risk_before, axis=0)
avg_risk_after = np.nanmean(all_risk_after, axis=0)

mean_before = np.nanmean(avg_risk_before)
std_before = np.nanstd(avg_risk_before)

mean_after = np.nanmean(avg_risk_after)
std_after = np.nanstd(avg_risk_after)

print(f"Before Alert - Mean: {mean_before:.2f}, Std Dev: {std_before:.2f}")
print(f"After Alert  - Mean: {mean_after:.2f}, Std Dev: {std_after:.2f}")

# Prepare data for box plot
risk_data = pd.DataFrame({
    "Risk Score": np.concatenate([avg_risk_before, avg_risk_after]),
    "Condition": ["Before Alert"] * len(avg_risk_before) + ["After Alert"] * len(avg_risk_after)
})

# Plot box plots
fig = go.Figure()

fig.add_trace(go.Box(
    y=risk_data.loc[risk_data["Condition"] == "Before Alert", "Risk Score"],
    name="Before Alert",
    marker_color='blue',
    boxmean=True  # Show mean point
))

fig.add_trace(go.Box(
    y=risk_data.loc[risk_data["Condition"] == "After Alert", "Risk Score"],
    name="After Alert",
    marker_color='red',
    boxmean=True  # Show mean point
))

fig.update_layout(
    yaxis_title="Average Adjudicated Risk Score",
    width=800,
    height=600,
    font=dict(size=16),
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=False
)

fig.update_yaxes(
    range=[0, 100],  # assuming risk scores are percentages
    showgrid=False
)

fig.write_image("box_plot.png", width=800, height=600)

fig.show()

# %%

import pandas as pd
import numpy as np
import plotly.express as px

# Load data
adj_df = pd.read_csv('adjudication_results_final.csv').iloc[:, 17:]
cases_df = pd.read_csv('adjudication_cases_survey.csv')

adjudicators = 3
questions = [f'Q{i+1}' for i in range(23)]
cases = [f'Case_{i+1}' for i in range(20)]

# Prepare storage
records = []

# Loop through adjudicators
for adjudicator in range(1, adjudicators + 1):
    data = {}
    for i, case in enumerate(cases):
        data[case] = adj_df.iloc[adjudicator + 1, i * 23:(i + 1) * 23].values.flatten()

    structured_df = pd.DataFrame(data, index=questions)
    structured_df["question"] = structured_df.index.map({f'Q{i+1}': adj_df.iloc[0, i] for i in range(23)})

    # Extract before/after scores (Q18 and Q19)
    risk_before = structured_df.loc["Q18"].drop("question").astype(float)
    risk_after = structured_df.loc["Q19"].drop("question").astype(float)

    # Store in long format for boxplotting
    for val in risk_before:
        records.append({"Adjudicator": f"Adjudicator {adjudicator}", "Condition": "Before Alert", "Risk Score": val})
    for val in risk_after:
        records.append({"Adjudicator": f"Adjudicator {adjudicator}", "Condition": "After Alert", "Risk Score": val})

# Convert to DataFrame
plot_df = pd.DataFrame(records)

# Plot using Plotly Express
fig = px.box(
    plot_df,
    x="Condition",
    y="Risk Score",
    color="Condition",
    facet_col="Adjudicator",
    boxmode="group",
    points="all",
    color_discrete_map={"Before Alert": "blue", "After Alert": "red"},
)

# Clean up facet titles
for i in range(1, adjudicators + 1):
    fig.layout.annotations[i - 1].text = f"Adjudicator {i}"

# Layout styling
fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    yaxis_title="Adjudicated Risk Score",
    font=dict(size=16),
    height=600,
    showlegend=False
)

# Remove x-axis titles
fig.update_xaxes(title_text="")

# Set consistent y-axis range
fig.update_yaxes(range=[0, 100], showgrid=False)

fig.write_image("box_plot_per_adjudicator.png", width=1200, height=600)

fig.show()


# %%
