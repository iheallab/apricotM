#%%

import pandas as pd

adj_df = pd.read_csv('adjudication_results_final.csv').iloc[:, 17:]
cases_df = pd.read_csv('adjudication_cases_survey.csv')

#%%

adjudicators = 3

# Reshape the dataframe to have one row per patient case and one column per question
questions = [f'Q{i+1}' for i in range(23)]
cases = [f'Case_{i+1}' for i in range(20)]

use_questions = ['Q17', 'Q14', 'Q15', 'Q3', 'Q5', 'Q8', 'Q11']

question_mapping_main = {
    'Q17': 'Alert identified increased acuity',
    'Q14': 'Alert changed management',
    'Q15': 'Alert timing sufficient',
    'Q3': 'Factors relevant to increased acuity',
    'Q5': 'MV recommendation appropriate',
    'Q8': 'VP recommendation appropriate',
    'Q11': 'CRRT recommendation appropriate',
}

statistics = {
    question_mapping_main[q]: [] for q in use_questions
}

for adjudicator in range(1, adjudicators+1):
    data = {}
    for i, case in enumerate(cases):
        data[case] = adj_df.iloc[adjudicator + 1, i*23:(i+1)*23].values.flatten()

    structured_df = pd.DataFrame(data, index=questions)
    
    # Create a mapping of the questions to the values in the first row of adj_1_df
    question_mapping = {f'Q{i+1}': adj_df.iloc[0, i] for i in range(23)}

    structured_df["question"] = structured_df.index
    structured_df["question"] = structured_df["question"].map(question_mapping)

    # Pivot the structured_df so each case is a row
    pivoted_df = structured_df.T
    pivoted_df = pivoted_df.drop('question')
    pivoted_df.index.name = 'Case'
    pivoted_df.reset_index(inplace=True)

    pivoted_df = pivoted_df[use_questions]

    # Combine the cases_df and pivoted_df

    combined_df = pd.concat([cases_df, pivoted_df], axis=1)

    # Map question numbers to question names

    combined_df.rename(columns=question_mapping_main, inplace=True)
        
    # Get statistics for positive answers ('Yes') for each question
    
    for question in combined_df.columns[-7:]:
        question_data = combined_df[question]
        question_stats = question_data.value_counts(normalize=True)
        
        # Append the statistics to the dictionary
        statistics[question].append(question_stats.get('Yes', 0))


#%%

# Convert the statistics dictionary to a DataFrame
statistics_df = pd.DataFrame(statistics)
statistics_df.index = [f'Adjudicator {i+1}' for i in range(adjudicators)]

summary = statistics_df.agg(['mean', 'std', 'min', 'max']).T
summary.columns = ['Mean', 'SD', 'Min', 'Max']
summary = summary.round(3) * 100
summary

#%%

# Convert summary table into a new format
summary_table = summary.copy()
summary_table["Agreement (%)"] = summary_table.apply(
    lambda row: f"{row['Mean']:.2f} (±{row['SD']:.2f})", axis=1
)
summary_table = summary_table[["Agreement (%)"]]
summary_table
# Save summary table to CSV
summary_table.to_csv("adjudication_summary.csv", index=True)

#%%

import plotly.graph_objects as go

# Extract data for plotting
questions = summary.index.tolist()
means = summary["Mean"].tolist()
stds = summary["SD"].tolist()

# Create bar plot with error bars
fig = go.Figure()

fig.add_trace(go.Bar(
    x=questions,
    y=means,
    error_y=dict(type='data', array=stds, visible=True),
    textposition='outside',
    marker_color='#1f77b4'
))

# Layout settings
fig.update_layout(
    xaxis_title="",
    yaxis_title="Mean Agreement (%)",
    yaxis=dict(range=[0, 100]),
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='black'),
    height=600
)

fig.show()


# %%


import pandas as pd
import pingouin as pg

# Load data
adj_df = pd.read_csv('adjudication_results_final.csv').iloc[:, 17:]
cases_df = pd.read_csv('adjudication_cases_survey.csv')

adjudicators = 3
questions = [f'Q{i+1}' for i in range(23)]
cases = [f'Case_{i+1}' for i in range(20)]
use_questions = ['Q17', 'Q14', 'Q15', 'Q3', 'Q5', 'Q8', 'Q11']

# Question name mapping
question_mapping = {
    'Q17': 'Alert identified ↑ acuity',
    'Q14': 'Alert changed management',
    'Q15': 'Alert timing sufficient',
    'Q3': 'Factors relevant to acuity ↑',
    'Q5': 'MV recommendation appropriate',
    'Q8': 'VP recommendation appropriate',
    'Q11': 'CRRT recommendation appropriate',
}

# Create long-format dataframe: one row per (case, question, adjudicator)
records = []

for adjudicator in range(1, adjudicators + 1):
    data = {}
    for i, case in enumerate(cases):
        data[case] = adj_df.iloc[adjudicator + 1, i * 23:(i + 1) * 23].values.flatten()
    structured_df = pd.DataFrame(data, index=questions)
    
    for case in cases:
        for q in use_questions:
            response = structured_df.loc[q, case]
            response_binary = 1 if response == 'Yes' else 0
            records.append({
                "case": case,
                "question": question_mapping[q],
                "adjudicator": f"Adjudicator {adjudicator}",
                "rating": response_binary
            })

df_long = pd.DataFrame(records)

# Now calculate ICC using Pingouin
icc_results = pg.intraclass_corr(data=df_long, targets='case', raters='adjudicator', ratings='rating')
icc_results = icc_results[['Type', 'ICC', 'CI95%']].round(3)

# Display results
print(icc_results)

#%%

import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Prep: adjudicator-wise response table per question
adjudicators = 3
questions_of_interest = {
    'Q17': 'Alert idenfied increase in patient acuity',
    'Q14': 'Alert would have changed my management of the patient',
    'Q15': 'The time of the alert would have been sufficient to prevent the adverse event',
    'Q3': "The identified factors are relevant to identify the reason for the patient's increased acuity",
    'Q5': "MV recommendation was appropriate",
    'Q8': 'VP recommendation was appropriate',
    'Q11': 'CRRT recommendation was appropriate',
}

# Build: dictionary with {question: adjudicator_df}
question_agreements = {label: pd.DataFrame() for label in questions_of_interest.values()}
cases = [f'Case_{i+1}' for i in range(20)]

for adjudicator in range(1, adjudicators + 1):
    data = {}
    for i, case in enumerate(cases):
        data[case] = adj_df.iloc[adjudicator + 1, i * 23:(i + 1) * 23].values.flatten()
    structured_df = pd.DataFrame(data, index=questions)
    
    for qid, label in questions_of_interest.items():
        if label not in question_agreements:
            question_agreements[label] = pd.DataFrame()
        answers = structured_df.loc[qid].map(lambda x: 1 if x == 'Yes' else 0)
        question_agreements[label][f'Adjudicator {adjudicator}'] = answers.values

# Compute Cohen’s Kappa for each pair per question
from itertools import combinations

kappa_results = []

for question, df in question_agreements.items():
    pairs = combinations(df.columns, 2)
    for a1, a2 in pairs:
        kappa = cohen_kappa_score(df[a1], df[a2])
        kappa_results.append({
            'Question': question,
            'Pair': f'{a1} vs {a2}',
            'Cohen\'s Kappa': round(kappa, 3)
        })

kappa_df = pd.DataFrame(kappa_results)
print("Cohen's Kappa:")
print(kappa_df)

#%%

import pandas as pd
from sklearn.metrics import cohen_kappa_score
from itertools import combinations
import numpy as np

# === Step 1: Compute kappa values per question ===
adjudicators = 3
questions_of_interest = {
    'Q17': 'Alert idenfied increase in patient acuity',
    'Q14': 'Alert would have changed my management of the patient',
    'Q15': 'The time of the alert would have been sufficient to prevent the adverse event',
    'Q3': "The identified factors are relevant to identify the reason for the patient's increased acuity",
    'Q5': "MV recommendation was appropriate",
    'Q8': 'VP recommendation appropriate',
    'Q11': 'CRRT recommendation appropriate',
}

question_agreements = {label: pd.DataFrame() for label in questions_of_interest.values()}
cases = [f'Case_{i+1}' for i in range(20)]

for adjudicator in range(1, adjudicators + 1):
    data = {}
    for i, case in enumerate(cases):
        data[case] = adj_df.iloc[adjudicator + 1, i * 23:(i + 1) * 23].values.flatten()
    structured_df = pd.DataFrame(data, index=questions)

    for qid, label in questions_of_interest.items():
        answers = structured_df.loc[qid].map(lambda x: 1 if x == 'Yes' else 0)
        question_agreements[label][f'Adjudicator {adjudicator}'] = answers.values

# === Step 2: Calculate mean and std of kappa values ===
mean_kappa_summary = []

for question, df in question_agreements.items():
    kappas = []
    for a1, a2 in combinations(df.columns, 2):
        kappa = cohen_kappa_score(df[a1], df[a2])
        kappas.append(kappa)
    
    mean_kappa = np.mean(kappas)
    std_kappa = np.std(kappas)
    label = f"{mean_kappa:.2f} (±{std_kappa:.2f})"

    mean_kappa_summary.append({
        "Question": question,
        "Mean Kappa": mean_kappa,
        "SD": std_kappa,
        "Label": label
    })

mean_kappa_df = pd.DataFrame(mean_kappa_summary)

mean_kappa_df.to_csv("mean_kappa_summary.csv", index=False)

mean_kappa_df

#%%

from scipy.spatial.distance import dice
from itertools import combinations
import numpy as np

# Dice similarity is 1 - dice distance
def dice_similarity(a, b):
    return 1 - dice(a, b)

# For each question, compute average pairwise Dice similarity across adjudicators
dice_results = []

for question, df in question_agreements.items():
    # Ensure binary format
    bin_df = df.applymap(lambda x: 1 if x == 'Yes' or x == 1 else 0)

    similarities = []
    for a1, a2 in combinations(bin_df.columns, 2):
        d = dice_similarity(bin_df[a1].values, bin_df[a2].values)
        similarities.append(d)
    
    avg_dice = round(np.mean(similarities), 3)
    dice_results.append({
        'Question': question,
        'Average Dice Similarity': avg_dice
    })

dice_df = pd.DataFrame(dice_results)
print("Average Dice Agreement Across All Adjudicators:")
print(dice_df)


# %%

import pandas as pd

# Load data
adj_df = pd.read_csv('adjudication_results_final.csv').iloc[:, 17:]
cases_df = pd.read_csv('adjudication_cases_survey.csv')

adjudicators = 3
questions = [f'Q{i+1}' for i in range(23)]
cases = [f'Case_{i+1}' for i in range(20)]
use_questions = ['Q17', 'Q14', 'Q15', 'Q3', 'Q5', 'Q8', 'Q11']

# Mapping of questions to readable labels
question_mapping = {
    'Q17': 'Alert identified increased acuity',
    'Q14': 'Alert changed management',
    'Q15': 'Alert timing sufficient',
    'Q3': 'Factors relevant to increased acuity',
    'Q5': 'MV recommendation appropriate',
    'Q8': 'VP recommendation appropriate',
    'Q11': 'CRRT recommendation appropriate',
}

# Mapping to suggested therapy flags
therapy_flags = {
    'MV recommendation appropriate': 'suggested_mv',
    'VP recommendation appropriate': 'suggested_vp',
    'CRRT recommendation appropriate': 'suggested_crrt',
}

concensus = [1, 2, 3]

# Step 1: Count "Yes" responses per adjudicator for each question and case
yes_counts = {question_mapping[q]: [0] * len(cases) for q in use_questions}

for adjudicator in range(1, adjudicators + 1):
    data = {}
    for i, case in enumerate(cases):
        data[case] = adj_df.iloc[adjudicator + 1, i * 23:(i + 1) * 23].values.flatten()
    structured_df = pd.DataFrame(data, index=questions)

    for q in use_questions:
        question_label = question_mapping[q]
        for idx, case in enumerate(cases):
            response = structured_df.loc[q, case]
            if response == 'Yes':
                yes_counts[question_label][idx] += 1
                
all_results = pd.DataFrame()

for criteria in concensus:

    # Step 2: Convert to binary (1 if at least one adjudicator said "Yes")
    at_least_one_yes = {
        q: [1 if count >= criteria else 0 for count in yes_counts[q]]
        for q in yes_counts
    }

    # Step 3: Compute positive % overall, and for recommended and not recommended therapy cases
    summary_data = []

    for question, results in at_least_one_yes.items():
        # Base values
        total_all = len(results)
        positive_all = sum(results)
        pct_all = round((positive_all / total_all) * 100, 1)

        # For therapy questions: recommended / not recommended
        if question in therapy_flags:
            flag_col = therapy_flags[question]
            is_recommended = cases_df[flag_col] == "Recommended"
            is_not_recommended = cases_df[flag_col] == "Not Recommended"

            results_recommended = [res for res, flag in zip(results, is_recommended) if flag]
            results_not_recommended = [res for res, flag in zip(results, is_not_recommended) if flag]

            pos_rec = sum(results_recommended)
            pos_not_rec = sum(results_not_recommended)

            total_rec = len(results_recommended)
            total_not_rec = len(results_not_recommended)

            pct_rec = round((pos_rec / total_rec) * 100, 1) if total_rec > 0 else None
            pct_not_rec = round((pos_not_rec / total_not_rec) * 100, 1) if total_not_rec > 0 else None
        else:
            pct_rec = pct_not_rec = None

        summary_data.append({
            "Question": question,
            f"Positive (≥{criteria} Yes) – All": pct_all,
            f"Positive (≥{criteria} Yes) – Recommended": pct_rec,
            f"Positive (≥{criteria} Yes) – Not Recommended": pct_not_rec
        })

    # Convert to DataFrame
    summary_df = pd.DataFrame(summary_data)
    summary_df.set_index("Question", inplace=True)

    print(summary_df)
    
    all_results = pd.concat([all_results, summary_df], axis=1)


# %%

import plotly.express as px

# Prepare and filter data
df_melted = all_results.reset_index().melt(id_vars='Question', var_name='Threshold', value_name='Agreement')
df_melted = df_melted[df_melted['Threshold'].str.contains('All')]

# Rename thresholds for clarity

df_melted['Threshold'] = df_melted['Threshold'].replace({
    'Positive (≥1 Yes) – All': 'One-or-More',
    'Positive (≥2 Yes) – All': 'Majority',
    'Positive (≥3 Yes) – All': 'Unanimous'
})

# Use a sequential color scale (manually mapped for increasing intensity)
color_map = {
    'One-or-More': '#a6cee3',
    'Majority': '#1f78b4',
    'Unanimous': '#08306b'
}

fig = px.bar(
    df_melted,
    x='Question',
    y='Agreement',
    color='Threshold',
    barmode='group',
    labels={'Agreement': 'Agreement (%)'},
    color_discrete_map=color_map,
    text='Agreement'
)

# Update layout for white background and readable labels
fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis_tickangle=30,
    height=500,
    font=dict(color='black'),
    yaxis=dict(range=[0, 100]),
    xaxis_title='',
)

# Display values on top of bars
fig.update_traces(textposition='outside')

# Save the figure as an image
fig.write_image("agreement_by_question_and_threshold.png")

fig.show()

# %%

import plotly.express as px

# Prepare and filter data
df_melted = all_results.reset_index().melt(id_vars='Question', var_name='Threshold', value_name='Agreement')
df_melted = df_melted[df_melted['Threshold'].str.contains('All')]

# Rename thresholds for clarity
df_melted['Threshold'] = df_melted['Threshold'].replace({
    'Positive (≥1 Yes) – All': 'One-or-More',
    'Positive (≥2 Yes) – All': 'Majority',
    'Positive (≥3 Yes) – All': 'Unanimous'
})

# Sequential color mapping
color_map = {
    'One-or-More': '#a6cee3',
    'Majority': '#1f78b4',
    'Unanimous': '#08306b'
}

# Horizontal bar plot
fig = px.bar(
    df_melted,
    x='Agreement',
    y='Question',
    color='Threshold',
    orientation='h',
    barmode='group',
    labels={'Agreement': 'Agreement (%)'},
    color_discrete_map=color_map,
    text='Agreement'
)

# Reverse the order of questions in the bar plot
question_order = df_melted['Question'].unique()[::-1]  # reversed order

fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=650,
    font=dict(color='black', size=16),
    xaxis=dict(range=[0, 100], title_font=dict(size=18)),
    yaxis=dict(
        title='',
        categoryorder='array',
        categoryarray=question_order
    ),
)

# Ensure values are outside and consistently formatted
fig.update_traces(
    textposition='outside',
    textfont_size=16,
    insidetextanchor='start',
    cliponaxis=False
)

# Save image
fig.write_image("agreement_by_question_and_threshold.png")

fig.show()



# %%
