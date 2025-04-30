#%%

# Import libraries

import pandas as pd
import numpy as np
import pickle
from scipy import stats
import re
import os

from variables import OUTPUT_DIR, PROSP_DATA_DIR

if not os.path.exists(f"{OUTPUT_DIR}/analyses/all_patient_characteristics"):
    os.makedirs(f"{OUTPUT_DIR}/analyses/all_patient_characteristics")

#%%

# Load admissions

admissions_eicu = pd.read_csv(f"{OUTPUT_DIR}/final_data/admissions_eicu.csv")
admissions_eicu = admissions_eicu.rename(
    {
        "patientunitstayid": "icustay_id",
        "patienthealthsystemstayid": "merged_enc_id",
        "uniquepid": "patient_deiden_id",
        "unitdischargeoffset": "icu_los",
    },
    axis=1,
)

#%%

# Split into sets

with open(f"{OUTPUT_DIR}/final_data/ids.pkl", "rb") as f:
    ids = pickle.load(f)
    ids_train, ids_val, ids_ext, ids_temp = (
        ids["train"],
        ids["val"],
        ids["ext_test"],
        ids["temp_test"],
    )

ids_dev = ids_train[0] + ids_val[0]

develop = admissions_eicu[admissions_eicu["icustay_id"].isin(ids_dev)]
external = admissions_eicu[admissions_eicu["icustay_id"].isin(ids_ext)]

# %%

import plotly.graph_objects as go
import pandas as pd

# Count ICU admissions per hospital
hospital_counts = admissions_eicu['hospitalid'].value_counts().sort_values(ascending=False)

# Find index where count drops below 800
cutoff_index = (hospital_counts >= 800).sum() - 0.5

# Create bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    x=list(range(len(hospital_counts))),
    y=hospital_counts.values,
    marker_color='steelblue',
    name='ICU Admissions'
))

# Add vertical cutoff line at the cutoff index
fig.add_shape(
    type='line',
    x0=cutoff_index,
    x1=cutoff_index,
    y0=0,
    y1=max(hospital_counts.values),
    line=dict(color='red', width=2, dash='dash'),
)

# Add annotation for the cutoff
fig.add_annotation(
    x=cutoff_index,
    y=max(hospital_counts.values) * 0.95,
    text="Cutoff",
    showarrow=True,
    arrowhead=2,
    ax=40,
    ay=-40,
    font=dict(color='red', size=16)
)

fig.update_layout(
    xaxis=dict(
        title=dict(text="Hospital", font=dict(size=18)),
        tickfont=dict(size=14)
    ),
    yaxis=dict(
        title=dict(text="Number of ICU Admissions", font=dict(size=18)),
        tickfont=dict(size=14)
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=False,
    margin=dict(t=60, b=60, l=80, r=40),
)

# Save plot as a high-resolution PNG file
output_path = f"{OUTPUT_DIR}/analyses/all_patient_characteristics/hospital_counts.png"
fig.write_image(output_path, width=1200, height=800, scale=2)

fig.show()


# %%
