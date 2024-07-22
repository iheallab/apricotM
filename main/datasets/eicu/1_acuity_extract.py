# %%
# Import libraries

import pandas as pd
import numpy as np

import os

# %%

# Get ventilation from Care Plan table

careplan = pd.read_csv("careplangeneral.csv")

careplan = careplan[
    (careplan["cplgroup"] == "Airway") | (careplan["cplgroup"] == "Ventilation")
]
careplan = careplan.sort_values(by=["patientunitstayid", "cplitemoffset"])

map_vent = {
    "Not intubated/normal airway": "non-invasive",
    "Spontaneous - adequate": "non-invasive",
    "Intubated/oral ETT": "invasive",
    "Ventilated - with daily extubation evaluation": "invasive",
    "Spontaneous - tenuous": "non-invasive",
    "Non-invasive ventilation": "non-invasive",
    "Ventilated - with no daily extubation trial": "invasive",
    "Ventilated - rapid wean/extubation": "invasive",
    "Intubated/trach-chronic": "invasive",
    "Intubated/trach-acute": "invasive",
    "Ventilated - chronic dependency": "invasive",
    "Not intubated/partial airway obstruction": "non-invasive",
    "Intubated/oral ETT - difficult": "invasive",
    "Intubated/nasal ETT": "invasive",
    "Intubated/nasal ETT - difficult": "invasive",
}

print(len(careplan["patientunitstayid"].unique()))
print(careplan["cplitemvalue"].value_counts())

# %%

# Get ventilation events from Respiratory Charting table

respchart = pd.read_csv(
    "respiratoryCharting.csv",
    usecols=["patientunitstayid", "respchartoffset", "respchartvaluelabel"],
)

devices = [
    "PEEP",
    "Total RR",
    "Vent Rate",
    "Tidal Volume (set)",
    "Tidal Volume Observed (VT)",
    "Tidal Volume, Delivered",
]

labels = respchart["respchartvaluelabel"].value_counts()

labels = pd.DataFrame(data={"label": labels.index, "count": labels.values})

ventchart = respchart[respchart["respchartvaluelabel"].isin(devices)]

print(len(ventchart["patientunitstayid"].unique()))

# %%

# Get invasive ventilation events

ventchart["MV"] = "invasive"
careplan["MV"] = careplan["cplitemvalue"].map(map_vent)

ventchart = ventchart.sort_values(by=["patientunitstayid", "respchartoffset"])

careplan["activeupondischarge"] = careplan["activeupondischarge"].astype(int)

active_disch = careplan[careplan["activeupondischarge"] == 1]

active_disch = list(active_disch["patientunitstayid"].unique())

columns = ["patientunitstayid", "respchartoffset", "respchartvaluelabel", "MV"]

careplan = careplan.rename(
    {"cplitemoffset": "respchartoffset", "cplitemvalue": "respchartvaluelabel"}, axis=1
)

ventchart = (
    pd.concat([ventchart.loc[:, columns], careplan.loc[:, columns]], axis=0)
    .sort_values(by=["patientunitstayid", "respchartoffset"])
    .reset_index(drop=True)
)

ventchart = ventchart.sort_values(by=["patientunitstayid", "respchartoffset"])


ventchart.loc[(ventchart["MV"] == "invasive"), "MV"] = 1
ventchart.loc[(ventchart["MV"] == "non-invasive"), "MV"] = 0


def duration_mv(df):
    df = df.sort_values(["respchartoffset"]).reset_index(drop=True)
    df["next_time"] = df["respchartoffset"].shift(-1)
    df.loc[df["next_time"].isnull(), "next_time"] = df.loc[
        df["next_time"].isnull(), "respchartoffset"
    ]
    eid = df.iloc[0]["patientunitstayid"]
    df["mv_diff"] = (df.MV.diff(1) != 0).cumsum()
    df = df[df["MV"] == 1]
    mv_duration = pd.DataFrame(
        {
            "patientunitstayid": eid,
            "start": df.groupby("mv_diff").respchartoffset.first(),
            "end": df.groupby("mv_diff").next_time.last(),
        }
    ).reset_index(drop=True)
    return mv_duration


mv_ids = ventchart.loc[ventchart["MV"] == 1, "patientunitstayid"].unique().tolist()
ventchart = ventchart[ventchart["patientunitstayid"].isin(mv_ids)]
mv_duration = (
    ventchart.groupby(["patientunitstayid"]).apply(duration_mv).reset_index(drop=True)
)

mv_duration["procedure"] = "mv"

# %%

# Get vasopressor events

pressor = pd.read_csv("infusiondrug.csv")

filter_list = [
    "dopamine",
    "epinephrine",
    "norepinephrine",
    "phenylephrine",
    "vasopressin",
]

pressor["drugname"] = pressor["drugname"].fillna("missing")

pressor = pressor[pressor["drugname"].str.contains("|".join(filter_list), case=False)]

print(len(pressor["patientunitstayid"].unique()))

pressor = pressor.sort_values(by=["patientunitstayid", "infusionoffset"])

med_counts = pressor["drugname"].value_counts()


def calculate_sequence_numbers(group):
    time_diff = group["infusionoffset"].diff().fillna(0)
    sequences = (time_diff >= 1440).cumsum()
    group["seq"] = sequences + 1
    return group


pressor = pressor.groupby("patientunitstayid").apply(calculate_sequence_numbers)

pressor = pressor.reset_index(drop=True)

pressor = (
    pressor.groupby(["patientunitstayid", "seq"])["infusionoffset"]
    .agg(["min", "max"])
    .reset_index()
)

pressor = pressor.rename({"min": "start", "max": "end"}, axis=1)

pressor["procedure"] = "vp"

print(len(pressor["patientunitstayid"].unique()))

# %%

# Get CRRT events

crrt = pd.read_csv("treatment.csv")

items = crrt["treatmentstring"].value_counts()

items = pd.DataFrame({"label": items.index, "count": items.values})

crrt = crrt[
    (crrt["treatmentstring"] == "renal|dialysis|C V V H D")
    | (crrt["treatmentstring"] == "renal|dialysis|C V V H")
]


print(len(crrt["patientunitstayid"].unique()))

crrt = crrt.sort_values(by=["patientunitstayid", "treatmentoffset"])

admissions = pd.read_csv("patient.csv")

admissions = admissions[
    admissions["patientunitstayid"].isin(crrt["patientunitstayid"].unique())
]

print(len(crrt["patientunitstayid"].unique()))

crrt = crrt.sort_values(by=["patientunitstayid", "treatmentoffset"])

crrt["activeupondischarge"] = crrt["activeupondischarge"].astype(int)

disch = crrt.groupby(["patientunitstayid"])["activeupondischarge"].sum().reset_index()

disch.loc[disch["activeupondischarge"] > 1, "activeupondischarge"] = 1

print(disch["activeupondischarge"].value_counts())

crrt = (
    crrt.groupby(["patientunitstayid"])["treatmentoffset"]
    .agg(["min", "max"])
    .reset_index()
)

crrt = crrt.rename({"min": "start", "max": "end"}, axis=1)

crrt = crrt.merge(
    admissions.loc[
        :, ["patientunitstayid", "unitdischargeoffset", "unitdischargelocation"]
    ],
    how="left",
    on="patientunitstayid",
)

crrt = crrt.merge(disch, how="left", on="patientunitstayid")

crrt.loc[(crrt["activeupondischarge"] == 1), "end"] = crrt.loc[
    (crrt["activeupondischarge"] == 1), "unitdischargeoffset"
]

crrt["procedure"] = "crrt"

# %%

# Extract blood transfusion

cols = [
    "patientunitstayid",
    "intakeoutputoffset",
    "celllabel",
    "cellvaluenumeric",
    "cellpath",
]

# Specify the chunk size (number of rows to read at a time)
chunk_size = 1000000
chunks = []
header_written = False


count = 1

for chunk in pd.read_csv(
    "intakeOutput.csv", chunksize=chunk_size, low_memory=False, usecols=cols
):

    chunk = chunk[
        (chunk["cellpath"].str.contains("blood"))
        & (~chunk["cellpath"].str.contains("output"))
        & (~chunk["cellpath"].str.contains("Cryoprecipitate"))
    ]

    chunk.to_csv("transfusion.csv", mode="a", index=False, header=not header_written)

    print(f"Processed chunk {count}")
    count += 1

    header_written = True

# %%

# Get massive blood transfusion events

transfusion = pd.read_csv("transfusion.csv")

print(len(transfusion["patientunitstayid"].unique()))

transfusion["cellvaluenumeric"] = transfusion["cellvaluenumeric"] / 200

transfusion = transfusion.sort_values(by=["patientunitstayid", "intakeoutputoffset"])

transfusion = transfusion[transfusion["intakeoutputoffset"] >= 0]

# Convert Admission_Time_Minutes to datetime
transfusion["Admission_Time"] = pd.to_datetime(
    transfusion["intakeoutputoffset"], unit="m"
)


# Function to calculate cumulative values for the past 24 hours
def calculate_cumulative_24h(row, df):
    # Filter previous 24 hours data for the same patient
    previous_24h = df[
        (df["patientunitstayid"] == row["patientunitstayid"])
        & (df["Admission_Time"] >= (row["Admission_Time"] - pd.Timedelta(hours=24)))
        & (df["Admission_Time"] <= row["Admission_Time"])
    ]
    # Calculate cumulative value
    cumulative_24h = previous_24h["cellvaluenumeric"].sum()
    return cumulative_24h


# Apply the function to each row
transfusion["Cumulative_Value_24h"] = transfusion.apply(
    lambda row: calculate_cumulative_24h(row, transfusion), axis=1
)

transfusion = transfusion[transfusion["Cumulative_Value_24h"] >= 10]

transfusion = transfusion.rename({"intakeoutputoffset": "start"}, axis=1)

transfusion["end"] = transfusion["start"]

transfusion["procedure"] = "bt"

print(len(transfusion["patientunitstayid"].unique()))

# %%

# Merge all procedures

cols = ["patientunitstayid", "start", "end", "procedure"]

procedures = pd.concat(
    [
        mv_duration.loc[:, cols],
        pressor.loc[:, cols],
        crrt.loc[:, cols],
        transfusion.loc[:, cols],
    ]
)

procedures = procedures.sort_values(by=["patientunitstayid", "start"])

print(procedures["procedure"].value_counts())

# %%

# Generate 4h intervals

icustays = pd.read_csv("patient.csv")

icustay_ids = icustays["patientunitstayid"].unique()


# Define a custom function to generate 4-hour intervals for each row
def generate_intervals(row):
    start_time = 0
    end_time = row["unitdischargeoffset"]
    intervals = range(start_time, end_time, 240)
    return intervals


# Apply the function to each row in the DataFrame
icustays["interval"] = icustays.apply(generate_intervals, axis=1)

# Explode the Timestamps column to expand it into separate rows
icustays = icustays.explode("interval").reset_index(drop=True)

print(len(icustays["patientunitstayid"].unique()))


# %%

# Label acuity for 4h intervals

icustays = icustays.rename({"interval": "shift_start"}, axis=1)
icustays["shift_end"] = icustays["shift_start"] + 240

icustays = icustays.merge(procedures, how="outer", on=["patientunitstayid"])

print(len(icustays["patientunitstayid"].unique()))

icustays["ind_start"] = icustays["start"] - icustays["shift_start"]
icustays["ind_end"] = icustays["shift_end"] - icustays["end"]

icustays["ind_start"] = icustays["ind_start"].fillna(0)
icustays["ind_end"] = icustays["ind_end"].fillna(0)

icustays.loc[
    (icustays["ind_start"] >= 240) | (icustays["ind_end"] >= 240), "procedure"
] = np.nan

icustays_filt = icustays.drop_duplicates(
    subset=["patientunitstayid", "shift_start", "procedure"]
)

print(len(icustays_filt["patientunitstayid"].unique()))

icustays_filt["procedure"] = icustays_filt["procedure"].fillna("missing")

icustays_filt["proc_ind"] = 0
icustays_filt.loc[(icustays_filt["procedure"] != "missing"), "proc_ind"] = 1

df_pivot = icustays_filt.pivot_table(
    index=["patientunitstayid", "shift_start"],
    columns="procedure",
    values="proc_ind",
    fill_value=None,
)

df_pivot.drop("missing", inplace=True, axis=1)

df_pivot.fillna(0, inplace=True)

df_pivot.reset_index(inplace=True)


# Define a custom function to determine stability
def determine_stability(row):
    if 1 in row.values[2:]:
        return "unstable"
    else:
        return "stable"


# Apply the custom function to create the 'Stability' column
df_pivot["final_state"] = df_pivot.apply(determine_stability, axis=1)

print(df_pivot["final_state"].value_counts())

df_pivot = df_pivot.sort_values(by=["patientunitstayid", "shift_start"])

df_pivot.to_csv("acuity_states.csv", index=None)
