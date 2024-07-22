# %%
import pandas as pd
import numpy as np

import os

ditems = pd.read_csv("d_items.csv")


# Item ids for ventilation, transfusion, vasopressors, and crrt
mv = [225792]
bt = [229620]
vp = [222315, 221749, 221289, 221906, 229617, 221662]
crrt = [225802, 225803, 225809]

all_proc = mv + bt + vp + crrt

# %%

# Extract all procedures
chunk_size = 100000
chunks = []

count = 1
# Read the CSV file in chunks
for chunk in pd.read_csv("procedureevents.csv", chunksize=chunk_size):
    chunk = chunk[chunk["itemid"].isin(all_proc)]
    chunks.append(chunk)
    print(f"Processed chunk {count}")
    count += 1

chartevents = pd.concat(chunks)
chartevents.to_csv("procedures.csv", index=None)

del chunks


# %%

# Extract vasopressors
chunk_size = 100000
chunks = []

count = 1
# Read the CSV file in chunks
for chunk in pd.read_csv("inputevents.csv", chunksize=chunk_size):
    chunk = chunk[chunk["itemid"].isin(all_proc)]
    chunks.append(chunk)
    print(f"Processed chunk {count}")
    count += 1

chartevents = pd.concat(chunks)
chartevents.to_csv("vasopressors.csv", index=None)

del chunks

# %%

# Merge all procedures

columns = ["subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid"]

proced = pd.read_csv("procedures.csv", usecols=columns)
pressor = pd.read_csv("vasopressors.csv", usecols=columns)

print(pressor["itemid"].value_counts())

all_proced = pd.concat([proced, pressor], axis=0).sort_values(
    by=["stay_id", "starttime"]
)

map_proced = {
    "mv": mv,
    "vp": vp,
    "crrt": crrt,
    "bt": bt,
}

inverted_dict = {value: key for key, values in map_proced.items() for value in values}

all_proced["procedure"] = all_proced["itemid"].map(inverted_dict)

print(all_proced["procedure"].value_counts())

# %%

# Generate 4h intervals

icustays = pd.read_csv(
    "icustays.csv", usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"]
)

icustay_ids = icustays["stay_id"].unique()


# Define a custom function to generate 4-hour intervals for each row
def generate_intervals(row):
    start_time = pd.to_datetime(row["intime"])
    end_time = pd.to_datetime(row["outtime"])
    intervals = pd.date_range(start=start_time, end=end_time, freq="4H")
    return intervals


# Apply the function to each row in the DataFrame
icustays["Timestamps"] = icustays.apply(generate_intervals, axis=1)

# Explode the Timestamps column to expand it into separate rows
icustays = icustays.explode("Timestamps").reset_index(drop=True)

print(len(icustays["stay_id"].unique()))

# %%

icustays = icustays.rename({"Timestamps": "shift_start"}, axis=1)
icustays["shift_end"] = icustays["shift_start"] + pd.Timedelta(hours=4)

icustays = icustays.merge(
    all_proced, how="outer", on=["subject_id", "hadm_id", "stay_id"]
)

print(len(icustays["stay_id"].unique()))

icustays["ind_start"] = (
    pd.to_datetime(icustays["starttime"]) - pd.to_datetime(icustays["shift_start"])
) / np.timedelta64(1, "h")
icustays["ind_end"] = (
    pd.to_datetime(icustays["shift_end"]) - pd.to_datetime(icustays["endtime"])
) / np.timedelta64(1, "h")

icustays["ind_start"] = icustays["ind_start"].fillna(0)
icustays["ind_end"] = icustays["ind_end"].fillna(0)

icustays.loc[(icustays["ind_start"] >= 4) | (icustays["ind_end"] >= 4), "procedure"] = (
    np.nan
)

icustays_filt = icustays.drop_duplicates(subset=["shift_start", "procedure"])

print(len(icustays["stay_id"].unique()))

icustays_filt["procedure"] = icustays_filt["procedure"].fillna("missing")

icustays_filt["proc_ind"] = 0
icustays_filt.loc[(icustays_filt["procedure"] != "missing"), "proc_ind"] = 1

df_pivot = icustays_filt.pivot_table(
    index=["stay_id", "shift_start"],
    columns="procedure",
    values="proc_ind",
    fill_value=None,
)

df_pivot.drop("missing", inplace=True, axis=1)

df_pivot.fillna(0, inplace=True)

df_pivot.reset_index(inplace=True)


# Define a custom function to determine stability
def determine_stability(row):
    if 1 in row.values:
        return "unstable"
    else:
        return "stable"


# Apply the custom function to create the 'Stability' column
df_pivot["final_state"] = df_pivot.apply(determine_stability, axis=1)

print(df_pivot["final_state"].value_counts())

df_pivot = df_pivot.sort_values(by=["stay_id", "shift_start"])

df_pivot.to_csv("acuity_states.csv", index=None)
