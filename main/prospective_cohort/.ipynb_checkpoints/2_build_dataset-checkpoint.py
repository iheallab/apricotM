#%%

# Import libraries
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from variables import time_window, PROSP_DATA_DIR, MODEL_DIR, VAR_MAP

#%%
# Load sequential data

seq = pd.read_csv(f"{PROSP_DATA_DIR}/final/seq.csv")

#%%
# Load outcomes data

outcomes = pd.read_csv(f"{PROSP_DATA_DIR}/final/outcomes.csv")

#%%

# Filter ICU stays with missing data in any 4h interval

seq = seq[seq["shift_id"].isin(outcomes["shift_id"].unique())]

missing1 = outcomes.loc[
    ~outcomes["shift_id"].isin(seq["shift_id"].unique()), "icustay_id"
].unique()
missing2 = seq.loc[
    ~seq["shift_id"].isin(outcomes["shift_id"].unique()), "icustay_id"
].unique()

missing = list(set(list(missing1) + list(missing2)))

seq = seq[~seq["icustay_id"].isin(missing)]
outcomes = outcomes[~outcomes["icustay_id"].isin(missing)]

print(f'ICU stays in sequential data: {len(seq["icustay_id"].unique())}')
print(f'ICU stays in outcome data: {len(outcomes["icustay_id"].unique())}')

print(f'Shifts in sequential data: {len(seq["shift_id"].unique())}')
print(f'Shifts in outcome data: {len(outcomes["shift_id"].unique())}')

#%%

# Build static data

static = pd.read_csv(f"{PROSP_DATA_DIR}/final/static.csv")

static = static[static["icustay_id"].isin(seq["icustay_id"].unique())]

#%%

# Scale data

with open(f"{MODEL_DIR}/scalers_static.pkl", "rb") as f:
    scalers = pickle.load(f)

print("* Scaling static...")

static["sex"] = scalers["scaler_gender"].transform(static["sex"])
static["race"] = scalers["scaler_race"].transform(static["race"])


def impute_and_scale_static(df):
    exclude = ["icustay_id"]
    columns = [c for c in df.columns if c not in exclude]

    scaler = scalers["scaler_static"]
    df[columns] = scaler.transform(df[columns])

    df = df.reset_index()
    return df


static = impute_and_scale_static(static)

with open(f"{MODEL_DIR}/scalers_seq.pkl", "rb") as f:
    scalers = pickle.load(f)

VALUE_COLS = ["value"]


def _standardize_variable(group):

    variable_code = int(group["variable_code"].values[0])

    scaler = scalers[f"scaler{variable_code}"]
    group[VALUE_COLS] = scaler.transform(group[VALUE_COLS])

    return group


def standardize_variables(seq):

    variables = seq["variable"].unique()
    print(f"Variables after imputation: {len(variables)}")

    seq = (
        seq.groupby("variable")
        .apply(_standardize_variable)
        .sort_values(by=["icustay_id", "hours"])
    )
    variables = seq["variable"].unique()
    print(f"Variables after standardization: {len(variables)}")

    return seq


print("* Scaling seq...")

seq = standardize_variables(seq)

print(f'ICU stays in outcomes data: {len(outcomes["icustay_id"].unique())}')
print(f'ICU stays in sequential data: {len(seq["icustay_id"].unique())}')

seq = seq.reset_index(drop=True)

hours_scaler = scalers["scaler_hours"]

seq[["hours"]] = hours_scaler.transform(seq[["hours"]])

seq.drop("variable", axis=1, inplace=True)

#%%

# Map labels to numeric values

map_transitions = {
    "stable-stable": 0,
    "unstable-unstable": 1,
    "unstable-stable": 2,
    "stable-discharge": 4,
    "stable-unstable": 3,
    "unstable-dead": 5,
    "unstable-discharge": 4,
    "stable-dead": 5,
}


outcomes["transition"] = outcomes["transition"].map(map_transitions)

map_transitions = {
    "no mv-no mv": 0,
    "mv-mv": 1,
    "mv-no mv": 2,
    "no mv-mv": 3,
}

outcomes["transition_mv"] = outcomes["transition_mv"].map(map_transitions)

map_transitions = {
    "no vp-no vp": 0,
    "vp-vp": 1,
    "vp-no vp": 2,
    "no vp-vp": 3,
}

outcomes["transition_vp"] = outcomes["transition_vp"].map(map_transitions)

map_transitions = {
    "no crrt-no crrt": 0,
    "crrt-crrt": 1,
    "crrt-no crrt": 2,
    "no crrt-crrt": 3,
}

outcomes["transition_crrt"] = outcomes["transition_crrt"].map(map_transitions)

outcomes = outcomes.sort_values(by=["icustay_id", "shift_start"]).reset_index(drop=True)

map_states = {
    "dead": 3,
    "unstable": 2,
    "stable": 1,
    "discharge": 0,
}

outcomes["final_state"] = outcomes["final_state"].replace(map_states)

print(f'Number of shifts in outcome data: {len(outcomes["shift_id"].unique())}')
print(f'Number of shifts in sequential data: {len(seq["shift_id"].unique())}')

outcomes = outcomes.reset_index(drop=True)
seq = seq.reset_index(drop=True)

# %%

# Prepare sequential data for input

seq_len = seq.groupby("shift_id").size().reset_index().rename({0: "seq_len"}, axis=1)

outcomes = outcomes.merge(seq_len, on="shift_id", how="left")

with open(f"{MODEL_DIR}/apricotm/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

max_seq_len = best_params["seq_len"]

seq_len = outcomes["seq_len"].tolist()

qkv = np.zeros((len(outcomes), max_seq_len, 4))
X = seq.iloc[:, :4].values

start = 0
for i in range(len(outcomes)):
    if seq_len[i] <= max_seq_len:
        end = start + seq_len[i]
        qkv[i, : seq_len[i], 0] = X[start:end, 2]
        qkv[i, : seq_len[i], 1] = X[start:end, 1]
        qkv[i, : seq_len[i], 2] = X[start:end, 3]
        qkv[i, : seq_len[i], 3] = X[start:end, 0]
    else:
        end = start + seq_len[i]
        trunc = end - max_seq_len
        qkv[i, :, 0] = X[trunc:end, 2]
        qkv[i, :, 1] = X[trunc:end, 1]
        qkv[i, :, 2] = X[trunc:end, 3]
        qkv[i, :, 3] = X[trunc:end, 0]

    start += seq_len[i]

#%%

# Prepare static data for input

static = static.sort_values(by=["icustay_id"]).reset_index(drop=True)

static = static.set_index("icustay_id")
static = static.loc[outcomes["icustay_id"].tolist(), :]
static = static.reset_index()

static = static.iloc[:, 2:]

static = static.to_numpy()

# %%
# Prepare outcome data for input

# y = outcomes["final_state"].values
# y_trans = outcomes["transition"].values
# y_mv = outcomes["transition_mv"].values
# y_pressor = outcomes["transition_vp"].values
# y_crrt = outcomes["transition_crrt"].values

# from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import OneHotEncoder

y = outcomes["final_state"].values
enc = OneHotEncoder()
y = enc.fit_transform(y.reshape(-1, 1)).toarray()

y_trans = outcomes["transition"].values
enc = OneHotEncoder()
y_trans = enc.fit_transform(y_trans.reshape(-1, 1)).toarray()

y_mv = outcomes["transition_mv"].values
enc = OneHotEncoder()
y_mv = enc.fit_transform(y_mv.reshape(-1, 1)).toarray()

y_pressor = outcomes["transition_vp"].values
enc = OneHotEncoder()
y_pressor = enc.fit_transform(y_pressor.reshape(-1, 1)).toarray()

y_crrt = outcomes["transition_crrt"].values
enc = OneHotEncoder()
y_crrt = enc.fit_transform(y_crrt.reshape(-1, 1)).toarray()

y_trans = np.concatenate(
    [y_trans[:, 2:4], y_mv[:, 2:], y_pressor[:, 2:], y_crrt[:, 2:]], axis=1
)

print(f"ICU admissions prospective: {len(np.unique(qkv[:,0,3]))}")

# enc = OneHotEncoder(categories=[[0, 1, 2, 3]])
# y_trans = enc.fit_transform(y_trans.reshape(-1, 1)).toarray()

# enc = OneHotEncoder(categories=[[0, 1, 2, 3]])
# y_mv = enc.fit_transform(y_mv.reshape(-1, 1)).toarray()

# enc = OneHotEncoder(categories=[[0, 1, 2, 3]])
# y_pressor = enc.fit_transform(y_pressor.reshape(-1, 1)).toarray()

# enc = OneHotEncoder(categories=[[0, 1, 2, 3]])
# y_crrt = enc.fit_transform(y_crrt.reshape(-1, 1)).toarray()

# enc = OneHotEncoder(categories=[[0, 1, 2, 3]])
# y_main = enc.fit_transform(y.reshape(-1, 1)).toarray()

# y_trans = np.concatenate(
#     (y_trans[:, 2:4], y_mv[:, 2:], y_pressor[:, 2:], y_crrt[:, 2:]), axis=1
# )

#%%

# Save dataset into hdf5 file

import h5py

with h5py.File(f"{PROSP_DATA_DIR}/dataset.h5", "w", libver="latest") as hf:

    # Save train data
    group = hf.create_group("prospective")
    group.create_dataset("X", data=qkv)
    group.create_dataset("static", data=static)
    group.create_dataset("y_trans", data=y_trans)
    group.create_dataset("y_main", data=y)
