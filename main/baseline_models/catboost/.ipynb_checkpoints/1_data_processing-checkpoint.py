#%%
# Import libraries

import pandas as pd
import numpy as np
import pickle
import os

from variables import time_window, OUTPUT_DIR, PROSP_DIR, MODEL_DIR

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

#%%

# Load data

seq = pd.read_csv("%s/final_data/seq.csv" % OUTPUT_DIR, compression="gzip")
static = pd.read_csv("%s/final_data/static.csv" % OUTPUT_DIR)
outcomes = pd.read_csv("%s/final_data/outcomes.csv" % OUTPUT_DIR)

outcomes = outcomes[outcomes["interval"] != 0]

seq = seq[seq["shift_id"].isin(outcomes["shift_id"].unique())]

with open("%s/model/variable_mapping.pkl" % OUTPUT_DIR, "rb") as f:
    variable_map = pickle.load(f)

seq["variable"] = seq["variable_code"].map(variable_map)

seq.drop("variable_code", inplace=True, axis=1)

#%%

# Filter shift IDs

missing = outcomes[~outcomes["shift_id"].isin(seq["shift_id"].unique())]

outcomes = outcomes[~outcomes["icustay_id"].isin(missing["icustay_id"].unique())]
seq = seq[~seq["icustay_id"].isin(missing["icustay_id"].unique())]
static = static[static["icustay_id"].isin(outcomes["icustay_id"].unique())]

outcomes = outcomes.reset_index(drop=True)
seq = seq.reset_index(drop=True)
static = static.reset_index(drop=True)

#%%

# Load prospective admissions

# Load sequential data

seq_prosp = pd.read_csv(f"{PROSP_DIR}/final/seq.csv")

# Load outcomes data

outcomes_prosp = pd.read_csv(f"{PROSP_DIR}/final/outcomes.csv")

# Filter ICU stays with missing data in any 4h interval

seq_prosp = seq_prosp[seq_prosp["shift_id"].isin(outcomes_prosp["shift_id"].unique())]

missing1 = outcomes_prosp.loc[
    ~outcomes_prosp["shift_id"].isin(seq_prosp["shift_id"].unique()), "icustay_id"
].unique()
missing2 = seq_prosp.loc[
    ~seq_prosp["shift_id"].isin(outcomes_prosp["shift_id"].unique()), "icustay_id"
].unique()

missing = list(set(list(missing1) + list(missing2)))

seq_prosp = seq_prosp[~seq_prosp["icustay_id"].isin(missing)]
outcomes_prosp = outcomes_prosp[~outcomes_prosp["icustay_id"].isin(missing)]

print(f'ICU stays in sequential data: {len(seq_prosp["icustay_id"].unique())}')
print(f'ICU stays in outcome data: {len(outcomes_prosp["icustay_id"].unique())}')

print(f'Shifts in sequential data: {len(seq_prosp["shift_id"].unique())}')
print(f'Shifts in outcome data: {len(outcomes_prosp["shift_id"].unique())}')

# Build static data

static_prosp = pd.read_csv(f"{PROSP_DIR}/final/static.csv")

static_prosp = static_prosp[
    static_prosp["icustay_id"].isin(seq_prosp["icustay_id"].unique())
]

# Scale data

with open(f"{OUTPUT_DIR}/model/scalers_static.pkl", "rb") as f:
    scalers = pickle.load(f)

print("* Scaling static_prosp...")

static_prosp["sex"] = scalers["scaler_gender"].transform(static_prosp["sex"])
static_prosp["race"] = scalers["scaler_race"].transform(static_prosp["race"])


def impute_and_scale_static(df):
    exclude = ["icustay_id"]
    columns = [c for c in df.columns if c not in exclude]

    scaler = scalers["scaler_static"]
    df[columns] = scaler.transform(df[columns])

    df = df.reset_index()
    return df


static_prosp = impute_and_scale_static(static_prosp)

with open(f"{OUTPUT_DIR}/model/scalers_seq.pkl", "rb") as f:
    scalers = pickle.load(f)

VALUE_COLS = ["value"]


def _standardize_variable(group):

    variable_code = int(group["variable_code"].values[0])

    scaler = scalers[f"scaler{variable_code}"]
    group[VALUE_COLS] = scaler.transform(group[VALUE_COLS])

    return group


def standardize_variables(seq_prosp):

    variables = seq_prosp["variable"].unique()
    print(f"Variables after imputation: {len(variables)}")

    seq_prosp = (
        seq_prosp.groupby("variable")
        .apply(_standardize_variable)
        .sort_values(by=["icustay_id", "hours"])
    )
    variables = seq_prosp["variable"].unique()
    print(f"Variables after standardization: {len(variables)}")

    return seq_prosp


print("* Scaling seq_prosp...")

seq_prosp = standardize_variables(seq_prosp)

print(f'ICU stays in outcomes_prosp data: {len(outcomes_prosp["icustay_id"].unique())}')
print(f'ICU stays in sequential data: {len(seq_prosp["icustay_id"].unique())}')

seq_prosp = seq_prosp.reset_index(drop=True)

hours_scaler = scalers["scaler_hours"]

seq_prosp[["hours"]] = hours_scaler.transform(seq_prosp[["hours"]])

outcomes_prosp.drop("shift_start", axis=1, inplace=True)
static_prosp.drop("index", axis=1, inplace=True)
seq_prosp.drop("variable_code", axis=1, inplace=True)


ids_prosp = list(seq_prosp["icustay_id"].unique())

outcomes_prosp.drop(labels=["dead_ind", "dischg_ind"], axis=1, inplace=True)

#%%

# Merge prospective data with dataset

seq = pd.concat([seq, seq_prosp], axis=0)
outcomes = pd.concat([outcomes, outcomes_prosp], axis=0)
static = pd.concat([static, static_prosp], axis=0)

#%%

# Calculate statistical features

seq_mean = (
    seq.groupby(by=["icustay_id", "shift_id", "variable"])["value"].mean().reset_index()
)
seq_std = (
    seq.groupby(by=["icustay_id", "shift_id", "variable"])["value"].std().reset_index()
)
seq_min = (
    seq.groupby(by=["icustay_id", "shift_id", "variable"])["value"].min().reset_index()
)
seq_max = (
    seq.groupby(by=["icustay_id", "shift_id", "variable"])["value"].max().reset_index()
)

#%%

# Tabularize sequential data

seq_mean["variable"] = seq_mean["variable"] + "_mean"

seq_mean = seq_mean.set_index(["icustay_id", "shift_id", "variable"])

multi_index = seq_mean.index
data = seq_mean.values.flatten()
seq_mean = pd.Series(data, index=multi_index)
print("Unstacking")
seq_mean = seq_mean.unstack("variable")

seq_mean = seq_mean.reset_index()

seq_std["variable"] = seq_std["variable"] + "_std"

seq_std = seq_std.set_index(["icustay_id", "shift_id", "variable"])

multi_index = seq_std.index
data = seq_std.values.flatten()
seq_std = pd.Series(data, index=multi_index)
print("Unstacking")
seq_std = seq_std.unstack("variable")

seq_std = seq_std.reset_index()

seq_min["variable"] = seq_min["variable"] + "_min"

seq_min = seq_min.set_index(["icustay_id", "shift_id", "variable"])

multi_index = seq_min.index
data = seq_min.values.flatten()
seq_min = pd.Series(data, index=multi_index)
print("Unstacking")
seq_min = seq_min.unstack("variable")

seq_min = seq_min.reset_index()

seq_max["variable"] = seq_max["variable"] + "_max"

seq_max = seq_max.set_index(["icustay_id", "shift_id", "variable"])

multi_index = seq_max.index
data = seq_max.values.flatten()
seq_max = pd.Series(data, index=multi_index)
print("Unstacking")
seq_max = seq_max.unstack("variable")

seq_max = seq_max.reset_index()

seq = pd.concat(
    [seq_mean, seq_std.iloc[:, 2:], seq_min.iloc[:, 2:], seq_max.iloc[:, 2:]], axis=1
)

del seq_mean, seq_std, seq_min, seq_max

seq_missing = seq.iloc[:, 2:61].isnull().astype(int)

seq = pd.concat([seq, seq_missing], axis=1)

del seq_missing

#%%

# Combine static and temporal data

outcomes = outcomes.set_index("shift_id")
outcomes = outcomes.loc[seq["shift_id"].tolist(), :]
outcomes = outcomes.reset_index()

static = static.set_index("icustay_id")
static = static.loc[outcomes["icustay_id"].tolist(), :]
static = static.reset_index()

seq.drop("shift_id", axis=1, inplace=True)

seq = pd.concat([seq, static.iloc[:, 1:]], axis=1)

feature_names = seq.columns[1:].tolist()

seq = seq.iloc[:, :].values


#%%

# Prepare targets

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

print(outcomes["transition"].value_counts())

map_transitions = {
    "no mv-no mv": 0,
    "mv-mv": 1,
    "mv-no mv": 2,
    "no mv-mv": 3,
}

outcomes["transition_mv"] = outcomes["transition_mv"].map(map_transitions)

print(outcomes["transition_mv"].value_counts())

map_transitions = {
    "no vp-no vp": 0,
    "vp-vp": 1,
    "vp-no vp": 2,
    "no vp-vp": 3,
}

outcomes["transition_vp"] = outcomes["transition_vp"].map(map_transitions)

print(outcomes["transition_vp"].value_counts())

map_transitions = {
    "no crrt-no crrt": 0,
    "crrt-crrt": 1,
    "crrt-no crrt": 2,
    "no crrt-crrt": 3,
}

outcomes["transition_crrt"] = outcomes["transition_crrt"].map(map_transitions)

print(outcomes["transition_crrt"].value_counts())

map_states = {
    "dead": 3,
    "unstable": 2,
    "stable": 1,
    "discharge": 0,
}

outcomes["final_state"] = outcomes["final_state"].map(map_states)

print(outcomes["final_state"].value_counts())

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

#%%

# Split data into train, val, ext_test, and temp_test

import pickle

with open("%s/final_data/ids.pkl" % OUTPUT_DIR, "rb") as f:
    ids = pickle.load(f)
    ids_train, ids_val, ids_ext, ids_temp = (
        ids["train"],
        ids["val"],
        ids["ext_test"],
        ids["temp_test"],
    )

ids_train = ids_train[0]
ids_val = ids_val[0]

ids_train = np.isin(seq[:, 0], ids_train)
ids_val = np.isin(seq[:, 0], ids_val)
ids_ext = np.isin(seq[:, 0], ids_ext)
ids_temp = np.isin(seq[:, 0], ids_temp)
ids_prosp = np.isin(seq[:, 0], ids_prosp)

# Impute input data

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")

imputer = imputer.fit(seq[ids_train][:, 1:])

seq = imputer.transform(seq[:, 1:])

X_train = seq[ids_train][:, 1:]
y_trans_train = y_trans[ids_train]
y_train = y[ids_train]

X_val = seq[ids_val][:, 1:]
y_trans_val = y_trans[ids_val]
y_val = y[ids_val]

X_ext = seq[ids_ext][:, 1:]
y_trans_ext = y_trans[ids_ext]
y_ext = y[ids_ext]

X_temp = seq[ids_temp][:, 1:]
y_trans_temp = y_trans[ids_temp]
y_temp = y[ids_temp]

X_prosp = seq[ids_prosp][:, 1:]
y_trans_prosp = y_trans[ids_prosp]
y_prosp = y[ids_prosp]

del seq

#%%

# Save hdf5 file

import h5py

with h5py.File("%s/dataset_catboost.h5" % MODEL_DIR, "w", libver="latest") as hf:

    # Save train data
    training_group = hf.create_group("training")
    training_group.create_dataset("X", data=X_train)
    training_group.create_dataset("y_trans", data=y_trans_train)
    training_group.create_dataset("y_main", data=y_train)

    # Save validation data
    validation_group = hf.create_group("validation")
    validation_group.create_dataset("X", data=X_val)
    validation_group.create_dataset("y_trans", data=y_trans_val)
    validation_group.create_dataset("y_main", data=y_val)

    # Save external test data
    external_test_group = hf.create_group("external_test")
    external_test_group.create_dataset("X", data=X_ext)
    external_test_group.create_dataset("y_trans", data=y_trans_ext)
    external_test_group.create_dataset("y_main", data=y_ext)

    # Save temporal test data
    temporal_test_group = hf.create_group("temporal_test")
    temporal_test_group.create_dataset("X", data=X_temp)
    temporal_test_group.create_dataset("y_trans", data=y_trans_temp)
    temporal_test_group.create_dataset("y_main", data=y_temp)

    # Save prospective data
    prospective_test_group = hf.create_group("prospective")
    prospective_test_group.create_dataset("X", data=X_prosp)
    prospective_test_group.create_dataset("y_trans", data=y_trans_prosp)
    prospective_test_group.create_dataset("y_main", data=y_prosp)


#%%

with open("%s/feature_names.pkl" % MODEL_DIR, "wb") as f:
    pickle.dump(feature_names, f, protocol=2)

# %%
