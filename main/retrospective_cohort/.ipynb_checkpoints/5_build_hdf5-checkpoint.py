#%%

# Import libraries

import pandas as pd
import numpy as np
from variables import time_window, DATA_DIR, OUTPUT_DIR

#%%

# Load data

seq = pd.read_csv("%s/final_data/seq.csv" % OUTPUT_DIR, compression="gzip")
static = pd.read_csv("%s/final_data/static.csv" % OUTPUT_DIR)
outcomes = pd.read_csv("%s/final_data/outcomes.csv" % OUTPUT_DIR)

outcomes = outcomes[outcomes["interval"] != 0]

seq = seq[seq["shift_id"].isin(outcomes["shift_id"].unique())]

# %%

# Filter shift IDs

missing = outcomes[~outcomes["shift_id"].isin(seq["shift_id"].unique())]

outcomes = outcomes[~outcomes["icustay_id"].isin(missing["icustay_id"].unique())]
seq = seq[~seq["icustay_id"].isin(missing["icustay_id"].unique())]
static = static[static["icustay_id"].isin(outcomes["icustay_id"].unique())]

outcomes = outcomes.reset_index(drop=True)
seq = seq.reset_index(drop=True)
static = static.reset_index(drop=True)

print(f"ICU admissions sequential: {len(seq['icustay_id'].unique())}")
print(f"ICU admissions static: {len(static['icustay_id'].unique())}")
print(f"ICU admissions outcome: {len(outcomes['icustay_id'].unique())}")

# %%

# Prepare sequential input data

seq_len = seq.groupby("shift_id").size().reset_index().rename({0: "seq_len"}, axis=1)

outcomes = outcomes.merge(seq_len, on="shift_id")

max_seq_len = 512

seq_len = outcomes["seq_len"].tolist()

qkv = np.zeros((len(outcomes), max_seq_len, 4))
X = seq.iloc[:, :-1].values

start = 0
for i in range(len(outcomes)):
    if seq_len[i] <= max_seq_len:
        end = start + seq_len[i]
        qkv[i, : seq_len[i], 0] = X[start:end, 1]
        qkv[i, : seq_len[i], 1] = X[start:end, 3]
        qkv[i, : seq_len[i], 2] = X[start:end, 2]
        qkv[i, : seq_len[i], 3] = X[start:end, 0]
    else:
        end = start + seq_len[i]
        trunc = end - max_seq_len
        qkv[i, :, 0] = X[trunc:end, 1]
        qkv[i, :, 1] = X[trunc:end, 3]
        qkv[i, :, 2] = X[trunc:end, 2]
        qkv[i, :, 3] = X[trunc:end, 0]

    start += seq_len[i]


#%%

# Prepare static input data

static = static.sort_values(by=["icustay_id"]).reset_index(drop=True)

static = static.set_index("icustay_id")
static = static.loc[outcomes["icustay_id"].tolist(), :]
static = static.reset_index()

static = static.to_numpy()

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

ids_train = np.isin(qkv[:, 0, 3], ids_train)
ids_val = np.isin(qkv[:, 0, 3], ids_val)
ids_ext = np.isin(qkv[:, 0, 3], ids_ext)
ids_temp = np.isin(qkv[:, 0, 3], ids_temp)


X_train = qkv[ids_train]
static_train = static[ids_train]
y_trans_train = y_trans[ids_train]
y_train = y[ids_train]

X_val = qkv[ids_val]
static_val = static[ids_val]
y_trans_val = y_trans[ids_val]
y_val = y[ids_val]

X_ext = qkv[ids_ext]
static_ext = static[ids_ext]
y_trans_ext = y_trans[ids_ext]
y_ext = y[ids_ext]

X_temp = qkv[ids_temp]
static_temp = static[ids_temp]
y_trans_temp = y_trans[ids_temp]
y_temp = y[ids_temp]

print(f"ICU admissions train: {len(np.unique(X_train[:,0,3]))}")
print(f"ICU admissions val: {len(np.unique(X_val[:,0,3]))}")
print(f"ICU admissions ext: {len(np.unique(X_ext[:,0,3]))}")
print(f"ICU admissions temp: {len(np.unique(X_temp[:,0,3]))}")


# %%

# Save hdf5 file

import h5py

with h5py.File("%s/final_data/dataset.h5" % OUTPUT_DIR, "w", libver="latest") as hf:

    # Save train data
    training_group = hf.create_group("training")
    training_group.create_dataset("X", data=X_train)
    training_group.create_dataset("static", data=static_train)
    training_group.create_dataset("y_trans", data=y_trans_train)
    training_group.create_dataset("y_main", data=y_train)

    # Save validation data
    validation_group = hf.create_group("validation")
    validation_group.create_dataset("X", data=X_val)
    validation_group.create_dataset("static", data=static_val)
    validation_group.create_dataset("y_trans", data=y_trans_val)
    validation_group.create_dataset("y_main", data=y_val)

    # Save external test data
    external_test_group = hf.create_group("external_test")
    external_test_group.create_dataset("X", data=X_ext)
    external_test_group.create_dataset("static", data=static_ext)
    external_test_group.create_dataset("y_trans", data=y_trans_ext)
    external_test_group.create_dataset("y_main", data=y_ext)

    # Save temporal test data
    temporal_test_group = hf.create_group("temporal_test")
    temporal_test_group.create_dataset("X", data=X_temp)
    temporal_test_group.create_dataset("static", data=static_temp)
    temporal_test_group.create_dataset("y_trans", data=y_trans_temp)
    temporal_test_group.create_dataset("y_main", data=y_temp)

# %%
