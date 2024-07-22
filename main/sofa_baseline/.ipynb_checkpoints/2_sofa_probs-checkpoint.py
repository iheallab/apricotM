#%%

# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import pickle
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from variables import SOFA_DIR, OUTPUT_DIR, PROSP_DIR, DATA_DIR, time_window

# Create directory to save results

if not os.path.exists(f"{SOFA_DIR}/results"):
    os.makedirs(f"{SOFA_DIR}/results")


#%%

# Load SOFA scores

sofa = pd.read_csv(
    f"{SOFA_DIR}/sofa_acuity.csv",
    usecols=["icustay_id", "interval", "total"],
)

sofa["score"] = MinMaxScaler().fit_transform(sofa["total"].values.reshape((-1, 1)))

sofa["interval"] = sofa["interval"] + 1
sofa["shift_id"] = sofa["icustay_id"].astype(str) + "_" + sofa["interval"].astype(str)

sofa["dead"] = sofa["score"]
sofa["stable-unstable"] = sofa["score"]
sofa["no mv-mv"] = sofa["score"]
sofa["no vp- vp"] = sofa["score"]
sofa["no crrt-crrt"] = sofa["score"]

#%%

# Load acuity labels for all cohorts

outcomes = pd.read_csv(f"{OUTPUT_DIR}/final_data/outcomes.csv")
outcomes_prosp = pd.read_csv(f"{PROSP_DIR}/final/outcomes.csv")

keep_cols = [
    "icustay_id",
    "interval",
    "shift_id",
    "final_state",
    "transition",
    "transition_mv",
    "transition_vp",
    "transition_crrt",
]

outcomes = outcomes[keep_cols]
outcomes_prosp = outcomes_prosp[keep_cols]

outcomes = pd.concat([outcomes, outcomes_prosp], axis=0)

del outcomes_prosp

outcomes = outcomes[outcomes["interval"] != 0]

#%%

# Prepare outcomes

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

map_states = {
    "dead": 3,
    "unstable": 2,
    "stable": 1,
    "discharge": 0,
}

outcomes["final_state"] = outcomes["final_state"].map(map_states)

print(outcomes["final_state"].value_counts())

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

y = np.concatenate(
    [y, y_trans[:, 2:4], y_mv[:, 2:], y_pressor[:, 2:], y_crrt[:, 2:]], axis=1
)

columns = [
    "discharge",
    "stable",
    "unstable",
    "dead",
    "unstable-stable",
    "stable-unstable",
    "mv-no mv",
    "no mv-mv",
    "vp-no vp",
    "no vp- vp",
    "crrt-no crrt",
    "no crrt-crrt",
]

outcomes[columns] = y

#%%

# Filter shift IDs

outcomes = outcomes[outcomes["shift_id"].isin(sofa["shift_id"].unique())]
sofa = sofa[sofa["shift_id"].isin(outcomes["shift_id"].unique())]

outcomes = outcomes.reset_index(drop=True)
sofa = sofa.reset_index(drop=True)

print(len(sofa["icustay_id"].unique()))
print(len(outcomes["icustay_id"].unique()))

print(len(sofa["shift_id"].unique()))
print(len(outcomes["shift_id"].unique()))

#%%

# Load ids

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

ids_prosp = pd.read_csv("%s/final/admissions.csv" % PROSP_DIR, usecols=["icustay_id"])
ids_prosp = ids_prosp["icustay_id"].tolist()

#%%

# Split SOFA scores

sofa_val = sofa[sofa["icustay_id"].isin(ids_val)]
sofa_ext = sofa[sofa["icustay_id"].isin(ids_ext)]
sofa_temp = sofa[sofa["icustay_id"].isin(ids_temp)]
sofa_prosp = sofa[sofa["icustay_id"].isin(ids_prosp)]

cohorts = [
    ["int", sofa_val],
    ["ext", sofa_ext],
    ["temp", sofa_temp],
    ["prosp", sofa_prosp],
]

for cohort in cohorts:

    group = cohort[1]

    group.to_csv(f"{SOFA_DIR}/results/{cohort[0]}_pred_labels.csv", index=False)

#%%

# Split outcomes

outcomes_val = outcomes[outcomes["icustay_id"].isin(ids_val)]
outcomes_ext = outcomes[outcomes["icustay_id"].isin(ids_ext)]
outcomes_temp = outcomes[outcomes["icustay_id"].isin(ids_temp)]
outcomes_prosp = outcomes[outcomes["icustay_id"].isin(ids_prosp)]

cohorts = [
    ["int", outcomes_val],
    ["ext", outcomes_ext],
    ["temp", outcomes_temp],
    ["prosp", outcomes_prosp],
]

for cohort in cohorts:

    group = cohort[1]

    group.to_csv(f"{SOFA_DIR}/results/{cohort[0]}_true_labels.csv", index=False)

# %%
