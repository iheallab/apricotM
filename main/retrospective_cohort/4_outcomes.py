#%%

# Import libraries

import pandas as pd
import numpy as np
from variables import time_window, DATA_DIR, OUTPUT_DIR

#%%

# Build eICU outcomes

outcomes_eicu = pd.read_csv("%s/eicu/acuity_states.csv" % DATA_DIR)
admissions_eicu = pd.read_csv("%s/final_data/admissions_eicu.csv" % OUTPUT_DIR)

outcomes_eicu = outcomes_eicu[
    outcomes_eicu["patientunitstayid"].isin(admissions_eicu["patientunitstayid"])
]

outcomes_eicu = outcomes_eicu.rename({"patientunitstayid": "icustay_id"}, axis=1)
outcomes_eicu = outcomes_eicu.sort_values(by=["icustay_id", "shift_start"])

outcomes_eicu["interval"] = outcomes_eicu.groupby("icustay_id").cumcount()

outcomes_eicu["shift_id"] = (
    outcomes_eicu["icustay_id"].astype(str)
    + "_"
    + outcomes_eicu["interval"].astype(str)
)

print(len(outcomes_eicu["shift_id"].unique()))

print(admissions_eicu["unitdischargelocation"].value_counts())

dead_ids = admissions_eicu[admissions_eicu["unitdischargelocation"] == "Death"]

dead_ids = dead_ids["patientunitstayid"].unique()

dischg_ids = admissions_eicu[admissions_eicu["unitdischargelocation"] == "Home"]

dischg_ids = dischg_ids["patientunitstayid"].unique()

last_state_indices = outcomes_eicu.groupby("icustay_id")["interval"].transform("idxmax")

outcomes_eicu.loc[
    (outcomes_eicu["icustay_id"].isin(dischg_ids))
    & (outcomes_eicu.index == last_state_indices),
    "final_state",
] = "discharge"
outcomes_eicu.loc[
    (outcomes_eicu["icustay_id"].isin(dead_ids))
    & (outcomes_eicu.index == last_state_indices),
    "final_state",
] = "dead"

print(outcomes_eicu["final_state"].value_counts())

outcomes_eicu.drop("shift_start", axis=1, inplace=True)

#%%

# Build MIMIC outcomes

outcomes_mimic = pd.read_csv("%s/mimic/acuity_states.csv" % DATA_DIR)
admissions_mimic = pd.read_csv("%s/final_data/admissions_mimic.csv" % OUTPUT_DIR)

outcomes_mimic = outcomes_mimic[
    outcomes_mimic["stay_id"].isin(admissions_mimic["stay_id"])
]

outcomes_mimic = outcomes_mimic.rename({"stay_id": "icustay_id"}, axis=1)
outcomes_mimic = outcomes_mimic.sort_values(by=["icustay_id", "shift_start"])

outcomes_mimic["interval"] = outcomes_mimic.groupby("icustay_id").cumcount()

outcomes_mimic["shift_id"] = (
    outcomes_mimic["icustay_id"].astype(str)
    + "_"
    + outcomes_mimic["interval"].astype(str)
)

print(len(outcomes_mimic["shift_id"].unique()))

dead_ids = pd.read_csv("%s/mimic/admissions.csv" % DATA_DIR)

dead_ids.dropna(subset=["deathtime"], inplace=True)

print(len(dead_ids["subject_id"].unique()))

dead_ids = admissions_mimic.merge(dead_ids, how="inner", on="subject_id")

dead_ids["ind1"] = (
    pd.to_datetime(dead_ids["deathtime"]) - pd.to_datetime(dead_ids["intime"])
) / np.timedelta64(1, "h")

dead_ids["ind2"] = (
    pd.to_datetime(dead_ids["outtime"]) - pd.to_datetime(dead_ids["deathtime"])
) / np.timedelta64(1, "h")

dead_ids = dead_ids[(dead_ids["ind1"] > 0) & (dead_ids["ind2"] > 0)]

dead_ids = dead_ids.drop_duplicates(subset=["stay_id"])

dead_ids = dead_ids[dead_ids["discharge_location"] == "DIED"]

dead_ids = dead_ids["stay_id"].unique()

dischg_ids = pd.read_csv("%s/mimic/admissions.csv" % DATA_DIR)

dischg_ids = dischg_ids[dischg_ids["discharge_location"] == "HOME"]

print(len(dischg_ids["hadm_id"].unique()))

dischg_ids = admissions_mimic.merge(
    dischg_ids, how="inner", on=["subject_id", "hadm_id"]
)

dischg_ids["discharge"] = (
    pd.to_datetime(dischg_ids["dischtime"]) - pd.to_datetime(dischg_ids["outtime"])
) / np.timedelta64(1, "h")

sample = dischg_ids.groupby("hadm_id")["discharge"].min()

dischg_ids = dischg_ids[dischg_ids["discharge"].isin(sample)]

dischg_ids.drop_duplicates(subset=["hadm_id"], inplace=True)

dischg_ids = dischg_ids[
    (dischg_ids["discharge"] >= -24) & (dischg_ids["discharge"] <= 24)
]

print(len(dischg_ids["stay_id"].unique()))

dischg_ids = dischg_ids["stay_id"].unique()

last_state_indices = outcomes_mimic.groupby("icustay_id")["interval"].transform(
    "idxmax"
)

outcomes_mimic.loc[
    (outcomes_mimic["icustay_id"].isin(dischg_ids))
    & (outcomes_mimic.index == last_state_indices),
    "final_state",
] = "discharge"
outcomes_mimic.loc[
    (outcomes_mimic["icustay_id"].isin(dead_ids))
    & (outcomes_mimic.index == last_state_indices),
    "final_state",
] = "dead"

print(outcomes_mimic["final_state"].value_counts())

outcomes_mimic.drop("shift_start", axis=1, inplace=True)
outcomes_mimic.drop("bt", axis=1, inplace=True)

#%%
# Build UF outcomes

outcomes_uf = pd.read_csv("%s/uf/acuity_states.csv" % DATA_DIR)
admissions_uf = pd.read_csv("%s/final_data/admissions_uf.csv" % OUTPUT_DIR)

outcomes_uf = outcomes_uf[outcomes_uf["icustay_id"].isin(admissions_uf["icustay_id"])]

print(outcomes_uf["final_state"].value_counts())

outcomes_uf["shift_id"] = (
    outcomes_uf["icustay_id"].astype(str) + "_" + outcomes_uf["interval"].astype(str)
)


outcomes_uf = outcomes_uf[
    ["icustay_id", "mv", "pressor", "crrt", "final_state", "shift_id", "interval"]
]

outcomes_uf = outcomes_uf.rename({"pressor": "vp"}, axis=1)

#%%

# Merge all outcomes

outcomes = pd.concat([outcomes_eicu, outcomes_mimic, outcomes_uf])

outcomes = outcomes.sort_values(by=["icustay_id", "interval"])

del outcomes_mimic, outcomes_eicu, outcomes_uf

#%%

# Convert time window if greater than 4h

if time_window > 4:

    factor = int(time_window / 4)

    outcomes["final_state"] = outcomes["final_state"].map(
        {"stable": 0, "unstable": 1, "discharge": 2, "dead": 3}
    )

    outcomes = (
        outcomes.groupby(by=["icustay_id", outcomes["interval"] // factor])
        .agg(
            {
                "icustay_id": "first",
                "crrt": "max",
                "mv": "max",
                "vp": "max",
                "final_state": "max",  # Take the max value of final_state within the group
                "interval": "first",  # Take the first value of interval within the group
                "shift_id": "first",  # Take first shift_id value within the group
            }
        )
        .reset_index(drop=True)
    )

    outcomes["interval"] = outcomes.groupby("icustay_id").cumcount()

    outcomes["shift_id"] = (
        outcomes["icustay_id"].astype(str) + "_" + outcomes["interval"].astype(str)
    )

    outcomes["final_state"] = outcomes["final_state"].map(
        {0: "stable", 1: "unstable", 2: "discharge", 3: "dead"}
    )


#%%

# Label transitions

outcomes["transition"] = np.nan
outcomes["transition_mv"] = np.nan
outcomes["transition_vp"] = np.nan
outcomes["transition_crrt"] = np.nan


outcomes.loc[(outcomes["mv"] == 1), "mv"] = "mv"
outcomes.loc[(outcomes["mv"] == 0), "mv"] = "no mv"
outcomes.loc[(outcomes["vp"] == 1), "vp"] = "vp"
outcomes.loc[(outcomes["vp"] == 0), "vp"] = "no vp"
outcomes.loc[(outcomes["crrt"] == 1), "crrt"] = "crrt"
outcomes.loc[(outcomes["crrt"] == 0), "crrt"] = "no crrt"


def transitions(group):
    group["transition"] = group["final_state"].shift(1) + "-" + group["final_state"]
    group[f"transition_mv"] = group["mv"].shift(1) + "-" + group["mv"]
    group[f"transition_vp"] = group["vp"].shift(1) + "-" + group["vp"]
    group[f"transition_crrt"] = group["crrt"].shift(1) + "-" + group["crrt"]
    return group


outcomes = outcomes.groupby("icustay_id").apply(transitions).reset_index(drop=True)

outcomes.to_csv("%s/final_data/outcomes.csv" % OUTPUT_DIR, index=None)

# %%
