#%%
# Import libraries

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.utils import shuffle
from variables import time_window, DATA_DIR, OUTPUT_DIR

#%%

# Load all admissions

admissions_eicu = pd.read_csv("%s/eicu/patient.csv" % DATA_DIR)
admissions_mimic = pd.read_csv("%s/mimic/icustays.csv" % DATA_DIR)
admissions_uf = pd.read_csv("%s/uf/icustays.csv" % DATA_DIR)

print("-" * 20 + "Initial ICU stays" + "-" * 20)

print(f"eICU: {len(admissions_eicu)}")
print(f"MIMIC: {len(admissions_mimic)}")
print(f"UF: {len(admissions_uf)}")

#%%

# Create cohort flow table

cohort_flow = pd.DataFrame(data=[], columns=["eICU", "MIMIC", "UF"])

cohort_flow = pd.concat(
    [
        cohort_flow,
        pd.DataFrame(
            data=[[len(admissions_eicu), len(admissions_mimic), len(admissions_uf)]],
            index=["Initial"],
            columns=cohort_flow.columns,
        ),
    ],
    axis=0,
)

#%%

# Filter admissions by vitals presence

print("-" * 20 + "Filtering by vitals presence" + "-" * 20)

use_vitals_eicu = [
    "Heart Rate",
    "Respiratory Rate",
    "O2 Saturation",
    "Non-Invasive BP Diastolic",
    "Non-Invasive BP Systolic",
    "Temperature (C)",
    "Temperature (F)",
]

map_var = pd.read_csv("%s/variable_mapping.csv" % DATA_DIR)

map_var = map_var[map_var["eicu"].isin(use_vitals_eicu)]

use_vitals_uf = map_var["uf"].tolist()
use_vitals_mimic = map_var["mimic"].tolist()

map_mimic = pd.read_csv("%s/mimic/d_items.csv" % DATA_DIR)
map_mimic = map_mimic[map_mimic["label"].isin(use_vitals_mimic)]

use_vitals_mimic = map_mimic["itemid"].tolist()


vitals = pd.read_csv(
    "%s/eicu/vitals.csv" % DATA_DIR,
    usecols=[
        "patientunitstayid",
        "nursingchartoffset",
        "nursingchartcelltypevalname",
        "nursingchartvalue",
    ],
)

vitals = vitals[vitals["nursingchartoffset"] >= 0]

vitals = vitals[vitals["nursingchartcelltypevalname"].isin(use_vitals_eicu)]

vitals["nursingchartcelltypevalname"] = vitals["nursingchartcelltypevalname"].replace(
    {"Temperature (F)": "Temperature (C)"}
)

filtered_vitals = vitals.groupby("patientunitstayid")[
    "nursingchartcelltypevalname"
].nunique()

filtered_vitals_eicu = list(filtered_vitals[filtered_vitals == 6].index.values)

dropped_eicu = len(admissions_eicu) - len(filtered_vitals_eicu)

print(f"eICU stays dropped: {len(admissions_eicu) - len(filtered_vitals_eicu)}")

admissions_eicu = admissions_eicu[
    admissions_eicu["patientunitstayid"].isin(filtered_vitals_eicu)
]

del vitals, filtered_vitals

vitals = pd.read_csv("%s/mimic/all_events.csv" % DATA_DIR)

temp = pd.read_csv("%s/mimic/temp_conv.csv" % DATA_DIR)

vitals = pd.concat([vitals, temp])

vitals = vitals[vitals["itemid"].isin(use_vitals_mimic)]

filtered_vitals = vitals.groupby("stay_id")["itemid"].nunique()

filtered_vitals_mimic = list(filtered_vitals[filtered_vitals == 6].index.values)

dropped_mimic = len(admissions_mimic) - len(filtered_vitals_mimic)

print(f"MIMIC stays dropped: {len(admissions_mimic) - len(filtered_vitals_mimic)}")

admissions_mimic = admissions_mimic[
    admissions_mimic["stay_id"].isin(filtered_vitals_mimic)
]

del vitals, filtered_vitals, temp

vitals = pd.read_csv("%s/uf/seq.csv" % DATA_DIR)

vitals = vitals[vitals["variable"].isin(use_vitals_uf)]

filtered_vitals = vitals.groupby("icustay_id")["variable"].nunique()

filtered_vitals_uf = list(filtered_vitals[filtered_vitals == 6].index.values)

dropped_uf = len(admissions_uf) - len(filtered_vitals_uf)

print(f"UF stays dropped: {len(admissions_uf) - len(filtered_vitals_uf)}")

admissions_uf = admissions_uf[admissions_uf["icustay_id"].isin(filtered_vitals_uf)]

del vitals, filtered_vitals

cohort_flow = pd.concat(
    [
        cohort_flow,
        pd.DataFrame(
            data=[[dropped_eicu, dropped_mimic, dropped_uf]],
            index=["Vitals Dropped"],
            columns=cohort_flow.columns,
        ),
    ],
    axis=0,
)

#%%

# Filter admissions by missigness of basic information

print("-" * 20 + "Filtering by basic information info" + "-" * 20)

age_eicu = pd.read_csv(
    "%s/eicu/patient.csv" % DATA_DIR,
    usecols=["patientunitstayid", "age", "gender", "ethnicity", "unitdischargestatus"],
)
age_eicu = age_eicu[
    age_eicu["patientunitstayid"].isin(admissions_eicu["patientunitstayid"].unique())
]
age_eicu = age_eicu.dropna(subset=["age", "gender", "ethnicity", "unitdischargestatus"])
age_eicu = age_eicu["patientunitstayid"].tolist()

dropped_eicu = len(admissions_eicu) - len(age_eicu)

print(f"eICU stays dropped: {len(admissions_eicu) - len(age_eicu)}")

admissions_eicu = admissions_eicu[admissions_eicu["patientunitstayid"].isin(age_eicu)]

age_mimic = pd.read_csv(
    "%s/mimic/icustays.csv" % DATA_DIR, usecols=["stay_id", "subject_id", "hadm_id"]
)
age_mimic = age_mimic[age_mimic["stay_id"].isin(admissions_mimic["stay_id"].unique())]
age_mimic = age_mimic.merge(
    pd.read_csv(
        "%s/mimic/patients.csv" % DATA_DIR,
        usecols=["subject_id", "anchor_age", "gender"],
    ),
    how="left",
    on="subject_id",
)
age_mimic = age_mimic.merge(
    pd.read_csv(
        "%s/mimic/admissions.csv" % DATA_DIR,
        usecols=["subject_id", "hadm_id", "discharge_location", "race"],
    ),
    how="left",
    on=["subject_id", "hadm_id"],
)
age_mimic = age_mimic.dropna(
    subset=["anchor_age", "gender", "race", "discharge_location"]
)
age_mimic = age_mimic["stay_id"].tolist()

dropped_mimic = len(admissions_mimic) - len(age_mimic)

print(f"MIMIC stays dropped: {len(admissions_mimic) - len(age_mimic)}")

admissions_mimic = admissions_mimic[admissions_mimic["stay_id"].isin(age_mimic)]

age_uf = pd.read_csv(
    "%s/uf/static.csv" % DATA_DIR, usecols=["icustay_id", "age", "sex", "race"]
)
age_uf = age_uf[age_uf["icustay_id"].isin(admissions_uf["icustay_id"].unique())]
age_uf = age_uf.dropna(subset=["age", "sex", "race"])
age_uf = age_uf["icustay_id"].tolist()

dropped_uf = len(admissions_uf) - len(age_uf)

print(f"UF stays dropped: {len(admissions_uf) - len(age_uf)}")

admissions_uf = admissions_uf[admissions_uf["icustay_id"].isin(age_uf)]

cohort_flow = pd.concat(
    [
        cohort_flow,
        pd.DataFrame(
            data=[[dropped_eicu, dropped_mimic, dropped_uf]],
            index=["Basic Info Dropped"],
            columns=cohort_flow.columns,
        ),
    ],
    axis=0,
)

#%%

# Filter admissions by ICU length of stay

print("-" * 20 + "Filtering by ICU length of stay" + "-" * 20)

los_eicu = pd.read_csv(
    "%s/eicu/patient.csv" % DATA_DIR,
    usecols=["patientunitstayid", "unitdischargeoffset"],
)
los_eicu = los_eicu[
    los_eicu["patientunitstayid"].isin(admissions_eicu["patientunitstayid"].unique())
]

lower_lim = time_window * 2 * 60

los_eicu = los_eicu[los_eicu["unitdischargeoffset"] <= 43200]
los_eicu = los_eicu[los_eicu["unitdischargeoffset"] >= lower_lim]
los_eicu = los_eicu["patientunitstayid"].tolist()

dropped_eicu = len(admissions_eicu) - len(los_eicu)

print(f"eICU stays dropped: {len(admissions_eicu) - len(los_eicu)}")

admissions_eicu = admissions_eicu[admissions_eicu["patientunitstayid"].isin(los_eicu)]

los_mimic = pd.read_csv("%s/mimic/icustays.csv" % DATA_DIR, usecols=["stay_id", "los"])
los_mimic = los_mimic[los_mimic["stay_id"].isin(admissions_mimic["stay_id"].unique())]

lower_lim = (time_window * 2) / 24

los_mimic = los_mimic[los_mimic["los"] <= 30]
los_mimic = los_mimic[los_mimic["los"] >= lower_lim]
los_mimic = los_mimic["stay_id"].tolist()

dropped_mimic = len(admissions_mimic) - len(los_mimic)

print(f"MIMIC stays dropped: {len(admissions_mimic) - len(los_mimic)}")

admissions_mimic = admissions_mimic[admissions_mimic["stay_id"].isin(los_mimic)]

los_uf = pd.read_csv(
    "%s/uf/icustays.csv" % DATA_DIR,
    usecols=["icustay_id", "enter_datetime", "exit_datetime"],
)
los_uf = los_uf[los_uf["icustay_id"].isin(los_uf["icustay_id"].unique())]
los_uf["icu_los"] = (
    pd.to_datetime(los_uf["exit_datetime"]) - pd.to_datetime(los_uf["enter_datetime"])
) / np.timedelta64(1, "h")

lower_lim = time_window * 2

los_uf = los_uf[los_uf["icu_los"] <= 720]
los_uf = los_uf[los_uf["icu_los"] >= lower_lim]
los_uf = los_uf["icustay_id"].tolist()

dropped_uf = len(admissions_uf) - len(los_uf)

print(f"UF stays dropped: {len(admissions_uf) - len(los_uf)}")

admissions_uf = admissions_uf[admissions_uf["icustay_id"].isin(los_uf)]

cohort_flow = pd.concat(
    [
        cohort_flow,
        pd.DataFrame(
            data=[[dropped_eicu, dropped_mimic, dropped_uf]],
            index=["ICU LOS Dropped"],
            columns=cohort_flow.columns,
        ),
    ],
    axis=0,
)

#%%

# Filter admissions by outcome data

print("-" * 20 + "Filtering by outcome presence" + "-" * 20)

outcomes_eicu = pd.read_csv("%s/eicu/acuity_states.csv" % DATA_DIR)
outcomes_eicu = outcomes_eicu[
    outcomes_eicu["patientunitstayid"].isin(
        admissions_eicu["patientunitstayid"].unique()
    )
]
outcomes_eicu = list(outcomes_eicu["patientunitstayid"].unique())

dropped_eicu = len(admissions_eicu) - len(outcomes_eicu)

print(f"eICU stays dropped: {len(admissions_eicu) - len(outcomes_eicu)}")

admissions_eicu = admissions_eicu[
    admissions_eicu["patientunitstayid"].isin(outcomes_eicu)
]

outcomes_mimic = pd.read_csv("%s/mimic/acuity_states.csv" % DATA_DIR)
outcomes_mimic = outcomes_mimic[
    outcomes_mimic["stay_id"].isin(admissions_mimic["stay_id"].unique())
]
outcomes_mimic = list(outcomes_mimic["stay_id"].unique())

dropped_mimic = len(admissions_mimic) - len(outcomes_mimic)

print(f"MIMIC stays dropped: {len(admissions_mimic) - len(outcomes_mimic)}")

admissions_mimic = admissions_mimic[admissions_mimic["stay_id"].isin(outcomes_mimic)]

outcomes_uf = pd.read_csv("%s/uf/acuity_states.csv" % DATA_DIR)
outcomes_uf = outcomes_uf[
    outcomes_uf["icustay_id"].isin(admissions_uf["icustay_id"].unique())
]
outcomes_uf = list(outcomes_uf["icustay_id"].unique())

dropped_uf = len(admissions_uf) - len(outcomes_uf)

print(f"UF stays dropped: {len(admissions_uf) - len(outcomes_uf)}")

admissions_uf = admissions_uf[admissions_uf["icustay_id"].isin(outcomes_uf)]

cohort_flow = pd.concat(
    [
        cohort_flow,
        pd.DataFrame(
            data=[[dropped_eicu, dropped_mimic, dropped_uf]],
            index=["Outcomes Dropped"],
            columns=cohort_flow.columns,
        ),
    ],
    axis=0,
)

#%%

# Save final cohorts

print("-" * 20 + "Filtered ICU stays" + "-" * 20)

print(f"eICU: {len(admissions_eicu)}")
print(f"MIMIC: {len(admissions_mimic)}")
print(f"UF: {len(admissions_uf)}")

if not os.path.exists(f"{OUTPUT_DIR}/final_data"):
    os.makedirs(f"{OUTPUT_DIR}/final_data")

admissions_eicu.to_csv("%s/final_data/admissions_eicu.csv" % OUTPUT_DIR, index=None)
admissions_mimic.to_csv("%s/final_data/admissions_mimic.csv" % OUTPUT_DIR, index=None)
admissions_uf.to_csv("%s/final_data/admissions_uf.csv" % OUTPUT_DIR, index=None)

#%%

# Split cohorts by IDs

print("-" * 20 + "Split ICU stays" + "-" * 20)

## eICU data split by hospital IDs

admissions_eicu = pd.read_csv("%s/final_data/admissions_eicu.csv" % OUTPUT_DIR)

hospital_counts = admissions_eicu.groupby("hospitalid")["patientunitstayid"].nunique()

print("eICU Development")

train_hospitals = hospital_counts[hospital_counts >= 800].index.values

train_set = admissions_eicu[admissions_eicu["hospitalid"].isin(train_hospitals)]

print(f"Hospitals: {len(train_hospitals)}")

train_set = train_set["patientunitstayid"].tolist()

print(f"ICU stays: {len(train_set)}")

cohort_flow = pd.concat(
    [
        cohort_flow,
        pd.DataFrame(
            data=[[len(train_hospitals), 0, 1]],
            index=["Development Hospitals"],
            columns=cohort_flow.columns,
        ),
    ],
    axis=0,
)

print("eICU External")

ext_test_set = admissions_eicu[~admissions_eicu["hospitalid"].isin(train_hospitals)]

test_hospitals = hospital_counts[hospital_counts < 800].index.values

print(f"Hospitals: {len(test_hospitals)}")

ext_test_set = ext_test_set["patientunitstayid"].tolist()

print(f"ICU stays: {len(ext_test_set)}")

cohort_flow = pd.concat(
    [
        cohort_flow,
        pd.DataFrame(
            data=[[len(test_hospitals), 1, 0]],
            index=["External Hospitals"],
            columns=cohort_flow.columns,
        ),
    ],
    axis=0,
)

## MIMIC data for external validation

print("MIMIC External")

admissions_mimic = pd.read_csv("%s/final_data/admissions_mimic.csv" % OUTPUT_DIR)

admissions_mimic = admissions_mimic["stay_id"].tolist()

cohort_flow = pd.concat(
    [
        cohort_flow,
        pd.DataFrame(
            data=[[len(ext_test_set), len(admissions_mimic), 0]],
            index=["External ICU stays"],
            columns=cohort_flow.columns,
        ),
    ],
    axis=0,
)

ext_test_set = ext_test_set + admissions_mimic

print(f"ICU stays: {len(admissions_mimic)}")

print(f"Total External Validation ICU stays: {len(ext_test_set)}")

## UF data split by time period

admissions_uf = pd.read_csv("%s/final_data/admissions_uf.csv" % OUTPUT_DIR)

admissions_uf["enter_datetime"] = pd.to_datetime(admissions_uf["enter_datetime"])
admissions_uf["exit_datetime"] = pd.to_datetime(admissions_uf["exit_datetime"])

period1_start = "2012-01-01"
period1_end = "2018-01-01"
period2_start = "2018-01-01"
period2_end = "2019-12-31"

period1_admissions = admissions_uf[
    (admissions_uf["enter_datetime"] >= period1_start)
    & (admissions_uf["enter_datetime"] < period1_end)
]
period2_admissions = admissions_uf[
    (admissions_uf["enter_datetime"] >= period2_start)
    & (admissions_uf["enter_datetime"] <= period2_end)
]

period1_admissions = period1_admissions["icustay_id"].tolist()
period2_admissions = period2_admissions["icustay_id"].tolist()

print("UF Develop")
print(f"ICU stays: {len(period1_admissions)}")

cohort_flow = pd.concat(
    [
        cohort_flow,
        pd.DataFrame(
            data=[[len(train_set), 0, len(period1_admissions)]],
            index=["Development ICU stays"],
            columns=cohort_flow.columns,
        ),
    ],
    axis=0,
)

print("UF Temporal")
print(f"ICU stays: {len(period2_admissions)}")

cohort_flow = pd.concat(
    [
        cohort_flow,
        pd.DataFrame(
            data=[[0, 0, len(period2_admissions)]],
            index=["Temporal ICU stays"],
            columns=cohort_flow.columns,
        ),
    ],
    axis=0,
)

train_set = train_set + period1_admissions
temp_test_set = period2_admissions

train_set = shuffle(train_set)

print("-" * 20 + "Total ICU stays per set" + "-" * 20)

print(f"Development ICU stays: {len(train_set)}")
print(f"External Validation ICU stays: {len(ext_test_set)}")
print(f"Temporal Validation ICU stays: {len(temp_test_set)}")

cohort_flow["Total"] = cohort_flow.sum(axis=1)

cohort_flow.to_csv("%s/final_data/cohort_flow.csv" % OUTPUT_DIR)

#%%

# Split development and save IDs

from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
fold_train = []
fold_val = []
for i, (train_index, test_index) in enumerate(kf.split(train_set)):
    fold_train.append(train_index.tolist())
    fold_val.append(test_index.tolist())

# Change index for different fold number
ids_train = []
ids_val = []

for i in range(len(fold_train)):
    fold_train_n = fold_train[i]
    fold_val_n = fold_val[i]
    ids_train_fold = []
    ids_val_fold = []
    for j in range(len(fold_train_n)):
        ids_train_fold.append(train_set[fold_train_n[j]])
    for j in range(len(fold_val_n)):
        ids_val_fold.append(train_set[fold_val_n[j]])
    ids_train.append(ids_train_fold)
    ids_val.append(ids_val_fold)

with open("%s/final_data/ids.pkl" % OUTPUT_DIR, "wb") as f:
    pickle.dump(
        {
            "train": ids_train,
            "val": ids_val,
            "ext_test": ext_test_set,
            "temp_test": temp_test_set,
        },
        f,
        protocol=2,
    )

# %%
