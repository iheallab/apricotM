#%%

# Import libraries

import pandas as pd
import numpy as np
import os

from variables import DATA_DIR, OUTPUT_DIR, SOFA_DIR, PROSP_DIR, time_window

if not os.path.exists(SOFA_DIR):
    os.makedirs(SOFA_DIR)

variables = pd.read_csv(f"{DATA_DIR}/sofa_variables.csv")

#%%

# Build eICU SOFA

icustay_ids = pd.read_csv(
    "%s/final_data/admissions_eicu.csv" % OUTPUT_DIR, usecols=["patientunitstayid"]
)
icustay_ids = icustay_ids["patientunitstayid"].tolist()

vitals = pd.read_csv(
    "%s/eicu/vitals.csv" % DATA_DIR,
    usecols=[
        "patientunitstayid",
        "nursingchartoffset",
        "nursingchartcelltypevalname",
        "nursingchartvalue",
    ],
)

use_vitals = [
    "O2 Saturation",
    "Non-Invasive BP Diastolic",
    "Non-Invasive BP Systolic",
]

vitals = vitals[vitals["nursingchartcelltypevalname"].isin(use_vitals)]

vitals["nursingchartvalue"] = vitals["nursingchartvalue"].astype(float)

vitals = vitals.rename(
    {
        "patientunitstayid": "icustay_id",
        "nursingchartoffset": "hours",
        "nursingchartcelltypevalname": "label",
        "nursingchartvalue": "value",
    },
    axis=1,
)

vitals = vitals[["icustay_id", "hours", "label", "value"]]

vitals["hours"] = vitals["hours"] / 60

vitals["value"] = vitals["value"].astype(float)

labs = pd.read_csv(
    "%s/eicu/lab.csv" % DATA_DIR,
    usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"],
)

use_labs = [
    "paO2",
    "creatinine",
    "total bilirubin",
    "platelets x 1000",
    "FiO2",
]

labs = labs[labs["labname"].isin(use_labs)]

labs = labs.rename(
    {
        "patientunitstayid": "icustay_id",
        "labresultoffset": "hours",
        "labname": "label",
        "labresult": "value",
    },
    axis=1,
)

labs = labs.dropna(subset=["value"])

labs["hours"] = labs["hours"] / 60

labs["value"] = labs["value"].astype(float)

seq_eicu = pd.concat([vitals, labs], axis=0)

del labs
del vitals

resp = pd.read_csv(
    "%s/eicu/respiratoryCharting.csv" % DATA_DIR,
    usecols=[
        "patientunitstayid",
        "respchartoffset",
        "respchartvaluelabel",
        "respchartvalue",
    ],
)

use_resp = [
    "FiO2",
]

resp = resp[resp["respchartvaluelabel"].isin(use_resp)]

resp = resp.rename(
    {
        "patientunitstayid": "icustay_id",
        "respchartoffset": "hours",
        "respchartvaluelabel": "label",
        "respchartvalue": "value",
    },
    axis=1,
)

resp = resp.dropna(subset=["value"])

resp["hours"] = resp["hours"] / 60

resp["value"] = resp["value"].str.replace("%", "")

resp["value"] = resp["value"].astype(float)

seq_eicu = pd.concat([seq_eicu, resp], axis=0)

del resp

meds = pd.read_csv(
    "%s/eicu/infusionDrug.csv" % DATA_DIR,
    usecols=["patientunitstayid", "infusionoffset", "drugname", "drugamount"],
)

meds = meds.rename(
    {
        "patientunitstayid": "icustay_id",
        "infusionoffset": "hours",
        "drugname": "label",
        "drugamount": "value",
    },
    axis=1,
)

meds = meds.dropna(subset=["value"])

meds.loc[
    meds["label"].str.contains(
        r"\b{}\b".format("norepinephrine"), case=False, regex=True
    ),
    "label",
] = "Norepinephrine"
meds.loc[
    meds["label"].str.contains(r"\b{}\b".format("dopamine"), case=False, regex=True),
    "label",
] = "Dopamine"
meds.loc[
    meds["label"].str.contains(r"\b{}\b".format("epinephrine"), case=False, regex=True),
    "label",
] = "Epinephrine"
meds.loc[
    meds["label"].str.contains(r"\b{}\b".format("dobutamine"), case=False, regex=True),
    "label",
] = "Dobutamine"


use_meds = ["Norepinephrine", "Dopamine", "Epinephrine", "Dobutamine"]

meds = meds[meds["label"].isin(use_meds)]

meds["hours"] = meds["hours"] / 60

meds["value"] = meds["value"].astype(float)

seq_eicu = pd.concat([seq_eicu, meds], axis=0)

del meds

scores = pd.read_csv(
    "%s/eicu/scores.csv" % DATA_DIR,
    usecols=[
        "patientunitstayid",
        "nursingchartoffset",
        "nursingchartcelltypevalname",
        "nursingchartvalue",
    ],
)

use_scores = ["GCS Total"]

scores = scores[scores["nursingchartcelltypevalname"].isin(use_scores)]

scores = scores[scores["nursingchartvalue"] != "Unable to score due to medication"]
scores = scores.dropna(subset=["nursingchartvalue"])
scores["nursingchartvalue"] = scores["nursingchartvalue"].astype(int)
scores = scores[
    (scores["nursingchartvalue"] <= 15) & (scores["nursingchartvalue"] >= 0)
]
scores["nursingchartcelltypevalname"] = "GCS"

scores = scores.rename(
    {
        "patientunitstayid": "icustay_id",
        "nursingchartoffset": "hours",
        "nursingchartcelltypevalname": "label",
        "nursingchartvalue": "value",
    },
    axis=1,
)

scores = scores[["icustay_id", "hours", "label", "value"]]

scores["hours"] = scores["hours"] / 60

scores["value"] = scores["value"].astype(float)

seq_eicu = pd.concat([seq_eicu, scores], axis=0)

del scores

urine = pd.read_csv("%s/eicu/urine_output.csv" % DATA_DIR)

urine = urine.rename(
    {
        "patientunitstayid": "icustay_id",
        "intakeoutputoffset": "hours",
        "celllabel": "label",
        "cellvaluenumeric": "value",
    },
    axis=1,
)

urine = urine[["icustay_id", "hours", "label", "value"]]

urine["hours"] = urine["hours"] / 60

urine["value"] = urine["value"].astype(float)

seq_eicu = pd.concat([seq_eicu, urine], axis=0)

del urine

vent = pd.read_csv("%s/eicu/acuity_states.csv" % DATA_DIR)

vent = vent[["patientunitstayid", "shift_start", "mv"]]

vent["shift_start"] = vent["shift_start"] / 60

vent = vent.rename(
    {
        "patientunitstayid": "icustay_id",
        "shift_start": "hours",
        "mv": "value",
    },
    axis=1,
)

vent["label"] = "MV"

vent["value"] = vent["value"].astype(float)

seq_eicu = pd.concat([seq_eicu, vent], axis=0)

del vent

seq_eicu = seq_eicu[seq_eicu["icustay_id"].isin(icustay_ids)]

#%%

# Build MIMIC SOFA

icustay_ids = pd.read_csv(
    "%s/final_data/admissions_mimic.csv" % OUTPUT_DIR, usecols=["stay_id"]
)
icustay_ids = icustay_ids["stay_id"].tolist()

admissions = pd.read_csv(
    "%s/mimic/icustays.csv" % DATA_DIR,
    usecols=["subject_id", "stay_id", "intime", "outtime"],
)
ditems = pd.read_csv("%s/mimic/d_items.csv" % DATA_DIR)

labs_vitals = pd.read_csv(
    "%s/mimic/all_events.csv" % DATA_DIR,
    usecols=["subject_id", "stay_id", "charttime", "itemid", "valuenum"],
)
labs_vitals = pd.concat(
    [
        labs_vitals,
        pd.read_csv(
            "%s/mimic/temp_conv.csv" % DATA_DIR,
            usecols=["subject_id", "stay_id", "charttime", "itemid", "valuenum"],
        ),
    ]
)

labs_vitals = labs_vitals.merge(admissions, on=["stay_id", "subject_id"])

labs_vitals["hours"] = (
    pd.to_datetime(labs_vitals["charttime"]) - pd.to_datetime(labs_vitals["intime"])
) / np.timedelta64(1, "h")


labitems = ditems[ditems["itemid"].isin(labs_vitals["itemid"].unique())]
labitems_ids = labitems["itemid"].unique()

map_labs = dict()

for i in range(len(labitems_ids)):
    map_labs[labitems_ids[i]] = labitems[labitems["itemid"] == labitems_ids[i]][
        "label"
    ].values[0]

labs_vitals["label"] = labs_vitals["itemid"].map(map_labs)


def remove_outliers(group):

    lower_threshold = group["valuenum"].quantile(0.01)
    upper_threshold = group["valuenum"].quantile(0.99)

    group_filtered = group[
        (group["valuenum"] >= lower_threshold) & (group["valuenum"] <= upper_threshold)
    ]

    return group_filtered


labs_vitals = labs_vitals.groupby("label").apply(remove_outliers)

labs_vitals = labs_vitals.reset_index(drop=True)

lab_variables = list(labs_vitals["label"].unique())

meds = pd.read_csv("%s/mimic/med_events.csv" % DATA_DIR)

meds = meds.merge(admissions, on=["stay_id", "subject_id"])


meds["hours"] = (
    pd.to_datetime(meds["starttime"]) - pd.to_datetime(meds["intime"])
) / np.timedelta64(1, "h")

meditems = ditems[ditems["itemid"].isin(meds["itemid"].unique())]
meditems_ids = meditems["itemid"].unique()

map_meds = dict()

for i in range(len(meditems_ids)):
    map_meds[meditems_ids[i]] = meditems[meditems["itemid"] == meditems_ids[i]][
        "label"
    ].values[0]

meds["label"] = meds["itemid"].map(map_meds)


def remove_outliers(group):

    lower_threshold = group["amount"].quantile(0.01)
    upper_threshold = group["amount"].quantile(0.99)

    group_filtered = group[
        (group["amount"] >= lower_threshold) & (group["amount"] <= upper_threshold)
    ]

    return group_filtered


meds = meds.groupby("label").apply(remove_outliers)

meds = meds.rename({"amount": "valuenum"}, axis=1)

meds["valuenum"] = 1

labs_vitals = labs_vitals.loc[:, ["stay_id", "hours", "label", "valuenum"]]
meds = meds.loc[:, ["stay_id", "hours", "label", "valuenum"]]

labs_vitals = labs_vitals.rename({"label": "variable"}, axis=1)
labs_vitals = labs_vitals.rename({"valuenum": "value"}, axis=1)

meds = meds.rename({"label": "variable"}, axis=1)
meds = meds.rename({"valuenum": "value"}, axis=1)

scores = pd.read_csv("%s/mimic/scores_raw.csv" % DATA_DIR)
scores = scores.merge(admissions, on=["stay_id"])

scores["hours"] = (
    pd.to_datetime(scores["charttime"]) - pd.to_datetime(scores["intime"])
) / np.timedelta64(1, "h")
scores = scores.loc[:, ["stay_id", "hours", "variable", "value"]]

seq_mimic = pd.concat([labs_vitals, meds, scores], axis=0).sort_values(
    by=["stay_id", "hours"]
)

del labs_vitals, meds, scores

seq_mimic = seq_mimic[seq_mimic["hours"] >= 0]

seq_mimic = seq_mimic[seq_mimic["variable"].isin(variables["mimic"].values)]

vent = pd.read_csv("%s/mimic/acuity_states.csv" % DATA_DIR)

vent = vent[["stay_id", "shift_start", "mv"]]

vent["interval"] = vent.groupby("stay_id").cumcount()

vent["shift_start"] = vent["interval"] * 4

vent.drop("interval", axis=1, inplace=True)

vent = vent.rename(
    {
        "shift_start": "hours",
        "mv": "value",
    },
    axis=1,
)

vent["variable"] = "MV"

vent["value"] = vent["value"].astype(float)

seq_mimic = pd.concat([seq_mimic, vent], axis=0)

del vent

urine = pd.read_csv("%s/mimic/urine_output.csv" % DATA_DIR)

urine = urine.merge(admissions, on=["stay_id"])

urine["hours"] = (
    pd.to_datetime(urine["charttime"]) - pd.to_datetime(urine["intime"])
) / np.timedelta64(1, "h")


urine["variable"] = "Urine"

urine = urine[["stay_id", "hours", "variable", "value"]]

urine["value"] = urine["value"].astype(float)

seq_mimic = pd.concat([seq_mimic, urine], axis=0)

del urine

variable_map = pd.Series(
    data=variables["eicu"].values, index=variables["mimic"].values
).to_dict()

variable_map.update({"MV": "MV", "Urine": "Urine"})

seq_mimic["variable"] = seq_mimic["variable"].map(variable_map)

seq_mimic = seq_mimic.rename({"variable": "label"}, axis=1)

seq_mimic = seq_mimic.rename(
    {
        "stay_id": "icustay_id",
    },
    axis=1,
)

seq_mimic = seq_mimic[seq_mimic["icustay_id"].isin(icustay_ids)]

#%%

# Build UF SOFA

icustay_ids = pd.read_csv(
    "%s/final_data/admissions_uf.csv" % OUTPUT_DIR, usecols=["icustay_id"]
)
icustay_ids = icustay_ids["icustay_id"].tolist()

vent = pd.read_csv("%s/uf/acuity_states.csv" % DATA_DIR)

vent = vent[vent["icustay_id"].isin(icustay_ids)]

vent = vent[["icustay_id", "start_hours", "mv"]]

vent = vent.rename(
    {
        "start_hours": "hours",
        "mv": "value",
    },
    axis=1,
)

vent["variable"] = "MV"

seq_uf = pd.read_csv("%s/uf/seq.csv" % DATA_DIR)

variables_uf = variables["uf"].tolist() + ["med_dobutamine_in_d5w"]

seq_uf = seq_uf[seq_uf["variable"].isin(variables_uf)]

seq_uf.drop("variable_code", axis=1, inplace=True)

seq_uf = pd.concat([seq_uf, vent], axis=0)

del vent

variable_map = pd.Series(
    data=variables["eicu"].values, index=variables["uf"].values
).to_dict()

variable_map.update({"MV": "MV", "med_dobutamine_in_d5w": "Dobutamine"})

seq_uf["variable"] = seq_uf["variable"].map(variable_map)

seq_uf = seq_uf.rename({"variable": "label"}, axis=1)

seq_uf = seq_uf[seq_uf["icustay_id"].isin(icustay_ids)]

#%%

# Build prospective SOFA

icustay_ids = pd.read_csv("%s/final/admissions.csv" % PROSP_DIR, usecols=["icustay_id"])
icustay_ids = icustay_ids["icustay_id"].tolist()

seq_prosp = pd.read_csv(f"{PROSP_DIR}/final/seq.csv")

vent = pd.read_csv(f"{PROSP_DIR}/intermediate/acuity_states_prosp.csv")

vent = vent[vent["icustay_id"].isin(icustay_ids)]

vent = vent[["icustay_id", "shift_start", "mv"]]

vent["interval"] = vent.groupby("icustay_id").cumcount()

vent["shift_start"] = vent["interval"] * 4

vent.drop("interval", axis=1, inplace=True)

vent = vent.rename(
    {
        "shift_start": "hours",
        "mv": "value",
    },
    axis=1,
)

vent["variable"] = "MV"

variables_uf = variables["mimic"].tolist()

seq_prosp = seq_prosp[seq_prosp["variable"].isin(variables_uf)]

seq_prosp.drop(["variable_code", "shift_id"], axis=1, inplace=True)

seq_prosp = pd.concat([seq_prosp, vent], axis=0)

del vent

variable_map = pd.Series(
    data=variables["eicu"].values, index=variables["mimic"].values
).to_dict()

variable_map.update({"MV": "MV"})

seq_prosp["variable"] = seq_prosp["variable"].map(variable_map)

seq_prosp = seq_prosp.rename({"variable": "label"}, axis=1)

seq_prosp = seq_prosp[seq_prosp["icustay_id"].isin(icustay_ids)]

#%%

# Merge all sequential data

seq = pd.concat([seq_eicu, seq_mimic, seq_uf, seq_prosp], axis=0)
del seq_eicu, seq_mimic, seq_uf, seq_prosp

print(len(seq["icustay_id"].unique()))
print(seq["label"].value_counts())

#%%

# Get worst values for each 4h interval

seq = seq.sort_values(by=["icustay_id", "hours"])

seq = seq[seq["hours"] >= 0]

seq = seq.sort_values(by=["icustay_id", "hours"]).reset_index(drop=True)

seq["interval"] = ((seq["hours"] // 4)).astype(int)

agg = {
    "Non-Invasive BP Diastolic": "min",
    "Non-Invasive BP Systolic": "min",
    "Dopamine": "max",
    "Epinephrine": "max",
    "Norepinephrine": "max",
    "Dobutamine": "max",
    "paO2": "min",
    "O2 Saturation": "min",
    "FiO2": "max",
    "MV": "max",
    "platelets x 1000": "min",
    "total bilirubin": "max",
    "GCS": "min",
    "creatinine": "max",
    "Urine": "sum",
}

min_vars = [key for key in agg.keys() if agg[key] == "min"]
max_vars = [key for key in agg.keys() if agg[key] == "max"]
sum_vars = [key for key in agg.keys() if agg[key] == "sum"]

seq_min = seq[seq["label"].isin(min_vars)]
seq_max = seq[seq["label"].isin(max_vars)]
seq_sum = seq[seq["label"].isin(sum_vars)]

seq_min = seq_min.pivot_table(
    index=["icustay_id", "interval"], columns="label", values="value", aggfunc="min"
).reset_index()

seq_max = seq_max.pivot_table(
    index=["icustay_id", "interval"], columns="label", values="value", aggfunc="max"
).reset_index()

seq_sum = seq_sum.pivot_table(
    index=["icustay_id", "interval"], columns="label", values="value", aggfunc="sum"
).reset_index()

seq = seq_min.merge(seq_max, how="outer", on=["icustay_id", "interval"])
seq = seq.merge(seq_sum, how="outer", on=["icustay_id", "interval"])

seq.drop_duplicates(subset=["icustay_id", "interval"], inplace=True)

del seq_max, seq_min, seq_sum

seq = seq.sort_values(by=["icustay_id", "interval"]).reset_index(drop=True)

#%%

# Impute missing values with normal values

impute = {
    "Non-Invasive BP Diastolic": 80,
    "Non-Invasive BP Systolic": 120,
    "paO2": 100,
    "O2 Saturation": 100,
    "FiO2": 21,
}

for key in impute.keys():

    seq[key] = seq[key].fillna(value=impute[key])

seq["MAP"] = seq["Non-Invasive BP Diastolic"] + (1 / 3) * (
    seq["Non-Invasive BP Systolic"] - seq["Non-Invasive BP Diastolic"]
)

seq.drop(
    labels=["Non-Invasive BP Diastolic", "Non-Invasive BP Systolic"],
    axis=1,
    inplace=True,
)

#%%

# Get worst measurements in previous 24-hour intervals at each hour

agg = {
    "MAP": "min",
    "Dopamine": "max",
    "Epinephrine": "max",
    "Norepinephrine": "max",
    "Dobutamine": "max",
    "paO2": "min",
    "O2 Saturation": "min",
    "FiO2": "max",
    "MV": "max",
    "platelets x 1000": "min",
    "total bilirubin": "max",
    "GCS": "min",
    "creatinine": "max",
    "Urine": "sum",
}

df = seq.sort_values(by=["icustay_id", "interval"])

df.drop("interval", axis=1, inplace=True)

df = df.groupby("icustay_id").rolling(window=6, min_periods=1).agg(agg)

df = df.reset_index()

df.drop("level_1", axis=1, inplace=True)

df.insert(column="interval", loc=1, value=seq["interval"].values)

#%%

# Calculate SOFA

# Calculate ratios
df["pf"] = df["paO2"] / (df["FiO2"] / 100)
df["sf"] = ((df["O2 Saturation"] / (df["FiO2"] / 100)) - 57) / 0.61

# Calculate SOFA scores
# If variable missing for 24-hour period, a score of 0 is assigned
df["cardio"] = np.nan
df.loc[
    (pd.isnull(df["cardio"]))
    & (
        (df["Dopamine"] > 15) | (df["Epinephrine"] > 0.1) | (df["Norepinephrine"] > 0.1)
    ),
    "cardio",
] = 4
df.loc[
    (pd.isnull(df["cardio"])) & ((df["Dopamine"] > 5) | (df["Epinephrine"] <= 0.1))
    | ((df["Norepinephrine"] <= 0.1)),
    "cardio",
] = 3
df.loc[
    (pd.isnull(df["cardio"])) & ((df["Dopamine"] <= 5) | (df["Dobutamine"] > 0)),
    "cardio",
] = 2
df.loc[(pd.isnull(df["cardio"])) & ((df["MAP"] < 70)), "cardio"] = 1
df["cardio"] = df["cardio"].fillna(0)

# For respiratory, using (worst PaO2) / (worst FiO2)
# If PaO2 is missing, use SpO2 conversion formula
df["resp"] = np.nan
df.loc[
    (pd.isnull(df["resp"])) & ((df["MV"] > 0) & (df["pf"] < 100)),
    "resp",
] = 4
df.loc[
    (pd.isnull(df["resp"])) & ((df["MV"] > 0) & (df["pf"] < 200)),
    "resp",
] = 3
df.loc[(pd.isnull(df["resp"])) & (df["pf"] < 300), "resp"] = 2
df.loc[(pd.isnull(df["resp"])) & (df["pf"] < 400), "resp"] = 1
df.loc[(pd.isnull(df["resp"])) & (df["pf"] >= 400), "resp"] = 0

df.loc[
    (pd.isnull(df["resp"])) & ((df["MV"] > 0) & (df["sf"] < 100)),
    "resp",
] = 4
df.loc[
    (pd.isnull(df["resp"])) & ((df["MV"] > 0) & (df["sf"] < 200)),
    "resp",
] = 3
df.loc[(pd.isnull(df["resp"])) & (df["sf"] < 300), "resp"] = 2
df.loc[(pd.isnull(df["resp"])) & (df["sf"] < 400), "resp"] = 1
df["resp"] = df["resp"].fillna(0)

df["coag"] = np.nan
df.loc[(pd.isnull(df["coag"])) & (df["platelets x 1000"] < 20), "coag"] = 4
df.loc[(pd.isnull(df["coag"])) & (df["platelets x 1000"] < 50), "coag"] = 3
df.loc[(pd.isnull(df["coag"])) & (df["platelets x 1000"] < 100), "coag"] = 2
df.loc[(pd.isnull(df["coag"])) & (df["platelets x 1000"] < 150), "coag"] = 1
df["coag"] = df["coag"].fillna(0)

df["liver"] = np.nan
df.loc[(pd.isnull(df["liver"])) & (df["total bilirubin"] > 12), "liver"] = 4
df.loc[(pd.isnull(df["liver"])) & (df["total bilirubin"] >= 6), "liver"] = 3
df.loc[(pd.isnull(df["liver"])) & (df["total bilirubin"] >= 2), "liver"] = 2
df.loc[(pd.isnull(df["liver"])) & (df["total bilirubin"] >= 1.2), "liver"] = 1
df["liver"] = df["liver"].fillna(0)

df["cns"] = np.nan
df.loc[(pd.isnull(df["cns"])) & (df["GCS"] < 6), "cns"] = 4
df.loc[(pd.isnull(df["cns"])) & (df["GCS"] <= 9), "cns"] = 3
df.loc[(pd.isnull(df["cns"])) & (df["GCS"] <= 12), "cns"] = 2
df.loc[(pd.isnull(df["cns"])) & (df["GCS"] <= 14), "cns"] = 1
df["cns"] = df["cns"].fillna(0)

df["renal"] = np.nan
df.loc[
    (pd.isnull(df["renal"])) & ((df["creatinine"] > 5) | (df["Urine"] < 200)),
    "renal",
] = 4
df.loc[
    (pd.isnull(df["renal"])) & ((df["creatinine"] >= 3.5) | (df["Urine"] < 500)),
    "renal",
] = 3
df.loc[(pd.isnull(df["renal"])) & (df["creatinine"] >= 2), "renal"] = 2
df.loc[(pd.isnull(df["renal"])) & (df["creatinine"] >= 1.2), "renal"] = 1
df["renal"] = df["renal"].fillna(0)

sofa = df[
    ["icustay_id", "interval", "cardio", "resp", "coag", "liver", "cns", "renal"]
].reset_index()
sofa["total"] = (
    sofa["cardio"]
    + sofa["resp"]
    + sofa["coag"]
    + sofa["liver"]
    + sofa["cns"]
    + sofa["renal"]
)

print(sofa["total"].describe())

#%%

# Define acuity


def acuity(group):

    baseline = group["total"].values[0]

    group.loc[(group["total"] >= (baseline + 2)), "final_state"] = "unstable"

    return group


sofa["final_state"] = "stable"

sofa = sofa.groupby("icustay_id").apply(acuity).reset_index(drop=True)

print(sofa["final_state"].value_counts())

if "index" in sofa.columns:

    sofa.drop("index", axis=1, inplace=True)

sofa["transition"] = np.nan


def transitions(group):
    group["transition"] = group["final_state"].shift(1) + "-" + group["final_state"]
    return group


sofa = sofa.groupby("icustay_id").apply(transitions).reset_index(drop=True)

sofa["shift_id"] = sofa["icustay_id"].astype(str) + "_" + sofa["interval"].astype(str)

sofa.to_csv(f"{SOFA_DIR}/sofa_acuity.csv", index=False)

# %%
