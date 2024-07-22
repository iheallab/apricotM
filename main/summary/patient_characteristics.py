#%%

# Import libraries

import pandas as pd
import numpy as np
import pickle
import os

from variables import OUTPUT_DIR, PROSP_DATA_DIR

if not os.path.exists(f"{OUTPUT_DIR}/summary/patient_characteristics"):
    os.makedirs(f"{OUTPUT_DIR}/summary/patient_characteristics")

#%%

# Load admissions

admissions_eicu = pd.read_csv(f"{OUTPUT_DIR}/final_data/admissions_eicu.csv")
admissions_mimic = pd.read_csv(f"{OUTPUT_DIR}/final_data/admissions_mimic.csv")
admissions_uf = pd.read_csv(f"{OUTPUT_DIR}/final_data/admissions_uf.csv")

#%%

# Convert LOS to days

admissions_eicu["unitdischargeoffset"] = admissions_eicu["unitdischargeoffset"] / 1440
admissions_uf.drop("icu_los", axis=1, inplace=True)
admissions_uf["icu_los"] = (
    pd.to_datetime(admissions_uf["exit_datetime"])
    - pd.to_datetime(admissions_uf["enter_datetime"])
) / np.timedelta64(1, "h")
admissions_uf["icu_los"] = admissions_uf["icu_los"] / 24


# %%

# Merge all admissions

admissions_eicu = admissions_eicu.rename(
    {
        "patientunitstayid": "icustay_id",
        "patienthealthsystemstayid": "merged_enc_id",
        "uniquepid": "patient_deiden_id",
        "unitdischargeoffset": "icu_los",
    },
    axis=1,
)

admissions_mimic = admissions_mimic.rename(
    {
        "stay_id": "icustay_id",
        "subject_id": "patient_deiden_id",
        "los": "icu_los",
        "hadm_id": "merged_enc_id",
    },
    axis=1,
)

cols = ["icustay_id", "patient_deiden_id", "merged_enc_id", "icu_los"]

admissions_eicu = admissions_eicu.loc[:, cols]
admissions_mimic = admissions_mimic.loc[:, cols]
admissions_uf = admissions_uf.loc[:, cols]

all_admissions = pd.concat(
    [admissions_eicu, admissions_mimic, admissions_uf], axis=0
).reset_index(drop=True)

#%%

# Load static and merge

static = pd.read_csv(f"{OUTPUT_DIR}/final_data/static.csv")

with open(f"{OUTPUT_DIR}/model/scalers_static.pkl", "rb") as f:
    scalers = pickle.load(f)

static.iloc[:, 1:] = scalers["scaler_static"].inverse_transform(static.iloc[:, 1:])

static["sex"] = scalers["scaler_gender"].inverse_transform(static["sex"].astype(int))
static["race"] = scalers["scaler_race"].inverse_transform(static["race"].astype(int))

all_admissions = all_admissions.merge(static, on="icustay_id")


#%%

# Load outcomes and merge

outcomes = pd.read_csv(f"{OUTPUT_DIR}/final_data/outcomes.csv")

mv = outcomes[outcomes["mv"] == "mv"]
vp = outcomes[outcomes["vp"] == "vp"]
crrt = outcomes[outcomes["crrt"] == "crrt"]
bt = outcomes[outcomes["bt"] == "bt"]
dead = outcomes[outcomes["final_state"] == "dead"]

mv = mv["icustay_id"].unique()
vp = vp["icustay_id"].unique()
crrt = crrt["icustay_id"].unique()
bt = bt["icustay_id"].unique()
dead = dead["icustay_id"].unique()

all_admissions["mv"] = 0
all_admissions.loc[(all_admissions["icustay_id"].isin(mv)), "mv"] = 1

all_admissions["vp"] = 0
all_admissions.loc[(all_admissions["icustay_id"].isin(vp)), "vp"] = 1

all_admissions["crrt"] = 0
all_admissions.loc[(all_admissions["icustay_id"].isin(crrt)), "crrt"] = 1

all_admissions["bt"] = 0
all_admissions.loc[(all_admissions["icustay_id"].isin(bt)), "bt"] = 1

all_admissions["died"] = 0
all_admissions.loc[(all_admissions["icustay_id"].isin(dead)), "died"] = 1

#%%

# Select characteristics

chars = [
    "patient_deiden_id",
    "merged_enc_id",
    "icustay_id",
    "age",
    "sex",
    "bmi",
    "race",
    "icu_los",
    "chf_poa",
    "copd_poa",
    "renal_disease_poa",
    "charlson_comorbidity_total_score",
    "mv",
    "vp",
    "crrt",
    "bt",
    "died",
]

all_admissions = all_admissions.loc[:, chars]

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

develop = all_admissions[all_admissions["icustay_id"].isin(ids_dev)]
external = all_admissions[all_admissions["icustay_id"].isin(ids_ext)]
temporal = all_admissions[all_admissions["icustay_id"].isin(ids_temp)]

#%%

# Distributions

numeric = ["age", "bmi", "icu_los", "charlson_comorbidity_total_score"]

develop_num_hist = develop.loc[:, numeric].hist()
external_num_hist = external.loc[:, numeric].hist()
temporal_num_hist = temporal.loc[:, numeric].hist()


# %%

# Development Set Characteristics

# Numeric characteristics

numeric_skew = ["age", "bmi", "icu_los", "charlson_comorbidity_total_score"]

develop_num_skew = (
    develop.loc[:, numeric_skew].describe().loc[["50%", "25%", "75%"]].round(1)
)
develop_num_skew = develop_num_skew.apply(
    lambda x: f"{x['50%']} ({x['25%']}-{x['75%']})", axis=0
)

# Binary characteristics

binary = ["chf_poa", "copd_poa", "renal_disease_poa", "mv", "vp", "crrt", "bt", "died"]

develop_bin = pd.DataFrame(
    data={
        "perc": ((develop.loc[:, binary].sum() / len(develop)) * 100).round(1),
        "count": develop.loc[:, binary].sum().astype("int"),
    }
)
develop_bin = develop_bin.apply(lambda x: f"{int(x['count'])} ({x['perc']}%)", axis=1)

# Gender and race

develop_gender = pd.DataFrame(
    data={
        "perc": ((len(develop[develop["sex"] == "Female"]) / len(develop)) * 100),
        "count": len(develop[develop["sex"] == "Female"]),
    },
    index=["Female"],
)
develop_race_b = pd.DataFrame(
    data={
        "perc": ((len(develop[develop["race"] == "black"]) / len(develop)) * 100),
        "count": len(develop[develop["race"] == "black"]),
    },
    index=["Black"],
)
develop_race_w = pd.DataFrame(
    data={
        "perc": ((len(develop[develop["race"] == "white"]) / len(develop)) * 100),
        "count": len(develop[develop["race"] == "white"]),
    },
    index=["White"],
)
develop_race_o = pd.DataFrame(
    data={
        "perc": ((len(develop[develop["race"] == "other"]) / len(develop)) * 100),
        "count": len(develop[develop["race"] == "other"]),
    },
    index=["Other"],
)

develop_socio = pd.concat(
    [develop_gender, develop_race_b, develop_race_w, develop_race_o], axis=0
).round(1)

develop_socio = develop_socio.apply(
    lambda x: f"{int(x['count'])} ({x['perc']}%)", axis=1
)

# Number of patients and admissions

develop_pat_count = pd.DataFrame(
    data=[
        len(develop["patient_deiden_id"].unique()),
        len(develop["merged_enc_id"].unique()),
        len(develop["icustay_id"].unique()),
    ],
    index=[
        "Number of patients",
        "Number of hospital encounters",
        "Number of ICU admissions",
    ],
)

develop_chars = pd.concat(
    [develop_pat_count, develop_num_skew, develop_socio, develop_bin],
    axis=0,
)

develop_chars = develop_chars.rename({0: "Development"}, axis=1)

#%%

# External Set Characteristics

# Numeric characteristics

numeric_skew = ["age", "bmi", "icu_los", "charlson_comorbidity_total_score"]

external_num_skew = (
    external.loc[:, numeric_skew].describe().loc[["50%", "25%", "75%"]].round(1)
)
external_num_skew = external_num_skew.apply(
    lambda x: f"{x['50%']} ({x['25%']}-{x['75%']})", axis=0
)

# Binary characteristics

binary = ["chf_poa", "copd_poa", "renal_disease_poa", "mv", "vp", "crrt", "bt", "died"]

external_bin = pd.DataFrame(
    data={
        "perc": ((external.loc[:, binary].sum() / len(external)) * 100).round(1),
        "count": external.loc[:, binary].sum().astype("int"),
    }
)
external_bin = external_bin.apply(lambda x: f"{int(x['count'])} ({x['perc']}%)", axis=1)

# Gender and race

external_gender = pd.DataFrame(
    data={
        "perc": ((len(external[external["sex"] == "Female"]) / len(external)) * 100),
        "count": len(external[external["sex"] == "Female"]),
    },
    index=["Female"],
)
external_race_b = pd.DataFrame(
    data={
        "perc": ((len(external[external["race"] == "black"]) / len(external)) * 100),
        "count": len(external[external["race"] == "black"]),
    },
    index=["Black"],
)
external_race_w = pd.DataFrame(
    data={
        "perc": ((len(external[external["race"] == "white"]) / len(external)) * 100),
        "count": len(external[external["race"] == "white"]),
    },
    index=["White"],
)
external_race_o = pd.DataFrame(
    data={
        "perc": ((len(external[external["race"] == "other"]) / len(external)) * 100),
        "count": len(external[external["race"] == "other"]),
    },
    index=["Other"],
)

external_socio = pd.concat(
    [external_gender, external_race_b, external_race_w, external_race_o], axis=0
).round(1)

external_socio = external_socio.apply(
    lambda x: f"{int(x['count'])} ({x['perc']}%)", axis=1
)

# Number of patients and admissions

external_pat_count = pd.DataFrame(
    data=[
        len(external["patient_deiden_id"].unique()),
        len(external["merged_enc_id"].unique()),
        len(external["icustay_id"].unique()),
    ],
    index=[
        "Number of patients",
        "Number of hospital encounters",
        "Number of ICU admissions",
    ],
)

external_chars = pd.concat(
    [
        external_pat_count,
        external_num_skew,
        external_socio,
        external_bin,
    ],
    axis=0,
)

external_chars = external_chars.rename({0: "External"}, axis=1)

#%%

# Temporal Set Characteristics

# Numeric characteristics

numeric_skew = ["age", "bmi", "icu_los", "charlson_comorbidity_total_score"]

temporal_num_skew = (
    temporal.loc[:, numeric_skew].describe().loc[["50%", "25%", "75%"]].round(1)
)
temporal_num_skew = temporal_num_skew.apply(
    lambda x: f"{x['50%']} ({x['25%']}-{x['75%']})", axis=0
)

# Binary characteristics

binary = ["chf_poa", "copd_poa", "renal_disease_poa", "mv", "vp", "crrt", "bt", "died"]

temporal_bin = pd.DataFrame(
    data={
        "perc": ((temporal.loc[:, binary].sum() / len(temporal)) * 100).round(1),
        "count": temporal.loc[:, binary].sum().astype("int"),
    }
)
temporal_bin = temporal_bin.apply(lambda x: f"{int(x['count'])} ({x['perc']}%)", axis=1)

# Gender and race

temporal_gender = pd.DataFrame(
    data={
        "perc": ((len(temporal[temporal["sex"] == "Female"]) / len(temporal)) * 100),
        "count": len(temporal[temporal["sex"] == "Female"]),
    },
    index=["Female"],
)
temporal_race_b = pd.DataFrame(
    data={
        "perc": ((len(temporal[temporal["race"] == "black"]) / len(temporal)) * 100),
        "count": len(temporal[temporal["race"] == "black"]),
    },
    index=["Black"],
)
temporal_race_w = pd.DataFrame(
    data={
        "perc": ((len(temporal[temporal["race"] == "white"]) / len(temporal)) * 100),
        "count": len(temporal[temporal["race"] == "white"]),
    },
    index=["White"],
)
temporal_race_o = pd.DataFrame(
    data={
        "perc": ((len(temporal[temporal["race"] == "other"]) / len(temporal)) * 100),
        "count": len(temporal[temporal["race"] == "other"]),
    },
    index=["Other"],
)

temporal_socio = pd.concat(
    [temporal_gender, temporal_race_b, temporal_race_w, temporal_race_o], axis=0
).round(1)

temporal_socio = temporal_socio.apply(
    lambda x: f"{int(x['count'])} ({x['perc']}%)", axis=1
)

# Number of patients and admissions

temporal_pat_count = pd.DataFrame(
    data=[
        len(temporal["patient_deiden_id"].unique()),
        len(temporal["merged_enc_id"].unique()),
        len(temporal["icustay_id"].unique()),
    ],
    index=[
        "Number of patients",
        "Number of hospital encounters",
        "Number of ICU admissions",
    ],
)

temporal_chars = pd.concat(
    [
        temporal_pat_count,
        temporal_num_skew,
        temporal_socio,
        temporal_bin,
    ],
    axis=0,
)

temporal_chars = temporal_chars.rename({0: "Temporal"}, axis=1)

#%%

# Add prospective patients

prosp = pd.read_csv(f"{PROSP_DATA_DIR}/final/admissions.csv")

prosp["icu_los"] = (
    pd.to_datetime(prosp["exit_datetime"]) - pd.to_datetime(prosp["enter_datetime"])
) / np.timedelta64(1, "h")
prosp["icu_los"] = prosp["icu_los"] / 24

cols = ["icustay_id", "patient_deiden_id", "merged_enc_id", "icu_los"]

prosp = prosp.loc[:, cols]

static = pd.read_csv(f"{PROSP_DATA_DIR}/final/static.csv")

prosp = prosp.merge(static, on="icustay_id")

outcomes = pd.read_csv(f"{PROSP_DATA_DIR}/final/outcomes.csv")

mv = outcomes[outcomes["mv"] == "mv"]
vp = outcomes[outcomes["vp"] == "vp"]
crrt = outcomes[outcomes["crrt"] == "crrt"]
dead = outcomes[outcomes["final_state"] == "dead"]

mv = mv["icustay_id"].unique()
vp = vp["icustay_id"].unique()
crrt = crrt["icustay_id"].unique()
dead = dead["icustay_id"].unique()

prosp["mv"] = 0
prosp.loc[(prosp["icustay_id"].isin(mv)), "mv"] = 1

prosp["vp"] = 0
prosp.loc[(prosp["icustay_id"].isin(vp)), "vp"] = 1

prosp["crrt"] = 0
prosp.loc[(prosp["icustay_id"].isin(crrt)), "crrt"] = 1

prosp["bt"] = 0

prosp["died"] = 0
prosp.loc[(prosp["icustay_id"].isin(dead)), "died"] = 1


chars = [
    "patient_deiden_id",
    "icustay_id",
    "merged_enc_id",
    "age",
    "sex",
    "bmi",
    "race",
    "icu_los",
    "chf_poa",
    "copd_poa",
    "renal_disease_poa",
    "charlson_comorbidity_total_score",
    "mv",
    "vp",
    "crrt",
    "bt",
    "died",
]

prosp = prosp.loc[:, chars]

# Numeric characteristics

numeric_skew = ["age", "bmi", "icu_los", "charlson_comorbidity_total_score"]

prosp["bmi"] = prosp["bmi"].replace({"MISSING": np.nan})
prosp["bmi"] = prosp["bmi"].astype(float)

prosp_num_skew = (
    prosp.loc[:, numeric_skew].describe().loc[["50%", "25%", "75%"]].round(1)
)
prosp_num_skew = prosp_num_skew.apply(
    lambda x: f"{x['50%']} ({x['25%']}-{x['75%']})", axis=0
)

# Binary characteristics

binary = ["chf_poa", "copd_poa", "renal_disease_poa", "mv", "vp", "crrt", "bt", "died"]

prosp_bin = pd.DataFrame(
    data={
        "perc": ((prosp.loc[:, binary].sum() / len(prosp)) * 100).round(1),
        "count": prosp.loc[:, binary].sum().astype("int"),
    }
)
prosp_bin = prosp_bin.apply(lambda x: f"{int(x['count'])} ({x['perc']}%)", axis=1)

# Gender and race

prosp_gender = pd.DataFrame(
    data={
        "perc": ((len(prosp[prosp["sex"] == "Female"]) / len(prosp)) * 100),
        "count": len(prosp[prosp["sex"] == "Female"]),
    },
    index=["Female"],
)
prosp_race_b = pd.DataFrame(
    data={
        "perc": ((len(prosp[prosp["race"] == "black"]) / len(prosp)) * 100),
        "count": len(prosp[prosp["race"] == "black"]),
    },
    index=["Black"],
)
prosp_race_w = pd.DataFrame(
    data={
        "perc": ((len(prosp[prosp["race"] == "white"]) / len(prosp)) * 100),
        "count": len(prosp[prosp["race"] == "white"]),
    },
    index=["White"],
)
prosp_race_o = pd.DataFrame(
    data={
        "perc": ((len(prosp[prosp["race"] == "other"]) / len(prosp)) * 100),
        "count": len(prosp[prosp["race"] == "other"]),
    },
    index=["Other"],
)

prosp_socio = pd.concat(
    [prosp_gender, prosp_race_b, prosp_race_w, prosp_race_o], axis=0
).round(1)

prosp_socio = prosp_socio.apply(lambda x: f"{int(x['count'])} ({x['perc']}%)", axis=1)

# Number of patients and admissions

prosp_pat_count = pd.DataFrame(
    data=[
        len(prosp["patient_deiden_id"].unique()),
        len(prosp["merged_enc_id"].unique()),
        len(prosp["icustay_id"].unique()),
    ],
    index=[
        "Number of patients",
        "Number of hospital encounters",
        "Number of ICU admissions",
    ],
)

prosp_chars = pd.concat(
    [prosp_pat_count, prosp_num_skew, prosp_socio, prosp_bin], axis=0
)

prosp_chars = prosp_chars.rename({0: "Prospective"}, axis=1)


#%%

# Merge all characteristics

all_chars = pd.concat(
    [develop_chars, external_chars, temporal_chars, prosp_chars], axis=1
)


# %%

# Perform statistical tests

from scipy.stats import shapiro, f_oneway, kruskal, mannwhitneyu
from statsmodels.stats.proportion import proportions_chisquare
import re

numerical_vars = ["age", "bmi", "icu_los", "charlson_comorbidity_total_score"]

sup = ["a", "b", "c", "d"]

cohorts = all_chars.columns.tolist()


def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans("".join(normal), "".join(super_s))
    return x.translate(res)


all_data = [develop, external, temporal, prosp]

count_cohort = 0

for cohort in all_data:

    count = 0

    for comp in all_data:

        superscript = sup[count]

        for var in numerical_vars:

            stat, p_value = mannwhitneyu(cohort[var], comp[var])

            if p_value < 0.05:

                all_chars.loc[var, cohorts[count_cohort]] = (
                    all_chars.loc[var, cohorts[count_cohort]]
                    + "\u2E34"
                    + get_super(superscript)
                )

        count += 1

    count_cohort += 1

# Extract categorical variables
categorical_vars = [
    "Female",
    "Black",
    "White",
    "Other",
    "chf_poa",
    "copd_poa",
    "renal_disease_poa",
    "mv",
    "vp",
    "crrt",
    "died",
]

observed_table = all_chars.loc[
    categorical_vars, ["Development", "External", "Temporal", "Prospective"]
].applymap(lambda x: int(re.search(r"(\d+)", str(x)).group(1)))


for cohort in cohorts:

    count = 0

    for comp in cohorts:

        superscript = sup[count]

        for var in categorical_vars:

            count1 = observed_table.loc[var, cohort]
            nobs1 = all_chars.loc["Number of ICU admissions", cohort]

            count2 = observed_table.loc[var, comp]
            nobs2 = all_chars.loc["Number of ICU admissions", comp]

            counts = np.array([count1, count2])
            nobs = np.array([nobs1, nobs2])

            chisq, pvalue, table = proportions_chisquare(counts, nobs)

            if pvalue < 0.05:

                all_chars.loc[var, cohort] = (
                    all_chars.loc[var, cohort] + "\u2E34" + get_super(superscript)
                )

        count += 1

# Regular expression pattern to match text after closing parenthesis
pattern = r"\)([^\s])"

# Function to replace matched text with a space
def replace_text(value):
    return re.sub(pattern, r") ", value)


# Apply the replacement operation to each element in the dataframe
all_chars.iloc[3:] = all_chars.iloc[3:].applymap(replace_text)

print(all_chars)

# Write HTML string to a file
with open(f"{OUTPUT_DIR}/summary/patient_characteristics/patient_chacteristics.html", 'w') as f:
    f.write(all_chars.to_html())

# %%

# Save table as csv

all_chars.to_csv(f"{OUTPUT_DIR}/summary/patient_characteristics/patient_chacteristics.csv")

# %%
