#%%

# Import libraries

import pandas as pd
import numpy as np
import pickle
from scipy import stats
import re
import os

from variables import OUTPUT_DIR, PROSP_DATA_DIR

if not os.path.exists(f"{OUTPUT_DIR}/analyses/all_patient_characteristics"):
    os.makedirs(f"{OUTPUT_DIR}/analyses/all_patient_characteristics")

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

# Load sequential

seq = pd.read_csv(f"{OUTPUT_DIR}/final_data/seq.csv", compression="gzip")

with open(f"{OUTPUT_DIR}/model/variable_mapping.pkl", "rb") as f:
    variable_map = pickle.load(f)

seq["variable"] = seq["variable_code"].map(variable_map)

#%%

# Map characteristics

char_dict = {
    "Number of patients": "Number of patients",
    "Number of ICU admissions": "Number of ICU admissions",
    "Number of hospital encounters": "Number of hospital encounters",
    "age": "Age, years",
    "bmi": "BMI, kg/m^2",
    "icu_los": "ICU length of stay, days",
    "charlson_comorbidity_total_score": "CCI",
    "Female": "Female",
    "Black": "Black",
    "White": "White",
    "Other": "Other",
    "aids_poa": "Acquired immunodeficiency syndrome",
    "cancer_poa": "Cancer",
    "cerebrovascular_poa": "Cerebrovascular disease",
    "chf_poa": "Congestive heart failure",
    "copd_poa": "Chronic obstructive pulmonary disease",
    "dementia_poa": "Dementia",
    "diabetes_w_o_complications_poa": "Diabetes without complications",
    "diabetes_w_complications_poa": "Diabetes with complications",
    "m_i_poa": "Myocardial infarction ",
    "metastatic_carcinoma_poa": "Metastatic carcinoma",
    "mild_liver_disease_poa": "Mild liver disease",
    "moderate_severe_liver_disease_poa": "Moderate severe liver disease",
    "paraplegia_hemiplegia_poa": "Paraplegia hemiplegia",
    "peptic_ulcer_disease_poa": "Peptic ulcer disease",
    "peripheral_vascular_disease_poa": "Peripheral vascular disease",
    "renal_disease_poa": "Renal disease",
    "rheumatologic_poa": "Rheumatologic disease",
    "mv": "MV",
    "vp": "VP",
    "crrt": "CRRT",
    "bt": "BT",
    "died": "Deceased",
    "ALT": "ALT",
    "AST": "AST",
    "Absolute Count - Basos": "Basophils",
    "Absolute Count - Eos": "Eosinophils",
    "Absolute Count - Lymphs": "Lymphocytes",
    "Absolute Count - Monos": "Monocytes",
    "Albumin": "Albumin",
    "Anion gap": "Anion gap",
    "Arterial Base Excess": "Arterial base excess",
    "Arterial CO2 Pressure": "Arterial CO2 pressure",
    "Arterial O2 Saturation": "Arterial O2 Saturation",
    "Arterial O2 pressure": "Arterial O2 pressure",
    "Brain Natiuretic Peptide (BNP)": "BNP",
    "C Reactive Protein (CRP)": "CRP",
    "Calcium non-ionized": "Calcium non-ionized",
    "Chloride (serum)": "Chloride",
    "Creatinine (serum)": "Creatinine",
    "Direct Bilirubin": "Bilirubin direct",
    "Glucose (serum)": "Glucose",
    "Heart Rate": "Heart rate",
    "Hematocrit (whole blood - calc)": "Hematocrit",
    "Hemoglobin": "Hemoglobin",
    "INR": "INR",
    "Ionized Calcium": "Calcium ionized",
    "Lactic Acid": "Lactate",
    "Non Invasive Blood Pressure diastolic": "DBP",
    "Non Invasive Blood Pressure systolic": "SBP",
    "O2 saturation pulseoxymetry": "SPO2",
    "PH (Arterial)": "Arterial PH",
    "Platelet Count": "Platelets",
    "Potassium (serum)": "Potassium",
    "Respiratory Rate": "Respiratory rate",
    "Sodium (serum)": "Sodium",
    "Specific Gravity (urine)": "Specific gravity urine",
    "Temperature Celsius": "Body temperature",
    "Total Bilirubin": "Bilirubin total",
    "Troponin-T": "Troponin-T",
    "WBC": "WBC",
    "cam": "CAM",
    "gcs": "GCS",
    "rass": "RASS",
    "Heparin Sodium": "Heparin sodium",
    "Propofol": "Propofol",
    "Folic Acid": "Folic acid",
    "Amiodarone": "Amiodarone",
    "Fentanyl": "Fentanyl",
    "Dexmedetomidine (Precedex)": "Dexmedetomidine",
    "Digoxin (Lanoxin)": "Digoxin",
}

cat_dict = {
    "Number of patients": "Number",
    "Number of hospital encounters": "Number",
    "Number of ICU admissions": "Number",
    "age": "Basic information",
    "bmi": "Basic information",
    "icu_los": "Basic information",
    "charlson_comorbidity_total_score": "Comorbidity index",
    "Female": "Basic information",
    "Black": "Race",
    "White": "Race",
    "Other": "Race",
    "aids_poa": "Comorbidities",
    "cancer_poa": "Comorbidities",
    "cerebrovascular_poa": "Comorbidities",
    "chf_poa": "Comorbidities",
    "copd_poa": "Comorbidities",
    "dementia_poa": "Comorbidities",
    "diabetes_w_o_complications_poa": "Comorbidities",
    "diabetes_w_complications_poa": "Comorbidities",
    "m_i_poa": "Comorbidities",
    "metastatic_carcinoma_poa": "Comorbidities",
    "mild_liver_disease_poa": "Comorbidities",
    "moderate_severe_liver_disease_poa": "Comorbidities",
    "paraplegia_hemiplegia_poa": "Comorbidities",
    "peptic_ulcer_disease_poa": "Comorbidities",
    "peripheral_vascular_disease_poa": "Comorbidities",
    "renal_disease_poa": "Comorbidities",
    "rheumatologic_poa": "Comorbidities",
    "mv": "Life-sustaining therapies",
    "vp": "Life-sustaining therapies",
    "crrt": "Life-sustaining therapies",
    "bt": "Life-sustaining therapies",
    "died": "Outcomes",
    "ALT": "Laboratory values",
    "AST": "Laboratory values",
    "Absolute Count - Basos": "Laboratory values",
    "Absolute Count - Eos": "Laboratory values",
    "Absolute Count - Lymphs": "Laboratory values",
    "Absolute Count - Monos": "Laboratory values",
    "Albumin": "Laboratory values",
    "Anion gap": "Laboratory values",
    "Arterial Base Excess": "Laboratory values",
    "Arterial CO2 Pressure": "Laboratory values",
    "Arterial O2 Saturation": "Laboratory values",
    "Arterial O2 pressure": "Laboratory values",
    "Brain Natiuretic Peptide (BNP)": "Laboratory values",
    "C Reactive Protein (CRP)": "Laboratory values",
    "Calcium non-ionized": "Laboratory values",
    "Chloride (serum)": "Laboratory values",
    "Creatinine (serum)": "Laboratory values",
    "Direct Bilirubin": "Laboratory values",
    "Glucose (serum)": "Laboratory values",
    "Heart Rate": "Vital signs",
    "Hematocrit (whole blood - calc)": "Laboratory values",
    "Hemoglobin": "Laboratory values",
    "INR": "Laboratory values",
    "Ionized Calcium": "Laboratory values",
    "Lactic Acid": "Laboratory values",
    "Non Invasive Blood Pressure diastolic": "Vital signs",
    "Non Invasive Blood Pressure systolic": "Vital signs",
    "O2 saturation pulseoxymetry": "Vital signs",
    "PH (Arterial)": "Laboratory values",
    "Platelet Count": "Laboratory values",
    "Potassium (serum)": "Laboratory values",
    "Respiratory Rate": "Vital signs",
    "Sodium (serum)": "Laboratory values",
    "Specific Gravity (urine)": "Laboratory values",
    "Temperature Celsius": "Vital signs",
    "Total Bilirubin": "Laboratory values",
    "Troponin-T": "Laboratory values",
    "WBC": "Laboratory values",
    "cam": "Assessment scores",
    "gcs": "Assessment scores",
    "rass": "Assessment scores",
    "Heparin Sodium": "Medications",
    "Propofol": "Medications",
    "Folic Acid": "Medications",
    "Amiodarone": "Medications",
    "Fentanyl": "Medications",
    "Dexmedetomidine (Precedex)": "Medications",
    "Digoxin (Lanoxin)": "Medications",
}


#%%

# Load outcomes and merge

outcomes = pd.read_csv(f"{OUTPUT_DIR}/final_data/outcomes.csv")

mv = outcomes[outcomes["mv"] == "mv"]
vp = outcomes[outcomes["vp"] == "vp"]
crrt = outcomes[outcomes["crrt"] == "crrt"]
bt = outcomes[outcomes["bt"] == "bt"]
dead = outcomes[outcomes["final_state"] == "dead"]

mv = list(mv["icustay_id"].unique())
vp = list(vp["icustay_id"].unique())
crrt = list(crrt["icustay_id"].unique())
bt = list(bt["icustay_id"].unique())
dead = list(dead["icustay_id"].unique())

nonstable = list(set(mv + vp + crrt + bt + dead))

stable = list(
    all_admissions.loc[
        (~all_admissions["icustay_id"].isin(nonstable)), "icustay_id"
    ].values
)

all_stays = stable + nonstable

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

seq_develop = seq[seq["icustay_id"].isin(ids_dev)]
seq_external = seq[seq["icustay_id"].isin(ids_ext)]
seq_temporal = seq[seq["icustay_id"].isin(ids_temp)]

del seq, all_admissions

# %%

# Development Set Characteristics

# Split stable patients, MV, VP, CRRT and deceased patients

groups = [all_stays, stable, mv, vp, crrt, dead]
labels = ["Overall", "Stable", "MV", "VP", "CRRT", "Deceased"]

develop_all_chars = pd.DataFrame()

for i in range(len(groups)):

    group = groups[i]
    label = labels[i]

    develop_group = develop[develop["icustay_id"].isin(group)]

    seq_develop_group = seq_develop[seq_develop["icustay_id"].isin(group)]

    print(len(develop_group))

    # Numeric characteristics

    numeric = ["age", "bmi", "icu_los", "charlson_comorbidity_total_score"]

    develop_num = (
        develop_group.loc[:, numeric].describe().loc[["50%", "25%", "75%"]].round(1)
    )
    develop_num = develop_num.apply(
        lambda x: f"{x['50%']} ({x['25%']}-{x['75%']})", axis=0
    )

    # Binary characteristics

    comorb = [col for col in static.columns.tolist() if "_poa" in col]

    binary = ["mv", "vp", "crrt", "bt", "died"]

    binary = comorb + binary

    develop_bin = pd.DataFrame(
        data={
            "perc": (
                (develop_group.loc[:, binary].sum() / len(develop_group)) * 100
            ).round(1),
            "count": develop_group.loc[:, binary].sum().astype("int"),
        }
    )
    develop_bin = develop_bin.apply(
        lambda x: f"{int(x['count'])} ({x['perc']}%)", axis=1
    )

    # Gender and race

    develop_gender = pd.DataFrame(
        data={
            "perc": (
                (
                    len(develop_group[develop_group["sex"] == "Female"])
                    / len(develop_group)
                )
                * 100
            ),
            "count": len(develop_group[develop_group["sex"] == "Female"]),
        },
        index=["Female"],
    )
    develop_race_b = pd.DataFrame(
        data={
            "perc": (
                (
                    len(develop_group[develop_group["race"] == "black"])
                    / len(develop_group)
                )
                * 100
            ),
            "count": len(develop_group[develop_group["race"] == "black"]),
        },
        index=["Black"],
    )
    develop_race_w = pd.DataFrame(
        data={
            "perc": (
                (
                    len(develop_group[develop_group["race"] == "white"])
                    / len(develop_group)
                )
                * 100
            ),
            "count": len(develop_group[develop_group["race"] == "white"]),
        },
        index=["White"],
    )
    develop_race_o = pd.DataFrame(
        data={
            "perc": (
                (
                    len(develop_group[develop_group["race"] == "other"])
                    / len(develop_group)
                )
                * 100
            ),
            "count": len(develop_group[develop_group["race"] == "other"]),
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
            len(develop_group["patient_deiden_id"].unique()),
            len(develop_group["merged_enc_id"].unique()),
            len(develop_group["icustay_id"].unique()),
        ],
        index=[
            "Number of patients",
            "Number of hospital encounters",
            "Number of ICU admissions",
        ],
    )

    # Add sequential variables

    meds = [
        "Amiodarone",
        "Folic Acid",
        "Heparin Sodium",
        "Dexmedetomidine (Precedex)",
        "Propofol",
        "Fentanyl",
        "Digoxin (Lanoxin)",
    ]

    seq_meds = seq_develop_group[seq_develop_group["variable"].isin(meds)]
    seq_all = seq_develop_group[~seq_develop_group["variable"].isin(meds)]

    seq_meds = seq_meds.drop_duplicates(subset=["icustay_id", "variable"])

    seq_meds = pd.DataFrame(
        data={
            "perc": (
                (seq_meds["variable"].value_counts() / len(develop_group)) * 100
            ).round(1),
            "count": seq_meds["variable"].value_counts().astype("int"),
        }
    )
    seq_meds = seq_meds.apply(lambda x: f"{int(x['count'])} ({x['perc']}%)", axis=1)

    seq_all = seq_all.groupby(by=["variable", "variable_code"])["value"].describe()

    seq_all = seq_all.reset_index()

    with open(f"{OUTPUT_DIR}/model/scalers_seq.pkl", "rb") as f:
        scalers = pickle.load(f)

    for variable_code in seq_all["variable_code"].tolist():
        scaler = scalers[f"scaler{variable_code}"]
        seq_all.loc[
            (seq_all["variable_code"] == variable_code), "50%"
        ] = scaler.inverse_transform(
            seq_all.loc[
                (seq_all["variable_code"] == variable_code), "50%"
            ].values.reshape(-1, 1)
        )
        seq_all.loc[
            (seq_all["variable_code"] == variable_code), "25%"
        ] = scaler.inverse_transform(
            seq_all.loc[
                (seq_all["variable_code"] == variable_code), "25%"
            ].values.reshape(-1, 1)
        )
        seq_all.loc[
            (seq_all["variable_code"] == variable_code), "75%"
        ] = scaler.inverse_transform(
            seq_all.loc[
                (seq_all["variable_code"] == variable_code), "75%"
            ].values.reshape(-1, 1)
        )

    seq_all = seq_all.loc[:, ["variable", "50%", "25%", "75%"]].round(1)

    seq_all = seq_all.set_index("variable")

    seq_all = seq_all.apply(lambda x: f"{x['50%']} ({x['25%']}-{x['75%']})", axis=1)

    develop_chars = pd.concat(
        [develop_pat_count, develop_num, develop_socio, develop_bin, seq_all, seq_meds],
        axis=0,
    )

    develop_chars = develop_chars.rename({0: f"{label}"}, axis=1)

    develop_all_chars = pd.concat([develop_all_chars, develop_chars], axis=1)


proportions = binary + meds + develop_socio.index.tolist()

means = develop_all_chars[~develop_all_chars.index.isin(proportions)].index.tolist()[3:]

compare = ["MV", "VP", "CRRT", "Deceased"]


def extract_statistics(input_string):
    # Use regular expression to extract mean, 25th percentile, and 75th percentile
    match = re.match(
        r"(-?\d+(\.\d+)?) \((-?\d+(\.\d+)?)\-\s*(-?\d+(\.\d+)?)\)", input_string
    )

    if match:
        mean = float(match.group(1))
        percentile25 = float(match.group(3))
        percentile75 = float(match.group(5))

        return mean, percentile25, percentile75
    else:
        print("Invalid format. Unable to extract statistics.")
        return None


for comp in compare:

    sample1 = develop_all_chars.loc[:, "Stable"]
    sample2 = develop_all_chars.loc[:, comp]

    size_group1 = sample1.loc["Number of ICU admissions"]
    size_group2 = sample2.loc["Number of ICU admissions"]

    exclude1 = sample1[sample1.isna()].index.tolist()
    exclude2 = sample2[sample2.isna()].index.tolist()

    # Z score test for proportions

    subsample = [
        proportion
        for proportion in proportions
        if proportion not in exclude1 and proportion not in exclude2
    ]

    for sample in subsample:

        success_group1 = int(
            re.search(r"(\d+)\s*\((\d+(\.\d*)?|\.\d+)\%\)", sample1.loc[sample]).group(
                1
            )
        )
        success_group2 = int(
            re.search(r"(\d+)\s*\((\d+(\.\d*)?|\.\d+)\%\)", sample2.loc[sample]).group(
                1
            )
        )

        # Calculate sample proportions
        p1 = success_group1 / size_group1
        p2 = success_group2 / size_group2

        # Calculate pooled sample proportion
        pooled_proportion = (success_group1 + success_group2) / (
            size_group1 + size_group2
        )

        # Calculate standard error of the difference between proportions
        standard_error = (
            (pooled_proportion * (1 - pooled_proportion))
            * ((1 / size_group1) + (1 / size_group2))
        ) ** 0.5

        if standard_error > 0:

            # Calculate the test statistic
            z_score = (p1 - p2) / standard_error

            # Calculate the two-tailed p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

            if p_value < 0.05:

                develop_all_chars.loc[sample, comp] = (
                    develop_all_chars.loc[sample, comp] + "*"
                )

                # prosp_all_chars.loc[sample, f'p-value {comp}'] = '< 0.001'

            # else:

            #     develop_all_chars.loc[sample, comp] = (
            #         develop_all_chars.loc[sample, comp] + f" (p = {round(p_value, 3)})"
            #     )

        # else:
        #     develop_all_chars.loc[sample, f'p-value {comp}'] = None

    # T-test for means

    subsample = [
        mean for mean in means if mean not in exclude1 and mean not in exclude2
    ]

    for sample in subsample:

        mean1, percentile25_1, percentile75_1 = extract_statistics(sample1.loc[sample])
        mean2, percentile25_2, percentile75_2 = extract_statistics(sample2.loc[sample])

        # Estimate standard deviation using interquartile range (IQR)
        iqr1 = percentile75_1 - percentile25_1
        iqr2 = percentile75_2 - percentile25_2

        std1 = iqr1 / (2 * stats.norm.ppf(3 / 4))  # Assuming normal distribution
        std2 = iqr2 / (2 * stats.norm.ppf(3 / 4))  # Assuming normal distribution

        # Perform independent two-sample t-test
        t_statistic, p_value = stats.ttest_ind_from_stats(
            mean1, std1, size_group1, mean2, std2, size_group2, equal_var=False
        )

        if p_value < 0.05:

            develop_all_chars.loc[sample, comp] = (
                develop_all_chars.loc[sample, comp] + "*"
            )

            # prosp_all_chars.loc[sample, f'p-value {comp}'] = '< 0.001'

        # else:

        #     develop_all_chars.loc[sample, comp] = (
        #         develop_all_chars.loc[sample, comp] + f" (p = {round(p_value, 3)})"
        #     )


def format_numbers_with_comma(x):

    # Extract numbers using regular expression
    numbers = re.findall(r"\d+\.\d+|\d+", x)

    if len(numbers) == 2:

        # Format the first number with comma
        formatted_number = "{:,.0f}".format(int(numbers[0]))
        # Concatenate the percentage
        formatted_value = f"{formatted_number} ({numbers[1]}%)"

        if "*" in x:

            formatted_value = formatted_value + "*"

        output = formatted_value

    else:

        output = x

    return output


develop_all_chars.loc[proportions, :] = develop_all_chars.loc[proportions, :].astype(
    str
)

develop_all_chars.loc[proportions, :] = develop_all_chars.loc[proportions, :].applymap(
    format_numbers_with_comma
)

develop_all_chars["category"] = develop_all_chars.index.map(cat_dict)
develop_all_chars.index = develop_all_chars.index.map(char_dict)

categories = [
    "Number",
    "Basic information",
    "Race",
    "Comorbidity index",
    "Comorbidities",
    "Vital signs",
    "Assessment scores",
    "Laboratory values",
    "Medications",
    "Life-sustaining therapies",
    "Outcomes",
]

develop_all_chars["category"] = pd.Categorical(
    develop_all_chars["category"], categories=categories, ordered=True
)

develop_all_chars["variables"] = develop_all_chars.index.values

develop_all_chars = develop_all_chars.sort_values(by=["category", "variables"])

develop_all_chars.to_csv(f"{OUTPUT_DIR}/analyses/all_patient_characteristics/develop_characteristics.csv")

#%%

# External Set Characteristics

# Split stable patients, MV, VP, CRRT and deceased patients

groups = [all_stays, stable, mv, vp, crrt, dead]
labels = ["Overall", "Stable", "MV", "VP", "CRRT", "Deceased"]

external_all_chars = pd.DataFrame()

for i in range(len(groups)):

    group = groups[i]
    label = labels[i]

    external_group = external[external["icustay_id"].isin(group)]

    seq_external_group = seq_external[seq_external["icustay_id"].isin(group)]

    print(len(external_group))

    # Numeric characteristics

    numeric = ["age", "bmi", "icu_los", "charlson_comorbidity_total_score"]

    external_num = (
        external_group.loc[:, numeric].describe().loc[["50%", "25%", "75%"]].round(1)
    )
    external_num = external_num.apply(
        lambda x: f"{x['50%']} ({x['25%']}-{x['75%']})", axis=0
    )

    # Binary characteristics

    comorb = [col for col in static.columns.tolist() if "_poa" in col]

    binary = ["mv", "vp", "crrt", "bt", "died"]

    binary = comorb + binary

    external_bin = pd.DataFrame(
        data={
            "perc": (
                (external_group.loc[:, binary].sum() / len(external_group)) * 100
            ).round(1),
            "count": external_group.loc[:, binary].sum().astype("int"),
        }
    )
    external_bin = external_bin.apply(
        lambda x: f"{int(x['count'])} ({x['perc']}%)", axis=1
    )

    # Gender and race

    external_gender = pd.DataFrame(
        data={
            "perc": (
                (
                    len(external_group[external_group["sex"] == "Female"])
                    / len(external_group)
                )
                * 100
            ),
            "count": len(external_group[external_group["sex"] == "Female"]),
        },
        index=["Female"],
    )
    external_race_b = pd.DataFrame(
        data={
            "perc": (
                (
                    len(external_group[external_group["race"] == "black"])
                    / len(external_group)
                )
                * 100
            ),
            "count": len(external_group[external_group["race"] == "black"]),
        },
        index=["Black"],
    )
    external_race_w = pd.DataFrame(
        data={
            "perc": (
                (
                    len(external_group[external_group["race"] == "white"])
                    / len(external_group)
                )
                * 100
            ),
            "count": len(external_group[external_group["race"] == "white"]),
        },
        index=["White"],
    )
    external_race_o = pd.DataFrame(
        data={
            "perc": (
                (
                    len(external_group[external_group["race"] == "other"])
                    / len(external_group)
                )
                * 100
            ),
            "count": len(external_group[external_group["race"] == "other"]),
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
            len(external_group["patient_deiden_id"].unique()),
            len(external_group["merged_enc_id"].unique()),
            len(external_group["icustay_id"].unique()),
        ],
        index=[
            "Number of patients",
            "Number of hospital encounters",
            "Number of ICU admissions",
        ],
    )

    # Add sequential variables

    meds = [
        "Amiodarone",
        "Folic Acid",
        "Heparin Sodium",
        "Dexmedetomidine (Precedex)",
        "Propofol",
        "Fentanyl",
        "Digoxin (Lanoxin)",
    ]

    seq_meds = seq_external_group[seq_external_group["variable"].isin(meds)]
    seq_all = seq_external_group[~seq_external_group["variable"].isin(meds)]

    seq_meds = seq_meds.drop_duplicates(subset=["icustay_id", "variable"])

    seq_meds = pd.DataFrame(
        data={
            "perc": (
                (seq_meds["variable"].value_counts() / len(external_group)) * 100
            ).round(1),
            "count": seq_meds["variable"].value_counts().astype("int"),
        }
    )
    seq_meds = seq_meds.apply(lambda x: f"{int(x['count'])} ({x['perc']}%)", axis=1)

    seq_all = seq_all.groupby(by=["variable", "variable_code"])["value"].describe()

    seq_all = seq_all.reset_index()

    with open(f"{OUTPUT_DIR}/model/scalers_seq.pkl", "rb") as f:
        scalers = pickle.load(f)

    for variable_code in seq_all["variable_code"].tolist():
        scaler = scalers[f"scaler{variable_code}"]
        seq_all.loc[
            (seq_all["variable_code"] == variable_code), "50%"
        ] = scaler.inverse_transform(
            seq_all.loc[
                (seq_all["variable_code"] == variable_code), "50%"
            ].values.reshape(-1, 1)
        )
        seq_all.loc[
            (seq_all["variable_code"] == variable_code), "25%"
        ] = scaler.inverse_transform(
            seq_all.loc[
                (seq_all["variable_code"] == variable_code), "25%"
            ].values.reshape(-1, 1)
        )
        seq_all.loc[
            (seq_all["variable_code"] == variable_code), "75%"
        ] = scaler.inverse_transform(
            seq_all.loc[
                (seq_all["variable_code"] == variable_code), "75%"
            ].values.reshape(-1, 1)
        )

    seq_all = seq_all.loc[:, ["variable", "50%", "25%", "75%"]].round(1)

    seq_all = seq_all.set_index("variable")

    seq_all = seq_all.apply(lambda x: f"{x['50%']} ({x['25%']}-{x['75%']})", axis=1)

    external_chars = pd.concat(
        [
            external_pat_count,
            external_num,
            external_socio,
            external_bin,
            seq_all,
            seq_meds,
        ],
        axis=0,
    )

    external_chars = external_chars.rename({0: f"{label}"}, axis=1)

    external_all_chars = pd.concat([external_all_chars, external_chars], axis=1)


proportions = binary + meds + external_socio.index.tolist()

means = external_all_chars[~external_all_chars.index.isin(proportions)].index.tolist()[
    3:
]

compare = ["MV", "VP", "CRRT", "Deceased"]


def extract_statistics(input_string):
    # Use regular expression to extract mean, 25th percentile, and 75th percentile
    match = re.match(
        r"(-?\d+(\.\d+)?) \((-?\d+(\.\d+)?)\-\s*(-?\d+(\.\d+)?)\)", input_string
    )

    if match:
        mean = float(match.group(1))
        percentile25 = float(match.group(3))
        percentile75 = float(match.group(5))

        return mean, percentile25, percentile75
    else:
        print("Invalid format. Unable to extract statistics.")
        return None


for comp in compare:

    sample1 = external_all_chars.loc[:, "Stable"]
    sample2 = external_all_chars.loc[:, comp]

    size_group1 = sample1.loc["Number of ICU admissions"]
    size_group2 = sample2.loc["Number of ICU admissions"]

    exclude1 = sample1[sample1.isna()].index.tolist()
    exclude2 = sample2[sample2.isna()].index.tolist()

    # Z score test for proportions

    subsample = [
        proportion
        for proportion in proportions
        if proportion not in exclude1 and proportion not in exclude2
    ]

    for sample in subsample:

        success_group1 = int(
            re.search(r"(\d+)\s*\((\d+(\.\d*)?|\.\d+)\%\)", sample1.loc[sample]).group(
                1
            )
        )
        success_group2 = int(
            re.search(r"(\d+)\s*\((\d+(\.\d*)?|\.\d+)\%\)", sample2.loc[sample]).group(
                1
            )
        )

        # Calculate sample proportions
        p1 = success_group1 / size_group1
        p2 = success_group2 / size_group2

        # Calculate pooled sample proportion
        pooled_proportion = (success_group1 + success_group2) / (
            size_group1 + size_group2
        )

        # Calculate standard error of the difference between proportions
        standard_error = (
            (pooled_proportion * (1 - pooled_proportion))
            * ((1 / size_group1) + (1 / size_group2))
        ) ** 0.5

        if standard_error > 0:

            # Calculate the test statistic
            z_score = (p1 - p2) / standard_error

            # Calculate the two-tailed p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

            if p_value < 0.05:

                external_all_chars.loc[sample, comp] = (
                    external_all_chars.loc[sample, comp] + "*"
                )

                # prosp_all_chars.loc[sample, f'p-value {comp}'] = '< 0.001'

            # else:

            #     external_all_chars.loc[sample, comp] = (
            #         external_all_chars.loc[sample, comp] + f" (p = {round(p_value, 3)})"
            #     )

        # else:
        #     external_all_chars.loc[sample, f'p-value {comp}'] = None

    # T-test for means

    subsample = [
        mean for mean in means if mean not in exclude1 and mean not in exclude2
    ]

    for sample in subsample:

        mean1, percentile25_1, percentile75_1 = extract_statistics(sample1.loc[sample])
        mean2, percentile25_2, percentile75_2 = extract_statistics(sample2.loc[sample])

        # Estimate standard deviation using interquartile range (IQR)
        iqr1 = percentile75_1 - percentile25_1
        iqr2 = percentile75_2 - percentile25_2

        std1 = iqr1 / (2 * stats.norm.ppf(3 / 4))  # Assuming normal distribution
        std2 = iqr2 / (2 * stats.norm.ppf(3 / 4))  # Assuming normal distribution

        # Perform independent two-sample t-test
        t_statistic, p_value = stats.ttest_ind_from_stats(
            mean1, std1, size_group1, mean2, std2, size_group2, equal_var=False
        )

        if p_value < 0.05:

            external_all_chars.loc[sample, comp] = (
                external_all_chars.loc[sample, comp] + "*"
            )

            # prosp_all_chars.loc[sample, f'p-value {comp}'] = '< 0.001'

        # else:

        #     external_all_chars.loc[sample, comp] = (
        #         external_all_chars.loc[sample, comp] + f" (p = {round(p_value, 3)})"
        #     )


def format_numbers_with_comma(x):

    # Extract numbers using regular expression
    numbers = re.findall(r"\d+\.\d+|\d+", x)

    if len(numbers) == 2:

        # Format the first number with comma
        formatted_number = "{:,.0f}".format(int(numbers[0]))
        # Concatenate the percentage
        formatted_value = f"{formatted_number} ({numbers[1]}%)"

        if "*" in x:

            formatted_value = formatted_value + "*"

        output = formatted_value

    else:

        output = x

    return output


external_all_chars.loc[proportions, :] = external_all_chars.loc[proportions, :].astype(
    str
)

external_all_chars.loc[proportions, :] = external_all_chars.loc[
    proportions, :
].applymap(format_numbers_with_comma)


external_all_chars["category"] = external_all_chars.index.map(cat_dict)
external_all_chars.index = external_all_chars.index.map(char_dict)

categories = [
    "Number",
    "Basic information",
    "Race",
    "Comorbidity index",
    "Comorbidities",
    "Vital signs",
    "Assessment scores",
    "Laboratory values",
    "Medications",
    "Life-sustaining therapies",
    "Outcomes",
]

external_all_chars["category"] = pd.Categorical(
    external_all_chars["category"], categories=categories, ordered=True
)

external_all_chars["variables"] = external_all_chars.index.values

external_all_chars = external_all_chars.sort_values(by=["category", "variables"])

external_all_chars.to_csv(f"{OUTPUT_DIR}/analyses/all_patient_characteristics/external_characteristics.csv")

#%%

# Temporal Set Characteristics

# Split stable patients, MV, VP, CRRT and deceased patients

groups = [all_stays, stable, mv, vp, crrt, dead]
labels = ["Overall", "Stable", "MV", "VP", "CRRT", "Deceased"]

temporal_all_chars = pd.DataFrame()

for i in range(len(groups)):

    group = groups[i]
    label = labels[i]

    temporal_group = temporal[temporal["icustay_id"].isin(group)]

    seq_temporal_group = seq_temporal[seq_temporal["icustay_id"].isin(group)]

    print(len(temporal_group))

    # Numeric characteristics

    numeric = ["age", "bmi", "icu_los", "charlson_comorbidity_total_score"]

    temporal_num = (
        temporal_group.loc[:, numeric].describe().loc[["50%", "25%", "75%"]].round(1)
    )
    temporal_num = temporal_num.apply(
        lambda x: f"{x['50%']} ({x['25%']}-{x['75%']})", axis=0
    )

    # Binary characteristics

    comorb = [col for col in static.columns.tolist() if "_poa" in col]

    binary = ["mv", "vp", "crrt", "bt", "died"]

    binary = comorb + binary

    temporal_bin = pd.DataFrame(
        data={
            "perc": (
                (temporal_group.loc[:, binary].sum() / len(temporal_group)) * 100
            ).round(1),
            "count": temporal_group.loc[:, binary].sum().astype("int"),
        }
    )
    temporal_bin = temporal_bin.apply(
        lambda x: f"{int(x['count'])} ({x['perc']}%)", axis=1
    )

    # Gender and race

    temporal_gender = pd.DataFrame(
        data={
            "perc": (
                (
                    len(temporal_group[temporal_group["sex"] == "Female"])
                    / len(temporal_group)
                )
                * 100
            ),
            "count": len(temporal_group[temporal_group["sex"] == "Female"]),
        },
        index=["Female"],
    )
    temporal_race_b = pd.DataFrame(
        data={
            "perc": (
                (
                    len(temporal_group[temporal_group["race"] == "black"])
                    / len(temporal_group)
                )
                * 100
            ),
            "count": len(temporal_group[temporal_group["race"] == "black"]),
        },
        index=["Black"],
    )
    temporal_race_w = pd.DataFrame(
        data={
            "perc": (
                (
                    len(temporal_group[temporal_group["race"] == "white"])
                    / len(temporal_group)
                )
                * 100
            ),
            "count": len(temporal_group[temporal_group["race"] == "white"]),
        },
        index=["White"],
    )
    temporal_race_o = pd.DataFrame(
        data={
            "perc": (
                (
                    len(temporal_group[temporal_group["race"] == "other"])
                    / len(temporal_group)
                )
                * 100
            ),
            "count": len(temporal_group[temporal_group["race"] == "other"]),
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
            len(temporal_group["patient_deiden_id"].unique()),
            len(temporal_group["merged_enc_id"].unique()),
            len(temporal_group["icustay_id"].unique()),
        ],
        index=[
            "Number of patients",
            "Number of hospital encounters",
            "Number of ICU admissions",
        ],
    )

    # Add sequential variables

    meds = [
        "Amiodarone",
        "Folic Acid",
        "Heparin Sodium",
        "Dexmedetomidine (Precedex)",
        "Propofol",
        "Fentanyl",
        "Digoxin (Lanoxin)",
    ]

    seq_meds = seq_temporal_group[seq_temporal_group["variable"].isin(meds)]
    seq_all = seq_temporal_group[~seq_temporal_group["variable"].isin(meds)]

    seq_meds = seq_meds.drop_duplicates(subset=["icustay_id", "variable"])

    seq_meds = pd.DataFrame(
        data={
            "perc": (
                (seq_meds["variable"].value_counts() / len(temporal_group)) * 100
            ).round(1),
            "count": seq_meds["variable"].value_counts().astype("int"),
        }
    )
    seq_meds = seq_meds.apply(lambda x: f"{int(x['count'])} ({x['perc']}%)", axis=1)

    seq_all = seq_all.groupby(by=["variable", "variable_code"])["value"].describe()

    seq_all = seq_all.reset_index()

    with open(f"{OUTPUT_DIR}/model/scalers_seq.pkl", "rb") as f:
        scalers = pickle.load(f)

    for variable_code in seq_all["variable_code"].tolist():
        scaler = scalers[f"scaler{variable_code}"]
        seq_all.loc[
            (seq_all["variable_code"] == variable_code), "50%"
        ] = scaler.inverse_transform(
            seq_all.loc[
                (seq_all["variable_code"] == variable_code), "50%"
            ].values.reshape(-1, 1)
        )
        seq_all.loc[
            (seq_all["variable_code"] == variable_code), "25%"
        ] = scaler.inverse_transform(
            seq_all.loc[
                (seq_all["variable_code"] == variable_code), "25%"
            ].values.reshape(-1, 1)
        )
        seq_all.loc[
            (seq_all["variable_code"] == variable_code), "75%"
        ] = scaler.inverse_transform(
            seq_all.loc[
                (seq_all["variable_code"] == variable_code), "75%"
            ].values.reshape(-1, 1)
        )

    seq_all = seq_all.loc[:, ["variable", "50%", "25%", "75%"]].round(1)

    seq_all = seq_all.set_index("variable")

    seq_all = seq_all.apply(lambda x: f"{x['50%']} ({x['25%']}-{x['75%']})", axis=1)

    temporal_chars = pd.concat(
        [
            temporal_pat_count,
            temporal_num,
            temporal_socio,
            temporal_bin,
            seq_all,
            seq_meds,
        ],
        axis=0,
    )

    temporal_chars = temporal_chars.rename({0: f"{label}"}, axis=1)

    temporal_all_chars = pd.concat([temporal_all_chars, temporal_chars], axis=1)


proportions = binary + meds + temporal_socio.index.tolist()

means = temporal_all_chars[~temporal_all_chars.index.isin(proportions)].index.tolist()[
    3:
]

compare = ["MV", "VP", "CRRT", "Deceased"]


def extract_statistics(input_string):
    # Use regular expression to extract mean, 25th percentile, and 75th percentile
    match = re.match(
        r"(-?\d+(\.\d+)?) \((-?\d+(\.\d+)?)\-\s*(-?\d+(\.\d+)?)\)", input_string
    )

    if match:
        mean = float(match.group(1))
        percentile25 = float(match.group(3))
        percentile75 = float(match.group(5))

        return mean, percentile25, percentile75
    else:
        print("Invalid format. Unable to extract statistics.")
        return None


for comp in compare:

    sample1 = temporal_all_chars.loc[:, "Stable"]
    sample2 = temporal_all_chars.loc[:, comp]

    size_group1 = sample1.loc["Number of ICU admissions"]
    size_group2 = sample2.loc["Number of ICU admissions"]

    exclude1 = sample1[sample1.isna()].index.tolist()
    exclude2 = sample2[sample2.isna()].index.tolist()

    # Z score test for proportions

    subsample = [
        proportion
        for proportion in proportions
        if proportion not in exclude1 and proportion not in exclude2
    ]

    for sample in subsample:

        success_group1 = int(
            re.search(r"(\d+)\s*\((\d+(\.\d*)?|\.\d+)\%\)", sample1.loc[sample]).group(
                1
            )
        )
        success_group2 = int(
            re.search(r"(\d+)\s*\((\d+(\.\d*)?|\.\d+)\%\)", sample2.loc[sample]).group(
                1
            )
        )

        # Calculate sample proportions
        p1 = success_group1 / size_group1
        p2 = success_group2 / size_group2

        # Calculate pooled sample proportion
        pooled_proportion = (success_group1 + success_group2) / (
            size_group1 + size_group2
        )

        # Calculate standard error of the difference between proportions
        standard_error = (
            (pooled_proportion * (1 - pooled_proportion))
            * ((1 / size_group1) + (1 / size_group2))
        ) ** 0.5

        if standard_error > 0:

            # Calculate the test statistic
            z_score = (p1 - p2) / standard_error

            # Calculate the two-tailed p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

            if p_value < 0.05:

                temporal_all_chars.loc[sample, comp] = (
                    temporal_all_chars.loc[sample, comp] + "*"
                )

                # prosp_all_chars.loc[sample, f'p-value {comp}'] = '< 0.001'

            # else:

            #     temporal_all_chars.loc[sample, comp] = (
            #         temporal_all_chars.loc[sample, comp] + f" (p = {round(p_value, 3)})"
            #     )

        # else:
        #     temporal_all_chars.loc[sample, f'p-value {comp}'] = None

    # T-test for means

    subsample = [
        mean for mean in means if mean not in exclude1 and mean not in exclude2
    ]

    for sample in subsample:

        mean1, percentile25_1, percentile75_1 = extract_statistics(sample1.loc[sample])
        mean2, percentile25_2, percentile75_2 = extract_statistics(sample2.loc[sample])

        # Estimate standard deviation using interquartile range (IQR)
        iqr1 = percentile75_1 - percentile25_1
        iqr2 = percentile75_2 - percentile25_2

        std1 = iqr1 / (2 * stats.norm.ppf(3 / 4))  # Assuming normal distribution
        std2 = iqr2 / (2 * stats.norm.ppf(3 / 4))  # Assuming normal distribution

        # Perform independent two-sample t-test
        t_statistic, p_value = stats.ttest_ind_from_stats(
            mean1, std1, size_group1, mean2, std2, size_group2, equal_var=False
        )

        if p_value < 0.05:

            temporal_all_chars.loc[sample, comp] = (
                temporal_all_chars.loc[sample, comp] + "*"
            )

            # prosp_all_chars.loc[sample, f'p-value {comp}'] = '< 0.001'

        # else:

        #     temporal_all_chars.loc[sample, comp] = (
        #         temporal_all_chars.loc[sample, comp] + f" (p = {round(p_value, 3)})"
        #     )


def format_numbers_with_comma(x):

    # Extract numbers using regular expression
    numbers = re.findall(r"\d+\.\d+|\d+", x)

    if len(numbers) == 2:

        # Format the first number with comma
        formatted_number = "{:,.0f}".format(int(numbers[0]))
        # Concatenate the percentage
        formatted_value = f"{formatted_number} ({numbers[1]}%)"

        if "*" in x:

            formatted_value = formatted_value + "*"

        output = formatted_value

    else:

        output = x

    return output


temporal_all_chars.loc[proportions, :] = temporal_all_chars.loc[proportions, :].astype(
    str
)

temporal_all_chars.loc[proportions, :] = temporal_all_chars.loc[
    proportions, :
].applymap(format_numbers_with_comma)

temporal_all_chars["category"] = temporal_all_chars.index.map(cat_dict)
temporal_all_chars.index = temporal_all_chars.index.map(char_dict)

categories = [
    "Number",
    "Basic information",
    "Race",
    "Comorbidity index",
    "Comorbidities",
    "Vital signs",
    "Assessment scores",
    "Laboratory values",
    "Medications",
    "Life-sustaining therapies",
    "Outcomes",
]

temporal_all_chars["category"] = pd.Categorical(
    temporal_all_chars["category"], categories=categories, ordered=True
)

temporal_all_chars["variables"] = temporal_all_chars.index.values

temporal_all_chars = temporal_all_chars.sort_values(by=["category", "variables"])

temporal_all_chars.to_csv(f"{OUTPUT_DIR}/analyses/all_patient_characteristics/temporal_characteristics.csv")

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

mv = list(mv["icustay_id"].unique())
vp = list(vp["icustay_id"].unique())
crrt = list(crrt["icustay_id"].unique())
dead = list(dead["icustay_id"].unique())

nonstable = list(set(mv + vp + crrt + dead))

stable = list(prosp.loc[(~prosp["icustay_id"].isin(nonstable)), "icustay_id"].values)

all_stays = stable + nonstable

prosp["mv"] = 0
prosp.loc[(prosp["icustay_id"].isin(mv)), "mv"] = 1

prosp["vp"] = 0
prosp.loc[(prosp["icustay_id"].isin(vp)), "vp"] = 1

prosp["crrt"] = 0
prosp.loc[(prosp["icustay_id"].isin(crrt)), "crrt"] = 1

prosp["bt"] = 0

prosp["died"] = 0
prosp.loc[(prosp["icustay_id"].isin(dead)), "died"] = 1


# Add sequential variables

seq_prosp = pd.read_csv(f"{PROSP_DATA_DIR}/final/seq.csv")

# Split stable patients, MV, VP, CRRT and deceased patients

groups = [all_stays, stable, mv, vp, crrt, dead]
labels = ["Overall", "Stable", "MV", "VP", "CRRT", "Deceased"]

prosp_all_chars = pd.DataFrame()

for i in range(len(groups)):

    group = groups[i]
    label = labels[i]

    prosp_group = prosp[prosp["icustay_id"].isin(group)]

    seq_prosp_group = seq_prosp[seq_prosp["icustay_id"].isin(group)]

    print(len(prosp_group))

    # Numeric characteristics

    numeric = ["age", "bmi", "icu_los", "charlson_comorbidity_total_score"]

    prosp_num = (
        prosp_group.loc[:, numeric].describe().loc[["50%", "25%", "75%"]].round(1)
    )
    prosp_num = prosp_num.apply(lambda x: f"{x['50%']} ({x['25%']}-{x['75%']})", axis=0)

    # Binary characteristics

    comorb = [col for col in static.columns.tolist() if "_poa" in col]

    binary = ["mv", "vp", "crrt", "bt", "died"]

    binary = comorb + binary

    prosp_bin = pd.DataFrame(
        data={
            "perc": ((prosp_group.loc[:, binary].sum() / len(prosp_group)) * 100).round(
                1
            ),
            "count": prosp_group.loc[:, binary].sum().astype("int"),
        }
    )
    prosp_bin = prosp_bin.apply(lambda x: f"{int(x['count'])} ({x['perc']}%)", axis=1)

    # Gender and race

    prosp_gender = pd.DataFrame(
        data={
            "perc": (
                (len(prosp_group[prosp_group["sex"] == "Female"]) / len(prosp_group))
                * 100
            ),
            "count": len(prosp_group[prosp_group["sex"] == "Female"]),
        },
        index=["Female"],
    )
    prosp_race_b = pd.DataFrame(
        data={
            "perc": (
                (len(prosp_group[prosp_group["race"] == "black"]) / len(prosp_group))
                * 100
            ),
            "count": len(prosp_group[prosp_group["race"] == "black"]),
        },
        index=["Black"],
    )
    prosp_race_w = pd.DataFrame(
        data={
            "perc": (
                (len(prosp_group[prosp_group["race"] == "white"]) / len(prosp_group))
                * 100
            ),
            "count": len(prosp_group[prosp_group["race"] == "white"]),
        },
        index=["White"],
    )
    prosp_race_o = pd.DataFrame(
        data={
            "perc": (
                (len(prosp_group[prosp_group["race"] == "other"]) / len(prosp_group))
                * 100
            ),
            "count": len(prosp_group[prosp_group["race"] == "other"]),
        },
        index=["Other"],
    )

    prosp_socio = pd.concat(
        [prosp_gender, prosp_race_b, prosp_race_w, prosp_race_o], axis=0
    ).round(1)

    prosp_socio = prosp_socio.apply(
        lambda x: f"{int(x['count'])} ({x['perc']}%)", axis=1
    )

    # Number of patients and admissions

    prosp_pat_count = pd.DataFrame(
        data=[
            len(prosp_group["patient_deiden_id"].unique()),
            len(prosp_group["merged_enc_id"].unique()),
            len(prosp_group["icustay_id"].unique()),
        ],
        index=[
            "Number of patients",
            "Number of hospital encounters",
            "Number of ICU admissions",
        ],
    )

    # Add sequential variables

    meds = [
        "Amiodarone",
        "Folic Acid",
        "Heparin Sodium",
        "Dexmedetomidine (Precedex)",
        "Propofol",
        "Fentanyl",
        "Digoxin (Lanoxin)",
    ]

    seq_meds = seq_prosp_group[seq_prosp_group["variable"].isin(meds)]
    seq_all = seq_prosp_group[~seq_prosp_group["variable"].isin(meds)]

    seq_meds = seq_meds.drop_duplicates(subset=["icustay_id", "variable"])

    seq_meds = pd.DataFrame(
        data={
            "perc": (
                (seq_meds["variable"].value_counts() / len(prosp_group)) * 100
            ).round(1),
            "count": seq_meds["variable"].value_counts().astype("int"),
        }
    )
    seq_meds = seq_meds.apply(lambda x: f"{int(x['count'])} ({x['perc']}%)", axis=1)

    seq_all = seq_all.groupby(by=["variable", "variable_code"])["value"].describe()

    seq_all = seq_all.reset_index()

    seq_all = seq_all.loc[:, ["variable", "50%", "25%", "75%"]].round(1)

    seq_all = seq_all.set_index("variable")

    seq_all = seq_all.apply(lambda x: f"{x['50%']} ({x['25%']}-{x['75%']})", axis=1)

    prosp_chars = pd.concat(
        [prosp_pat_count, prosp_num, prosp_socio, prosp_bin, seq_all, seq_meds], axis=0
    )

    prosp_chars = prosp_chars.rename({0: f"{label}"}, axis=1)

    prosp_all_chars = pd.concat([prosp_all_chars, prosp_chars], axis=1)


proportions = binary + meds + prosp_socio.index.tolist()

proportions = [
    proportion
    for proportion in proportions
    if proportion in prosp_all_chars.index.tolist()
]

means = prosp_all_chars[~prosp_all_chars.index.isin(proportions)].index.tolist()[3:]

means = [mean for mean in means if mean in prosp_all_chars.index.tolist()]


compare = ["MV", "VP", "CRRT", "Deceased"]


def extract_statistics(input_string):
    # Use regular expression to extract mean, 25th percentile, and 75th percentile
    match = re.match(
        r"(-?\d+(\.\d+)?) \((-?\d+(\.\d+)?)\-\s*(-?\d+(\.\d+)?)\)", input_string
    )

    if match:
        mean = float(match.group(1))
        percentile25 = float(match.group(3))
        percentile75 = float(match.group(5))

        return mean, percentile25, percentile75
    else:
        print("Invalid format. Unable to extract statistics.")
        return None


for comp in compare:

    sample1 = prosp_all_chars.loc[:, "Stable"]
    sample2 = prosp_all_chars.loc[:, comp]

    size_group1 = sample1.loc["Number of ICU admissions"]
    size_group2 = sample2.loc["Number of ICU admissions"]

    exclude1 = sample1[sample1.isna()].index.tolist()
    exclude2 = sample2[sample2.isna()].index.tolist()

    # Z score test for proportions

    subsample = [
        proportion
        for proportion in proportions
        if proportion not in exclude1 and proportion not in exclude2
    ]

    for sample in subsample:

        success_group1 = int(
            re.search(r"(\d+)\s*\((\d+(\.\d*)?|\.\d+)\%\)", sample1.loc[sample]).group(
                1
            )
        )
        success_group2 = int(
            re.search(r"(\d+)\s*\((\d+(\.\d*)?|\.\d+)\%\)", sample2.loc[sample]).group(
                1
            )
        )

        # Calculate sample proportions
        p1 = success_group1 / size_group1
        p2 = success_group2 / size_group2

        # Calculate pooled sample proportion
        pooled_proportion = (success_group1 + success_group2) / (
            size_group1 + size_group2
        )

        # Calculate standard error of the difference between proportions
        standard_error = (
            (pooled_proportion * (1 - pooled_proportion))
            * ((1 / size_group1) + (1 / size_group2))
        ) ** 0.5

        if standard_error > 0:

            # Calculate the test statistic
            z_score = (p1 - p2) / standard_error

            # Calculate the two-tailed p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

            if p_value < 0.05:

                prosp_all_chars.loc[sample, comp] = (
                    prosp_all_chars.loc[sample, comp] + "*"
                )

                # prosp_all_chars.loc[sample, f'p-value {comp}'] = '< 0.001'

            # else:

            #     prosp_all_chars.loc[sample, comp] = (
            #         prosp_all_chars.loc[sample, comp] + f" (p = {round(p_value, 3)})"
            #     )

            # prosp_all_chars.loc[sample, f'p-value {comp}'] = round(p_value, 3)

        # else:
        #     prosp_all_chars.loc[sample, f'p-value {comp}'] = None

    # T-test for means

    subsample = [
        mean for mean in means if mean not in exclude1 and mean not in exclude2
    ]

    for sample in subsample:

        mean1, percentile25_1, percentile75_1 = extract_statistics(sample1.loc[sample])
        mean2, percentile25_2, percentile75_2 = extract_statistics(sample2.loc[sample])

        # Estimate standard deviation using interquartile range (IQR)
        iqr1 = percentile75_1 - percentile25_1
        iqr2 = percentile75_2 - percentile25_2

        std1 = iqr1 / (2 * stats.norm.ppf(3 / 4))  # Assuming normal distribution
        std2 = iqr2 / (2 * stats.norm.ppf(3 / 4))  # Assuming normal distribution

        # Perform independent two-sample t-test
        t_statistic, p_value = stats.ttest_ind_from_stats(
            mean1, std1, size_group1, mean2, std2, size_group2, equal_var=False
        )

        if p_value < 0.05:

            prosp_all_chars.loc[sample, comp] = prosp_all_chars.loc[sample, comp] + "*"

            # prosp_all_chars.loc[sample, f'p-value {comp}'] = '< 0.001'

        # else:

        #     prosp_all_chars.loc[sample, comp] = (
        #         prosp_all_chars.loc[sample, comp] + f" (p = {round(p_value, 3)})"
        #     )


def format_numbers_with_comma(x):

    # Extract numbers using regular expression
    numbers = re.findall(r"\d+\.\d+|\d+", x)

    if len(numbers) == 2:

        # Format the first number with comma
        formatted_number = "{:,.0f}".format(int(numbers[0]))
        # Concatenate the percentage
        formatted_value = f"{formatted_number} ({numbers[1]}%)"

        if "*" in x:

            formatted_value = formatted_value + "*"

        output = formatted_value

    else:

        output = x

    return output


prosp_all_chars.loc[proportions, :] = prosp_all_chars.loc[proportions, :].astype(str)

prosp_all_chars.loc[proportions, :] = prosp_all_chars.loc[proportions, :].applymap(
    format_numbers_with_comma
)


prosp_all_chars["category"] = prosp_all_chars.index.map(cat_dict)
prosp_all_chars.index = prosp_all_chars.index.map(char_dict)

categories = [
    "Number",
    "Basic information",
    "Race",
    "Comorbidity index",
    "Comorbidities",
    "Vital signs",
    "Assessment scores",
    "Laboratory values",
    "Medications",
    "Life-sustaining therapies",
    "Outcomes",
]

prosp_all_chars["category"] = pd.Categorical(
    prosp_all_chars["category"], categories=categories, ordered=True
)

prosp_all_chars["variables"] = prosp_all_chars.index.values

prosp_all_chars = prosp_all_chars.sort_values(by=["category", "variables"])

prosp_all_chars.to_csv(f"{OUTPUT_DIR}/analyses/all_patient_characteristics/prosp_characteristics.csv")

# %%
