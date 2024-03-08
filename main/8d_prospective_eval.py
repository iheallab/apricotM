#%%

# Import libraries
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

DATA_DIR = "/home/contreras.miguel/deepacu"

#%%

# Load prospective admissions

admissions = pd.read_csv(f"{DATA_DIR}/prosp_new/admissions.csv")

print("-" * 20 + "Initial ICU stays" + "-" * 20)

print(f"Prospective cohort: {len(admissions)}")

#%%

# Filter by vitals presence
print("-" * 20 + "Filtering by vitals presence" + "-" * 20)

vitals = pd.read_csv(f"{DATA_DIR}/prosp_new/vitals.csv")

use_vitals = [
    "heart rate",
    "spo2",
    "respiratory_rate",
    "temp_celsius",
    "noninvasive_diastolic",
    "noninvasive_systolic",
]

vitals = vitals[vitals["variable"].isin(use_vitals)]

filtered_vitals = vitals.groupby("icustay_id")["variable"].nunique()

filtered_vitals = list(filtered_vitals[filtered_vitals == 5].index.values)

print(f"ICU stays dropped: {len(admissions) - len(filtered_vitals)}")

admissions = admissions[admissions["icustay_id"].isin(filtered_vitals)]

#%%

# Filter admissions by missigness of basic information
print("-" * 20 + "Filtering by basic info presence" + "-" * 20)

static = pd.read_csv(f"{DATA_DIR}/prosp_new/static.csv")

static = static[static["icustay_id"].isin(admissions["icustay_id"])]

basic = static.dropna()
basic = basic[basic["race"] != "missing"]
basic = basic[basic["bmi"] != "MISSING"]
basic = basic["icustay_id"].tolist()

print(f"ICU stays dropped: {len(admissions) - len(basic)}")

admissions = admissions[admissions["icustay_id"].isin(basic)]

#%%

# Filter admissions by outcome data

print("-" * 20 + "Filtering by outcome presence" + "-" * 20)

outcomes = pd.read_csv(f"{DATA_DIR}/prosp_new/acuity_states_prosp.csv")

outcomes = outcomes[outcomes["icustay_id"].isin(admissions["icustay_id"].unique())]
outcomes = list(outcomes["icustay_id"].unique())

print(f"ICU stays dropped: {len(admissions) - len(outcomes)}")

admissions = admissions[admissions["icustay_id"].isin(outcomes)]

#%%

# Filter admissions by discharge info

print("-" * 20 + "Filtering by discharge info" + "-" * 20)

station = admissions.dropna(subset=["to_station"])

print(f"ICU stays dropped: {len(admissions) - len(station)}")

admissions = admissions.dropna(subset=["to_station"])


#%%

# Build sequential data

vitals = pd.read_csv(f"{DATA_DIR}/prosp_new/vitals.csv")
labs = pd.read_csv(f"{DATA_DIR}/prosp_new/labs.csv")
meds = pd.read_csv(f"{DATA_DIR}/prosp_new/meds.csv")
scores = pd.read_csv(f"{DATA_DIR}/prosp_new/scores.csv")

meds["value"] = 1

vitals["variable"] = vitals["variable"].replace(
    {
        "heart_rate": "vital_clean_heart_rate",
        "spo2": "vital_clean_spo2",
        "fio2_resp": "vital_fio2",
        "respiratory_rate": "vital_clean_respiratory_rate",
        "temp_celsius": "vital_clean_temp_celsius",
        "noninvasive_diastolic": "vital_bp_non_invasive_diastolic",
        "noninvasive_systolic": "vital_bp_non_invasive_systolic",
        "etco2": "vital_clean_etco2",
        "peep": "vital_clean_peep",
        "tidal_volume_exhaled": "vital_tidal_volume",
        "o2_l_min": "vital_clean_o2_l_min",
        "pip": "vital_clean_pip",
    }
)

labs["variable"] = "lab_" + labs["variable"]

meds["variable"] = meds["variable"].str.split().str[0]
meds["variable"] = meds["variable"].str.lower()
meds["variable"] = "med_" + meds["variable"]


meds["variable"] = meds["variable"].replace(
    {
        "med_folic": "med_folic_acid",
        "med_propofol": "med_propofol",
        "med_heparin": "med_heparin_sodium_(porcine)",
        "med_amiodarone": "med_amiodarone_hcl",
        "med_fentanyl": "med_fentanyl",
        "med_dexmedetomidine": "med_dexmedetomidine_hcl_[200_mcg/2ml]_|_sodium_chloride",
        "med_norepinephrine": "med_norepinephrine_bitartrate",
        "med_phenylephrine": "med_phenylephrine_hcl_(pressors)",
        "med_epinephrine": "med_epinephrine",
        "med_vasopressin": "med_vasopressin",
    }
)

scores["variable"] = scores["variable"].replace(
    {"glasgow_coma_adult_score": "gcs_total", "rass": "rass_score", "cam": "cam_score"}
)

scores["variable"] = "score_" + scores["variable"]

seq = (
    pd.concat([vitals, labs, meds, scores], axis=0)
    .sort_values(by=["icustay_id", "time"])
    .reset_index(drop=True)
)

del vitals, labs, meds, scores

# seq['time'] = pd.to_datetime(seq['time'])

# seq.set_index('time', inplace=True)

# seq = seq.groupby(['icustay_id', 'variable']).resample('20T').mean().reset_index()

seq = seq.dropna()

seq["hours"] = (
    pd.to_datetime(seq["time"]) - pd.to_datetime(seq["enter_datetime"])
) / np.timedelta64(1, "h")

seq = seq[seq["hours"] >= 0]

seq = seq[seq["icustay_id"].isin(admissions["icustay_id"])]


def remove_outliers(group):

    # Q1 = group['value'].quantile(0.25)
    # Q3 = group['value'].quantile(0.75)
    # IQR = Q3 - Q1
    # lower_threshold = Q1 - 1.5 * IQR
    # upper_threshold = Q3 + 1.5 * IQR

    lower_threshold = group["value"].quantile(0.01)
    upper_threshold = group["value"].quantile(0.99)

    group_filtered = group[
        (group["value"] >= lower_threshold) & (group["value"] <= upper_threshold)
    ]

    return group_filtered


seq = seq.groupby("variable").apply(remove_outliers)

seq = seq.reset_index(drop=True)

seq = seq.sort_values(by=["icustay_id", "hours"])

print(len(seq["icustay_id"].unique()))

#%%

# Extract and encode variables

import pickle

with open(f"{DATA_DIR}/model/variable_mapping.pkl", "rb") as f:
    variable_map = pickle.load(f)

final_map = pd.read_csv(f"{DATA_DIR}/final_var_map.csv")
variables = final_map["uf"].tolist()

seq = seq[seq["variable"].isin(variables)]


map_eicu_uf = dict()
for i in range(len(final_map)):
    map_eicu_uf[final_map.loc[i, "uf"]] = final_map.loc[i, "mimic"]

seq["variable"] = seq["variable"].map(map_eicu_uf)

print(len(seq["variable"].unique()))

inverted_map = {}
for key, values in variable_map.items():
    inverted_map[values] = key

seq["variable_code"] = seq["variable"].map(inverted_map)

seq = seq[
    [
        "icustay_id",
        "variable",
        "variable_code",
        "hours",
        "value",
    ]
]

seq = seq[seq["hours"] >= 0]

seq = seq.sort_values(by=["icustay_id", "hours"])

seq = seq.dropna().reset_index(drop=True)

seq["interval"] = ((seq["hours"] // 4) + 1).astype(int)

seq["shift_id"] = seq["icustay_id"].astype(str) + "_" + seq["interval"].astype(str)

seq.drop("interval", axis=1, inplace=True)

seq.to_csv(f"{DATA_DIR}/prosp_new/seq.csv", index=None)

#%%

# Build outcomes

outcomes = pd.read_csv(f"{DATA_DIR}/prosp_new/acuity_states_prosp.csv")

outcomes = outcomes[outcomes["icustay_id"].isin(admissions["icustay_id"].unique())]

outcomes["interval"] = outcomes.groupby("icustay_id").cumcount()

outcomes["shift_id"] = (
    outcomes["icustay_id"].astype(str) + "_" + outcomes["interval"].astype(str)
)

outcomes["transition"] = np.nan

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

outcomes = outcomes[outcomes["interval"] != 0]

seq = seq[seq["shift_id"].isin(outcomes["shift_id"].unique())]

# seq = seq[seq['icustay_id'].isin(outcomes['icustay_id'].unique())]

print(len(seq["icustay_id"].unique()))
print(len(outcomes["icustay_id"].unique()))

print(len(seq["shift_id"].unique()))
print(len(outcomes["shift_id"].unique()))

#%%

# Filter ICU stays with missing data in any 4h interval

missing1 = outcomes.loc[
    ~outcomes["shift_id"].isin(seq["shift_id"].unique()), "icustay_id"
].unique()
missing2 = seq.loc[
    ~seq["shift_id"].isin(outcomes["shift_id"].unique()), "icustay_id"
].unique()

missing = list(set(list(missing1) + list(missing2)))

print(len(missing))

seq = seq[~seq["icustay_id"].isin(missing)]
outcomes = outcomes[~outcomes["icustay_id"].isin(missing)]

print(len(seq["icustay_id"].unique()))
print(len(outcomes["icustay_id"].unique()))

print(len(seq["shift_id"].unique()))
print(len(outcomes["shift_id"].unique()))

#%%

# Build static data

static = pd.read_csv(f"{DATA_DIR}/prosp_new/static.csv")

static = static[static["icustay_id"].isin(seq["icustay_id"].unique())]

static["sex"] = static["sex"].map({"FEMALE": "Female", "MALE": "Male"})
static["race"] = static["race"].map({"AA": "black", "WHITE": "white", "OTHER": "other"})

numeric_feat = ["sex", "age", "race", "bmi"]

comob_cols = [c for c in static.columns.tolist() if "_poa" in c]

select_feat = (
    ["icustay_id"] + numeric_feat + comob_cols + ["charlson_comorbidity_total_score"]
)

static = static.loc[:, select_feat]

#%%

# Scale data

with open(f"{DATA_DIR}/model/scalers_static.pkl", "rb") as f:
    scalers = pickle.load(f)

static["sex"] = scalers["scaler_gender"].transform(static["sex"])
static["race"] = scalers["scaler_race"].transform(static["race"])


def impute_and_scale_static(df):
    # exclude = ['icustay_id', 'gender', 'race']
    exclude = ["icustay_id"]
    columns = [c for c in df.columns if c not in exclude]

    scaler = scalers["scaler_static"]
    df[columns] = scaler.transform(df[columns])

    df = df.reset_index()
    print(f"sum na static: {((df.isna()).sum() != 0).sum()}")
    return df


static = impute_and_scale_static(static)


with open(f"{DATA_DIR}/model/scalers_seq.pkl", "rb") as f:
    scalers = pickle.load(f)

VALUE_COLS = ["value"]


def _standardize_variable(group):

    variable_code = group["variable_code"].values[0]

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

print(len(outcomes["icustay_id"].unique()))
print(len(seq["icustay_id"].unique()))

seq = seq.reset_index(drop=True)

hours_scaler = scalers["scaler_hours"]

seq[["hours"]] = hours_scaler.transform(seq[["hours"]])

seq.drop("variable", axis=1, inplace=True)

#%%

admissions = pd.read_csv(f"{DATA_DIR}/prosp_new/admissions.csv")

admissions = admissions[admissions["icustay_id"].isin(outcomes["icustay_id"].unique())]

dead_ids = admissions[admissions["to_station"].str.contains("expired", case=False)]

dead_ids = dead_ids["icustay_id"].unique()

dischg_ids = admissions[admissions["to_station"].str.contains("home", case=False)]

dischg_ids = dischg_ids["icustay_id"].unique()


outcomes["dead_ind"] = 0
outcomes.loc[(outcomes["icustay_id"].isin(dead_ids)), "dead_ind"] = 1

outcomes["dischg_ind"] = 0
outcomes.loc[(outcomes["icustay_id"].isin(dischg_ids)), "dischg_ind"] = 1

# Identify the index of the last state for each icustay_id
last_state_indices = outcomes.groupby("icustay_id")["interval"].transform("idxmax")

# Assign 'discharge' to the last state if dead_indicator is 0, otherwise assign 'dead'
outcomes.loc[
    (outcomes["dead_ind"] == 1) & (outcomes.index == last_state_indices), "final_state"
] = "dead"
outcomes.loc[
    (outcomes["dischg_ind"] == 1) & (outcomes.index == last_state_indices),
    "final_state",
] = "discharge"

# outcomes.loc[(outcomes['dead_ind'] == 0) & (outcomes.index == last_state_indices), 'transition'] = 'discharge'
# outcomes.loc[(outcomes['dead_ind'] == 1) & (outcomes.index == last_state_indices), 'transition'] = 'dead'

print(outcomes["final_state"].value_counts())
print(len(outcomes["icustay_id"].unique()))

#%%

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


outcomes = outcomes.sort_values(by=["icustay_id", "shift_start"]).reset_index(drop=True)

map_states = {
    "dead": 3,
    "unstable": 2,
    "stable": 1,
    "discharge": 0,
}

outcomes["final_state"] = outcomes["final_state"].replace(map_states)

print(outcomes["final_state"].value_counts())

# outcomes = outcomes[outcomes['shift_id'].isin(seq['shift_id'].unique())]
# seq = seq[seq['shift_id'].isin(outcomes['shift_id'].unique())]


print(outcomes["final_state"].value_counts())

print(len(outcomes["shift_id"].unique()))
print(len(seq["shift_id"].unique()))

outcomes = outcomes.reset_index(drop=True)
seq = seq.reset_index(drop=True)

# %%
seq_len = seq.groupby("shift_id").size().reset_index().rename({0: "seq_len"}, axis=1)

outcomes = outcomes.merge(seq_len, on="shift_id", how="left")


# outcomes['seq_len'] = outcomes['seq_len'].fillna(0)

# outcomes['seq_len_cum'] = outcomes.groupby('icustay_id')['seq_len'].cumsum()

# outcomes['seq_len_cum'] = outcomes['seq_len_cum'].astype(int)

# print(outcomes['seq_len_cum'].min())

max_seq_len = 128

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

    # curr_id = outcomes.loc[i, 'icustay_id']
    # if i+1 < len(outcomes):
    #     next_id = outcomes.loc[i+1, 'icustay_id']
    # else:
    #     next_id = 0

    # if curr_id != next_id:
    #     start += seq_len[i]

    start += seq_len[i]

#%%

y = outcomes["final_state"].values
y_trans = outcomes["transition"].values
y_mv = outcomes["transition_mv"].values
y_pressor = outcomes["transition_vp"].values
y_crrt = outcomes["transition_crrt"].values

static = static.sort_values(by=["icustay_id"]).reset_index(drop=True)

static = static.set_index("icustay_id")
static = static.loc[outcomes["icustay_id"].tolist(), :]
static = static.reset_index()

static = static.iloc[:, 2:]

static = static.to_numpy()
# %%

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(categories=[[0, 1, 2, 3]])
y_trans = enc.fit_transform(y_trans.reshape(-1, 1)).toarray()

enc = OneHotEncoder(categories=[[0, 1, 2, 3]])
y_mv = enc.fit_transform(y_mv.reshape(-1, 1)).toarray()

enc = OneHotEncoder(categories=[[0, 1, 2, 3]])
y_pressor = enc.fit_transform(y_pressor.reshape(-1, 1)).toarray()

enc = OneHotEncoder(categories=[[0, 1, 2, 3]])
y_crrt = enc.fit_transform(y_crrt.reshape(-1, 1)).toarray()

enc = OneHotEncoder(categories=[[0, 1, 2, 3]])
y_main = enc.fit_transform(y.reshape(-1, 1)).toarray()

y_trans = np.concatenate(
    (y_trans[:, 2:4], y_mv[:, 2:], y_pressor[:, 2:], y_crrt[:, 2:]), axis=1
)

print(np.unique(y_trans[:, 1], return_counts=True))
print(np.unique(y_trans[:, 3], return_counts=True))
print(np.unique(y_trans[:, 5], return_counts=True))
print(np.unique(y_trans[:, 7], return_counts=True))


#%%

# Load model

from model.clin_transfo.clin_transfo import Clintransfo
import torch

# Load model architecture

model_architecture = torch.load(
    f"{DATA_DIR}/model/clin_transfo/clintransfo_architecture.pth"
)
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = Clintransfo(
    d_model=model_architecture["d_model"],
    d_hidden=model_architecture["d_hidden"],
    d_input=model_architecture["d_input"],
    d_static=model_architecture["d_static"],
    d_output=model_architecture["d_output"],
    max_code=model_architecture["max_code"],
    q=model_architecture["q"],
    v=model_architecture["v"],
    h=model_architecture["h"],
    N=model_architecture["N"],
    device=DEVICE,
    dropout=model_architecture["dropout"],
    mask=model_architecture["mask"],
).to(DEVICE)


# Load model weights

model.load_state_dict(
    torch.load(
        f"{DATA_DIR}/model/clin_transfo/clintransfo_weights.pth", map_location=DEVICE
    )
)

#%%

# Convert data to tensors

X = torch.FloatTensor(qkv)
static = torch.FloatTensor(static)

y = np.concatenate([y_main, y_trans], axis=1)
y = torch.FloatTensor(y)

#%%

# Prospective validation

BATCH_SIZE = 256

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch.autograd import Variable

y_true_class = np.zeros((len(X), 12))
y_pred_prob = np.zeros((len(X), 12))

for patient in range(0, len(X), BATCH_SIZE):
    inputs = Variable(X[patient : patient + BATCH_SIZE]).to(DEVICE)
    static_input = Variable(static[patient : patient + BATCH_SIZE]).to(DEVICE)
    labels = Variable(y[patient : patient + BATCH_SIZE]).to(DEVICE)

    pred_y = model(inputs, static_input)

    y_true_class[patient : patient + BATCH_SIZE, :] = labels.to("cpu").numpy()
    y_pred_prob[patient : patient + BATCH_SIZE, :] = pred_y.to("cpu").detach().numpy()


#%%

print("-" * 40)
print("Prospective Validation")

aucs = []
for i in range(y_pred_prob.shape[1]):
    if len(np.unique(y_true_class[:, i])) > 1:
        ind_auc = roc_auc_score(y_true_class[:, i], y_pred_prob[:, i])
    else:
        ind_auc = "N/A"
    aucs.append(ind_auc)

print(f"val_roc_auc: {aucs}")


aucs = []
for i in range(y_pred_prob.shape[1]):
    if len(np.unique(y_true_class[:, i])) > 1:
        precision, recall, _ = precision_recall_curve(
            y_true_class[:, i], y_pred_prob[:, i]
        )
        val_pr_auc = auc(recall, precision)
    else:
        val_pr_auc = "N/A"
    aucs.append(val_pr_auc)

print(f"val_pr_auc: {aucs}")

#%%

icustay_ids = X[:, 0, 3].reshape(-1, 1)

cols = [
    "icustay_id",
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

pred_labels = np.concatenate([icustay_ids, y_pred_prob], axis=1)
pred_labels = pd.DataFrame(pred_labels, columns=cols)

true_labels = np.concatenate([icustay_ids, y_true_class], axis=1)
true_labels = pd.DataFrame(true_labels, columns=cols)

true_labels.to_csv("results/apricot_tuned/prosp_true_labels.csv", index=None)
pred_labels.to_csv("results/apricot_tuned/prosp_pred_labels.csv", index=None)
