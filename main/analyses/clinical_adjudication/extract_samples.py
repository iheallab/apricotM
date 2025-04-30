#%%
# Import libraries

import pandas as pd
import h5py
from variables import PROSP_DATA_DIR, MODEL_DIR, OUTPUT_DIR, time_window
import captum.attr as attr
import numpy as np
import tqdm

#%%

# Read tables

admissions = pd.read_csv(f"{PROSP_DATA_DIR}/final/admissions.csv")
outcomes = pd.read_csv(f"{PROSP_DATA_DIR}/final/outcomes.csv")
predictions = pd.read_csv(f"{MODEL_DIR}/results/prosp_pred_labels.csv")
ground_truth = pd.read_csv(f"{MODEL_DIR}/results/prosp_true_labels.csv")

ground_truth = ground_truth.rename({"stable-unstable": "gt_unstable", "no mv-mv": "gt_mv", "no vp- vp": "gt_vp", "no crrt-crrt": "gt_crrt"}, axis=1)
predictions = pd.concat([predictions, ground_truth[["gt_unstable", "gt_mv", "gt_vp", "gt_crrt"]]], axis=1)

predictions = predictions.reset_index().rename({"index": "sample_id"}, axis=1)

admissions = admissions[admissions["icustay_id"].isin(predictions["icustay_id"].unique())].reset_index(drop=True)
outcomes = outcomes[outcomes["icustay_id"].isin(predictions["icustay_id"].unique())].reset_index(drop=True)

print(len(admissions["icustay_id"].unique()))
print(len(outcomes["icustay_id"].unique()))
print(len(predictions["icustay_id"].unique()))

predictions = pd.concat([predictions, outcomes[["shift_id"]]], axis=1)

predictions["hours"] = (predictions.groupby("icustay_id").cumcount() + 1) * time_window
predictions = predictions.merge(admissions[["icustay_id", "patient_deiden_id", "enter_datetime"]], on="icustay_id", how="inner")

# Convert 'enter_datetime' to datetime format if it's not already in that format
predictions['enter_datetime'] = pd.to_datetime(predictions['enter_datetime'])

# Create the 'time' column by adding 'hours' as a timedelta to 'enter_datetime'
predictions['time'] = predictions['enter_datetime'] + pd.to_timedelta(predictions['hours'], unit='h')

condition = (outcomes["transition"] == "stable-stable") | (outcomes["transition"] == "stable-unstable")

outcomes = outcomes[condition].reset_index(drop=True)
predictions = predictions[condition].reset_index(drop=True)

# Define a function to drop rows after the first occurrence of "stable-unstable" per icustay_id
def drop_after_first_unstable(group):
    # Find the index of the first "stable-unstable" occurrence
    unstable_index = group[group['transition'] == 'stable-unstable'].index.min()
    # Return rows up to the first "stable-unstable" occurrence
    if pd.notna(unstable_index):  # Check if there was any "stable-unstable" occurrence
        return group.loc[:unstable_index]
    return group  # Return the entire group if "stable-unstable" was not found

outcomes = outcomes.groupby("icustay_id", group_keys=False).apply(drop_after_first_unstable).reset_index().reset_index(drop=True)

predictions = predictions[predictions["shift_id"].isin(outcomes["shift_id"])].reset_index(drop=True)
admissions = admissions[admissions["icustay_id"].isin(predictions["icustay_id"].unique())]

thresholds = pd.read_csv(f"{MODEL_DIR}/episode_prediction/stable-unstable_episode_metrics.csv").rename({"Unnamed: 0": "cohort"}, axis=1)
threshold = thresholds.loc[(thresholds["cohort"] == "prosp") & (thresholds["Precision"] == 0.33), "Threshold"].values[0]

# %%

predictions["alert"] = (predictions["stable-unstable"] >= threshold).astype(int)
subsample = list(predictions.loc[predictions["alert"] == 1, "icustay_id"].unique())

admissions = admissions[admissions["icustay_id"].isin(subsample)]
predictions = predictions[predictions["icustay_id"].isin(subsample)]
outcomes = outcomes[outcomes["icustay_id"].isin(subsample)]

sample_ids = predictions["sample_id"].tolist()

# %%

thresholds_mv = pd.read_csv(f"{MODEL_DIR}/episode_prediction/no mv-mv_episode_metrics.csv").rename({"Unnamed: 0": "cohort"}, axis=1)
thresholds_vp = pd.read_csv(f"{MODEL_DIR}/episode_prediction/no vp- vp_episode_metrics.csv").rename({"Unnamed: 0": "cohort"}, axis=1)
thresholds_crrt = pd.read_csv(f"{MODEL_DIR}/episode_prediction/no crrt-crrt_episode_metrics.csv").rename({"Unnamed: 0": "cohort"}, axis=1)

threshold_mv = thresholds_mv.loc[(thresholds_mv["cohort"] == "prosp") & (thresholds_mv["Precision"] == 0.33), "Threshold"].values[0]
threshold_vp = thresholds_vp.loc[(thresholds_vp["cohort"] == "prosp") & (thresholds_vp["Precision"] == 0.25), "Threshold"].values[0]
threshold_crrt = thresholds_crrt.loc[(thresholds_crrt["cohort"] == "prosp") & (thresholds_crrt["Precision"] == 0.05), "Threshold"].values[0]


# %%

predictions["suggested_mv"] = (predictions["no mv-mv"] >= threshold_mv).astype(int)
predictions["suggested_vp"] = (predictions["no vp- vp"] >= threshold_vp).astype(int)
predictions["suggested_crrt"] = (predictions["no crrt-crrt"] >= threshold_vp).astype(int)

alerts = predictions[predictions["alert"] == 1]

alerts = alerts.reset_index().rename({"index": "alert_id"}, axis=1)
alert_ids = alerts["alert_id"].values.reshape((-1,1))

alert_ids = np.repeat(alert_ids, 512, axis=0)
alert_ids = alert_ids.reshape((len(alerts), 512, 1))

# %%

# Load model

from apricotm import ApricotM
import torch

# Load model architecture

model_architecture = torch.load(f"{MODEL_DIR}/apricotm_architecture.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device {DEVICE}")

model = ApricotM(
    d_model=model_architecture["d_model"],
    d_hidden=model_architecture["d_hidden"],
    d_input=model_architecture["d_input"],
    d_static=model_architecture["d_static"],
    max_code=model_architecture["max_code"],
    n_layer=model_architecture["n_layer"],
    device=DEVICE,
    dropout=model_architecture["dropout"],
).to(DEVICE)


# Load model weights

model.load_state_dict(torch.load(f"{MODEL_DIR}/apricotm_weights.pth"))

# Load best parameters

import pickle

with open("%s/best_params.pkl" % MODEL_DIR, "rb") as f:
    best_params = pickle.load(f)

seq_len = best_params["seq_len"]

#%%

with open(f"{OUTPUT_DIR}/model/variable_mapping.pkl", "rb") as f:
    variable_map = pickle.load(f)

char_dict = {
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
    "EtCO2": "EtCO2",
    "Glucose (serum)": "Glucose",
    "Heart Rate": "Heart rate",
    "Hematocrit (whole blood - calc)": "Hematocrit",
    "Hemoglobin": "Hemoglobin",
    "INR": "INR",
    "Inspired O2 Fraction": "Inspired O2 Fraction",
    "Ionized Calcium": "Calcium ionized",
    "Lactic Acid": "Lactate",
    "Non Invasive Blood Pressure diastolic": "DBP",
    "Non Invasive Blood Pressure systolic": "SBP",
    "O2 Flow": "O2 flow",
    "O2 saturation pulseoxymetry": "SPO2",
    "PH (Arterial)": "Arterial PH",
    "Peak Insp. Pressure": "PIP",
    "Platelet Count": "Platelets",
    "Potassium (serum)": "Potassium",
    "Respiratory Rate": "Respiratory rate",
    "Sodium (serum)": "Sodium",
    "Specific Gravity (urine)": "Specific gravity urine",
    "Temperature Celsius": "Body temperature",
    "Tidal Volume (observed)": "Tidal volume",
    "Total Bilirubin": "Bilirubin total",
    "Total PEEP Level": "PEEP",
    "Troponin-T": "Troponin-T",
    "WBC": "WBC",
    "cam": "CAM",
    "gcs": "GCS",
    "rass": "RASS",
    "Heparin Sodium": "Heparin sodium",
    "Propofol": "Propofol",
    "Phenylephrine": "Phenylephrine",
    "Folic Acid": "Folic acid",
    "Norepinephrine": "Norepinephrine",
    "Amiodarone": "Amiodarone",
    "Fentanyl": "Fentanyl",
    "Dexmedetomidine (Precedex)": "Dexmedetomidine",
    "Digoxin (Lanoxin)": "Digoxin",
    "Vasopressin": "Vasopressin",
    "Epinephrine": "Epinephrine",
    "Dopamine": "Dopamine",
}

char_static_dict = {
    "age": "Age",
    "sex": "Gender",
    "race": "Race",
    "bmi": "BMI",
    "aids_poa": "AIDS",
    "cancer_poa": "Cancer",
    "cerebrovascular_poa": "Cerebrovascular Disease",
    "chf_poa": "CHF",
    "copd_poa": "COPD",
    "dementia_poa": "Dementia",
    "diabetes_w_o_complications_poa": "Diabetes w/o Complications",
    "diabetes_w_complications_poa": "Diabetes with Complications",
    "m_i_poa": "Myocardial Infarction",
    "metastatic_carcinoma_poa": "Metastatic Carcinoma",
    "mild_liver_disease_poa": "Mild Liver Disease",
    "moderate_severe_liver_disease_poa": "Moderate/Severe Liver Disease",
    "paraplegia_hemiplegia_poa": "Paraplegia/Hemiplegia",
    "peptic_ulcer_disease_poa": "Peptic Ulcer Disease",
    "peripheral_vascular_disease_poa": "Peripheral Vascular Disease",
    "renal_disease_poa": "Renal Disease",
    "rheumatologic_poa": "Rheumatologic Disease",
    "charlson_comorbidity_total_score": "CCI",
}


#%%

# Interpretability all sets

ig_variables = attr.LayerIntegratedGradients(model, model.embedding_variable)
ig_values_time = attr.LayerIntegratedGradients(model, model.conv1d)
ig_static = attr.IntegratedGradients(model.ffn_static)


integrated_gradients = attr.IntegratedGradients(model)

from torch.autograd import Variable

BATCH_SIZE = 4
    
print(f"Computing integrated gradients for prospective")

with h5py.File(f"{PROSP_DATA_DIR}/dataset.h5", "r") as f:
    data = f["prospective"]
    X = data["X"][:]
    static = data["static"][:]
    y_trans = data["y_trans"][:]
    y_main = data["y_main"][:]

y = np.concatenate([y_main, y_trans], axis=1)

X = X[sample_ids,:,:]
y = y[sample_ids,:]
static = static[sample_ids,:]

# # Find indices where values in X[:,:,3] are in subsamples
# samples = np.where(np.isin(X[:, 0, 3], subsample))[0]

# X = X[samples]
# y = y[samples]
# static = static[samples]

targets = [
    "alert",
    "suggested_mv",
    "suggested_vp",
    "suggested_crrt",
]

target_imp = [5, 7, 9, 11]

for i, target in zip(target_imp, targets):
    
    condition = (predictions["alert"] == 1).values
    X_sub = X[condition]
    y_sub = y[condition]
    static_sub = static[condition]
    predictions_sub = predictions[condition]
    
    variable_codes = X_sub[:, :, 1].copy()
    values = X_sub[:, :, 2].copy()
    
    icustay_ids = X_sub[:, :, 3].copy()
    
    X_sub = np.concatenate([X_sub, alert_ids], axis=2)
    
    alert_ids_flat = X_sub[:, :, 4].copy()
    
    ig_importance_static = np.zeros((len(X_sub), model_architecture["d_static"]))
    ig_importance_variable = np.zeros((len(X_sub), seq_len))
    ig_importance_value_time = np.zeros((len(X_sub), seq_len))
    
    X_sub = torch.FloatTensor(X_sub).to(DEVICE)
    static_sub = torch.FloatTensor(static_sub)

    print(f"Computing integrated gradients for {target}")

    for patient in tqdm.trange(0, len(X_sub), BATCH_SIZE):
        inputs = Variable(X_sub[patient : patient + BATCH_SIZE]).to(DEVICE)
        static_input = Variable(static_sub[patient : patient + BATCH_SIZE]).to(DEVICE)

        attributions_ig, _ = ig_variables.attribute(
            (inputs, static_input), target=i, return_convergence_delta=True
        )

        attributions_ig = attributions_ig.sum(dim=2).squeeze(0)
        attributions_ig = attributions_ig / torch.norm(attributions_ig)

        ig_importance_variable[
            patient : patient + BATCH_SIZE, :
        ] = attributions_ig.to("cpu").numpy()

        attributions_ig, _ = ig_values_time.attribute((inputs, static_input), target=i, return_convergence_delta=True)

        attributions_ig = attributions_ig.sum(dim=1).squeeze(0)
        attributions_ig = attributions_ig / torch.norm(attributions_ig)

        ig_importance_value_time[patient:patient+BATCH_SIZE,:] = attributions_ig.to('cpu').numpy()

        attributions_ig, _ = ig_static.attribute(
            static_input, target=i, return_convergence_delta=True
        )

        ig_importance_static[
            patient : patient + BATCH_SIZE, :
        ] = attributions_ig.to("cpu").numpy()


    ig_temporal = pd.DataFrame(
        {
            "icustay_id": icustay_ids.flatten(),
            "alert_id": alert_ids_flat.flatten(),
            "code": variable_codes.flatten(),
            "ig_variable": ig_importance_variable.flatten(),
            "ig_value_time": ig_importance_value_time.flatten(),
            "value": values.flatten(),
        }
    )
    
    ig_temporal["variable"] = ig_temporal["code"].map(variable_map)

    ig_temporal["variable"] = ig_temporal["variable"].replace(char_dict)
            
    ig_temporal = ig_temporal.merge(alerts[["alert_id", target]], on="alert_id", how="left")
    
    # Sort ig_value_time based on the target value
    ig_temporal = ig_temporal.groupby("alert_id").apply(lambda x: x.sort_values(by="ig_value_time", ascending=x[target].max() == 0)).reset_index(drop=True)
    
    # ig_temporal = ig_temporal.sort_values(by=["alert_id", "ig_value_time"], ascending=False)
    ig_temporal = ig_temporal.drop_duplicates(subset=["alert_id", "variable"], keep="first")

    top_factors = ig_temporal.groupby('alert_id').head(3).reset_index(drop=True)
    
    # Add a rank column to assign a rank to the factors for each alert_id
    top_factors['rank'] = top_factors.groupby('alert_id').cumcount() + 1

    # Pivot the table to create separate columns for each factor
    top_factors_pivoted = top_factors.pivot(index='alert_id', columns='rank', values='variable').reset_index()

    # Rename columns to have more descriptive names
    top_factors_pivoted.columns = ['alert_id', f'Factor 1_{target}', f'Factor 2_{target}', f'Factor 3_{target}']
    
    alerts = alerts.merge(top_factors_pivoted, on="alert_id", how="inner")

# %%

# alerts = alerts.rename({"stable-unstable": "risk_score"}, axis=1)
# alerts["risk_score"] = alerts["stable-unstable"] + alerts["unstable"]
alerts["risk_score"] = alerts["unstable"]

gt_labels = alerts[["patient_deiden_id", "icustay_id", "time", "gt_unstable", "gt_mv", "gt_vp", "gt_crrt"]].drop_duplicates()

gt_labels.to_csv("gt_labels.csv", index=False)

# factor_columns = []

# for target in targets:
    
#     factor_columns += [f"Factor {i}_{target}" for i in range(1, 4)]

# alerts = alerts[[
#     "patient_deiden_id",
#     "icustay_id",
#     "enter_datetime",
#     "time",
#     "risk_score",
#     "suggested_mv",
#     "suggested_vp",
#     "suggested_crrt"] + factor_columns]

# alerts["risk_score"] = (alerts["risk_score"]*100).astype(int)

# therapies = ["mv", "vp", "crrt"]

# for therapy in therapies:
    
#     # for i in range(1, 4):
    
#     #     alerts.loc[alerts[f"suggested_{therapy}"] == 0, f"Factor {i}_suggested_{therapy}"] = ""
        
#     alerts[f"suggested_{therapy}"] = alerts[f"suggested_{therapy}"].map({0: "Not Recommended", 1: "Recommended"})
    
    
# alerts.to_csv("alerts.csv", index=False)

# %%
