#%%

# Import libraries

import pandas as pd
import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt
import tqdm

import torch
import captum.attr as attr

from variables import MODEL_DIR, OUTPUT_DIR

import os

if not os.path.exists(f"{MODEL_DIR}/integrated_gradients"):
    os.makedirs(f"{MODEL_DIR}/integrated_gradients")

#%%

# Load model

from apricotm import ApricotM
import torch

# Load model architecture

model_architecture = torch.load(f"{MODEL_DIR}/apricotm_architecture.pth")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

cohorts = ["validation", "external_test", "temporal_test"]

BATCH_SIZE = 50

for cohort in cohorts:
    
    print(f"Computing integrated gradients for {cohort}")

    with h5py.File(f"{OUTPUT_DIR}/final_data/dataset.h5", "r") as f:
        data = f[cohort]
        X = data["X"][:]
        static = data["static"][:]
        y_trans = data["y_trans"][:]
        y_main = data["y_main"][:]

    y = np.concatenate([y_main, y_trans], axis=1)

    static = static[:, 1:]

    samples = 10000

    random_sample = np.random.choice(len(X), samples, replace=False)

    X = X[random_sample]
    y = y[random_sample]
    static = static[random_sample]

    variable_codes = X[:, :, 1].copy()
    values = X[:, :, 2].copy()

    X = torch.FloatTensor(X).to(DEVICE)
    static = torch.FloatTensor(static)

    targets = [
        "discharge",
        "stable",
        "unstable",
        "deceased",
        "unstable-stable",
        "stable-unstable",
        "mv-no mv",
        "no mv-mv",
        "vp-no vp",
        "no vp-vp",
        "crrt-no crrt",
        "no crrt-crrt",
    ]
    
    target_imp = [0, 1, 2, 3, 4, 5, 7, 9, 11]

    ig_importance_static = np.zeros((samples, model_architecture["d_static"]))
    ig_importance_variable = np.zeros((samples, seq_len))
    ig_importance_value_time = np.zeros((samples, seq_len))

    for i in target_imp:
        
        print(f"Computing integrated gradients for {targets[i]}")

        for patient in tqdm.trange(0, samples, BATCH_SIZE):
            inputs = Variable(X[patient : patient + BATCH_SIZE]).to(DEVICE)
            static_input = Variable(static[patient : patient + BATCH_SIZE]).to(DEVICE)

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
                "code": variable_codes.flatten(),
                "ig_variable": ig_importance_variable.flatten(),
                "ig_value_time": ig_importance_value_time.flatten(),
                "value": values.flatten(),
            }
        )

        static_feat = (
            pd.read_csv(f"{OUTPUT_DIR}/final_data/static.csv").columns[1:].tolist()
        )

        ig_static_df = pd.DataFrame(ig_importance_static, columns=static_feat)

        ig_static_df = (
            ig_static_df.abs()
            .mean(axis=0)
            .reset_index()
            .rename({"index": "variable", 0: "ig_variable"}, axis=1)
        )

        ig_static_df["variable"] = ig_static_df["variable"].replace(char_static_dict)

        ig_temporal["variable"] = ig_temporal["code"].map(variable_map)

        ig_temporal["variable"] = ig_temporal["variable"].replace(char_dict)

        ig_temporal["ig_variable"] = ig_temporal["ig_variable"].abs()

        ig_temporal['ig_value_time'] = ig_temporal['ig_value_time'].abs()
        
        ig_temporal["ig_variable"] = ig_temporal["ig_variable"] + ig_temporal['ig_value_time']

        ig_temporal = (
            ig_temporal.groupby("variable")["ig_variable"].mean().reset_index()
        )

        ig_temporal = pd.concat([ig_temporal, ig_static_df])

        ig_temporal = ig_temporal.sort_values(by="ig_variable", ascending=False)

        ig_temporal.to_csv(
            f"{MODEL_DIR}/integrated_gradients/{cohort}_integrated_gradients_{targets[i]}.csv",
            index=None,
        )
