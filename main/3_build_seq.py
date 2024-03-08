#%%

# Import libraries

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = '/home/contreras.miguel/deepacu'

#%%

# Build eICU sequential

icustay_ids = pd.read_csv('%s/final_data/admissions_eicu.csv' % DATA_DIR, usecols=['patientunitstayid'])
icustay_ids = icustay_ids['patientunitstayid'].tolist()

vitals = pd.read_csv('%s/eicu/vitals.csv' % DATA_DIR, usecols=['patientunitstayid', 'nursingchartoffset', 'nursingchartcelltypevalname', 'nursingchartvalue'])

use_vitals = ['Heart Rate', 'Respiratory Rate', 'O2 Saturation', 'Non-Invasive BP Diastolic', 'Non-Invasive BP Systolic', 'Temperature (C)', 'Temperature (F)', 'End Tidal CO2']

vitals = vitals[vitals['nursingchartcelltypevalname'].isin(use_vitals)]

vitals['nursingchartvalue'] = vitals['nursingchartvalue'].astype(float)

def fahrenheit_to_celsius(fahrenheit):
    celsius = (fahrenheit - 32) * 5 / 9
    return celsius

vitals.loc[(vitals['nursingchartcelltypevalname'] == 'Temperature (F)'), 'nursingchartvalue'] = fahrenheit_to_celsius(vitals.loc[(vitals['nursingchartcelltypevalname'] == 'Temperature (F)'), 'nursingchartvalue'])

vitals['nursingchartcelltypevalname'] = vitals['nursingchartcelltypevalname'].replace({'Temperature (F)': 'Temperature (C)'})

vitals = vitals.rename({'patientunitstayid': 'icustay_id', 'nursingchartoffset': 'hours', 'nursingchartcelltypevalname': 'label', 'nursingchartvalue': 'value'}, axis=1)

vitals = vitals[['icustay_id', 'hours', 'label', 'value']]

vitals['hours'] = vitals['hours'] / 60

vitals['value'] = vitals['value'].astype(float)

labs = pd.read_csv('%s/eicu/lab.csv' % DATA_DIR, usecols=['patientunitstayid', 'labresultoffset', 'labname', 'labresult'])

labs = labs.rename({'patientunitstayid': 'icustay_id', 'labresultoffset': 'hours', 'labname': 'label', 'labresult': 'value'}, axis=1)

labs = labs.dropna(subset=['value'])

labs['hours'] = labs['hours'] / 60

labs['value'] = labs['value'].astype(float)

seq_eicu = pd.concat([vitals, labs], axis=0)

del labs
del vitals

resp = pd.read_csv('%s/eicu/respiratoryCharting.csv' % DATA_DIR, usecols=['patientunitstayid', 'respchartoffset', 'respchartvaluelabel', 'respchartvalue'])

use_resp = ['FiO2', 'PEEP', 'Tidal Volume Observed (VT)', 'Peak Insp. Pressure', 'Oxygen Flow Rate']

resp = resp[resp['respchartvaluelabel'].isin(use_resp)]

resp = resp.rename({'patientunitstayid': 'icustay_id', 'respchartoffset': 'hours', 'respchartvaluelabel': 'label', 'respchartvalue': 'value'}, axis=1)

resp = resp.dropna(subset=['value'])

resp['hours'] = resp['hours'] / 60

resp['value'] = resp['value'].str.replace('%', '')

resp['value'] = resp['value'].astype(float)

seq_eicu = pd.concat([seq_eicu, resp], axis=0)

del resp

meds = pd.read_csv('%s/eicu/infusionDrug.csv' % DATA_DIR, usecols=['patientunitstayid', 'infusionoffset', 'drugname', 'drugamount'])

meds = meds.rename({'patientunitstayid': 'icustay_id', 'infusionoffset': 'hours', 'drugname': 'label', 'drugamount': 'value'}, axis=1)

meds = meds.dropna(subset=['value'])

meds.loc[meds['label'].str.contains(r'\b{}\b'.format('norepinephrine'), case=False, regex=True), 'label'] = 'Norepinephrine'
meds.loc[meds['label'].str.contains(r'\b{}\b'.format('dopamine'), case=False, regex=True), 'label'] = 'Dopamine'
meds.loc[meds['label'].str.contains(r'\b{}\b'.format('epinephrine'), case=False, regex=True), 'label'] = 'Epinephrine'
meds.loc[meds['label'].str.contains(r'\b{}\b'.format('phenylephrine'), case=False, regex=True), 'label'] = 'Phenylephrine'
meds.loc[meds['label'].str.contains(r'\b{}\b'.format('vasopressin'), case=False, regex=True), 'label'] = 'Vasopressin'

meds['hours'] = meds['hours'] / 60

meds['value'] = meds['value'].astype(float)

meds['value'] = 1

seq_eicu = pd.concat([seq_eicu, meds], axis=0)

del meds

scores = pd.read_csv('%s/eicu/scores.csv' % DATA_DIR, usecols=['patientunitstayid', 'nursingchartoffset', 'nursingchartcelltypevalname', 'nursingchartvalue'])

cam = scores[scores['nursingchartcelltypevalname'] == 'Delirium Score']
cam = cam[(cam['nursingchartvalue'] == 'Yes') | (cam['nursingchartvalue'] == 'No')]
cam['nursingchartvalue'] = cam['nursingchartvalue'].map({'Yes': 1, 'No': 0})
cam['nursingchartcelltypevalname'] = 'CAM'

gcs = scores[scores['nursingchartcelltypevalname'] == 'GCS Total']
gcs = gcs[gcs['nursingchartvalue'] != 'Unable to score due to medication']
gcs = gcs.dropna(subset=['nursingchartvalue'])
gcs['nursingchartvalue'] = gcs['nursingchartvalue'].astype(int)
gcs = gcs[(gcs['nursingchartvalue'] <= 15) & (gcs['nursingchartvalue'] >= 0)]
gcs['nursingchartcelltypevalname'] = 'GCS'

rass = scores[scores['nursingchartcelltypevalname'] == 'Sedation Score']
rass['nursingchartvalue'] = rass['nursingchartvalue'].astype(int)
rass = rass[(rass['nursingchartvalue'] <= 4) & (rass['nursingchartvalue'] >= -5)]
rass['nursingchartcelltypevalname'] = 'RASS'

del scores

scores = pd.concat([cam, gcs, rass], axis=0)

scores = scores.rename({'patientunitstayid': 'icustay_id', 'nursingchartoffset': 'hours', 'nursingchartcelltypevalname': 'label', 'nursingchartvalue': 'value'}, axis=1)

scores = scores[['icustay_id', 'hours', 'label', 'value']]

scores['hours'] = scores['hours'] / 60

scores['value'] = scores['value'].astype(float)

seq_eicu = pd.concat([seq_eicu, scores], axis=0)

del scores, cam, gcs, rass

seq_eicu = seq_eicu.sort_values(by=['icustay_id', 'hours'])

seq_eicu = seq_eicu[seq_eicu['hours'] >= 0]

seq_eicu = seq_eicu[seq_eicu['icustay_id'].isin(icustay_ids)]

def remove_outliers(group):
    
    # Q1 = group['value'].quantile(0.25)
    # Q3 = group['value'].quantile(0.75)
    # IQR = Q3 - Q1
    # lower_threshold = Q1 - 1.5 * IQR
    # upper_threshold = Q3 + 1.5 * IQR
    
    lower_threshold = group['value'].quantile(0.01)
    upper_threshold = group['value'].quantile(0.99)

    group_filtered = group[(group['value'] >= lower_threshold) & (group['value'] <= upper_threshold)]

    return group_filtered

seq_eicu = seq_eicu.groupby('label').apply(remove_outliers)

seq_eicu = seq_eicu.reset_index(drop=True)

seq_eicu = seq_eicu.sort_values(by=['icustay_id', 'hours'])

map_var = pd.read_csv('%s/final_var_map.csv' % DATA_DIR)
feat_eicu = map_var['eicu'].tolist()

seq_eicu = seq_eicu[seq_eicu['label'].isin(feat_eicu)]

seq_eicu = seq_eicu.rename({'label': 'variable'}, axis=1)

print(len(seq_eicu['variable'].unique()))
print(len(seq_eicu['icustay_id'].unique()))

# vars_eicu = seq_eicu['variable'].unique()

# %%

# Build MIMIC sequential

icustay_ids = pd.read_csv('%s/final_data/admissions_mimic.csv' % DATA_DIR, usecols=['stay_id'])
icustay_ids = icustay_ids['stay_id'].tolist()

admissions = pd.read_csv('%s/mimic/icustays.csv' % DATA_DIR, usecols=['subject_id', 'stay_id', 'intime', 'outtime'])
ditems = pd.read_csv('%s/mimic/d_items.csv' % DATA_DIR)

labs_vitals = pd.read_csv('%s/mimic/all_events.csv' % DATA_DIR, usecols=['subject_id', 'stay_id', 'charttime', 'itemid', 'valuenum'])
labs_vitals = pd.concat([labs_vitals, pd.read_csv('mimic/temp_conv.csv', usecols=['subject_id', 'stay_id', 'charttime', 'itemid', 'valuenum'])])

labs_vitals = labs_vitals.merge(admissions, on=['stay_id', 'subject_id'])

labs_vitals['hours'] = (pd.to_datetime(labs_vitals['charttime']) - pd.to_datetime(labs_vitals['intime'])) / np.timedelta64(1, 'h')


labitems = ditems[ditems['itemid'].isin(labs_vitals['itemid'].unique())]
labitems_ids = labitems['itemid'].unique()

map_labs = dict()

for i in range(len(labitems_ids)):
    map_labs[labitems_ids[i]] = labitems[labitems['itemid'] == labitems_ids[i]]['label'].values[0]

labs_vitals['label'] = labs_vitals['itemid'].map(map_labs)

def remove_outliers(group):
    
    # Q1 = group['valuenum'].quantile(0.25)
    # Q3 = group['valuenum'].quantile(0.75)
    # IQR = Q3 - Q1
    # lower_threshold = Q1 - 1.5 * IQR
    # upper_threshold = Q3 + 1.5 * IQR
    
    lower_threshold = group['valuenum'].quantile(0.01)
    upper_threshold = group['valuenum'].quantile(0.99)
    
    group_filtered = group[(group['valuenum'] >= lower_threshold) & (group['valuenum'] <= upper_threshold)]

    return group_filtered

labs_vitals = labs_vitals.groupby('label').apply(remove_outliers)

labs_vitals = labs_vitals.reset_index(drop=True)

lab_variables = list(labs_vitals['label'].unique())

meds = pd.read_csv('%s/mimic/med_events.csv' % DATA_DIR)

meds = meds.merge(admissions, on=['stay_id', 'subject_id'])


meds['hours'] = (pd.to_datetime(meds['starttime']) - pd.to_datetime(meds['intime'])) / np.timedelta64(1, 'h')

meditems = ditems[ditems['itemid'].isin(meds['itemid'].unique())]
meditems_ids = meditems['itemid'].unique()

map_meds = dict()

for i in range(len(meditems_ids)):
    map_meds[meditems_ids[i]] = meditems[meditems['itemid'] == meditems_ids[i]]['label'].values[0]

meds['label'] = meds['itemid'].map(map_meds)

def remove_outliers(group):
    
    # Q1 = group['amount'].quantile(0.25)
    # Q3 = group['amount'].quantile(0.75)
    # IQR = Q3 - Q1
    # lower_threshold = Q1 - 1.5 * IQR
    # upper_threshold = Q3 + 1.5 * IQR
    
    lower_threshold = group['amount'].quantile(0.01)
    upper_threshold = group['amount'].quantile(0.99)
    
    group_filtered = group[(group['amount'] >= lower_threshold) & (group['amount'] <= upper_threshold)]

    return group_filtered

meds = meds.groupby('label').apply(remove_outliers)

meds = meds.rename({'amount': 'valuenum'}, axis=1)

meds['valuenum'] = 1

labs_vitals = labs_vitals.loc[:, ['stay_id', 'hours', 'label', 'valuenum']]
meds = meds.loc[:, ['stay_id', 'hours', 'label', 'valuenum']]

labs_vitals = labs_vitals.rename({'label': 'variable'}, axis=1)
labs_vitals = labs_vitals.rename({'valuenum': 'value'}, axis=1)

meds = meds.rename({'label': 'variable'}, axis=1)
meds = meds.rename({'valuenum': 'value'}, axis=1)

scores = pd.read_csv('%s/mimic/scores_raw.csv' % DATA_DIR)
scores = scores.merge(admissions, on=['stay_id'])

scores['hours'] = (pd.to_datetime(scores['charttime']) - pd.to_datetime(scores['intime'])) / np.timedelta64(1, 'h')
scores = scores.loc[:, ['stay_id', 'hours', 'variable', 'value']]

seq_mimic = pd.concat([labs_vitals, meds, scores], axis=0).sort_values(by=['stay_id', 'hours'])

del labs_vitals, meds, scores

seq_mimic = seq_mimic[seq_mimic['stay_id'].isin(icustay_ids)]

seq_mimic = seq_mimic[seq_mimic['hours'] >= 0]

seq_mimic = seq_mimic.rename({'stay_id': 'icustay_id'}, axis=1)

map_var = pd.read_csv('%s/final_var_map.csv' % DATA_DIR)
feat_mimic = map_var['mimic'].tolist()

seq_mimic = seq_mimic[seq_mimic['variable'].isin(feat_mimic)]

print(len(seq_mimic['variable'].unique()))
print(len(seq_mimic['icustay_id'].unique()))

# %%

# Build UF sequential

icustay_ids = pd.read_csv('%s/final_data/admissions_uf.csv' % DATA_DIR, usecols=['icustay_id'])
icustay_ids = icustay_ids['icustay_id'].tolist()

seq_uf = pd.read_csv('%s/uf/seq.csv' % DATA_DIR)

seq_uf = seq_uf[seq_uf['icustay_id'].isin(icustay_ids)]

map_var = pd.read_csv('%s/final_var_map.csv' % DATA_DIR)
feat_uf = map_var['uf'].tolist()

seq_uf = seq_uf[seq_uf['variable'].isin(feat_uf)]

def remove_outliers(group):
    
    # Q1 = group['value'].quantile(0.25)
    # Q3 = group['value'].quantile(0.75)
    # IQR = Q3 - Q1
    # lower_threshold = Q1 - 1.5 * IQR
    # upper_threshold = Q3 + 1.5 * IQR
    
    lower_threshold = group['value'].quantile(0.01)
    upper_threshold = group['value'].quantile(0.99)
    
    group_filtered = group[(group['value'] >= lower_threshold) & (group['value'] <= upper_threshold)]

    return group_filtered

seq_uf = seq_uf.groupby('variable').apply(remove_outliers).reset_index(drop=True)

meds = [med for med in seq_uf['variable'].unique() if 'med_' in med]

seq_uf.loc[seq_uf['variable'].isin(meds), 'value'] = 1

print(len(seq_uf['variable'].unique()))
print(len(seq_uf['icustay_id'].unique()))

seq_uf.drop('variable_code', axis=1, inplace=True)

# %%

# Standardize variables

map_var = pd.read_csv('%s/final_var_map.csv' % DATA_DIR)
map_uf_mimic = {}

for i in range(len(map_var)):
    map_uf_mimic[map_var.loc[i, 'uf']] = map_var.loc[i, 'mimic']

seq_uf['variable'] = seq_uf['variable'].map(map_uf_mimic)

print(len(seq_uf['variable'].unique()))

map_eicu_mimic = {}

for i in range(len(map_var)):
    map_eicu_mimic[map_var.loc[i, 'eicu']] = map_var.loc[i, 'mimic']

seq_eicu['variable'] = seq_eicu['variable'].map(map_eicu_mimic)

print(len(seq_eicu['variable'].unique()))


seq = pd.concat([seq_eicu, seq_mimic, seq_uf])

del seq_mimic, seq_eicu, seq_uf

seq = seq.dropna()

print(len(seq['variable'].unique()))
print(len(seq['icustay_id'].unique()))

seq = seq.sort_values(by=['icustay_id', 'hours']).reset_index(drop=True)

seq['interval'] = ((seq['hours'] // 4) + 1).astype(int)

def convert_variables_to_indices(seq):
    print("* Converting variables to indices...")
    var_idx, var_label = pd.factorize(seq["variable"])
    var_idx = var_idx + 1  # 0 will be for padding
    seq["variable_code"] = var_idx

    variable_code_mapping = {i + 1: label for (i, label) in enumerate(var_label)}
    variable_code_mapping[0] = "<PAD>"
    return seq, variable_code_mapping

seq, variable_mapping = convert_variables_to_indices(seq)

with open("%s/model/variable_mapping.pkl" % DATA_DIR, "wb") as f:
    pickle.dump(variable_mapping, f, protocol=2)

seq['shift_id'] = seq['icustay_id'].astype(str) + '_' + seq['interval'].astype(str)

seq.drop('interval', axis=1, inplace=True)

with open("%s/final_data/ids.pkl" % DATA_DIR, "rb") as f:
    ids = pickle.load(f)
    ids_train = ids["train"]

ids_train = ids_train[0]

SET_TRAIN = set(ids_train)
VALUE_COLS = ["value"]

scalers = {}

def _standardize_variable(group):
    ids_present = set(group["icustay_id"].unique())
    train_ids_present = ids_present.intersection(SET_TRAIN)
    variable_value = group["variable"].values[0]
    
    variable_code = group['variable_code'].values[0]
    
    if len(train_ids_present) > 0 and variable_value != 'mob_level_of_assistance' and variable_value != 'brain_status':
        scaler = MinMaxScaler()
        scaler.fit(group[group["icustay_id"].isin(train_ids_present)][VALUE_COLS])
        # scaler = scalers[f'scaler{variable_code}']
        group[VALUE_COLS] = scaler.transform(group[VALUE_COLS])
        scalers[f'scaler{variable_code}'] = scaler


    elif len(train_ids_present) == 0 and variable_value != 'mob_level_of_assistance' and variable_value != 'brain_status':
        print("Debug standardize")
        scaler = MinMaxScaler()
        group[VALUE_COLS] = scaler.fit_transform(group[VALUE_COLS])
        # scaler = scalers[f'scaler{variable_code}']
        # group[VALUE_COLS] = scaler.transform(group[VALUE_COLS])
        scalers[f'scaler{variable_code}'] = scaler


    return group


def standardize_variables(df):

    variables = df["variable"].unique()
    print(f'Variables after imputation: {len(variables)}')
    
    df = df.groupby("variable").apply(_standardize_variable).sort_values(by=["icustay_id", "hours"])
    variables = df["variable"].unique()
    print(f'Variables after standardization: {len(variables)}')

    return df


def get_hours_scaler(df):
    hours = df[["icustay_id", "hours"]]
    scaler = MinMaxScaler()
    scaler.fit(hours[hours["icustay_id"].isin(ids_train)][["hours"]])
    return scaler

print("* Scaling seq...")
hours = seq[["icustay_id", "hours"]]
hours_scaler = get_hours_scaler(seq)

scalers['scaler_hours'] = hours_scaler

seq = standardize_variables(seq)

print(len(seq['icustay_id'].unique()))
seq = seq.reset_index(drop=True)

seq[["hours"]] = hours_scaler.transform(seq[["hours"]])

seq.drop('variable', axis=1, inplace=True)

seq.to_csv("%s/final_data/seq.csv" % DATA_DIR, index=None)

pickle.dump(scalers, open('%s/model/scalers_seq.pkl' % DATA_DIR, 'wb'))

# %%
