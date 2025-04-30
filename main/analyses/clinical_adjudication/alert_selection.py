#%%
import pandas as pd
import random

alerts = pd.read_csv("alerts.csv")
gt_labels = pd.read_csv("gt_labels.csv")

alerts["alert_number"] = alerts.groupby("icustay_id").cumcount() + 1

# Keep only first alert

# alerts = alerts.drop_duplicates(subset=["patient_deiden_id"], keep="first").reset_index(drop=True)
alerts = alerts.reset_index()
alerts = alerts.rename({"index": "alert_id"}, axis=1)

gt_labels = gt_labels.reset_index()
gt_labels = gt_labels.rename({"index": "alert_id"}, axis=1)

# Function to balance sampling
def balanced_sample(survey_table, col, n_samples):
    recommended = survey_table[survey_table[col] == 'Recommended']
    not_recommended = survey_table[survey_table[col] == 'Not Recommended']
    
    n_each = n_samples // 2  # Equal split

    sampled_recommended = recommended.sample(min(n_each, len(recommended)), random_state=42)
    sampled_not_recommended = not_recommended.sample(min(n_each, len(not_recommended)), random_state=42)
    
    return pd.concat([sampled_recommended, sampled_not_recommended])

# Balancing for each column
sampled_mv = balanced_sample(alerts, 'suggested_mv', 20)
sampled_vp = balanced_sample(alerts, 'suggested_vp', 20)
sampled_crrt = balanced_sample(alerts, 'suggested_crrt', 20)

# Combining samples and ensuring uniqueness while selecting 20 final samples
final_sample = pd.concat([sampled_mv, sampled_vp, sampled_crrt]).drop_duplicates().sample(20, replace=True, random_state=42)

# Display or save the final sample
print(final_sample)

# Get internal stations

internal_stations = pd.read_csv("/home/contreras.miguel/daily_data/Cleaned/IICU_210501_240801/internal_stations_clean_0.csv")
internal_stations = internal_stations[internal_stations["patient_deiden_id"].isin(alerts["patient_deiden_id"].unique())].reset_index(drop=True)

internal_stations = internal_stations[["patient_deiden_id", "encounter_deiden_id", "enter_datetime", "exit_datetime", "location_type", "to_station", "to_station_class"]].merge(alerts[["patient_deiden_id", "alert_id", "time", "risk_score", "suggested_vp"]], on="patient_deiden_id", how="inner")

internal_stations["time"] = pd.to_datetime(internal_stations["time"])
internal_stations["enter_datetime"] = pd.to_datetime(internal_stations["enter_datetime"])
internal_stations["exit_datetime"] = pd.to_datetime(internal_stations["exit_datetime"])

internal_stations = internal_stations[(internal_stations["time"] >= internal_stations["enter_datetime"]) & (internal_stations["time"] <= internal_stations["exit_datetime"])].reset_index(drop=True)

print(internal_stations["location_type"].value_counts())
print(internal_stations["to_station_class"].value_counts())
print(alerts["suggested_vp"].value_counts())


#%%

# Get OR case schedule

or_case_1 = pd.read_csv("/data/daily_data/202101013_datadump/IICU_210501_231031/or_case_schedule_0.csv")
or_case_2 = pd.read_csv("/home/contreras.miguel/daily_data/Raw/IICU_231031_240801/or_case_schedule_0.csv")

or_case = pd.concat([or_case_1, or_case_2], axis=0).reset_index(drop=True)

# or_case = or_case[["patient_deiden_id", "sched_start_datetime", "procedure_end_datetime", "surgery_type"]].merge(alerts[["patient_deiden_id", "alert_id", "time", "risk_score"]], on="patient_deiden_id", how="inner")

or_case = or_case[["patient_deiden_id", "sched_start_datetime", "procedure_end_datetime", "surgery_type"]].merge(internal_stations, on="patient_deiden_id", how="inner")


or_case["sched_start_datetime"] = pd.to_datetime(or_case["sched_start_datetime"])
or_case["procedure_end_datetime"] = pd.to_datetime(or_case["procedure_end_datetime"])

or_case = or_case[(or_case["procedure_end_datetime"] <= or_case["exit_datetime"])].reset_index(drop=True)

or_case = or_case[(or_case["time"] >= or_case["enter_datetime"]) & (or_case["time"] <= or_case["procedure_end_datetime"])].reset_index(drop=True)

print(or_case["surgery_type"].value_counts())

internal_stations = internal_stations[internal_stations["location_type"] == "ICU"]
internal_stations = internal_stations[internal_stations["to_station_class"] == "Adult Unit"]

map_ids = internal_stations[["alert_id", "encounter_deiden_id"]]
map_ids = pd.Series(data=map_ids["encounter_deiden_id"].values, index=map_ids["alert_id"].values)

alerts = alerts[alerts["alert_id"].isin(internal_stations["alert_id"])]
alerts["encounter_deiden_id"] = alerts["alert_id"].map(map_ids)

alerts = alerts[~alerts["alert_id"].isin(or_case["alert_id"])]

gt_labels = gt_labels[gt_labels["alert_id"].isin(internal_stations["alert_id"])]
gt_labels["encounter_deiden_id"] = gt_labels["alert_id"].map(map_ids)

gt_labels = gt_labels[~gt_labels["alert_id"].isin(or_case["alert_id"])]

#%%

id_mapping = pd.read_excel("/data/daily_data/patient_id_mapping.xlsx", engine='openpyxl')
mrn_mapping1 = pd.read_excel("/data/daily_data/enrollment_log_202101013.xlsx", engine='openpyxl')
mrn_mapping2 = pd.read_excel("/data/daily_data/enrollment_log_201900354.xlsx", engine='openpyxl')

mrn_mapping = pd.concat([mrn_mapping1, mrn_mapping2], axis=0)

mrn_mapping = mrn_mapping[mrn_mapping["encounter_deiden_id"].isin(alerts["encounter_deiden_id"].unique())]
id_mapping = id_mapping[id_mapping["encounter_deiden_id"].isin(alerts["encounter_deiden_id"].unique())]

cases_table = id_mapping[["subject_deiden_id", "encounter_deiden_id", "patient_id"]].merge(mrn_mapping[["Admit Date", "encounter_deiden_id"]], on="encounter_deiden_id", how="inner")
cases_table = cases_table.rename({"subject_deiden_id": "patient_deiden_id"}, axis=1)

cases_table = cases_table.merge(alerts[["alert_id", "patient_deiden_id", "encounter_deiden_id", "enter_datetime", "time"]], on=["patient_deiden_id", "encounter_deiden_id"], how="inner")

cases_table = cases_table[~cases_table["patient_id"].str.contains("P")].reset_index(drop=True)
cases_table = cases_table.reset_index()

gt_labels = gt_labels[gt_labels["alert_id"].isin(cases_table["alert_id"])]

cases_table = cases_table.rename({
    "index": "case_id",
    "patient_id": "sequence_id",
    "Admit Date": "Hospital Admit Date",
    "enter_datetime": "ICU Admit Date/Time",
    "time": "Alert Date/Time"
}, axis=1)

cases_table["case_id"] = cases_table["case_id"] + 1

gt_labels.drop(columns=["alert_id"], inplace=True)

gt_labels = gt_labels.reset_index(drop=True).reset_index().rename({"index": "case_id"}, axis=1)

gt_labels["case_id"] = gt_labels["case_id"] + 1


#%%

alerts.drop(columns=["icustay_id", "enter_datetime", "time"], inplace=True)

survey_table = cases_table.merge(alerts, on=["patient_deiden_id", "encounter_deiden_id", "alert_id"], how="inner")

survey_table.drop(columns=["encounter_deiden_id", "alert_id"], inplace=True)
cases_table.drop(columns=["encounter_deiden_id", "alert_id"], inplace=True)

# %%

# Identify all unique patients where suggested_vp or suggested_crrt is "Recommended"
must_include = survey_table[(survey_table["suggested_vp"] == "Recommended") | (survey_table["suggested_crrt"] == "Recommended")]

# Keep only one entry per patient
must_include = must_include.drop_duplicates(subset=["patient_deiden_id"])

# Remove these selected patients from the original dataframe
remaining = survey_table[~survey_table["patient_deiden_id"].isin(must_include["patient_deiden_id"])]

# Determine how many more unique patients we need to reach 20
num_additional_samples = max(0, 20 - len(must_include))

# Randomly sample additional unique patients from the remaining cases
additional_samples = (
    remaining.drop_duplicates(subset=["patient_deiden_id"])
    .sample(n=min(len(remaining["patient_deiden_id"].unique()), num_additional_samples), random_state=42, replace=False)
)

# Combine the two sets and shuffle
survey_table = pd.concat([must_include, additional_samples]).sample(frac=1, random_state=42)
survey_table = survey_table.sort_values(by=["case_id"]).reset_index(drop=True)

cases_table = cases_table[cases_table["case_id"].isin(survey_table["case_id"].unique())].reset_index(drop=True)
gt_labels = gt_labels[gt_labels["case_id"].isin(survey_table["case_id"].unique())].reset_index(drop=True)

survey_table.drop(columns=["case_id"], inplace=True)
cases_table.drop(columns=["case_id"], inplace=True)
gt_labels.drop(columns=["case_id"], inplace=True)


survey_table = survey_table.reset_index().rename({"index": "case_id"}, axis=1)
survey_table["case_id"] = survey_table["case_id"] + 1

cases_table = cases_table.reset_index().rename({"index": "case_id"}, axis=1)
cases_table["case_id"] = cases_table["case_id"] + 1

gt_labels = gt_labels.reset_index().rename({"index": "case_id"}, axis=1)
gt_labels["case_id"] = gt_labels["case_id"] + 1

print(survey_table)
print(cases_table)
print(gt_labels)

# cases_table.to_csv("adjudication_cases.csv", index=False)
# survey_table.to_csv("adjudication_cases_survey.csv", index=False)

# %%
