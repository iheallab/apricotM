#%%
# Import libraries
import pandas as pd
import numpy as np
import os

from variables import time_window, PROSP_DATA_DIR, MODEL_DIR, VAR_MAP

# Specify the parent directory to extract data
parent_directory = "/data/daily_data/Cleaned/IICU_210501_231031/Data/final_data/"

if not os.path.exists(PROSP_DATA_DIR):
    os.makedirs(PROSP_DATA_DIR)

#%%

# Extract admissions

ind_file_path = os.path.join(parent_directory, "internal_stations_clean_0.csv")

all_admissions = pd.read_csv(ind_file_path)

all_admissions.drop_duplicates(
    subset=["patient_deiden_id", "enter_datetime"], inplace=True
)

all_admissions = all_admissions[all_admissions["location_type"] == "ICU"]

all_admissions["enter_datetime"] = pd.to_datetime(all_admissions["enter_datetime"])
all_admissions["exit_datetime"] = pd.to_datetime(all_admissions["exit_datetime"])

all_admissions["icu_los"] = (
    all_admissions["exit_datetime"] - all_admissions["enter_datetime"]
)
all_admissions = all_admissions[all_admissions["icu_los"] >= pd.Timedelta("1 hour")]

all_admissions = all_admissions.drop(columns=["icu_los"])

all_admissions.sort_values(by=["patient_deiden_id", "enter_datetime"], inplace=True)

all_admissions["grp"] = (
    (
        (
            (all_admissions["enter_datetime"] - pd.Timedelta("24 hours"))
            > all_admissions["exit_datetime"].shift()
        )
        & (all_admissions["enter_datetime"] > all_admissions["exit_datetime"].shift())
    )
    | ((all_admissions["merged_enc_id"] != all_admissions["merged_enc_id"].shift()))
).cumsum()


all_admissions = all_admissions.groupby(["merged_enc_id", "grp"]).agg(
    {
        "enter_datetime": "min",
        "exit_datetime": "max",
        "patient_deiden_id": pd.Series.unique,
        "to_station": "last",
    }
)

all_admissions = (
    all_admissions.reset_index()
    .drop(columns=["grp"])
    .sort_values(by=["merged_enc_id", "enter_datetime", "exit_datetime"])
)

all_admissions = all_admissions.reset_index()

all_admissions = all_admissions.rename({"index": "icustay_id"}, axis=1)

all_admissions["icustay_id"] = 40000000 + all_admissions["icustay_id"].astype(int)

all_admissions["enter_datetime"] = all_admissions["enter_datetime"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)
all_admissions["exit_datetime"] = all_admissions["exit_datetime"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)

all_admissions["patient_deiden_id"] = all_admissions["patient_deiden_id"].apply(
    lambda x: x[0]
)

patients = all_admissions["patient_deiden_id"].unique()
icu_stays = all_admissions["icustay_id"].unique()

print(f"Initial patients: {len(patients)}")
print(f"Initial ICU stays: {len(icu_stays)}")

print(f'First date: {all_admissions["enter_datetime"].min()}')
print(f'Last date: {all_admissions["enter_datetime"].max()}')

enc_ids = all_admissions["merged_enc_id"].unique()

if not os.path.exists(f"{PROSP_DATA_DIR}/intermediate"):
    os.makedirs(f"{PROSP_DATA_DIR}/intermediate")

all_admissions.to_csv(f"{PROSP_DATA_DIR}/intermediate/admissions.csv", index=None)

#%%

# Define data extraction function


def get_data(icustays, df):

    df = df.merge(
        icustays[
            ["patient_deiden_id", "icustay_id", "enter_datetime", "exit_datetime"]
        ],
        on="patient_deiden_id",
    )
    df = df[(df["time"] >= df["enter_datetime"]) & (df["time"] <= df["exit_datetime"])]

    df = df.sort_values(by=["icustay_id", "time"])

    return df


#%%

# Extract respiratory, meds, and dialysis

# Extract respiratory file

ind_file_path = os.path.join(parent_directory, "respiratory_clean_0.csv")

respiratory_df = pd.read_csv(ind_file_path)

respiratory_df = respiratory_df[respiratory_df["merged_enc_id"].isin(enc_ids)]

respiratory_df = respiratory_df.rename({"respiratory_datetime": "time"}, axis=1)

respiratory_df = get_data(all_admissions, respiratory_df)

respiratory_df = respiratory_df.sort_values(by=["icustay_id", "time"])

icu_stays = len(respiratory_df["icustay_id"].unique())

print(f"ICU stays with respiratory data: {icu_stays}")

# Extract meds file

ind_file_path = os.path.join(parent_directory, "meds_clean_0.csv")

pressor_df = pd.read_csv(ind_file_path)

pressor_df = pressor_df[pressor_df["merged_enc_id"].isin(enc_ids)]

list_pressors = [
    "dopamine",
    "epinephrine",
    "norepinephrine",
    "phenylephrine",
    "vasopressin",
]
list_pressors = r"\b(?:{})\b".format("|".join(list_pressors))

pressor_df = pressor_df[
    pressor_df["med_order_display_name"].str.contains(
        list_pressors, case=False, regex=True
    )
]

pressor_df = pressor_df.rename({"taken_datetime": "time"}, axis=1)

pressor_df["patient_deiden_id"] = pressor_df["patient_deiden_id"].astype(str)

pressor_df = get_data(all_admissions, pressor_df)

pressor_df = pressor_df.dropna(subset=["icustay_id"])

pressor_df = pressor_df.dropna(subset=["end_datetime"])

icu_stays = len(pressor_df["icustay_id"].unique())

print(f"ICU stays with vasopressors data: {icu_stays}")

# Extract dialysis file

ind_file_path = os.path.join(parent_directory, "dialysis_clean_0.csv")

crrt_df = pd.read_csv(ind_file_path)

crrt_df = crrt_df[crrt_df["merged_enc_id"].isin(enc_ids)]

crrt_df = crrt_df.rename({"observation_datetime": "time"}, axis=1)

crrt_df = crrt_df[crrt_df["vital_sign_measure_name"] == "Treatment Type"]

crrt_df["crrt"] = 0

crrt_df.loc[(crrt_df["meas_value"].isin(["CVVHD", "CVVH", "CVVHDF"])), "crrt"] = 1

crrt_df["patient_deiden_id"] = crrt_df["patient_deiden_id"].astype(str)

crrt_df = get_data(all_admissions, crrt_df)

icu_stays = len(crrt_df["icustay_id"].unique())

print(f"ICU stays with dialysis data: {icu_stays}")


# %%

# Extract ventilation events

vent = respiratory_df.copy().reset_index(drop=True)

vent = vent[vent["station_type"] == "ICU"]

tidal_vol = vent[vent["measurement_name"] == "tidal_volume_exhaled"]
tidal_vol = tidal_vol[["icustay_id", "time"]].reset_index(drop=True)


vent = vent.dropna(subset=["device_type"])

vent["MV"] = 0

vent.loc[(vent["device_type"] == "ventilator"), "MV"] = 1


def duration_mv(all_admissions):
    all_admissions = all_admissions.sort_values(["time"]).reset_index(drop=True)
    all_admissions["next_time"] = all_admissions["time"].shift(-1)
    all_admissions.loc[
        all_admissions["next_time"].isnull(), "next_time"
    ] = all_admissions.loc[all_admissions["next_time"].isnull(), "time"]
    eid = all_admissions.iloc[0]["icustay_id"]
    all_admissions["mv_diff"] = (all_admissions.MV.diff(1) != 0).cumsum()
    all_admissions = all_admissions[all_admissions["MV"] == 1]
    imputed = all_admissions[
        all_admissions["measurement_name"] == "respiratory_device"
    ]["imputed"].min()
    mv_duration = pd.DataFrame(
        {
            "icustay_id": eid,
            "begin_mv": all_admissions.groupby("mv_diff").time.first(),
            "stop_mv": all_admissions.groupby("mv_diff").next_time.last(),
            "imputed": imputed,
        }
    ).reset_index(drop=True)
    return mv_duration


mv_ids = vent.loc[vent["MV"] == 1, "icustay_id"].unique().tolist()
vent = vent[vent["icustay_id"].isin(mv_ids)]

mv_duration = vent.groupby(["icustay_id"]).apply(duration_mv).reset_index(drop=True)

overlap = mv_duration.merge(tidal_vol, how="inner", on="icustay_id")

overlap = overlap[
    (overlap["begin_mv"] <= overlap["time"])
    & (overlap["time"] <= overlap["stop_mv"])
    & (overlap["imputed"] == 1)
]

overlap.drop("time", inplace=True, axis=1)

overlap = overlap.drop_duplicates()

# Perform left join
merged = pd.merge(
    mv_duration,
    overlap,
    on=["icustay_id", "begin_mv", "stop_mv"],
    how="left",
    indicator=True,
)

# Filter out rows where there's a match in the second DataFrame
mv_duration = merged[merged["_merge"] == "left_only"].drop(columns="_merge")
mv_duration = mv_duration.drop(columns=["imputed_x", "imputed_y"])


#%%
# Extract vasopressors events


def group_duration(group):
    group = group.sort_values(["time"]).reset_index(drop=True)

    if "icustay_id" not in group.columns:
        # Handle the case where 'icustay_id' is not present
        print("Warning: 'icustay_id' column is missing in the group. Defaulting to -1.")
        eid = -1  # You can use a default value or any other approach here
    else:
        # Get the ICU stay ID from the first row
        eid = group.iloc[0]["icustay_id"]

    start_time = None
    end_time = None

    from_time, to_time = [], []
    for row in group.itertuples():
        if not start_time:
            start_time = row.time
            end_time = row.end_datetime
        else:
            if end_time == row.time:
                end_time = row.end_datetime
            elif end_time < row.time:
                from_time.append(start_time)
                to_time.append(end_time)
                start_time = row.time
                end_time = row.end_datetime
            else:
                end_time = max(end_time, row.end_datetime)
    from_time.append(start_time)
    to_time.append(end_time)
    eids = [eid] * len(from_time)

    pressor_duration = pd.DataFrame(
        {"icustay_id": eids, "begin_pressor": from_time, "stop_pressor": to_time}
    )
    return pressor_duration


pressor_df = pressor_df.loc[:, ["icustay_id", "time", "end_datetime"]]

pressor_duration = (
    pressor_df.groupby(["icustay_id"]).apply(group_duration).reset_index(drop=True)
)

#%%

# Extract CRRT events


def duration_crrt(all_admissions):
    all_admissions = all_admissions.sort_values(["time"]).reset_index(drop=True)
    all_admissions["next_time"] = all_admissions["time"].shift(-1)
    all_admissions.loc[
        all_admissions["next_time"].isnull(), "next_time"
    ] = all_admissions.loc[all_admissions["next_time"].isnull(), "time"]
    eid = all_admissions.iloc[0]["icustay_id"]
    all_admissions["crrt_diff"] = (all_admissions.crrt.diff(1) != 0).cumsum()
    all_admissions = all_admissions[all_admissions["crrt"] == 1]
    mv_duration = pd.DataFrame(
        {
            "icustay_id": eid,
            "begin_crrt": all_admissions.groupby("crrt_diff").time.first(),
            "stop_crrt": all_admissions.groupby("crrt_diff").next_time.last(),
        }
    ).reset_index(drop=True)
    return mv_duration


crrt_ids = crrt_df.loc[crrt_df["crrt"] == 1, "icustay_id"].unique().tolist()
crrt_df = crrt_df[crrt_df["icustay_id"].isin(crrt_ids)]
crrt_duration = (
    crrt_df.groupby(["icustay_id"]).apply(duration_crrt).reset_index(drop=True)
)

#%%
# Merge all procedures

pressor_duration = pressor_duration.rename(
    {"begin_pressor": "starttime", "stop_pressor": "endtime"}, axis=1
)
mv_duration = mv_duration.rename(
    {"begin_mv": "starttime", "stop_mv": "endtime"}, axis=1
)
crrt_duration = crrt_duration.rename(
    {"begin_crrt": "starttime", "stop_crrt": "endtime"}, axis=1
)

pressor_duration["procedure"] = "vp"
mv_duration["procedure"] = "mv"
crrt_duration["procedure"] = "crrt"


all_proced = (
    pd.concat([pressor_duration, mv_duration, crrt_duration], axis=0)
    .sort_values(by=["icustay_id", "starttime"])
    .reset_index(drop=True)
)

#%%
# Create 4h intervals

all_admissions = all_admissions[
    ["patient_deiden_id", "icustay_id", "enter_datetime", "exit_datetime"]
]

# Define a custom function to generate 4-hour intervals for each row
def generate_intervals(row):
    start_time = pd.to_datetime(row["enter_datetime"])
    end_time = pd.to_datetime(row["exit_datetime"])
    intervals = pd.date_range(start=start_time, end=end_time, freq=f"{time_window}H")
    return intervals


# Apply the function to each row in the DataFrame
all_admissions["Timestamps"] = all_admissions.apply(generate_intervals, axis=1)

# Explode the Timestamps column to expand it into separate rows
all_admissions = all_admissions.explode("Timestamps").reset_index(drop=True)

# %%
# Create acuity labels

all_admissions = all_admissions.rename({"Timestamps": "shift_start"}, axis=1)
all_admissions["shift_end"] = all_admissions["shift_start"] + pd.Timedelta(
    hours=time_window
)

all_admissions = all_admissions.merge(all_proced, how="outer", on=["icustay_id"])

admission_numb = len(all_admissions["icustay_id"].unique())

print(f"ICU stays with procedures info: {admission_numb}")

all_admissions["ind_start"] = (
    pd.to_datetime(all_admissions["starttime"], format="mixed")
    - pd.to_datetime(all_admissions["shift_start"], format="mixed")
) / np.timedelta64(1, "h")
all_admissions["ind_end"] = (
    pd.to_datetime(all_admissions["shift_end"], format="mixed")
    - pd.to_datetime(all_admissions["endtime"], format="mixed")
) / np.timedelta64(1, "h")

all_admissions["ind_start"] = all_admissions["ind_start"].fillna(0)
all_admissions["ind_end"] = all_admissions["ind_end"].fillna(0)

all_admissions.loc[
    (all_admissions["ind_start"] >= time_window)
    | (all_admissions["ind_end"] >= time_window),
    "procedure",
] = np.nan

icustays_filt = all_admissions.drop_duplicates(subset=["shift_start", "procedure"])

admission_numb = len(all_admissions["icustay_id"].unique())

print(f"ICU stays with procedures in {time_window}h intervals: {admission_numb}")

icustays_filt["procedure"] = icustays_filt["procedure"].fillna("missing")

icustays_filt["proc_ind"] = 0
icustays_filt.loc[(icustays_filt["procedure"] != "missing"), "proc_ind"] = 1

df_pivot = icustays_filt.pivot_table(
    index=["icustay_id", "shift_start"],
    columns="procedure",
    values="proc_ind",
    fill_value=None,
)

procedures = ["mv", "vp", "crrt"]

missing_columns = [col for col in procedures if col not in df_pivot.columns]

for col in missing_columns:
    df_pivot[col] = 0

df_pivot.drop("missing", inplace=True, axis=1)

df_pivot.fillna(0, inplace=True)

df_pivot.reset_index(inplace=True)

# Define a custom function to determine stability
def determine_stability(row):
    if 1 in row.iloc[2:].values:
        return "unstable"
    else:
        return "stable"


# Apply the custom function to create the 'Stability' column
df_pivot["final_state"] = df_pivot.apply(determine_stability, axis=1)

icu_stays = len(df_pivot["icustay_id"].unique())

print(f'Final acuity states counts: {df_pivot["final_state"].value_counts()}')
print(f"ICU stays: {icu_stays}")

df_pivot = df_pivot.sort_values(by=["icustay_id", "shift_start"])

df_pivot.to_csv(f"{PROSP_DATA_DIR}/intermediate/acuity_states_prosp.csv", index=None)

#%%

# Extract vitals

if not os.path.exists(f"{PROSP_DATA_DIR}/intermediate/vitals"):
    os.makedirs(f"{PROSP_DATA_DIR}/intermediate/vitals")


all_admissions = pd.read_csv(f"{PROSP_DATA_DIR}/intermediate/admissions.csv")
enc_ids = all_admissions["merged_enc_id"].unique()

vitals = ["heart_rate", "respiratory", "blood_pressure", "temperature"]

for vital in vitals:

    ind_file_path = os.path.join(parent_directory, f"{vital}_clean_0.csv")

    vital_df = pd.read_csv(ind_file_path)

    vital_df = vital_df[vital_df["merged_enc_id"].isin(enc_ids)]

    vital_df.to_csv(f"{PROSP_DATA_DIR}/intermediate/vitals/{vital}.csv", index=None)

    del vital_df

hr = pd.read_csv(f"{PROSP_DATA_DIR}/intermediate/vitals/heart_rate.csv")
resp = pd.read_csv(f"{PROSP_DATA_DIR}/intermediate/vitals/respiratory.csv")
temp = pd.read_csv(f"{PROSP_DATA_DIR}/intermediate/vitals/temperature.csv")
bp = pd.read_csv(f"{PROSP_DATA_DIR}/intermediate/vitals/blood_pressure.csv")

hr = hr.rename({"vitals_datetime": "time", "heart_rate": "value"}, axis=1)
hr["variable"] = "heart_rate"
hr = hr[["patient_deiden_id", "time", "variable", "value"]]

temp = temp.rename(
    {"vitals_datetime": "time", "clean_core_body_temp_celsius": "value"}, axis=1
)
temp["variable"] = "temp_celsius"
temp = temp[["patient_deiden_id", "time", "variable", "value"]]

bp = bp[
    [
        "patient_deiden_id",
        "bp_datetime",
        "noninvasive_systolic",
        "noninvasive_diastolic",
    ]
]

bp = bp.dropna()

diastolic_df = pd.melt(
    bp,
    id_vars=["patient_deiden_id", "bp_datetime"],
    value_vars=["noninvasive_diastolic"],
    var_name="variable",
    value_name="value",
)

systolic_df = pd.melt(
    bp,
    id_vars=["patient_deiden_id", "bp_datetime"],
    value_vars=["noninvasive_systolic"],
    var_name="variable",
    value_name="value",
)

bp = (
    pd.concat([diastolic_df, systolic_df], ignore_index=True)
    .sort_values(by=["patient_deiden_id", "bp_datetime"])
    .reset_index(drop=True)
)
bp = bp.rename({"bp_datetime": "time"}, axis=1)

resp = resp[
    ["patient_deiden_id", "respiratory_datetime", "measurement_name", "measured_value"]
]
resp = resp.rename(
    {
        "respiratory_datetime": "time",
        "measurement_name": "variable",
        "measured_value": "value",
    },
    axis=1,
)

resp_vars = [
    "respiratory_rate",
    "spo2",
    "pip",
    "tidal_volume_exhaled",
    "fio2_resp",
    "etco2",
    "o2_l_min",
    "peep",
]
resp = resp[resp["variable"].isin(resp_vars)]

vitals = (
    pd.concat([hr, temp, bp, resp], axis=0)
    .sort_values(by=["patient_deiden_id", "time"])
    .reset_index(drop=True)
)

vitals = get_data(all_admissions, vitals)

vitals.to_csv(f"{PROSP_DATA_DIR}/intermediate/vitals.csv", index=None)

#%%

# Extract labs

# Check if the folder contains a respiratory.csv file
ind_file_path = os.path.join(parent_directory, "labs_clean_0.csv")

# Concatenate all DataFrames into a single DataFrame
labs = pd.read_csv(ind_file_path)

labs = labs[labs["merged_enc_id"].isin(enc_ids)]

labs = labs[
    [
        "patient_deiden_id",
        "measurement_datetime",
        "stamped_and_inferred_loinc_code",
        "value_as_number",
    ]
]
labs = labs.rename(
    {
        "measurement_datetime": "time",
        "stamped_and_inferred_loinc_code": "variable",
        "value_as_number": "value",
    },
    axis=1,
)

labs = labs.dropna()
labs = labs.sort_values(by=["patient_deiden_id", "time"]).reset_index(drop=True)

labs = get_data(all_admissions, labs)

labs.to_csv(f"{PROSP_DATA_DIR}/intermediate/labs.csv", index=None)

# %%
# Extract meds

# Check if the folder contains a respiratory.csv file
ind_file_path = os.path.join(parent_directory, "meds_clean_0.csv")

# Concatenate all DataFrames into a single DataFrame
meds = pd.read_csv(ind_file_path)

meds = meds[meds["merged_enc_id"].isin(enc_ids)]

meds = meds[
    [
        "patient_deiden_id",
        "taken_datetime",
        "med_order_display_name",
        "total_dose_character",
    ]
]
meds = meds.rename(
    {
        "taken_datetime": "time",
        "med_order_display_name": "variable",
        "total_dose_character": "value",
    },
    axis=1,
)

meds = meds.dropna()
meds = meds.sort_values(by=["patient_deiden_id", "time"]).reset_index(drop=True)

meds = get_data(all_admissions, meds)

meds.to_csv(f"{PROSP_DATA_DIR}/intermediate/meds.csv", index=None)

# %%
# Extract scores

if not os.path.exists(f"{PROSP_DATA_DIR}/intermediate/scores"):
    os.makedirs(f"{PROSP_DATA_DIR}/intermediate/scores")

scores = ["braden", "cam", "rass", "glasgow", "pain"]

for score in scores:

    # Check if the folder contains a respiratory.csv file
    ind_file_path = os.path.join(parent_directory, f"{score}_clean_0.csv")

    # Concatenate all DataFrames into a single DataFrame
    scores_df = pd.read_csv(ind_file_path)

    scores_df = scores_df[scores_df["merged_enc_id"].isin(enc_ids)]

    scores_df.to_csv(f"{PROSP_DATA_DIR}/intermediate/scores/{score}.csv", index=None)

    del scores_df

braden = pd.read_csv(f"{PROSP_DATA_DIR}/intermediate/scores/braden.csv")
cam = pd.read_csv(f"{PROSP_DATA_DIR}/intermediate/scores/cam.csv")
rass = pd.read_csv(f"{PROSP_DATA_DIR}/intermediate/scores/rass.csv")
gcs = pd.read_csv(f"{PROSP_DATA_DIR}/intermediate/scores/glasgow.csv")
pain = pd.read_csv(f"{PROSP_DATA_DIR}/intermediate/scores/pain.csv")

braden = braden[
    ["patient_deiden_id", "braden_datetime", "measurement_name", "measurement_value"]
]
braden = braden.rename(
    {
        "braden_datetime": "time",
        "measurement_name": "variable",
        "measurement_value": "value",
    },
    axis=1,
)


def cam_score(all_admissions):

    all_admissions.loc[all_admissions["meas_value"] == "Negative", "meas_value"] = 0
    all_admissions.loc[all_admissions["meas_value"] == "Positive", "meas_value"] = 1
    all_admissions.loc[all_admissions["meas_value"] == "Yes", "meas_value"] = 1
    all_admissions.loc[all_admissions["meas_value"] == "UTA", "meas_value"] = None
    all_admissions.loc[all_admissions["meas_value"] == "None", "meas_value"] = None
    all_admissions["meas_value"] = pd.to_numeric(all_admissions["meas_value"])
    all_admissions["meas_value"] = all_admissions["meas_value"].fillna(-1)
    all_admissions["meas_value"] = all_admissions["meas_value"].astype(np.int32)

    all_admissions = all_admissions.sort_values(
        by=["patient_deiden_id", "recorded_time"]
    )
    retired_mask = (
        all_admissions["disp_name"]
        == "*RETIRED* CALCULATING POSITIVE OR NEGATIVE FOR DELIRIUM"
    )
    new_mask = (
        (all_admissions["disp_name"] == "CAM Screening Results")
        | (
            all_admissions["disp_name"]
            == "CALCULATING POSITIVE OR NEGATIVE FOR DELIRIUM"
        )
        | (all_admissions["disp_name"] == "Screening positive for delirium")
    )

    all_admissions["retired_score"] = all_admissions.loc[
        retired_mask, "meas_value"
    ].astype(int)
    all_admissions["new_score"] = all_admissions.loc[new_mask, "meas_value"].astype(int)
    all_admissions = all_admissions.drop(columns=["meas_value", "disp_name"])
    all_admissions = pd.merge(
        all_admissions.loc[retired_mask].drop(columns=["new_score"]),
        all_admissions.loc[new_mask].drop(columns=["retired_score"]),
        how="inner",
        on=["patient_deiden_id", "recorded_time"],
    )

    # now, merge the two columns together
    all_admissions["cam_score"] = 0

    def merge(row):
        if row["retired_score"] != row["new_score"]:
            if row["new_score"] != -1:
                row["cam_score"] = row["new_score"]
            elif row["retired_score"] != -1:
                row["cam_score"] = row["retired_score"]
            else:
                row["cam_score"] = row["new_score"]
        else:
            row["cam_score"] = row["new_score"]
        return row

    all_admissions = all_admissions.apply(merge, axis=1)
    all_admissions = all_admissions.drop(columns=["retired_score", "new_score"])

    all_admissions = all_admissions[all_admissions["cam_score"] != -1]
    all_admissions = all_admissions[all_admissions["cam_score"].notna()]

    return all_admissions


cam = cam_score(cam)

cam["variable"] = "cam"
cam = cam[["patient_deiden_id", "recorded_time", "variable", "cam_score"]]
cam = cam.rename({"recorded_time": "time", "cam_score": "value"}, axis=1)

rass["variable"] = "rass"
rass = rass[["patient_deiden_id", "recorded_time", "variable", "meas_value"]]
rass = rass.rename({"recorded_time": "time", "meas_value": "value"}, axis=1)

gcs = gcs[
    [
        "patient_deiden_id",
        "glasgow_coma_datetime",
        "measurement_name",
        "measurement_value",
    ]
]
gcs = gcs[gcs["measurement_name"] == "glasgow_coma_adult_score"]
gcs = gcs.rename(
    {
        "glasgow_coma_datetime": "time",
        "measurement_name": "variable",
        "measurement_value": "value",
    },
    axis=1,
)


pain["value"] = pain["measurement_value"].replace(
    {
        "Patient Asleep": np.nan,
        "-999": np.nan,
        -999: np.nan,
        "4 (Mild)": 4,
        "5 (Moderate)": 5,
        "6 (Moderate)": 6,
        "3 (Mild)": 3,
        "7 (Severe)": 7,
        "8 (Severe)": 8,
        "2 (Mild)": 2,
        "10 (Severe)": 10,
        "9 (Severe)": 9,
        "1 (Mild)": 1,
        "No pain": 0,
        "6abdomen": 6,
        "Asleep": np.nan,
        "Off unit": np.nan,
    }
)

pain = pain.dropna(subset=["value"])
pain["variable"] = "pain"
pain = pain[["patient_deiden_id", "pain_datetime", "variable", "value"]]
pain = pain.rename({"pain_datetime": "time"}, axis=1)

scores = (
    pd.concat([braden, cam, rass, gcs, pain], axis=0)
    .sort_values(by=["patient_deiden_id", "time"])
    .reset_index(drop=True)
)

scores = get_data(all_admissions, scores)

scores.to_csv(f"{PROSP_DATA_DIR}/intermediate/scores.csv", index=None)

# %%
# Extract static data

ind_file_path = os.path.join(
    parent_directory, "all_generated_admission_variables_0.csv"
)

# Concatenate all DataFrames into a single DataFrame
all_adm_var = pd.read_csv(ind_file_path)

# Check if the folder contains a respiratory.csv file
ind_file_path = os.path.join(parent_directory, "admission_encounters_clean_0.csv")

# Concatenate all DataFrames into a single DataFrame
all_adm_info = pd.read_csv(ind_file_path)

all_adm_var = all_adm_var[all_adm_var["merged_enc_id"].isin(enc_ids)]
all_adm_var = all_adm_var[
    ["patient_deiden_id", "merged_enc_id", "age", "sex", "race", "bmi"]
]

cols = (
    ["merged_enc_id"]
    + [col for col in all_adm_info.columns.tolist() if "_poa" in col]
    + ["charlson_comorbidity_total_score"]
)
comords = all_adm_info.loc[:, cols]

static = all_adm_var.merge(comords, how="inner", on="merged_enc_id")

static = all_admissions[["merged_enc_id", "icustay_id"]].merge(
    static, how="left", on=["merged_enc_id"]
)

static.to_csv(f"{PROSP_DATA_DIR}/intermediate/static.csv", index=None)


# %%

# Filter ICU admissions

admissions = pd.read_csv(f"{PROSP_DATA_DIR}/intermediate/admissions.csv")

print("-" * 20 + "Initial ICU stays" + "-" * 20)

print(f"Prospective cohort: {len(admissions)}")

#%%

# Create cohort flow table

cohort_flow = pd.DataFrame(data=[], columns=["prosp"])

cohort_flow = pd.concat(
    [
        cohort_flow,
        pd.DataFrame(
            data=[[len(admissions)]],
            index=["Initial"],
            columns=cohort_flow.columns,
        ),
    ],
    axis=0,
)


#%%

# Filter by vitals presence
print("-" * 20 + "Filtering by vitals presence" + "-" * 20)

vitals = pd.read_csv(f"{PROSP_DATA_DIR}/intermediate/vitals.csv")

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

dropped = len(admissions) - len(filtered_vitals)

print(f"ICU stays dropped: {len(admissions) - len(filtered_vitals)}")

admissions = admissions[admissions["icustay_id"].isin(filtered_vitals)]

cohort_flow = pd.concat(
    [
        cohort_flow,
        pd.DataFrame(
            data=[[dropped]],
            index=["Vitals Dropped"],
            columns=cohort_flow.columns,
        ),
    ],
    axis=0,
)

#%%

# Filter admissions by missigness of basic information
print("-" * 20 + "Filtering by basic info presence" + "-" * 20)

static = pd.read_csv(f"{PROSP_DATA_DIR}/intermediate/static.csv")

static = static[static["icustay_id"].isin(admissions["icustay_id"])]

basic = static.dropna()
basic = basic[basic["race"] != "missing"]
basic = basic[basic["bmi"] != "MISSING"]
basic = basic["icustay_id"].tolist()

dropped = len(admissions) - len(basic)

print(f"ICU stays dropped: {len(admissions) - len(basic)}")

admissions = admissions[admissions["icustay_id"].isin(basic)]

cohort_flow = pd.concat(
    [
        cohort_flow,
        pd.DataFrame(
            data=[[dropped]],
            index=["Basic Info Dropped"],
            columns=cohort_flow.columns,
        ),
    ],
    axis=0,
)

#%%

# Filter admissions by outcome data

print("-" * 20 + "Filtering by outcome presence" + "-" * 20)

outcomes = pd.read_csv(f"{PROSP_DATA_DIR}/intermediate/acuity_states_prosp.csv")

outcomes = outcomes[outcomes["icustay_id"].isin(admissions["icustay_id"].unique())]
outcomes = list(outcomes["icustay_id"].unique())

dropped = len(admissions) - len(outcomes)

print(f"ICU stays dropped: {len(admissions) - len(outcomes)}")

admissions = admissions[admissions["icustay_id"].isin(outcomes)]

cohort_flow = pd.concat(
    [
        cohort_flow,
        pd.DataFrame(
            data=[[dropped]],
            index=["Outcomes Dropped"],
            columns=cohort_flow.columns,
        ),
    ],
    axis=0,
)


#%%

# Filter admissions by discharge info

print("-" * 20 + "Filtering by discharge info" + "-" * 20)

station = admissions.dropna(subset=["to_station"])

dropped = len(admissions) - len(station)

print(f"ICU stays dropped: {len(admissions) - len(station)}")

admissions = admissions.dropna(subset=["to_station"])

cohort_flow = pd.concat(
    [
        cohort_flow,
        pd.DataFrame(
            data=[[dropped]],
            index=["Discharge Info Dropped"],
            columns=cohort_flow.columns,
        ),
    ],
    axis=0,
)


#%%

# Filter admissions by ICU length of stay

print("-" * 20 + "Filtering by ICU LOS" + "-" * 20)

admissions["icu_los"] = pd.to_datetime(
    admissions["exit_datetime"], format="mixed"
) - pd.to_datetime(admissions["enter_datetime"], format="mixed")

admissions_exclude = admissions[
    (admissions["icu_los"] <= pd.Timedelta("30 days"))
    & (admissions["icu_los"] >= pd.Timedelta(f"{time_window*2} hours"))
]

dropped = len(admissions) - len(admissions_exclude)

print(f"ICU stays dropped: {len(admissions) - len(admissions_exclude)}")

admissions = admissions[
    (admissions["icu_los"] <= pd.Timedelta("30 days"))
    & (admissions["icu_los"] >= pd.Timedelta(f"{time_window*2} hours"))
]

admissions = admissions.drop(columns=["icu_los"])

cohort_flow = pd.concat(
    [
        cohort_flow,
        pd.DataFrame(
            data=[[dropped]],
            index=["ICU LOS Dropped"],
            columns=cohort_flow.columns,
        ),
    ],
    axis=0,
)

#%%

# Save final cohort

admissions = admissions.reset_index()

if not os.path.exists(f"{PROSP_DATA_DIR}/final"):
    os.makedirs(f"{PROSP_DATA_DIR}/final")

cohort_flow.to_csv("%s/final/cohort_flow.csv" % PROSP_DATA_DIR)

admissions.to_csv(f"{PROSP_DATA_DIR}/final/admissions.csv", index=False)

# %%
# Build sequential data

vitals = pd.read_csv(f"{PROSP_DATA_DIR}/intermediate/vitals.csv")
labs = pd.read_csv(f"{PROSP_DATA_DIR}/intermediate/labs.csv")
meds = pd.read_csv(f"{PROSP_DATA_DIR}/intermediate/meds.csv")
scores = pd.read_csv(f"{PROSP_DATA_DIR}/intermediate/scores.csv")

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

seq = seq.dropna()

seq["hours"] = (
    pd.to_datetime(seq["time"], format="mixed")
    - pd.to_datetime(seq["enter_datetime"], format="mixed")
) / np.timedelta64(1, "h")

seq = seq[seq["hours"] >= 0]

seq = seq[seq["icustay_id"].isin(admissions["icustay_id"])]


def remove_outliers(group):

    lower_threshold = group["value"].quantile(0.01)
    upper_threshold = group["value"].quantile(0.99)

    group_filtered = group[
        (group["value"] >= lower_threshold) & (group["value"] <= upper_threshold)
    ]

    return group_filtered


seq = seq.groupby("variable").apply(remove_outliers)

seq = seq.reset_index(drop=True)

seq = seq.sort_values(by=["icustay_id", "hours"])

icu_stays = len(seq["icustay_id"].unique())

print(f"ICU stays with sequential data: {icu_stays}")

import pickle

with open(f"{MODEL_DIR}/variable_mapping.pkl", "rb") as f:
    variable_map = pickle.load(f)

final_map = pd.read_csv(f"{VAR_MAP}/variable_mapping.csv")
variables = final_map["uf"].tolist()

seq = seq[seq["variable"].isin(variables)]

map_eicu_uf = dict()
for i in range(len(final_map)):
    map_eicu_uf[final_map.loc[i, "uf"]] = final_map.loc[i, "mimic"]

seq["variable"] = seq["variable"].map(map_eicu_uf)

var_numb = len(seq["variable"].unique())

print(f"Number of variables in sequential data: {var_numb}")

inverted_map = {}
for key, values in variable_map.items():
    inverted_map[values] = key

seq["variable_code"] = seq["variable"].map(inverted_map)

seq = seq.dropna(subset=["variable_code"])

var_numb = len(seq["variable"].unique())

print(f"Number of reduced variables in sequential data: {var_numb}")

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

seq["interval"] = ((seq["hours"] // time_window) + 1).astype(int)

seq["shift_id"] = seq["icustay_id"].astype(str) + "_" + seq["interval"].astype(str)

seq.drop("interval", axis=1, inplace=True)

seq.to_csv(f"{PROSP_DATA_DIR}/final/seq.csv", index=None)


#%%

# Build static data

static = pd.read_csv(f"{PROSP_DATA_DIR}/intermediate/static.csv")

static = static[static["icustay_id"].isin(seq["icustay_id"].unique())]

static["sex"] = static["sex"].map({"FEMALE": "Female", "MALE": "Male"})
static["race"] = static["race"].map({"AA": "black", "WHITE": "white", "OTHER": "other"})

numeric_feat = ["sex", "age", "race", "bmi"]

comob_cols = [c for c in static.columns.tolist() if "_poa" in c]

select_feat = (
    ["icustay_id"] + numeric_feat + comob_cols + ["charlson_comorbidity_total_score"]
)

static = static.loc[:, select_feat]

static.to_csv(f"{PROSP_DATA_DIR}/final/static.csv", index=None)

# %%

# Build outcomes

outcomes = pd.read_csv(f"{PROSP_DATA_DIR}/intermediate/acuity_states_prosp.csv")

# Label discharge and deceased states

admissions = pd.read_csv(f"{PROSP_DATA_DIR}/final/admissions.csv")

outcomes = outcomes[outcomes["icustay_id"].isin(admissions["icustay_id"].unique())]

admissions = admissions[admissions["icustay_id"].isin(outcomes["icustay_id"].unique())]

outcomes["interval"] = outcomes.groupby("icustay_id").cumcount()

outcomes["shift_id"] = (
    outcomes["icustay_id"].astype(str) + "_" + outcomes["interval"].astype(str)
)

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

print(f'Final states counts: {outcomes["final_state"].value_counts()}')

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

outcomes.to_csv(f"{PROSP_DATA_DIR}/final/outcomes.csv", index=None)
