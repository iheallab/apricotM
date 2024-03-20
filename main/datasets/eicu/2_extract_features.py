# %%
import pandas as pd
import numpy as np
import pickle
import os

# %%

# Filter by patients with ICU LOS > 4 and with acuity labels

acuity = pd.read_csv("acuity_states.csv")

icustays = pd.read_csv("patient.csv")

icustays = icustays[icustays["unitdischargeoffset"] > 240]

acuity = acuity[
    acuity["patientunitstayid"].isin(icustays["patientunitstayid"].unique())
]


# %%

# Extract vitals

vital_cols = [
    "patientunitstayid",
    "observationoffset",
    "temperature",
    "sao2",
    "heartrate",
    "respiration",
    "etco2",
    "systemicsystolic",
    "systemicdiastolic",
]

# Specify the chunk size (number of rows to read at a time)
chunk_size = 1000000
chunks = []
header_written = False
count = 1

# Read the CSV file in chunks
for chunk in pd.read_csv("vitalPeriodic.csv", chunksize=chunk_size, usecols=vital_cols):

    chunk = pd.melt(
        chunk,
        id_vars=["patientunitstayid", "observationoffset"],
        var_name="vitalname",
        value_name="vitalvalue",
    )

    chunk = chunk.dropna(subset=["vitalvalue"])

    chunk.to_csv("vitals.csv", mode="a", index=False, header=not header_written)

    # chunks.append(chunk)
    print(f"Processed chunk {count}")
    count += 1

    header_written = True

    del chunk

vitals = pd.concat(chunks)


vitals.to_csv("vitals.csv", index=None)

del chunks

# %%

# Extract scores

cols = [
    "patientunitstayid",
    "nursingchartoffset",
    "nursingchartcelltypecat",
    "nursingchartcelltypevalname",
    "nursingchartvalue",
]

# Specify the chunk size (number of rows to read at a time)
chunk_size = 1000000
chunks = []
header_written = False


count = 1

for chunk in pd.read_csv("nurseCharting.csv", chunksize=chunk_size, low_memory=False):

    chunk = chunk[chunk["nursingchartcelltypecat"] == "Scores"]

    chunk.to_csv("scores.csv", mode="a", index=False, header=not header_written)

    print(f"Processed chunk {count}")
    count += 1

    header_written = True


# %%

# Extract urine output

cols = ["patientunitstayid", "intakeoutputoffset", "celllabel", "cellvaluenumeric"]

# Specify the chunk size (number of rows to read at a time)
chunk_size = 1000000
chunks = []
header_written = False


count = 1

for chunk in pd.read_csv(
    "intakeOutput.csv", chunksize=chunk_size, low_memory=False, usecols=cols
):

    chunk = chunk[chunk["celllabel"] == "Urine"]

    chunk.to_csv("urine_output.csv", mode="a", index=False, header=not header_written)

    print(f"Processed chunk {count}")
    count += 1

    header_written = True


# %%
