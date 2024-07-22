# Import libraries

import pandas as pd
import numpy as np
import h5py
import os

from variables import OUTPUT_DIR, PROSP_DATA_DIR

if not os.path.exists(f"{OUTPUT_DIR}/summary/transition_prop"):
    os.makedirs(f"{OUTPUT_DIR}/summary/transition_prop")

# Load data

outcomes = pd.read_csv("%s/final_data/outcomes.csv" % OUTPUT_DIR)

outcomes = outcomes[outcomes["interval"] != 0]

import pickle

with open("%s/final_data/ids.pkl" % OUTPUT_DIR, "rb") as f:
    ids = pickle.load(f)
    ids_train, ids_val, ids_ext, ids_temp = (
        ids["train"],
        ids["val"],
        ids["ext_test"],
        ids["temp_test"],
    )

ids_train = ids_train[0]
ids_val = ids_val[0]

outcomes_train = outcomes[outcomes["icustay_id"].isin(ids_train)]
outcomes_val = outcomes[outcomes["icustay_id"].isin(ids_val)]
outcomes_ext = outcomes[outcomes["icustay_id"].isin(ids_ext)]
outcomes_temp = outcomes[outcomes["icustay_id"].isin(ids_temp)]

outcomes_develop = pd.concat([outcomes_train, outcomes_val], axis=0)

del outcomes_train, outcomes_val

outcomes_prosp = pd.read_csv(f"{PROSP_DATA_DIR}/final/outcomes.csv")


def extract_first_word(text):
    return text.split("-")[0]


outcomes_develop["prev_state"] = outcomes_develop["transition"].apply(
    extract_first_word
)
outcomes_ext["prev_state"] = outcomes_ext["transition"].apply(extract_first_word)
outcomes_temp["prev_state"] = outcomes_temp["transition"].apply(extract_first_word)
outcomes_prosp["prev_state"] = outcomes_prosp["transition"].apply(extract_first_word)

prop_develop = pd.DataFrame(
    outcomes_develop.groupby("prev_state")["final_state"].value_counts()
    / outcomes_develop.groupby("prev_state").size()
).applymap(lambda x: f"{x*100:.2f}")
prop_ext = pd.DataFrame(
    outcomes_ext.groupby("prev_state")["final_state"].value_counts()
    / outcomes_ext.groupby("prev_state").size()
).applymap(lambda x: f"{x*100:.2f}")
prop_temp = pd.DataFrame(
    outcomes_temp.groupby("prev_state")["final_state"].value_counts()
    / outcomes_temp.groupby("prev_state").size()
).applymap(lambda x: f"{x*100:.2f}")
prop_prosp = pd.DataFrame(
    outcomes_prosp.groupby("prev_state")["final_state"].value_counts()
    / outcomes_prosp.groupby("prev_state").size()
).applymap(lambda x: f"{x*100:.2f}")

prop_develop.to_csv(f"{OUTPUT_DIR}/summary/transition_prop/develop_transition_prop.csv")
prop_ext.to_csv(f"{OUTPUT_DIR}/summary/transition_prop/ext_transition_prop.csv")
prop_temp.to_csv(f"{OUTPUT_DIR}/summary/transition_prop/temp_transition_prop.csv")
prop_prosp.to_csv(f"{OUTPUT_DIR}/summary/transition_prop/prosp_transition_prop.csv")
