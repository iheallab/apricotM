#%%

# Import libraries

import pandas as pd
import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from variables import MODEL_DIR, OUTPUT_DIR

import os

if not os.path.exists(f"{MODEL_DIR}/examples"):
    os.makedirs(f"{MODEL_DIR}/examples")
    
threshold_dead = pd.read_csv(f"{MODEL_DIR}/episode_prediction/dead_episode_metrics.csv").rename({"Unnamed: 0": "cohort"}, axis=1)
threshold_dead = threshold_dead.loc[(threshold_dead["cohort"] == "ext")&(threshold_dead["Precision"] == 0.33), "Threshold"]

threshold_unstable = pd.read_csv(f"{MODEL_DIR}/episode_prediction/stable-unstable_episode_metrics.csv").rename({"Unnamed: 0": "cohort"}, axis=1)
threshold_unstable = threshold_unstable.loc[(threshold_unstable["cohort"] == "ext")&(threshold_unstable["Precision"] == 0.33), "Threshold"]

# Load prospective results

true_ext = pd.read_csv(f"{MODEL_DIR}/results/ext_true_labels.csv")
pred_ext = pd.read_csv(f"{MODEL_DIR}/calibration/ext_cal_probs.csv")

#%%

# Extract deceased patient example

# dead_ids = true_ext.loc[
#     (true_ext["dead"] == 1) & (pred_ext["dead"] > 0.5), "icustay_id"
# ].values

# dead_examples = true_ext[true_ext["icustay_id"].isin(dead_ids)]

# unstable_ids = dead_examples.loc[(dead_examples["stable-unstable"] == 1), "icustay_id"].unique()

# dead_examples = dead_examples[dead_examples["icustay_id"].isin(unstable_ids)]

# stable_ids = dead_examples.groupby("icustay_id").first().reset_index()

# stable_ids = stable_ids.loc[(stable_ids["stable"] == 1), "icustay_id"].unique()

# dead_examples = dead_examples[dead_examples["icustay_id"].isin(stable_ids)]

# dead_examples = dead_examples.groupby("icustay_id").size()

# dead_examples = dead_examples[dead_examples >= 24]

# dead_ids = dead_examples.index.values

# icustay_id = np.random.choice(dead_ids, 1)[0]

# print(f"Using patient: {int(icustay_id)}")

icustay_id = 38499331


pred_ext = pred_ext[true_ext["icustay_id"] == icustay_id].reset_index(drop=True)
true_ext = true_ext[true_ext["icustay_id"] == icustay_id].reset_index(drop=True)

time = list((np.arange(0, len(true_ext)) * 4))

true_ext["time"] = time

#%%

# Extract sequential data

with h5py.File(f"{OUTPUT_DIR}/final_data/dataset.h5", "r") as f:

    data = f["external_test"]
    seq = data["X"][:]

seq = seq.reshape(int(seq.shape[0] * seq.shape[1]), seq.shape[2])

seq = seq[np.where(seq[:, 3] == icustay_id)]

seq = pd.DataFrame(seq, columns=["time", "variable", "value", "icustay_id"])

seq = seq[seq["variable"] != 0]

with open(f"{OUTPUT_DIR}/model/scalers_seq.pkl", "rb") as f:
    scalers = pickle.load(f)

seq["time"] = scalers["scaler_hours"].inverse_transform(
    seq["time"].values.reshape(-1, 1)
)


def transform_var(group):
    variable_code = group["variable"].astype(int).values[0]

    scaler = scalers[f"scaler{variable_code}"]

    group["value"] = scaler.inverse_transform(group["value"].values.reshape(-1, 1))

    return group


seq = seq.groupby("variable").apply(transform_var).reset_index(drop=True)

with open(f"{OUTPUT_DIR}/model/variable_mapping.pkl", "rb") as f:
    variable_map = pickle.load(f)

seq["variable"] = seq["variable"].replace(variable_map)


#%%

# Load model

from apricotm import ApricotM
import torch
from torch.autograd import Variable
import captum.attr as attr

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

with open(f"{MODEL_DIR}/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

seq_len = best_params["seq_len"]

# Compute integrated gradients

with h5py.File(f"{OUTPUT_DIR}/final_data/dataset.h5", "r") as f:
    data = f["external_test"]
    X = data["X"][:]
    static = data["static"][:]
    y_trans = data["y_trans"][:]
    y_main = data["y_main"][:]

condition = np.where(X[:, 0, 3] == icustay_id)

X_sample = X[condition]

y = np.concatenate([y_main, y_trans], axis=1)
y = y[condition]

static = static[:, 1:]
static = static[condition]

variable_codes = X_sample[:, :, 1].copy()
values = X_sample[:, :, 2].copy()


X_sample = torch.FloatTensor(X_sample).to(DEVICE)
static = torch.FloatTensor(static)

ig_importance_variable = np.zeros((len(X_sample), seq_len))
ig_importance_value_time = np.zeros((len(X_sample), seq_len))

ig_variables = attr.LayerIntegratedGradients(model, model.embedding_variable)
ig_value_time = attr.LayerIntegratedGradients(model, model.conv1d)

BATCH_SIZE = 2

for patient in range(0, len(X_sample), BATCH_SIZE):
    inputs = Variable(X_sample[patient : patient + BATCH_SIZE]).to(DEVICE)
    static_input = Variable(static[patient : patient + BATCH_SIZE]).to(DEVICE)

    attributions_ig, _ = ig_variables.attribute(
        (inputs, static_input), target=3, return_convergence_delta=True
    )

    attributions_ig = attributions_ig.sum(dim=2).squeeze(0)
    attributions_ig = attributions_ig / torch.norm(attributions_ig)

    ig_importance_variable[patient : patient + BATCH_SIZE, :] = attributions_ig.to(
        "cpu"
    ).numpy()
    
    attributions_ig, _ = ig_value_time.attribute(
        (inputs, static_input), target=3, return_convergence_delta=True
    )

    attributions_ig = attributions_ig.sum(dim=1).squeeze(0)
    attributions_ig = attributions_ig / torch.norm(attributions_ig)

    ig_importance_value_time[patient : patient + BATCH_SIZE, :] = attributions_ig.to(
        "cpu"
    ).numpy()


# ig_temporal = pd.DataFrame({'code': variable_codes.flatten(), 'ig_variable': ig_importance_variable.flatten(), 'ig_value_time': ig_importance_value_time.flatten(), 'value': values.flatten()})
ig_temporal = pd.DataFrame(
    {"code": variable_codes.flatten(), "ig_variable": ig_importance_variable.flatten(), "ig_value_time": ig_importance_value_time.flatten()}
)

ig_temporal["ig_variable"] = ig_temporal["ig_variable"] + ig_temporal["ig_value_time"]

ig_temporal["variable"] = ig_temporal["code"].map(variable_map)

ig_temporal = ig_temporal[ig_temporal["code"] != 0].reset_index(drop=True)

seq = pd.concat([seq, ig_temporal.loc[:, "ig_variable"]], axis=1)

#%%

# Generate prediction uncertainty

n_iterations = 100

uncertainty = np.zeros((n_iterations, len(X_sample), 12))

for i in range(n_iterations):
    inputs = Variable(X_sample).to(DEVICE)
    static_input = Variable(static).to(DEVICE)

    pred_y = model(inputs, static_input)

    uncertainty[i, :, :] = pred_y.to("cpu").detach().numpy()


lower_bound = np.min(uncertainty, axis=0)
upper_bound = np.max(uncertainty, axis=0)


# %%

font = 16

# Create monitoring plot

fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(10, 12), sharex=True, gridspec_kw={"height_ratios": [1, 10, 10]}
)

outcome_columns = ["discharge", "stable", "unstable", "dead"]

acuity = pd.DataFrame()

acuity["state"] = true_ext[outcome_columns].idxmax(axis=1)
acuity["time"] = time

# Initialize lists to store start and end times
start_times = []
end_times = []
states = [acuity["state"].iloc[0]]

# Iterate through the DataFrame to identify state transitions
current_state = acuity["state"].iloc[0]
start_time = acuity["time"].iloc[0]

for index, row in acuity.iterrows():
    if row["state"] != current_state:
        end_time = acuity["time"].iloc[index - 1]
        start_times.append(start_time)
        end_times.append(end_time + 4)
        states.append(row["state"])
        current_state = row["state"]
        start_time = row["time"]

# Add the last state's start and end times
end_times.append(acuity["time"].iloc[-1] + 4)
start_times.append(start_time)

# Create a new DataFrame with start and end times
acuity = pd.DataFrame(
    {"state": states, "start_time": start_times, "end_time": end_times}
)

state_colors = {"unstable": "orange", "stable": "blue", "dead": "red"}

for i, row in acuity.iterrows():
    ax1.hlines(
        y=0,
        xmin=row["start_time"],
        xmax=row["end_time"],
        color=state_colors.get(row["state"], "gray"),
        linewidth=500,
        alpha=0.5,
    )


# Customize the plot
ax1.set_yticks([])  # Hide y-axis ticks
ax1.set_xlim(0, acuity["end_time"].max())
ax1.set_ylim(0, 0.01)


time = list((np.arange(0, len(true_ext)) * 4))
pred = pred_ext["unstable"]
# pred = pred_ext["stable-unstable"]

ax2.plot(time, pred, color="blue", label="Model prediction")
ax2.plot(time, [threshold_unstable] * len(time), "red", label="33% Precision Threshold")
ax2.fill_between(
    time,
    lower_bound[:, 2],
    upper_bound[:, 2],
    color="blue",
    alpha=0.3,
    label="Prediction uncertainty",
)


ax2.set_ylabel("Risk of instability", fontsize=font)
ax2.tick_params(labelsize=font)
ax2.set_xlim(0, acuity["end_time"].max())
ax2.set_ylim(0, 1)


time = list((np.arange(0, len(true_ext)) * 4))
pred = pred_ext["dead"]

ax3.plot(time, pred, color="blue", label="Model prediction")
ax3.plot(time, [threshold_dead] * len(time), "red", label="33% Precision Threshold")
ax3.fill_between(
    time,
    lower_bound[:, 3],
    upper_bound[:, 3],
    color="blue",
    alpha=0.3,
    label="Prediction uncertainty",
)


ax3.set_xlabel("Time in ICU (hours)", fontsize=font)
ax3.set_ylabel("Risk of mortality", fontsize=font)
ax3.tick_params(labelsize=font)
ax3.set_xlim(0, acuity["end_time"].max())
ax3.set_ylim(0, 1)
ax3.legend(fontsize=font, loc='upper right')

mv = true_ext[true_ext["no mv-mv"] == 1]

# for i in range(len(mv)):

if len(mv) > 0:

    mv_time = mv.iloc[0, -1]
    plt.text(mv_time-5, 2.4, "MV", fontsize=font, color="red")
    fig.axes[0].axvline(
        x=mv_time, color="red", linestyle="dotted", linewidth=2, label="Threshold"
    )
    fig.axes[1].axvline(
        x=mv_time, color="red", linestyle="dotted", linewidth=2, label="Threshold"
    )
    fig.axes[2].axvline(
        x=mv_time, color="red", linestyle="dotted", linewidth=2, label="Threshold"
    )


vp = true_ext[true_ext["no vp- vp"] == 1]

if len(vp) > 0:

    # for i in range(len(vp)):
    vp_time = vp.iloc[0, -1]
    plt.text(vp_time-5, 2.4, "VP", fontsize=font, color="red")
    fig.axes[0].axvline(
        x=vp_time, color="red", linestyle="dotted", linewidth=2, label="Threshold"
    )
    fig.axes[1].axvline(
        x=vp_time, color="red", linestyle="dotted", linewidth=2, label="Threshold"
    )
    fig.axes[2].axvline(
        x=vp_time, color="red", linestyle="dotted", linewidth=2, label="Threshold"
    )


plt.savefig(
    f"{MODEL_DIR}/examples/deceased_monitor.png",
    format="png",
    dpi=400,
)
plt.show()
plt.clf()

# %%

font = 16

# Create vitals chart
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(10, 12), sharex=True)

hr = seq[seq["variable"] == "Heart Rate"]

time_feat = hr["time"]
val_feat = hr["value"]
ig_feat = hr["ig_variable"]

# Plot heart rate values
ax1.plot(time_feat, val_feat, label="Heart Rate", color="blue")

normalized_gradients = (ig_feat - np.min(ig_feat)) / (np.max(ig_feat) - np.min(ig_feat))


for t, val, ig in zip(time_feat, val_feat, normalized_gradients):
    ax1.fill_betweenx(
        [min(val_feat), max(val_feat)], t, t + 0.1, color="orange", alpha=ig
    )


# Customize the plot
ax1.set_ylabel("Heart Rate (bpm)", fontsize=font, rotation=0, labelpad=70)
ax1.tick_params("y", labelsize=font)
ax1.set_xlim(0, max(time))
ax1.set_ylim(min(val_feat), max(val_feat))

hr = seq[seq["variable"] == "Non Invasive Blood Pressure systolic"]

time_feat = hr["time"]
val_feat = hr["value"]
ig_feat = hr["ig_variable"]

# Plot heart rate values
ax2.plot(time_feat, val_feat, label="Heart Rate", color="blue")

normalized_gradients = (ig_feat - np.min(ig_feat)) / (np.max(ig_feat) - np.min(ig_feat))


for t, val, ig in zip(time_feat, val_feat, normalized_gradients):
    ax2.fill_betweenx(
        [min(val_feat), max(val_feat)], t, t + 0.1, color="orange", alpha=ig
    )


# Customize the plot
ax2.set_ylabel("SBP (mmHg)", fontsize=font, rotation=0, labelpad=70)
ax2.tick_params("y", labelsize=font)
ax2.set_xlim(0, max(time))
ax2.set_ylim(min(val_feat), max(val_feat))

hr = seq[seq["variable"] == "Non Invasive Blood Pressure diastolic"]

time_feat = hr["time"]
val_feat = hr["value"]
ig_feat = hr["ig_variable"]

# Plot heart rate values
ax3.plot(time_feat, val_feat, label="Heart Rate", color="blue")

normalized_gradients = (ig_feat - np.min(ig_feat)) / (np.max(ig_feat) - np.min(ig_feat))


for t, val, ig in zip(time_feat, val_feat, normalized_gradients):
    ax3.fill_betweenx(
        [min(val_feat), max(val_feat)], t, t + 0.1, color="orange", alpha=ig
    )


# Customize the plot
ax3.set_ylabel("DBP (mmHg)", fontsize=font, rotation=0, labelpad=70)
ax3.tick_params("y", labelsize=font)
ax3.set_xlim(0, max(time))
ax3.set_ylim(min(val_feat), max(val_feat))


hr = seq[seq["variable"] == "O2 saturation pulseoxymetry"]

time_feat = hr["time"]
val_feat = hr["value"]
ig_feat = hr["ig_variable"]

# Plot heart rate values
ax4.plot(time_feat, val_feat, label="Heart Rate", color="blue")

normalized_gradients = (ig_feat - np.min(ig_feat)) / (np.max(ig_feat) - np.min(ig_feat))


for t, val, ig in zip(time_feat, val_feat, normalized_gradients):
    ax4.fill_betweenx(
        [min(val_feat), max(val_feat)], t, t + 0.1, color="orange", alpha=ig
    )


# Customize the plot
ax4.set_ylabel("SPO2 (%)", fontsize=font, rotation=0, labelpad=70)
ax4.tick_params("y", labelsize=font)
ax4.set_xlim(0, max(time))
ax4.set_ylim(min(val_feat), max(val_feat))

hr = seq[seq["variable"] == "Respiratory Rate"]

time_feat = hr["time"]
val_feat = hr["value"]
ig_feat = hr["ig_variable"]

# Plot heart rate values
ax5.plot(time_feat, val_feat, label="Heart Rate", color="blue")

normalized_gradients = (ig_feat - np.min(ig_feat)) / (np.max(ig_feat) - np.min(ig_feat))


for t, val, ig in zip(time_feat, val_feat, normalized_gradients):
    ax5.fill_betweenx(
        [min(val_feat), max(val_feat)], t, t + 0.1, color="orange", alpha=ig
    )


# Customize the plot
ax5.set_ylabel("RR (breaths/min)", fontsize=font, rotation=0, labelpad=70)
ax5.tick_params("y", labelsize=font)
ax5.set_xlim(0, max(time))
ax5.set_ylim(min(val_feat), max(val_feat))


hr = seq[seq["variable"] == "Temperature Celsius"]

time_feat = hr["time"]
val_feat = hr["value"]
ig_feat = hr["ig_variable"]

# Plot heart rate values
ax6.plot(time_feat, val_feat, label="Heart Rate", color="blue")

normalized_gradients = (ig_feat - np.min(ig_feat)) / (np.max(ig_feat) - np.min(ig_feat))


for t, val, ig in zip(time_feat, val_feat, normalized_gradients):
    ax6.fill_betweenx(
        [min(val_feat), max(val_feat)], t, t + 0.1, color="orange", alpha=ig
    )


# Customize the plot
ax6.set_ylabel("Temperature (C)", fontsize=font, rotation=0, labelpad=70)
ax6.tick_params("y", labelsize=font)
ax6.tick_params("x", labelsize=font)
ax6.set_xlim(0, max(time))
ax6.set_ylim(min(val_feat), max(val_feat))
ax6.set_xlabel("Time in ICU (hours)", fontsize=font)

plt.tight_layout()

plt.savefig(
    f"{MODEL_DIR}/examples/deceased_vitals.png",
    format="png",
    dpi=400,
)
plt.show()
plt.clf()

# %%
