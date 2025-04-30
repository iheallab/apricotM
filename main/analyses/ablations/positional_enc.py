#%% Import libraries
import pandas as pd
import numpy as np
import h5py
import time
import torch
import os
import pickle
import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.utils import shuffle
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch import optim
from torch.autograd import Variable

#%% Load variables
from variables import time_window, MODEL_DIR, OUTPUT_DIR, ANALYSIS_DIR
from apricotm import ApricotM

#%% Create analysis directory
if not os.path.exists(f"{ANALYSIS_DIR}/positional_enc"):
    os.makedirs(f"{ANALYSIS_DIR}/positional_enc")

#%% Load model architecture
model_architecture = torch.load(f"{MODEL_DIR}/apricotm_architecture.pth")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Load best hyperparameters
with open(f"{MODEL_DIR}/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

#%% Load data
with h5py.File(f"{OUTPUT_DIR}/final_data/dataset.h5", "r") as f:
    data = f["training"]
    X_train = data["X"][:]
    static_train = data["static"][:, 1:]
    y_trans_train = data["y_trans"][:]
    y_main_train = data["y_main"][:]

    data = f["validation"]
    X_val = data["X"][:]
    static_val = data["static"][:, 1:]
    y_trans_val = data["y_trans"][:]
    y_main_val = data["y_main"][:]

# Shuffle training set
X_train, static_train, y_main_train, y_trans_train = shuffle(X_train, static_train, y_main_train, y_trans_train)

# Merge targets
y_train = np.concatenate([y_main_train, y_trans_train], axis=1)
y_val = np.concatenate([y_main_val, y_trans_val], axis=1)

#%% Define constants
EPOCH = 100
BATCH_SIZE = best_params["batch_size"]
LR = best_params["learning_rate"]
seq_len = best_params["seq_len"]
max_code = 49
d_output = 12
d_input = X_train.shape[1]
d_static = static_train.shape[1]

nb_batches = int(len(X_train) / BATCH_SIZE) + 1
nb_batches_val = int(len(X_val) / BATCH_SIZE) + 1

# Tensor conversion
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
static_train = torch.FloatTensor(static_train)

X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val)
static_val = torch.FloatTensor(static_val)

# Class weights
class_weights = []
for i in range(d_output):
    negative = np.unique(y_train[:, i], return_counts=True)[1][0]
    positive = np.unique(y_train[:, i], return_counts=True)[1][1]
    class_weights.append(negative / positive)

loss_bin = BCELoss()

#%% Define early stopping
class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False, path=""):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, roc_pr_auc, model):
        if self.best_score is None:
            self.best_score = roc_pr_auc
            self.save_checkpoint(roc_pr_auc, model)
        elif roc_pr_auc < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(roc_pr_auc, model)
            self.best_score = roc_pr_auc
            self.counter = 0

    def save_checkpoint(self, roc_pr_auc, model):
        if self.verbose:
            print(
                f"Validation AUROC+AUPRC increased ({self.best_score:.6f} --> {roc_pr_auc:.6f}).  Saving model..."
            )
        torch.save(model.state_dict(), self.path)

#%% Function to train and validate
def train_and_validate(pe_condition):
    model = ApricotM(
        d_model=model_architecture["d_model"],
        d_hidden=model_architecture["d_hidden"],
        d_input=d_input,
        d_static=d_static,
        max_code=max_code,
        n_layer=model_architecture["n_layer"],
        device=DEVICE,
        dropout=model_architecture["dropout"],
        pe=pe_condition,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    early_stopping = EarlyStopping(
        patience=3,
        verbose=True,
        path=f"{ANALYSIS_DIR}/positional_enc/apricotm_pe_{pe_condition}.pth"
    )

    best_score = 0

    for epoch in range(EPOCH):
        model.train()
        train_loss = 0
        y_true_class = np.zeros((len(X_train), d_output))
        y_pred_prob = np.zeros((len(X_train), d_output))

        for patient in tqdm.trange(0, len(X_train), BATCH_SIZE):
            inputs = []
            for sample in X_train[patient : patient + BATCH_SIZE]:
                last_non_zero_index = torch.where(sample[:, 1] != 0)[0][-1].item()
                if last_non_zero_index >= seq_len:
                    adjusted_sample = sample[last_non_zero_index - seq_len + 1 : last_non_zero_index + 1]
                else:
                    padding = torch.zeros((seq_len - last_non_zero_index - 1, sample.shape[1]))
                    adjusted_sample = torch.cat((sample[: last_non_zero_index + 1], padding), dim=0)
                inputs.append(adjusted_sample)

            inputs = torch.stack(inputs)
            inputs = Variable(inputs).to(DEVICE)
            static_input = Variable(static_train[patient : patient + BATCH_SIZE]).to(DEVICE)
            labels = Variable(y_train[patient : patient + BATCH_SIZE]).to(DEVICE)

            optimizer.zero_grad()
            pred_y = model(inputs, static_input)

            loss = 0
            for i in range(d_output):
                loss_class = loss_bin(pred_y[:, i], labels[:, i]) * class_weights[i]
                loss += loss_class

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            y_true_class[patient : patient + BATCH_SIZE, :] = labels.to("cpu").numpy()
            y_pred_prob[patient : patient + BATCH_SIZE, :] = pred_y.to("cpu").detach().numpy()

        print("-" * 40)
        print(f"Epoch {epoch+1} Train loss: {train_loss / nb_batches:.4f}")

        # Validation
        model.eval()
        y_true_class_val = np.zeros((len(X_val), d_output))
        y_pred_prob_val = np.zeros((len(X_val), d_output))

        with torch.no_grad():
            for patient in range(0, len(X_val), BATCH_SIZE):
                inputs = []
                for sample in X_val[patient : patient + BATCH_SIZE]:
                    last_non_zero_index = torch.where(sample[:, 1] != 0)[0][-1].item()
                    if last_non_zero_index >= seq_len:
                        adjusted_sample = sample[last_non_zero_index - seq_len + 1 : last_non_zero_index + 1]
                    else:
                        padding = torch.zeros((seq_len - last_non_zero_index - 1, sample.shape[1]))
                        adjusted_sample = torch.cat((sample[: last_non_zero_index + 1], padding), dim=0)
                    inputs.append(adjusted_sample)

                inputs = torch.stack(inputs)
                inputs = Variable(inputs).to(DEVICE)
                static_input = Variable(static_val[patient : patient + BATCH_SIZE]).to(DEVICE)
                labels = Variable(y_val[patient : patient + BATCH_SIZE]).to(DEVICE)

                pred_y = model(inputs, static_input)

                y_true_class_val[patient : patient + BATCH_SIZE, :] = labels.to("cpu").numpy()
                y_pred_prob_val[patient : patient + BATCH_SIZE, :] = pred_y.to("cpu").detach().numpy()

        # Validation metrics
        aucs = []
        for i in range(d_output):
            aucs.append(roc_auc_score(y_true_class_val[:, i], y_pred_prob_val[:, i]))
        roc_auc_mean = np.mean(np.array(aucs)[[0,3,5,7,9,11]])

        aucs_pr = []
        for i in range(d_output):
            precision, recall, _ = precision_recall_curve(y_true_class_val[:, i], y_pred_prob_val[:, i])
            auc_pr = auc(recall, precision)
            aucs_pr.append(auc_pr)
        pr_auc_mean = np.mean(np.array(aucs_pr)[[0,3,5,7,9,11]])

        roc_pr_auc = roc_auc_mean + pr_auc_mean

        print(f"Validation AUROC: {roc_auc_mean:.4f}, AUPRC: {pr_auc_mean:.4f}, Sum: {roc_pr_auc:.4f}")

        if roc_pr_auc > best_score:
            best_score = roc_pr_auc

        early_stopping(roc_pr_auc, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return roc_auc_mean, pr_auc_mean

#%% Run training with and without PE
results = {}
for pe_condition in [False]:
    print("=" * 80)
    print(f"Training model with Positional Encoding = {pe_condition}")
    roc_auc, pr_auc = train_and_validate(pe_condition)
    results[pe_condition] = {"AUROC": roc_auc, "AUPRC": pr_auc}

#%% Save final results
results_df = pd.DataFrame(results).T
results_df.to_csv(f"{ANALYSIS_DIR}/positional_enc/pe_ablation_results.csv")
print(results_df)

    

