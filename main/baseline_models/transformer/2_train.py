#%%
# Import libraries

import pandas as pd
import numpy as np
import h5py
import os
import optuna

# Choose time window

from variables import time_window, MODEL_DIR

# time_window = 4
# MODEL_DIR = f"/home/contreras.miguel/deepacu/main/baseline_models/transformer/{time_window}h_window"

if not os.path.exists(f"{MODEL_DIR}/results"):
    os.makedirs(f"{MODEL_DIR}/results")

#%%

# Load data

with h5py.File("%s/dataset_transformer.h5" % MODEL_DIR, "r") as f:
    data = f["training"]
    X_train = data["X"][:]
    y_trans_train = data["y_trans"][:]
    y_main_train = data["y_main"][:]

    data = f["validation"]
    X_val = data["X"][:]
    y_trans_val = data["y_trans"][:]
    y_main_val = data["y_main"][:]


# %%

# Shuffle training data

from sklearn.utils import shuffle

X_train, y_main_train, y_trans_train = shuffle(X_train, y_main_train, y_trans_train)


#%%

# Combine main and transition outcomes

y_train = np.concatenate([y_main_train, y_trans_train], axis=1)
y_val = np.concatenate([y_main_val, y_trans_val], axis=1)


#%%

# Import Transformer model

from model.model import Transformer
import torch
from torch import nn
from torch.nn import Module, Embedding, Linear, Sequential, Dropout, ModuleList


#%%

# Define model parameters and hyperparameters

EPOCH = 100
BATCH_SIZE = 256
LR = 5e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"use DEVICE: {DEVICE}")

# Convert data to tensors

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val)

optimizer_name = "Adam"

input_size = X_train.shape[2]
d_output = 12

class_weights = []

for i in range(d_output):

    negative = np.unique(y_train[:, i], return_counts=True)[1][0]
    positive = np.unique(y_train[:, i], return_counts=True)[1][1]

    class_weight = negative / positive

    # class_weight = torch.tensor(negative/positive)
    class_weights.append(class_weight)

from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss

loss_bin = BCELoss()
loss_bin_mort = BCEWithLogitsLoss()


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        # self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, roc_pr_auc, model):

        if self.best_score is None:
            self.best_score = roc_pr_auc
            # self.save_checkpoint(roc_pr_auc, model)
        elif roc_pr_auc < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # self.save_checkpoint(roc_pr_auc, model)
            self.best_score = roc_pr_auc
            self.counter = 0


from torch import optim
from torch.autograd import Variable
import tqdm

from sklearn.metrics import auc, roc_auc_score, precision_recall_curve

# Define objective function
def objective(trial):
    # Define hyperparameters to be optimized
    h = trial.suggest_int("h", 2, 16, step=2)
    q = int(input_size / h)
    v = int(input_size / h)
    N = trial.suggest_int("N", 1, 6)
    LR = trial.suggest_loguniform("learning_rate", 1e-4, 5e-4)
    BATCH_SIZE = trial.suggest_categorical("batch_size", [64, 128, 256])
    dropout = trial.suggest_categorical("dropout", [0.2, 0.3, 0.4])

    net = Transformer(
        d_model=input_size,
        q=q,
        v=v,
        h=h,
        N=N,
        device=DEVICE,
        dropout=dropout,
        mask=False,
    ).to(DEVICE)

    nb_batches = int(len(X_train) / BATCH_SIZE) + 1
    nb_batches_val = int(len(X_val) / BATCH_SIZE) + 1

    if optimizer_name == "Adagrad":
        optimizer = optim.Adagrad(net.parameters(), lr=LR, weight_decay=1e-5)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
    elif optimizer_name == "RMS":
        optimizer = optim.RMSprop(net.parameters(), lr=LR, weight_decay=1e-5)

    early_stopping = EarlyStopping(patience=3, verbose=True)

    best_score = 0
    best_auroc = 0
    best_auprc = 0

    for epoch in range(EPOCH):
        train_loss = 0
        # total_accuracy = 0
        count = 0
        y_true_class = np.zeros((len(X_train), d_output))
        y_pred_prob = np.zeros((len(X_train), d_output))
        for patient in tqdm.trange(0, len(X_train), BATCH_SIZE):

            inputs = Variable(X_train[patient : patient + BATCH_SIZE]).to(DEVICE)
            labels = Variable(y_train[patient : patient + BATCH_SIZE]).to(DEVICE)
            optimizer.zero_grad()
            pred_y = net(inputs)

            loss = 0

            for i in range(d_output):
                loss_class = loss_bin(pred_y[:, i], labels[:, i]) * class_weights[i]

                loss = loss + loss_class

            loss.backward()
            train_loss += loss.data
            optimizer.step()

            y_true_class[patient : patient + BATCH_SIZE, :] = labels.to("cpu").numpy()
            y_pred_prob[patient : patient + BATCH_SIZE, :] = (
                pred_y.to("cpu").detach().numpy()
            )

            count += 1

        print("-" * 40)
        print(f"Epoch {epoch+1}")

        print("Train performance")
        print("loss: {}".format(train_loss / nb_batches))

        aucs = []
        for i in range(y_pred_prob.shape[1]):
            ind_auc = roc_auc_score(y_true_class[:, i], y_pred_prob[:, i])
            aucs.append(ind_auc)

        print(f"train_roc_auc: {np.mean(aucs)}, class_aucs: {aucs}")

        aucs = []
        for i in range(y_pred_prob.shape[1]):
            precision, recall, _ = precision_recall_curve(
                y_true_class[:, i], y_pred_prob[:, i]
            )
            val_pr_auc = auc(recall, precision)
            aucs.append(val_pr_auc)

        print(f"train_pr_auc: {np.mean(aucs)}, class_aucs: {aucs}")

        y_true_class = np.zeros((len(X_val), d_output))
        y_pred_prob = np.zeros((len(X_val), d_output))
        val_loss = 0
        for patient in range(0, len(X_val), BATCH_SIZE):

            inputs = Variable(X_val[patient : patient + BATCH_SIZE]).to(DEVICE)
            labels = Variable(y_val[patient : patient + BATCH_SIZE]).to(DEVICE)
            pred_y = net(inputs)

            loss = 0

            for i in range(d_output):
                loss_class = loss_bin(pred_y[:, i], labels[:, i]) * class_weights[i]

                loss = loss + loss_class

            val_loss += loss.data

            y_true_class[patient : patient + BATCH_SIZE, :] = labels.to("cpu").numpy()
            y_pred_prob[patient : patient + BATCH_SIZE, :] = (
                pred_y.to("cpu").detach().numpy()
            )

        print("-" * 40)
        print("Val performance")

        print("val_loss: {}".format(val_loss / nb_batches_val))

        aucs = []
        for i in range(y_pred_prob.shape[1]):
            ind_auc = roc_auc_score(y_true_class[:, i], y_pred_prob[:, i])
            aucs.append(ind_auc)

        print(f"val_roc_auc: {np.mean(aucs)}, class_aucs: {aucs}")

        aucs = np.array(aucs)

        roc_auc_mean = np.mean(aucs[[0, 3, 5, 7, 9, 11]])

        print(f"Overall AUROC: {roc_auc_mean}")

        aucs = []
        for i in range(y_pred_prob.shape[1]):
            precision, recall, _ = precision_recall_curve(
                y_true_class[:, i], y_pred_prob[:, i]
            )
            val_pr_auc = auc(recall, precision)
            aucs.append(val_pr_auc)

        print(f"val_pr_auc: {np.mean(aucs)}, class_aucs: {aucs}")

        aucs = np.array(aucs)

        pr_auc_mean = np.mean(aucs[[0, 3, 5, 7, 9, 11]])

        print(f"Overall AUPRC: {pr_auc_mean}")

        roc_pr_auc = roc_auc_mean + pr_auc_mean

        if roc_pr_auc > best_score:
            best_score = roc_pr_auc
            best_auroc = roc_auc_mean
            best_auprc = pr_auc_mean

        early_stopping(roc_pr_auc, net)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return best_score


# Create study
study = optuna.create_study(direction="maximize")

# Optimize hyperparameters
study.optimize(objective, n_trials=20)

# Get best hyperparameters
best_params = study.best_params
print("Best hyperparameters:", best_params)

import pickle

with open("%s/best_params.pkl" % MODEL_DIR, "wb") as f:
    pickle.dump(best_params, f, protocol=2)


# Define hyperparameters to be optimized
h = best_params["h"]
q = int(input_size / h)
v = int(input_size / h)
N = best_params["N"]
LR = best_params["learning_rate"]
BATCH_SIZE = best_params["batch_size"]
dropout = best_params["dropout"]

net = Transformer(
    d_model=input_size,
    q=q,
    v=v,
    h=h,
    N=N,
    device=DEVICE,
    dropout=dropout,
    mask=False,
).to(DEVICE)


optimizer_name = "Adam"

if optimizer_name == "Adagrad":
    optimizer = optim.Adagrad(net.parameters(), lr=LR, weight_decay=1e-5)
elif optimizer_name == "Adam":
    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
elif optimizer_name == "RMS":
    optimizer = optim.RMSprop(net.parameters(), lr=LR, weight_decay=1e-5)


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
                f"Validation AUROC-AUPRC increased ({self.best_score:.6f} --> {roc_pr_auc:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)


early_stopping = EarlyStopping(
    patience=3, verbose=True, path=f"{MODEL_DIR}/model_weights.pth"
)

nb_batches = int(len(X_train) / BATCH_SIZE) + 1
nb_batches_val = int(len(X_val) / BATCH_SIZE) + 1

best_score = 0
best_auroc = 0
best_auprc = 0

for epoch in range(EPOCH):
    train_loss = 0
    # total_accuracy = 0
    count = 0
    y_true_class = np.zeros((len(X_train), d_output))
    y_pred_prob = np.zeros((len(X_train), d_output))
    for patient in tqdm.trange(0, len(X_train), BATCH_SIZE):

        inputs = Variable(X_train[patient : patient + BATCH_SIZE]).to(DEVICE)
        labels = Variable(y_train[patient : patient + BATCH_SIZE]).to(DEVICE)
        optimizer.zero_grad()
        pred_y = net(inputs)

        loss = 0

        for i in range(d_output):
            loss_class = loss_bin(pred_y[:, i], labels[:, i]) * class_weights[i]

            loss = loss + loss_class

        loss.backward()
        train_loss += loss.data
        optimizer.step()

        y_true_class[patient : patient + BATCH_SIZE, :] = labels.to("cpu").numpy()
        y_pred_prob[patient : patient + BATCH_SIZE, :] = (
            pred_y.to("cpu").detach().numpy()
        )

        count += 1

    print("-" * 40)
    print(f"Epoch {epoch+1}")

    print("Train performance")
    print("loss: {}".format(train_loss / nb_batches))

    aucs = []
    for i in range(y_pred_prob.shape[1]):
        ind_auc = roc_auc_score(y_true_class[:, i], y_pred_prob[:, i])
        aucs.append(ind_auc)

    print(f"train_roc_auc: {np.mean(aucs)}, class_aucs: {aucs}")

    aucs = []
    for i in range(y_pred_prob.shape[1]):
        precision, recall, _ = precision_recall_curve(
            y_true_class[:, i], y_pred_prob[:, i]
        )
        val_pr_auc = auc(recall, precision)
        aucs.append(val_pr_auc)

    print(f"train_pr_auc: {np.mean(aucs)}, class_aucs: {aucs}")

    y_true_class = np.zeros((len(X_val), d_output))
    y_pred_prob = np.zeros((len(X_val), d_output))
    val_loss = 0
    for patient in range(0, len(X_val), BATCH_SIZE):

        inputs = Variable(X_val[patient : patient + BATCH_SIZE]).to(DEVICE)
        labels = Variable(y_val[patient : patient + BATCH_SIZE]).to(DEVICE)
        pred_y = net(inputs)

        loss = 0

        for i in range(d_output):
            loss_class = loss_bin(pred_y[:, i], labels[:, i]) * class_weights[i]

            loss = loss + loss_class

        val_loss += loss.data

        y_true_class[patient : patient + BATCH_SIZE, :] = labels.to("cpu").numpy()
        y_pred_prob[patient : patient + BATCH_SIZE, :] = (
            pred_y.to("cpu").detach().numpy()
        )

    print("-" * 40)
    print("Val performance")

    print("val_loss: {}".format(val_loss / nb_batches_val))

    aucs = []
    for i in range(y_pred_prob.shape[1]):
        ind_auc = roc_auc_score(y_true_class[:, i], y_pred_prob[:, i])
        aucs.append(ind_auc)

    print(f"val_roc_auc: {np.mean(aucs)}, class_aucs: {aucs}")

    aucs = np.array(aucs)

    roc_auc_mean = np.mean(aucs[[0, 3, 5, 7, 9, 11]])

    print(f"Overall AUROC: {roc_auc_mean}")

    aucs = []
    for i in range(y_pred_prob.shape[1]):
        precision, recall, _ = precision_recall_curve(
            y_true_class[:, i], y_pred_prob[:, i]
        )
        val_pr_auc = auc(recall, precision)
        aucs.append(val_pr_auc)

    print(f"val_pr_auc: {np.mean(aucs)}, class_aucs: {aucs}")

    aucs = np.array(aucs)

    pr_auc_mean = np.mean(aucs[[0, 3, 5, 7, 9, 11]])

    print(f"Overall AUPRC: {pr_auc_mean}")

    roc_pr_auc = roc_auc_mean + pr_auc_mean

    if roc_pr_auc > best_score:
        best_score = roc_pr_auc
        best_auroc = roc_auc_mean
        best_auprc = pr_auc_mean

    early_stopping(roc_pr_auc, net)

    if early_stopping.early_stop:
        print("Early stopping")
        break

# Save model architecture

model_architecture = {
    "q": q,
    "v": v,
    "h": h,
    "N": N,
    "dropout": dropout,
}

torch.save(model_architecture, "%s/model_architecture.pth" % MODEL_DIR)
