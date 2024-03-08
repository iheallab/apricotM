#%%

# Import libraries

import pandas as pd
import numpy as np
import h5py


DATA_DIR = "/home/contreras.miguel/deepacu"

#%%

# Load data

with h5py.File("%s/final_data/dataset.h5" % DATA_DIR, "r") as f:
    data = f["training"]
    X_train = data["X"][:]
    static_train = data["static"][:]
    y_trans_train = data["y_trans"][:]
    y_main_train = data["y_main"][:]

    data = f["validation"]
    X_val = data["X"][:]
    static_val = data["static"][:]
    y_trans_val = data["y_trans"][:]
    y_main_val = data["y_main"][:]

static_train = static_train[:, 1:]
static_val = static_val[:, 1:]


# %%

# Shuffle training data

from sklearn.utils import shuffle

X_train, static_train, y_main_train, y_trans_train = shuffle(
    X_train, static_train, y_main_train, y_trans_train
)


#%%

# Merge targets

y_train = np.concatenate([y_main_train, y_trans_train], axis=1)
y_val = np.concatenate([y_main_val, y_trans_val], axis=1)


#%%

# Define hyper parameters and model parameters

import torch
import math
import torch.nn as nn
from model.model import Strats


EPOCH = 10
BATCH_SIZE = 256
LR = 5e-4
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

print(f"use DEVICE: {DEVICE}")

d_model = 256
d_hidden = int(math.sqrt(d_model))
q = 32
v = 32
h = 8
N = 2
dropout = 0.4
mask = False
optimizer_name = "Adam"

d_input = X_train.shape[1]
d_static = static_train.shape[1]
d_output = 12


nb_batches = int(len(X_train) / BATCH_SIZE) + 1
nb_batches_val = int(len(X_val) / BATCH_SIZE) + 1

net = Strats(
    d_model=d_model,
    d_hidden=d_hidden,
    d_input=d_input,
    max_code=60,
    d_static=d_static,
    d_output=d_output,
    q=q,
    v=v,
    h=h,
    N=N,
    dropout=dropout,
    mask=mask,
    device=DEVICE,
).to(DEVICE)

# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     net = torch.nn.DataParallel(net)

model_parameters = filter(lambda p: p.requires_grad, net.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

for param in model_parameters:
    if len(param.size()) > 1:  # Apply initialization to weight parameters only
        nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")

print("Parameters: {}".format(params))

class_weights = []

for i in range(d_output):

    negative = np.unique(y_train[:, i], return_counts=True)[1][0]
    positive = np.unique(y_train[:, i], return_counts=True)[1][1]

    class_weight = negative / positive

    # class_weight = torch.tensor(negative/positive)
    class_weights.append(class_weight)

# class_weights = torch.tensor(class_weights,dtype=torch.float).to(DEVICE)

# loss_function = CrossEntropyLoss(weight=class_weights)

from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss

loss_bin = BCELoss()
loss_bin_mort = BCEWithLogitsLoss()


from torch import optim

if optimizer_name == "Adagrad":
    optimizer = optim.Adagrad(net.parameters(), lr=LR, weight_decay=1e-5)
elif optimizer_name == "Adam":
    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
elif optimizer_name == "RMS":
    optimizer = optim.RMSprop(net.parameters(), lr=LR, weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, factor=0.5, patience=5, threshold=0.01, mode="min"
)

#%%

# Convert data to tensors

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
static_train = torch.FloatTensor(static_train)

X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val)
static_val = torch.FloatTensor(static_val)


#%%

# Train model

from torch.autograd import Variable
import tqdm

from sklearn.metrics import auc, roc_auc_score, precision_recall_curve

for epoch in range(EPOCH):
    train_loss = 0
    # total_accuracy = 0
    count = 0
    y_true = np.zeros((len(X_train)))
    y_true_class = np.zeros((len(X_train), d_output))
    y_pred = np.zeros((len(X_train)))
    y_pred_prob = np.zeros((len(X_train), d_output))
    for patient in tqdm.trange(0, len(X_train), BATCH_SIZE):
        # for patient in range(0, len(X_train), BATCH_SIZE):
        inputs = Variable(X_train[patient : patient + BATCH_SIZE]).to(DEVICE)
        static_input = Variable(static_train[patient : patient + BATCH_SIZE]).to(DEVICE)
        labels = Variable(y_train[patient : patient + BATCH_SIZE]).to(DEVICE)
        optimizer.zero_grad()

        pred_y = net(inputs, static_input)

        loss = 0

        for i in range(d_output):
            loss_class = loss_bin(pred_y[:, i], labels[:, i]) * class_weights[i]

            loss = loss + loss_class

        # loss1 = loss_bin(pred_y[:,0], labels[:,0])
        # loss2 = loss_bin(pred_y[:,1], labels[:,1])
        # loss3 = loss_bin(pred_y[:,2], labels[:,2])
        # loss4 = loss_bin(pred_y[:,3], labels[:,3])
        # loss5 = loss_bin(pred_y[:,4], labels[:,4])
        # loss6 = loss_bin(pred_y[:,5], labels[:,5])
        # loss7 = loss_bin(pred_y[:,6], labels[:,6])
        # loss8 = loss_bin(pred_y[:,7], labels[:,7])
        # loss9 = loss_bin(pred_y[:,8], labels[:,8])
        # loss10 = loss_bin(pred_y[:,9], labels[:,9])
        # loss11 = loss_bin(pred_y[:,10], labels[:,10])
        # loss12 = loss_bin(pred_y[:,11], labels[:,11])

        # loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9 + loss10 + loss11 + loss12

        # params = torch.cat([x.view(-1) for x in net.module.mlp.parameters()])
        # l1_regularization_2 = 0.01 * torch.norm(params, 1)
        # loss = loss + l1_regularization_2
        # params = torch.cat([x.view(-1) for x in net.module.dense2.parameters()])
        # l1_regularization_2 = 0.01 * torch.norm(params, 1)
        # loss = loss + l1_regularization_2
        # params = torch.cat([x.view(-1) for x in net.module.dense3.parameters()])
        # l1_regularization_2 = 0.01 * torch.norm(params, 1)
        # loss = loss + l1_regularization_2
        # params = torch.cat([x.view(-1) for x in net.module.dense4.parameters()])
        # l1_regularization_2 = 0.01 * torch.norm(params, 1)
        # loss = loss + l1_regularization_2
        # params = torch.cat([x.view(-1) for x in net.module.dense5.parameters()])
        # l1_regularization_2 = 0.01 * torch.norm(params, 1)
        # loss = loss + l1_regularization_2
        # params = torch.cat([x.view(-1) for x in net.module.dense6.parameters()])
        # l1_regularization_2 = 0.01 * torch.norm(params, 1)
        # loss = loss + l1_regularization_2
        # params = torch.cat([x.view(-1) for x in net.module.dense7.parameters()])
        # l1_regularization_2 = 0.01 * torch.norm(params, 1)
        # loss = loss + l1_regularization_2
        # params = torch.cat([x.view(-1) for x in net.module.dense8.parameters()])
        # l1_regularization_2 = 0.01 * torch.norm(params, 1)
        # loss = loss + l1_regularization_2
        # params = torch.cat([x.view(-1) for x in net.module.main1.parameters()])
        # l1_regularization_2 = 0.01 * torch.norm(params, 1)
        # loss = loss + l1_regularization_2
        # params = torch.cat([x.view(-1) for x in net.module.main2.parameters()])
        # l1_regularization_2 = 0.01 * torch.norm(params, 1)
        # loss = loss + l1_regularization_2
        # params = torch.cat([x.view(-1) for x in net.module.main3.parameters()])
        # l1_regularization_2 = 0.01 * torch.norm(params, 1)
        # loss = loss + l1_regularization_2
        # params = torch.cat([x.view(-1) for x in net.module.main4.parameters()])
        # l1_regularization_2 = 0.01 * torch.norm(params, 1)
        # loss = loss + l1_regularization_2
        loss.backward()
        train_loss += loss.data
        optimizer.step()

        # for i in range(d_output):
        #     pred_y[:,i] = torch.nn.Sigmoid()(pred_y[:,i])

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

    y_true = np.zeros((len(X_val)))
    y_true_class = np.zeros((len(X_val), d_output))
    y_pred = np.zeros((len(X_val)))
    y_pred_prob = np.zeros((len(X_val), d_output))
    val_loss = 0
    for patient in range(0, len(X_val), BATCH_SIZE):
        inputs = Variable(X_val[patient : patient + BATCH_SIZE]).to(DEVICE)
        static_input = Variable(static_val[patient : patient + BATCH_SIZE]).to(DEVICE)
        labels = Variable(y_val[patient : patient + BATCH_SIZE]).to(DEVICE)

        pred_y = net(inputs, static_input)

        loss = 0

        for i in range(d_output):
            loss_class = loss_bin(pred_y[:, i], labels[:, i]) * class_weights[i]

            loss = loss + loss_class

        val_loss += loss.data

        # for i in range(d_output):
        #     pred_y[:,i] = torch.nn.Sigmoid()(pred_y[:,i])

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

    aucs = []
    for i in range(y_pred_prob.shape[1]):
        precision, recall, _ = precision_recall_curve(
            y_true_class[:, i], y_pred_prob[:, i]
        )
        val_pr_auc = auc(recall, precision)
        aucs.append(val_pr_auc)

    print(f"val_pr_auc: {np.mean(aucs)}, class_aucs: {aucs}")


# %%

# Save model weights

torch.save(net.state_dict(), "%s/model/model_weights.pth" % DATA_DIR)


# %%

# Save model architecture

model_architecture = {
    "d_model": d_model,
    "d_hidden": d_hidden,
    "d_input": d_input,
    "max_code": 60,
    "d_static": d_static,
    "d_output": d_output,
    "q": q,
    "v": v,
    "h": h,
    "N": N,
    "dropout": dropout,
    "mask": mask,
}

torch.save(model_architecture, "%s/model/model_architecture.pth" % DATA_DIR)


# %%
