#%%

# Import libraries

import pandas as pd
import numpy as np
import h5py

#%%

# Load data

with h5py.File('final_data/dataset.h5', "r") as f:
    data = f['validation']
    X_int = data['X'][:]
    static_int = data['static'][:]
    y_trans_int = data['y_trans'][:]
    y_main_int = data['y_main'][:]
    
    
    data = f['external_test']
    X_ext = data['X'][:]
    static_ext = data['static'][:]
    y_trans_ext = data['y_trans'][:]
    y_main_ext = data['y_main'][:]
    
    data = f['temporal_test']
    X_temp = data['X'][:]
    static_temp = data['static'][:]
    y_trans_temp = data['y_trans'][:]
    y_main_temp = data['y_main'][:]

static_int = static_int[:,1:]
static_ext = static_ext[:,1:]
static_temp = static_temp[:,1:]


#%%

# Load model

from model.model import Strats
import torch

# Load model architecture

model_architecture = torch.load('model/model_architecture.pth')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Strats(
    d_model=model_architecture['d_model'],
    d_hidden=model_architecture['d_hidden'],
    d_input=model_architecture['d_input'],
    d_static=model_architecture['d_static'],
    d_output=model_architecture['d_output'],
    max_code=model_architecture['max_code'],
    q=model_architecture['q'],
    v=model_architecture['v'],
    h=model_architecture['h'],
    N=model_architecture['N'],
    device=DEVICE,
    dropout=model_architecture['dropout'],
    mask=model_architecture['mask']
).to(DEVICE)


# Load model weights

model.load_state_dict(torch.load('model/model_weights.pth'))

#%%

# Convert data to tensors

BATCH_SIZE = 256

X_int_arr = X_int.copy()
X_int = torch.FloatTensor(X_int)
static_int = torch.FloatTensor(static_int)
y_int = np.concatenate([y_main_int, y_trans_int], axis=1)
y_int = torch.FloatTensor(y_int)

X_ext_arr = X_ext.copy()
X_ext = torch.FloatTensor(X_ext)
static_ext = torch.FloatTensor(static_ext)
y_ext = np.concatenate([y_main_ext, y_trans_ext], axis=1)
y_ext = torch.FloatTensor(y_ext)

X_temp_arr = X_temp.copy()
X_temp = torch.FloatTensor(X_temp)
static_temp = torch.FloatTensor(static_temp)
y_temp = np.concatenate([y_main_temp, y_trans_temp], axis=1)
y_temp = torch.FloatTensor(y_temp)



#%%

# Internal validation

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch.autograd import Variable

y_true_class = np.zeros((len(X_int), 12))
y_pred_prob = np.zeros((len(X_int), 12))
for patient in range(0, len(X_int), BATCH_SIZE):
    inputs = Variable(X_int[patient:patient+BATCH_SIZE]).to(DEVICE)
    static_input = Variable(static_int[patient:patient+BATCH_SIZE]).to(DEVICE)
    labels = Variable(y_int[patient:patient+BATCH_SIZE]).to(DEVICE) 

    pred_y = model(inputs, static_input)
    
    y_true_class[patient:patient+BATCH_SIZE, :] = labels.to('cpu').numpy()
    y_pred_prob[patient:patient+BATCH_SIZE, :] = pred_y.to('cpu').detach().numpy()

print('-'*40)
print('Internal Validation')

aucs = []
for i in range(y_pred_prob.shape[1]):
    ind_auc = roc_auc_score(y_true_class[:,i], y_pred_prob[:,i])
    aucs.append(ind_auc)

print(f'val_roc_auc: {np.mean(aucs)}, class_aucs: {aucs}')


aucs = []
for i in range(y_pred_prob.shape[1]):
    precision, recall, _ = precision_recall_curve(y_true_class[:,i], y_pred_prob[:,i])
    val_pr_auc = auc(recall, precision)
    aucs.append(val_pr_auc)

print(f'val_pr_auc: {np.mean(aucs)}, class_aucs: {aucs}')

cols = [
    'icustay_id',
    'discharge',
    'stable',
    'unstable',
    'dead',
    'unstable-stable',
    'stable-unstable',
    'mv-no mv',
    'no mv-mv',
    'vp-no vp',
    'no vp- vp',
    'crrt-no crrt',
    'no crrt-crrt'
]

pred_labels = np.concatenate([X_int_arr[:,0,3].reshape(-1,1), y_pred_prob], axis=1)
pred_labels = pd.DataFrame(pred_labels, columns=cols)

true_labels = np.concatenate([X_int_arr[:,0,3].reshape(-1,1), y_true_class], axis=1)
true_labels = pd.DataFrame(true_labels, columns=cols)

true_labels.to_csv('results/int_true_labels.csv', index=None)
pred_labels.to_csv('results/int_pred_labels.csv', index=None)


# %%

# External validation

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch.autograd import Variable

y_true_class = np.zeros((len(X_ext), 12))
y_pred_prob = np.zeros((len(X_ext), 12))

for patient in range(0, len(X_ext), BATCH_SIZE):
    inputs = Variable(X_ext[patient:patient+BATCH_SIZE]).to(DEVICE)
    static_input = Variable(static_ext[patient:patient+BATCH_SIZE]).to(DEVICE)
    labels = Variable(y_ext[patient:patient+BATCH_SIZE]).to(DEVICE) 

    pred_y = model(inputs, static_input)
    
    y_true_class[patient:patient+BATCH_SIZE, :] = labels.to('cpu').numpy()
    y_pred_prob[patient:patient+BATCH_SIZE, :] = pred_y.to('cpu').detach().numpy()
    


print('-'*40)
print('External Validation')

aucs = []
for i in range(y_pred_prob.shape[1]):
    ind_auc = roc_auc_score(y_true_class[:,i], y_pred_prob[:,i])
    aucs.append(ind_auc)

print(f'val_roc_auc: {np.mean(aucs)}, class_aucs: {aucs}')


aucs = []
for i in range(y_pred_prob.shape[1]):
    precision, recall, _ = precision_recall_curve(y_true_class[:,i], y_pred_prob[:,i])
    val_pr_auc = auc(recall, precision)
    aucs.append(val_pr_auc)

print(f'val_pr_auc: {np.mean(aucs)}, class_aucs: {aucs}')

cols = [
    'icustay_id',
    'discharge',
    'stable',
    'unstable',
    'dead',
    'unstable-stable',
    'stable-unstable',
    'mv-no mv',
    'no mv-mv',
    'vp-no vp',
    'no vp- vp',
    'crrt-no crrt',
    'no crrt-crrt'
]


pred_labels = np.concatenate([X_ext_arr[:,0,3].reshape(-1,1), y_pred_prob], axis=1)
pred_labels = pd.DataFrame(pred_labels, columns=cols)

true_labels = np.concatenate([X_ext_arr[:,0,3].reshape(-1,1), y_true_class], axis=1)
true_labels = pd.DataFrame(true_labels, columns=cols)

true_labels.to_csv('results/ext_true_labels.csv', index=None)
pred_labels.to_csv('results/ext_pred_labels.csv', index=None)


# %%

# Temporal validation

y_true_class = np.zeros((len(X_temp), 12))
y_pred_prob = np.zeros((len(X_temp), 12))
for patient in range(0, len(X_temp), BATCH_SIZE):
    inputs = Variable(X_temp[patient:patient+BATCH_SIZE]).to(DEVICE)
    static_input = Variable(static_temp[patient:patient+BATCH_SIZE]).to(DEVICE)
    labels = Variable(y_temp[patient:patient+BATCH_SIZE]).to(DEVICE) 

    pred_y = model(inputs, static_input)
    
    y_true_class[patient:patient+BATCH_SIZE, :] = labels.to('cpu').numpy()
    y_pred_prob[patient:patient+BATCH_SIZE, :] = pred_y.to('cpu').detach().numpy()

print('-'*40)

print('Temporal Validation')

aucs = []
for i in range(y_pred_prob.shape[1]):
    ind_auc = roc_auc_score(y_true_class[:,i], y_pred_prob[:,i])
    aucs.append(ind_auc)

print(f'val_roc_auc: {np.mean(aucs)}, class_aucs: {aucs}')

aucs = []
for i in range(y_pred_prob.shape[1]):
    precision, recall, _ = precision_recall_curve(y_true_class[:,i], y_pred_prob[:,i])
    val_pr_auc = auc(recall, precision)
    aucs.append(val_pr_auc)

print(f'val_pr_auc: {np.mean(aucs)}, class_aucs: {aucs}')

cols = [
    'icustay_id',
    'discharge',
    'stable',
    'unstable',
    'dead',
    'unstable-stable',
    'stable-unstable',
    'mv-no mv',
    'no mv-mv',
    'vp-no vp',
    'no vp- vp',
    'crrt-no crrt',
    'no crrt-crrt'
]

pred_labels = np.concatenate([X_temp_arr[:,0,3].reshape(-1,1), y_pred_prob], axis=1)
pred_labels = pd.DataFrame(pred_labels, columns=cols)

true_labels = np.concatenate([X_temp_arr[:,0,3].reshape(-1,1), y_true_class], axis=1)
true_labels = pd.DataFrame(true_labels, columns=cols)

true_labels.to_csv('results/temp_true_labels.csv', index=None)
pred_labels.to_csv('results/temp_pred_labels.csv', index=None)


# %%
