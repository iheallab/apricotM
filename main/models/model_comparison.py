import torch
import time
from torch.nn import BCELoss
import h5py
import numpy as np
from torch.autograd import Variable

from variables import time_window, MODEL_DIR, OUTPUT_DIR

from apricotm.apricotm import ApricotM
from apricott.apricott import ApricotT

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with h5py.File(f"{OUTPUT_DIR}/final_data/dataset.h5", "r") as f:
    data = f["validation"]
    data_1 = data["X"][:256, :, :]
    data_2 = data["static"][:256, 1:]
    target_1 = data["y_trans"][:256, :]
    target_2 = data["y_main"][:256, :]
    
target = np.concatenate([target_2, target_1], axis=1)

data_1 = Variable(torch.FloatTensor(data_1)).to(DEVICE)
data_2 = Variable(torch.FloatTensor(data_2)).to(DEVICE)
target = torch.FloatTensor(target).to(DEVICE)

model_architecture = torch.load(
    f"{MODEL_DIR}/apricotm/apricotm_architecture.pth"
)

model1 = ApricotM(
    d_model=model_architecture["d_model"],
    d_hidden=model_architecture["d_hidden"],
    d_input=model_architecture["d_input"],
    d_static=model_architecture["d_static"],
    max_code=model_architecture["max_code"],
    n_layer=model_architecture["n_layer"],
    device=DEVICE,
    dropout=model_architecture["dropout"],
).to(DEVICE)

# Measure training speed (forward pass + backward pass)
optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-5, weight_decay=1e-5)
criterion = BCELoss()

start_time = time.time()
output1 = model1(data_1, data_2)
loss1 = 0
for i in range(target.size(1)):
    loss_ind = criterion(output1, target)
    loss1 = loss1 + loss_ind
loss1.backward()
optimizer1.step()
training_time1 = time.time() - start_time

# Measure inference speed
with torch.no_grad():
    start_time = time.time()
    output1 = model1(data_1, data_2)
    inference_time1 = time.time() - start_time

# Measure memory usage
torch.cuda.reset_peak_memory_stats()

output1 = model1(data_1, data_2)
torch.cuda.synchronize()
memory_usage1 = torch.cuda.memory_allocated()

model_parameters = filter(lambda p: p.requires_grad, model1.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print("APRICOT-M:")
print("Parameters:", params)
print("Inference Time:", inference_time1)
print("Training Time:", training_time1)
print("Memory Usage:", memory_usage1)

del model1

model_architecture = torch.load(
    f"{MODEL_DIR}/apricott/apricott_architecture.pth"
)

model2 = ApricotT(
    d_model=model_architecture["d_model"],
    d_hidden=model_architecture["d_hidden"],
    d_input=model_architecture["d_input"],
    d_static=model_architecture["d_static"],
    max_code=model_architecture["max_code"],
    N=model_architecture["N"],
    h=model_architecture["h"],
    q=model_architecture["q"],
    v=model_architecture["v"],
    device=DEVICE,
    dropout=model_architecture["dropout"],
).to(DEVICE)


# Measure training speed (forward pass + backward pass)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-5, weight_decay=1e-5)
criterion = BCELoss()

start_time = time.time()
output2 = model2(data_1, data_2)
loss2 = 0
for i in range(target.size(1)):
    loss_ind = criterion(output2, target)
    loss2 = loss2 + loss_ind
loss2.backward()
optimizer2.step()
training_time2 = time.time() - start_time

with torch.no_grad():
    start_time = time.time()
    output2 = model2(data_1, data_2)
    inference_time2 = time.time() - start_time

# Measure memory usage
torch.cuda.reset_peak_memory_stats()

output2 = model2(data_1, data_2)
torch.cuda.synchronize()
memory_usage2 = torch.cuda.memory_allocated()

model_parameters = filter(lambda p: p.requires_grad, model2.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print("\nAPRICOT-T:")
print("Parameters:", params)
print("Inference Time:", inference_time2)
print("Training Time:", training_time2)
print("Memory Usage:", memory_usage2)
