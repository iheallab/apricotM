#%%
import pandas as pd
import numpy as np
import h5py
import time
import torch
import os
import psutil
import pickle
from apricotm import ApricotM
from apricott import ApricotT
from variables import MODEL_DIR, ANALYSIS_DIR, OUTPUT_DIR

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
def load_model(model_name, model_class, model_dir):
    model_architecture = torch.load(f"{model_dir}/{model_name}/model_architecture.pth")
    
    if model_class == ApricotM:
        model = model_class(
            d_model=model_architecture["d_model"],
            d_hidden=model_architecture["d_hidden"],
            d_input=model_architecture["d_input"],
            d_static=model_architecture["d_static"],
            n_layer=model_architecture["n_layer"],
            max_code=model_architecture["max_code"],
            device=DEVICE,
            dropout=model_architecture["dropout"],
        ).to(DEVICE)
    elif model_class == ApricotT:
        model = model_class(
            d_model=model_architecture["d_model"],
            d_hidden=model_architecture["d_hidden"],
            d_input=model_architecture["d_input"],
            d_static=model_architecture["d_static"],
            max_code=model_architecture["max_code"],
            q=model_architecture["q"],
            v=model_architecture["v"],
            h=model_architecture["h"],
            N=model_architecture["N"],
            device=DEVICE,
            dropout=model_architecture["dropout"],
        ).to(DEVICE)
    else:
        raise ValueError(f"Unsupported model class: {model_class}")

    model.load_state_dict(torch.load(f"{model_dir}/{model_name}/model_weights.pth", map_location=DEVICE))
    
    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in {model_name}: {num_params / 1e6:.2f}M")
    
    return model

def evaluate_model(model, data, seq_len, batch_size):
    X_int, static_int, y_int = data
    X_int = torch.FloatTensor(X_int)
    static_int = torch.FloatTensor(static_int)
    y_int = torch.FloatTensor(y_int)

    y_true_class = np.zeros((len(X_int), 12))
    y_pred_prob = np.zeros((len(X_int), 12))

    start_time = time.time()
    process = psutil.Process(os.getpid())
    # max_memory = 0
    max_gpu_memory = 0
    
    model.eval()
    
    inference_times = []
    memory_usages = []

    for patient in range(0, len(X_int), batch_size):
        inputs = []
        for sample in X_int[patient : patient + batch_size]:
            last_non_zero_index = torch.where(sample[:, 1] != 0)[0][-1].item()
            if last_non_zero_index >= seq_len:
                adjusted_sample = sample[last_non_zero_index - seq_len + 1 : last_non_zero_index + 1, :]
            else:
                padding = torch.zeros((seq_len - last_non_zero_index - 1, sample.shape[1]), dtype=sample.dtype)
                adjusted_sample = torch.cat((sample[: last_non_zero_index + 1], padding), dim=0)
            inputs.append(adjusted_sample)

        inputs = torch.stack(inputs).to(DEVICE)
        static_input = static_int[patient : patient + batch_size].to(DEVICE)
        labels = y_int[patient : patient + batch_size].to(DEVICE)
        
        torch.cuda.reset_peak_memory_stats(DEVICE)
        
        start_time = time.time()

        pred_y = model(inputs, static_input)
                
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        gpu_memory_usage = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)
        memory_usages.append(gpu_memory_usage)
        

        y_true_class[patient : patient + batch_size, :] = labels.cpu().numpy()
        y_pred_prob[patient : patient + batch_size, :] = pred_y.cpu().detach().numpy()

        # memory_usage = process.memory_info().rss / (1024 ** 2)
        # if memory_usage > max_memory:
        #     max_memory = memory_usage

        # # Track GPU memory usage
        # if DEVICE.type == "cuda":
        #     gpu_memory_usage = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)
        #     if gpu_memory_usage > max_gpu_memory:
        #         max_gpu_memory = gpu_memory_usage
    
    memory_usage = np.mean(memory_usages)
    inference_time = np.mean(inference_times)
    return inference_time, memory_usage

def load_data(cohort, data_dir):
    with h5py.File(f"{data_dir}/final_data/dataset.h5", "r") as f:
        data = f[cohort]
        X_int = data["X"][:]
        static_int = data["static"][:, 1:]
        y_trans_int = data["y_trans"][:]
        y_main_int = data["y_main"][:]
    y_int = np.concatenate([y_main_int, y_trans_int], axis=1)
    return X_int, static_int, y_int

#%%
seq_lens = [64, 128, 256, 512]
# cohorts = ["validation", "external_test", "temporal_test"]
cohorts = ["validation"]

results = []

for seq_len in seq_lens:
    for cohort in cohorts:
        data = load_data(cohort, ANALYSIS_DIR)
        # Subsample to 10000
        data = [d[:10000] for d in data]
        
        apricott_model = load_model("apricott", ApricotT, MODEL_DIR)
        with open(f"{MODEL_DIR}/apricott/best_params.pkl", "rb") as f:
            best_params = pickle.load(f)
        apricott_time, apricott_memory = evaluate_model(apricott_model, data, seq_len, 256)
        
        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats(DEVICE)
        
        apricotm_model = load_model("apricotm", ApricotM, MODEL_DIR)
        with open(f"{MODEL_DIR}/apricotm/best_params.pkl", "rb") as f:
            best_params = pickle.load(f)
        apricotm_time, apricotm_memory = evaluate_model(apricotm_model, data, seq_len, 256)

        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats(DEVICE)

        
        results.append({
            "seq_len": seq_len,
            "cohort": cohort,
            "apricotm_time": apricotm_time,
            "apricotm_memory": apricotm_memory,
            "apricott_time": apricott_time,
            "apricott_memory": apricott_memory,
        })
        
        print(f"Seq len: {seq_len}, Cohort: {cohort} - ApricotM time: {apricotm_time:.2f}s, ApricotM memory: {apricotm_memory:.2f}MB, ApricotT time: {apricott_time:.2f}s, ApricotT memory: {apricott_memory:.2f}MB")

results_df = pd.DataFrame(results)
print(results_df)
# results_df.to_csv(f"{ANALYSIS_DIR}/results_comp_speed.csv", index=False)

#%%

import pandas as pd
from variables import ANALYSIS_DIR
results_df = pd.read_csv(f"{ANALYSIS_DIR}/results_comp_speed.csv")
print(results_df)

# # %%

# with open(f"{MODEL_DIR}/apricotm/best_params.pkl", "rb") as f:
#     best_params = pickle.load(f)
    
# print(best_params["batch_size"])
    
# with open(f"{MODEL_DIR}/apricott/best_params.pkl", "rb") as f:
#     best_params = pickle.load(f)
    
# print(best_params["batch_size"])

# %%
