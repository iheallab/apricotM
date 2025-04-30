# Import libraries

import pandas as pd
import numpy as np
import h5py
import time
import torch
import os

#%%

from variables import time_window, MODEL_DIR, ANALYSIS_DIR

#%%

models = ["apricotm", "apricott"]

for model_name in models:

    # Create directory to save results

    if not os.path.exists(f"{ANALYSIS_DIR}/seq_len/{model_name}"):
        os.makedirs(f"{ANALYSIS_DIR}/seq_len/{model_name}")


    # Load model architecture
    
    if model_name == "apricotm":

        from models.apricotm import ApricotM

        model_architecture = torch.load(
            f"{MODEL_DIR}/apricotm/model_architecture.pth"
        )
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

        model.load_state_dict(torch.load(f"{MODEL_DIR}/apricotm/model_weights.pth", map_location=DEVICE))
        
    elif model_name == "apricott":
        
        from models.apricott import ApricotT

        model_architecture = torch.load(f"{MODEL_DIR}/apricott/model_architecture.pth")
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = ApricotT(
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


        # Load model weights

        model.load_state_dict(
            torch.load(f"{MODEL_DIR}/apricott/model_weights.pth", map_location=DEVICE)
        )
        
    else:
        
        raise ValueError('Invalid model')


    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print("Parameters: {}".format(params))

    model.eval()

    # Load best parameters

    import pickle

    with open(f"{MODEL_DIR}/{model_name}/best_params.pkl", "rb") as f:
        best_params = pickle.load(f)

    seq_lens = [64, 128, 256, 512]

    print(f"Best Parameters: {best_params}")

    # Load data

    cohorts = [["validation", "int"], ["external_test", "ext"], ["temporal_test", "temp"]]

    for seq_len in seq_lens:

        for cohort in cohorts:

            with h5py.File(f"{ANALYSIS_DIR}/final_data/dataset.h5", "r") as f:
                data = f[cohort[0]]
                X_int = data["X"][:]
                static_int = data["static"][:]
                y_trans_int = data["y_trans"][:]
                y_main_int = data["y_main"][:]

            static_int = static_int[:, 1:]

            # Convert data to tensors

            BATCH_SIZE = best_params["batch_size"]

            X_int_arr = X_int.copy()
            X_int = torch.FloatTensor(X_int)
            static_int = torch.FloatTensor(static_int)
            y_int = np.concatenate([y_main_int, y_trans_int], axis=1)
            y_int = torch.FloatTensor(y_int)

            # Validation

            from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
            from torch.autograd import Variable

            y_true_class = np.zeros((len(X_int), 12))
            y_pred_prob = np.zeros((len(X_int), 12))

            start_time = time.time()

            for patient in range(0, len(X_int), BATCH_SIZE):

                inputs = []
                for sample in X_int[patient : patient + BATCH_SIZE]:

                    # Find the last non-zero index in the first dimension (512)
                    last_non_zero_index = torch.where(sample[:, 1] != 0)[0][-1].item()

                    # Take the last 256 values if the last non-zero index is greater than or equal to 256
                    if last_non_zero_index >= seq_len:
                        adjusted_sample = sample[
                            last_non_zero_index - seq_len + 1 : last_non_zero_index + 1, :
                        ]
                    # Pad the sequence with zeros if the last non-zero index is less than 256
                    else:
                        padding = torch.zeros(
                            (seq_len - last_non_zero_index - 1, sample.shape[1]),
                            dtype=sample.dtype,
                        )
                        adjusted_sample = torch.cat(
                            (sample[: last_non_zero_index + 1], padding), dim=0
                        )

                    inputs.append(adjusted_sample)

                inputs = torch.stack(inputs)
                inputs = Variable(inputs).to(DEVICE)

                static_input = Variable(static_int[patient : patient + BATCH_SIZE]).to(DEVICE)
                labels = Variable(y_int[patient : patient + BATCH_SIZE]).to(DEVICE)

                pred_y = model(inputs, static_input)

                y_true_class[patient : patient + BATCH_SIZE, :] = labels.to("cpu").numpy()
                y_pred_prob[patient : patient + BATCH_SIZE, :] = (
                    pred_y.to("cpu").detach().numpy()
                )

            inference_time = time.time() - start_time
            inference_time_batch = inference_time / (int(len(X_int) / BATCH_SIZE))

            print("-"*20 + f"Sequence length: {seq_len}" + "-"*20)

            print(f"Total inference time: {inference_time} seconds")
            print(f"Batch inference time: {inference_time_batch} seconds")

            print("-" * 40)
            print(f"Validation {cohort[1]}")

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

            cols = [
                "icustay_id",
                "discharge",
                "stable",
                "unstable",
                "dead",
                "unstable-stable",
                "stable-unstable",
                "mv-no mv",
                "no mv-mv",
                "vp-no vp",
                "no vp- vp",
                "crrt-no crrt",
                "no crrt-crrt",
            ]

            pred_labels = np.concatenate(
                [X_int_arr[:, 0, 3].reshape(-1, 1), y_pred_prob], axis=1
            )
            pred_labels = pd.DataFrame(pred_labels, columns=cols)

            true_labels = np.concatenate(
                [X_int_arr[:, 0, 3].reshape(-1, 1), y_true_class], axis=1
            )
            true_labels = pd.DataFrame(true_labels, columns=cols)

            true_labels.to_csv(
                f"{ANALYSIS_DIR}/seq_len/{model_name}/{cohort[1]}_{seq_len}_true_labels.csv", index=None
            )
            pred_labels.to_csv(
                f"{ANALYSIS_DIR}/seq_len/{model_name}/{cohort[1]}_{seq_len}_pred_labels.csv", index=None
            )

