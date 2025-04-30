# import pandas as pd
# import os
# from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
# import numpy as np
# import plotly.graph_objs as go
# from plotly.subplots import make_subplots

# from variables import MODEL_DIR

# # Define sequence lengths and cohorts
# seq_lengths = [64, 128, 256, 512]
# cohorts = ["int", "ext", "temp"]
# models = ["apricott", "apricotm"]

# # Function to load prediction and true labels
# def load_results(model_name, seq_len, cohort):
#     pred_labels_path = f"{MODEL_DIR}/{model_name}/seq_len_results/{cohort}_pred_labels_seq_len_{seq_len}.csv"
#     true_labels_path = f"{MODEL_DIR}/{model_name}/seq_len_results/{cohort}_true_labels_seq_len_{seq_len}.csv"
    
#     pred_labels = pd.read_csv(pred_labels_path)
#     true_labels = pd.read_csv(true_labels_path)
    
#     return pred_labels, true_labels

# # Function to calculate AUROC and AUPRC
# def calculate_metrics(pred_labels, true_labels):
#     y_true = true_labels.iloc[:, 1:].values
#     y_pred = pred_labels.iloc[:, 1:].values
    
#     aucs_roc = []
#     aucs_pr = []
    
#     for i in range(y_pred.shape[1]):
#         auc_roc = roc_auc_score(y_true[:, i], y_pred[:, i])
#         precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
#         auc_pr = auc(recall, precision)
        
#         aucs_roc.append(auc_roc)
#         aucs_pr.append(auc_pr)
    
#     return np.mean(aucs_roc), np.mean(aucs_pr)

# # Initialize dictionaries to store metrics
# metrics = {metric: {model: {cohort: [] for cohort in cohorts} for model in models} for metric in ['AUROC', 'AUPRC']}

# # Compute metrics across sequence lengths, models, and cohorts
# for seq_len in seq_lengths:
#     for cohort in cohorts:
#         for model in models:
#             pred_labels, true_labels = load_results(model, seq_len, cohort)
#             auc_roc, auc_pr = calculate_metrics(pred_labels, true_labels)
            
#             metrics['AUROC'][model][cohort].append(auc_roc)
#             metrics['AUPRC'][model][cohort].append(auc_pr)

# # Function to create Plotly visualizations
# def plot_metrics_plotly(seq_lengths, metrics, cohorts):
#     for cohort in cohorts:
#         fig = make_subplots(rows=1, cols=2, subplot_titles=(f"AUROC - {cohort.upper()}", f"AUPRC - {cohort.upper()}"))

#         for idx, metric_name in enumerate(['AUROC', 'AUPRC']):
#             for model in models:
#                 fig.add_trace(
#                     go.Scatter(
#                         x=seq_lengths,
#                         y=metrics[metric_name][model][cohort],
#                         mode='lines+markers',
#                         name=f"{model.upper()} - {metric_name}",
#                         marker=dict(symbol='circle' if model == 'apricott' else 'x', size=10),
#                         line=dict(dash='solid' if model == 'apricott' else 'dash')
#                     ),
#                     row=1,
#                     col=idx + 1
#                 )

#         fig.update_layout(
#             title_text=f"Model Comparison for Cohort: {cohort.upper()}",
#             xaxis_title="Sequence Length",
#             yaxis_title="Metric Value",
#             legend_title="Models",
#             width=900,
#             height=500,
#             showlegend=True
#         )

#         fig.update_xaxes(title_text="Sequence Length", tickvals=seq_lengths)
#         fig.update_yaxes(title_text="Metric Value", range=[0, 1])

#         fig.show()

# # Plot metrics for each cohort
# plot_metrics_plotly(seq_lengths, metrics, cohorts)



import pandas as pd
import os
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from variables import MODEL_DIR

# Define sequence lengths and cohorts
seq_lengths = [64, 128, 256, 512]
# cohorts = ["int", "ext", "temp", "prosp"]
cohorts = ["prosp"]
models = ["apricott", "apricotm"]

# Function to load prediction and true labels
def load_results(model_name, seq_len, cohort):
    pred_labels_path = f"{MODEL_DIR}/{model_name}/seq_len_results/{cohort}_pred_labels_seq_len_{seq_len}.csv"
    true_labels_path = f"{MODEL_DIR}/{model_name}/seq_len_results/{cohort}_true_labels_seq_len_{seq_len}.csv"
    
    pred_labels = pd.read_csv(pred_labels_path)
    true_labels = pd.read_csv(true_labels_path)
    
    return pred_labels, true_labels

# Function to calculate AUROC and AUPRC
def calculate_metrics(pred_labels, true_labels):
    y_true = true_labels.iloc[:, 1:].values
    y_pred = pred_labels.iloc[:, 1:].values
    
    aucs_roc = []
    aucs_pr = []
    
    for i in range(y_pred.shape[1]):
        auc_roc = roc_auc_score(y_true[:, i], y_pred[:, i])
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        auc_pr = auc(recall, precision)
        
        aucs_roc.append(auc_roc)
        aucs_pr.append(auc_pr)
    
    return np.mean(aucs_roc), np.mean(aucs_pr)

# Initialize dictionaries to store metrics
metrics = {metric: {model: {cohort: [] for cohort in cohorts} for model in models} for metric in ['AUROC', 'AUPRC']}

# Compute metrics across sequence lengths, models, and cohorts
for seq_len in seq_lengths:
    for cohort in cohorts:
        for model in models:
            pred_labels, true_labels = load_results(model, seq_len, cohort)
            auc_roc, auc_pr = calculate_metrics(pred_labels, true_labels)
            
            metrics['AUROC'][model][cohort].append(auc_roc)
            metrics['AUPRC'][model][cohort].append(auc_pr)

# Function to create Plotly visualizations
def plot_metrics_plotly(seq_lengths, metrics, cohorts):
    model_display_names = {
        "apricott": "APRICOT-T",
        "apricotm": "APRICOT-M"
    }
    model_colors = {
        "apricott": "#201658",  # Dark purple
        "apricotm": "#1D24CA"   # Deep blue
    }
    
    for cohort in cohorts:
        fig = make_subplots(rows=1, cols=2)

        added_models = set()

        for idx, metric_name in enumerate(['AUROC', 'AUPRC']):
            for model in models:
                showlegend = False
                if model not in added_models:
                    showlegend = True
                    added_models.add(model)
                
                fig.add_trace(
                    go.Scatter(
                        x=seq_lengths,
                        y=metrics[metric_name][model][cohort],
                        mode='lines+markers',
                        name=model_display_names[model],
                        marker=dict(
                            symbol='circle' if model == 'apricott' else 'x',
                            size=10,
                            color=model_colors[model]
                        ),
                        line=dict(
                            dash='solid' if model == 'apricott' else 'dash',
                            color=model_colors[model]
                        ),
                        showlegend=showlegend
                    ),
                    row=1,
                    col=idx + 1
                )

        fig.update_layout(
            xaxis_title="Sequence Length",
            xaxis2_title="Sequence Length",
            yaxis_title="AUROC",
            yaxis2_title="AUPRC",
            legend_title="Models",
            width=900,
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                x=1.05,
                y=1,
                font=dict(size=16),
                bgcolor='rgba(255,255,255,0)',
                bordercolor='rgba(0,0,0,0)'
            ),
            font=dict(
                size=16  # Control general font size (ticks, etc.)
            )
        )

        fig.update_xaxes(
            tickvals=seq_lengths,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor='black',
            title_font=dict(size=18),
            tickfont=dict(size=16)
        )
        fig.update_yaxes(
            range=[0, 1],
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor='black',
            title_font=dict(size=18),
            tickfont=dict(size=16)
        )

        # Save figure
        fig.write_image(f"metrics_plot_{cohort}.png", scale=3)  # High resolution
        # Optionally also display
        fig.show()

# Plot metrics for each cohort
plot_metrics_plotly(seq_lengths, metrics, cohorts)
