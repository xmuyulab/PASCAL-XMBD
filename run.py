import argparse
import numpy as np
import pandas as pd
import os
from dataset import *
from labelPropagation import LabelSpreading
import datetime
from config import params



parser = argparse.ArgumentParser(description='Propagation Algorithm for Semi-supervised Cell-type Assignment and Labeling')
parser.add_argument("-data", type=str, help="Input dataset identifier", default='imc1')
args = parser.parse_args()

if __name__ == "__main__":

    dataname = args.data

    # --- Create a timestamped prefix for all output files ---
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = f"{dataname}_{timestamp}"

    # Define paths for saving results and models
    if not os.path.exists(params.results_path):
        os.mkdir(params.results_path)

    adata, X, y, id_to_class = read_for_label_propagation(dataname, lib_name='confident_cells', seed=params.seed)

    # Model Definition
    model = LabelSpreading(k=10, max_iter=300, alpha=0.9, tol=1e-15, min_shared=1, metric='euclidean')
    model.fit(X, y)

    # --- Prediction based on UNNORMALIZED scores ---
    unnorm_probs = pd.DataFrame(model.unnormalized_label_distributions_, index=adata.obs.index, columns=model.classes_)
    max_unnorm_p = unnorm_probs.max(axis=1)
    y_pred_int = unnorm_probs.idxmax(axis=1)
    
    # max/second max ratio calculation
    second_max_unnorm_p = np.partition(unnorm_probs.values, -2, axis=1)[:, -2]
    ratio = max_unnorm_p / np.clip(second_max_unnorm_p, params.eps, None)
    invalid = ~np.isfinite(ratio)
    ratio[invalid] = 0.0
    ratio_series = pd.Series(ratio, index=unnorm_probs.index, name='max_second_ratio')

    y_pred = y_pred_int.map(id_to_class)

    # Unknown recognition
    y_pred_after_th = y_pred.copy()
    y_pred_after_th = y_pred_after_th.where(ratio_series >= params.unknown_thres, "Unknown")
    y_pred_after_th = y_pred_after_th.replace('negative', 'Unknown')

    results = pd.DataFrame({
        'sample.id': adata.obs.index,
        'predict': y_pred_after_th.values
    })

    
    class_name_columns = {label: id_to_class[label] for label in model.classes_ if label in id_to_class}
    
    unnorm_probs_with_names = unnorm_probs.rename(columns=class_name_columns)

    res_filename = f"{prefix}_th{params.unknown_thres}_result.csv"
    results.to_csv(os.path.join(params.results_path, res_filename), index=False)

    prob_filename = f"{prefix}_th{params.unknown_thres}_unnorm_probs.csv"
    unnorm_probs_with_names.to_csv(os.path.join(params.results_path, prob_filename), index=False)
    
    print(f"Saved in: {params.results_path}")