from collections import defaultdict
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from sklearn.mixture import GaussianMixture
from config import params


def gaussian_mixture_intersections(mu1, sigma1, w1, mu2, sigma2, w2, eps):
    """
    Compute intersection point(s) between two 1D weighted Gaussian PDFs.

    We solve for x such that:
        w1 * N(x | mu1, sigma1^2) = w2 * N(x | mu2, sigma2^2)

    Taking logs and rearranging yields a quadratic equation:
        A x^2 + B x + C = 0
    """
    A = 0.5 / (sigma2**2) - 0.5 / (sigma1**2)
    B = mu1 / (sigma1**2) - mu2 / (sigma2**2)
    C = -mu1**2 / (2 * sigma1**2) + mu2**2 / (2 * sigma2**2) + np.log((sigma2 * w1) / (sigma1 * w2))

    roots = []

    if abs(A) < eps:
        if abs(B) < eps:
            return roots
        x = -C / B
        roots.append(x)
        return roots

    D = B**2 - 4 * A * C  # Discriminant

    if D < 0:
        return roots
    elif abs(D) < eps:
        x = -B / (2 * A)
        roots.append(x)
    else:
        sqrtD = np.sqrt(D)
        x1 = (-B + sqrtD) / (2 * A)
        x2 = (-B - sqrtD) / (2 * A)
        roots.extend([x1, x2])

    return roots

def two_stage_normalization(p, eps):
    lo = float(np.min(p))
    hi = float(np.max(p))


    if hi - lo < 0.3:
        print("Warning: Data range is very small, normalization may amplify noise")
    
    if hi < 1.0 - eps:
        right_mask = p > 0.5 + eps
        if np.any(right_mask):
            denom = max(hi - 0.5, eps)
            p[right_mask] = 0.5 + (p[right_mask] - 0.5) / denom * 0.5

    if lo > 0.0 + eps:
        left_mask = p < 0.5 - eps
        if np.any(left_mask):
            denom = max(0.5 - lo, eps)
            p[left_mask] = (p[left_mask] - lo) / denom * 0.5

    return p

def filter_cells_by_likelihood(adata, marker_conditions, prob_low, prob_high, celltype_list, marker_list):
    """
    Compute per-cell likelihoods for each (cell type / subtype) under marker-based conditions.
    """
    all_subtypes_dict = {} 
    
    for cell_type in celltype_list:
        conditions = marker_conditions[cell_type]
        # For each subtype rule under this cell type
        for sub_idx, condition in enumerate(conditions):
            condition_values = np.array([condition.get(marker) for marker in marker_list])
            post_prob = np.zeros((adata.n_obs, len(marker_list))) #(n_cell, n_marker)

            for marker_idx, marker in enumerate(marker_list):
                prob_values_low = prob_low[marker].values
                prob_values_high = prob_high[marker].values
                cond_value = condition_values[marker_idx]

                if cond_value == 1:
                    post_prob[:, marker_idx] = prob_values_high
                else:
                    post_prob[:, marker_idx] = prob_values_low
            

            subtype_probs = np.prod(post_prob, axis=1)

            if len(conditions)>1:
                celltype_name = f"{cell_type}_sub{sub_idx}"
            else:
                celltype_name = cell_type
            all_subtypes_dict[celltype_name] = subtype_probs

    all_celltype_probs = pd.DataFrame(all_subtypes_dict, index=adata.obs_names)

    return all_celltype_probs


def calculate_marker_expression_posteriors(adata, marker, seed=0):
    """
    Estimate per-cell posterior probabilities of a marker being "low" vs "high".

    This function fits a 2-component Gaussian Mixture Model (GMM) to the expression values
    of a single marker across all cells, then derives a threshold separating the two modes.
    Finally, it converts raw expression values into smooth probabilities using a logistic
    function centered at that threshold.
    """

    data = adata[:, marker].X.copy()

    print(marker, end=" ")
    min_val = np.min(data)
    max_val = np.max(data)
    data_fit = data.reshape(-1, 1)
        
    gmm = GaussianMixture(n_components=2, random_state=seed)
    gmm.fit(data_fit)


    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    weights = gmm.weights_
    
    sorted_indices = np.argsort(means)
    means_sorted = means[sorted_indices]
    covariances_sorted = covariances[sorted_indices]
    weights_sorted = weights[sorted_indices]

    mean1, mean2 = means_sorted[-2:]
    std1, std2 = np.sqrt(covariances_sorted[-2:])
    weight1, weight2 = weights_sorted[-2:]

    roots = gaussian_mixture_intersections(mean1, std1, weight1, mean2, std2, weight2, eps=params.eps)
    roots = [r for r in roots if min_val <= r <= max_val]

    if len(roots) > 0:
        threshold = np.sort(roots)[-1]
    else:
        threshold = 0.5 * (mean1 + mean2)
    print(f"Data range: [{min_val:.4f}, {max_val:.4f}], GMM(2) threshold: {threshold:.4f}")
    k = 10 / (max_val - min_val)
    high_prob_logistic = 1 / (1 + np.exp(-k * (data - threshold)))
    high_prob_logistic = two_stage_normalization(high_prob_logistic, eps=params.eps)
    low_prob_logistic = 1 - high_prob_logistic
    
    return low_prob_logistic, high_prob_logistic


def cal_marker_score(adata, marker_list, seed=0):

    cdf_values = pd.DataFrame(np.zeros((adata.n_obs, len(marker_list))), index=adata.obs_names, columns=marker_list)
    cdf_values_high = pd.DataFrame(np.zeros((adata.n_obs, len(marker_list))), index=adata.obs_names, columns=marker_list)
    
    for marker in marker_list:
        cdf_value, cdf_value_high = calculate_marker_expression_posteriors(adata, marker, seed=seed)
        cdf_values[marker] = cdf_value
        cdf_values_high[marker] = cdf_value_high
   
    return cdf_values, cdf_values_high


def get_autoAnno_data(adata_path, marker_path, lib_name, seed=0):
    """
    Load an AnnData object and generate confident pseudo-labels via marker-based auto-gating.

    Parameters
    ----------
    adata_path : str
        Path to the input .h5ad file.
    marker_path : str
        Path to the marker prior CSV file. Rows are cell types; columns are markers.
    lib_name : str
        Column name to store pseudo labels in `adata.obs`.
    seed : int, default=0
        Random seed.

    """

    # ----------- Load and preprocess data ----------- 
    adata = sc.read_h5ad(adata_path)
    prior_info = pd.read_csv(marker_path, index_col=0)

    marker_list = list(prior_info.columns)
    marker_conditions = defaultdict(list)
    celltype_list = []

    for cell_type in prior_info.index:
        marker_dict = prior_info.loc[cell_type].to_dict()
        if '#' in cell_type:
            cell_type = cell_type.split('#')[0]
        marker_conditions[cell_type].append(marker_dict)

        if cell_type not in celltype_list:
            celltype_list.append(cell_type)


    if not params.isArcsinhed:
        min_vals = adata.X.min(axis=0)
        adata_pos = adata.X - min_vals
        processed_adata_X = np.arcsinh(adata_pos / params.cofactor)
        processed_adata = anndata.AnnData(processed_adata_X, obs=adata.obs, var=adata.var)
    else:
        processed_adata = adata.copy()


    # ----------- Pseudo gating ----------- 
    try:
        cdf_values, cdf_values_high = cal_marker_score(processed_adata, marker_list, seed=seed)
    except ValueError as e:
        raise e


    pseudo_gating_adata = processed_adata[cdf_values_high.index]
    # Compute per-cell-type probabilities
    probs = filter_cells_by_likelihood(pseudo_gating_adata, marker_conditions, cdf_values, cdf_values_high, celltype_list, marker_list)
        
    probs = probs.replace(0, params.eps)
    predicted_probs = probs.max(axis=1)
    predicted_labels = probs.idxmax(axis=1)

    second_max_probs = np.partition(probs.values, -2, axis=1)[:, -2]
    conf = predicted_probs.values / (second_max_probs + params.eps)

    # high-confidence cells picking
    df_tmp = pd.DataFrame({
        "predicted_label": predicted_labels,
        "conf": conf,
        "first_prob": predicted_probs,
        "second_prob": second_max_probs}, index=probs.index)
    final_labels = pd.Series("Unlabeled", index=probs.index, dtype=str)
    top_selected = set()
    for ct in probs.columns:
        sub = df_tmp[df_tmp["predicted_label"] == ct]
        if len(sub) == 0:
            continue
        sub_sorted = sub.sort_values(by="conf", ascending=False)
        top_frac = params.top_percentage
        k = max(1, int(len(sub_sorted) * top_frac))
        selected_idx = sub_sorted.index[:k]
        top_selected.update(selected_idx)

    selected = list(top_selected)
    final_labels.loc[selected] = df_tmp.loc[selected, "predicted_label"]

    pseudo_gating_adata.obs[lib_name] = final_labels

    # Store high-confidence labels
    adata.obs[lib_name] = 'Unlabeled'
    adata.obs.loc[pseudo_gating_adata.obs_names, lib_name] = pseudo_gating_adata.obs[lib_name]
    # Store per-cell probability table in obsm
    adata.obsm[f'{lib_name}_probs'] = probs
 
    return adata