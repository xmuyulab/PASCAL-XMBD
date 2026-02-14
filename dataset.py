from auto_anno import get_autoAnno_data
import numpy as np
import scipy.sparse as sp
from typing import Optional
from config import MARKER_PRIOR_PATHS, TEST_DATA_PATHS

def load_marker_prior(dataset_name):
    try:
        return MARKER_PRIOR_PATHS[dataset_name]
    except KeyError:
        raise ValueError(f"dataset name {dataset_name} is unknown in MARKER_PRIOR_PATHS.")


def load_byName_test(dataset_name):
    try:
        return TEST_DATA_PATHS[dataset_name]
    except KeyError:
        raise ValueError(f"dataset name {dataset_name} is unknown in TEST_DATA_PATHS.")


def adata_to_array(
    adata,
    label_key: str,
    y_dtype = np.int8,
    X_dtype = np.float32,
):
    """
    Convert AnnData -> (X, y)

    - X: numpy array of shape (n_samples, n_features)
    - y: integer labels from adata.obs[label_key]
        - "Unlabeled" is encoded as -1
        - all other labels are encoded as 0, 1, 2, ... in alphabetical order
    """
    if label_key not in adata.obs:
        raise KeyError(f"obs not found: {label_key}")

    X_mat = adata.X
    if sp.issparse(X_mat):
        X = X_mat.toarray().astype(X_dtype, copy=False)
    else:
        X = np.asarray(X_mat, dtype=X_dtype)


    y_series = adata.obs[label_key].astype(str)

    # Build encoding map: "Unlabeled" -> -1, others sorted alphabetically
    unique_labels = sorted(set(y_series.unique()) - {"Unlabeled"})
    class_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    class_to_id["Unlabeled"] = -1

    id_to_class = {v: k for k, v in class_to_id.items()}

    # Generate encoded label array
    y = y_series.map(class_to_id).astype(y_dtype)

    classes = [id_to_class[i] for i in sorted(class_to_id.values()) if i >= 0]
    print('classes:', classes)

    return X, y, classes, class_to_id, id_to_class


def read_for_label_propagation(dataname, lib_name, seed=0):
    
    """
    Load an AnnData object and prepare inputs for label propagation.
    Parameters
    ----------
    dataname : str
        Dataset identifier used as a key for looking up:
        - the marker prior path (MARKER_PRIOR_PATHS[dataname])
        - the test data path (TEST_DATA_PATHS[dataname])

    lib_name : str
        Column name in `adata.obs` that contains the confident cell labels produced by
        the auto-gating step.
        Convention:
        - labeled cells are stored as string class names
        - unlabeled cells should be stored as "Unlabeled" (which will be encoded as -1)

    seed : int, default=0
    """

    marker_prior_path = load_marker_prior(dataname)
    adata_path = load_byName_test(dataname)

    adata = get_autoAnno_data(adata_path, marker_prior_path, lib_name=lib_name, seed=seed)
    
    X, y, classes, class_to_id, id_to_class = adata_to_array(
        adata,
        label_key=lib_name,
        y_dtype=np.int8,
        X_dtype=np.float32,
    )
    
    return adata, X, y, id_to_class