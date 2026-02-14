import numpy as np
import pandas as pd
from anndata import AnnData
import fcsparser


def read_fcs(
    path: str,
    meta_path:str = None,
    exclude_regex: str = "fsc|ssc|time|zombie",
    transform: bool = False,
    cofactor: int = 5
) -> AnnData:
    """Read a FCS file and return an `AnnData` object.

    Args:
        path: Path to the FCS file to be read.
        meta_path: Path to the cell meatdata.
        exclude_regex: Regex used to identify columns that do not correspond to markers. You can add names to the regex by including the lowercase name after a new `|` in the string.
        transform: Indicates whether data transformation processing is required.

    Returns:
        `AnnData` object containing the FCS data.
    """
    _, data = fcsparser.parse(path)

    is_marker = ~data.columns.str.lower().str.contains(exclude_regex)
    marker_cols = data.columns[is_marker]
    non_marker_cols = data.columns[~is_marker]

    
    obs_m = pd.read_csv(meta_path, index_col=0)

    if len(non_marker_cols) > 0:
        non_marker_data = data.loc[:, non_marker_cols]
        non_marker_data.index = obs_m.index
        obs_m = pd.concat([obs_m, non_marker_data], axis=1)

    if transform==True:
        data[marker_cols] = data[marker_cols].apply(lambda x: np.arcsinh(x / cofactor))

    adata = AnnData(
        X=data.loc[:, marker_cols].values.astype(np.float32),
        var=pd.DataFrame(index=marker_cols),
        obs=obs_m,
    )

    return adata


def read_CSV(
    path: str,
    meta_path:str = None,
    transform: bool = False,
    cofactor: int = 5
) -> AnnData:
    """Read a CSV file and return an `AnnData` object.

    Args:
        path: Path to the CSV file to be read.
        meta_path: Path to the cell meatdata.
        transform: Indicates whether data transformation processing is required.
        cofactor: 
    Returns:
        `AnnData` object containing the expression data.
    """
    
    data = pd.read_csv(path, index_col=0)
    obs_m = pd.read_csv(meta_path, index_col=0)

    if transform==True:
        data = data.apply(lambda x: np.arcsinh(x / cofactor))

    adata = AnnData(
        X=data.values.astype(np.float32),
        var=pd.DataFrame(index=data.columns),
        obs=obs_m,
    )

    return adata