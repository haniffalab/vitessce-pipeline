#!/usr/bin/env python3
"""
process_h5ad.py
====================================
Processes H5AD files into AnnData-Zarr
"""

from __future__ import annotations
import typing as T
import os
import fire
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import h5py
import zarr
import logging
import warnings
from scipy.sparse import spmatrix, csr_matrix, csc_matrix
from constants.suffixes import ANNDATA_ZARR_SUFFIX

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.INFO)


def h5ad_to_zarr(
    path: str = None,
    stem: str = "",
    adata: ad.AnnData = None,
    chunk_size: int = 10,
    batch_processing: bool = False,
    batch_size: int = 10000,
    consolidate_metadata: bool = True,
    **kwargs,
) -> str:
    """This function takes an AnnData object or path to an h5ad file,
    ensures data is of an appropriate data type for Vitessce
    and writes the object to Zarr.

    Args:
        path (str, optional): Path to the h5ad file. Defaults to None.
        stem (str, optional): Prefix for the output file. Defaults to "".
        adata (AnnData, optional): AnnData object to process. Supersedes `path`.
            Defaults to None.
        chunk_size (int, optional): Output Zarr column chunk size. Defaults to 10.
        batch_processing (bool, optional): If the expression matrix will be written to Zarr incrementally.
            Use to avoid loading the whole AnnData into memory. Defaults to False.
        batch_size (int, optional): The amount of rows (if matrix is in CSR format)
            or columns (if matrix is dense or in CSC format) of the expression matrix
            to process at a time when batch processing. Defaults to 10000.

    Raises:
        SystemError: If `batch_processing` is True and the matrix contains an `indptr` key
            but the matrix is not in scipy.sparse.csr_matrix nor scipy.sparse.csc_matrix format

    Returns:
        str: Output Zarr filename
    """

    if not adata:
        if not batch_processing:
            adata = sc.read(path)
        else:
            batch_size = max(1, batch_size)
            logging.info("Batch processing with batch size {}".format(batch_size))
            with h5py.File(path, "r") as f:
                # Load everything but X and layers
                adata = ad.AnnData(
                    obs=ad._io.h5ad.read_elem(f["obs"]) if "obs" in f else None,
                    var=ad._io.h5ad.read_elem(f["var"]) if "var" in f else None,
                    obsm=ad._io.h5ad.read_elem(f["obsm"]) if "obsm" in f else None,
                    obsp=ad._io.h5ad.read_elem(f["obsp"]) if "obsp" in f else None,
                    varm=ad._io.h5ad.read_elem(f["varm"]) if "varm" in f else None,
                    varp=ad._io.h5ad.read_elem(f["varp"]) if "varp" in f else None,
                    uns=ad._io.h5ad.read_elem(f["uns"]) if "uns" in f else None,
                )

    adata = preprocess_anndata(adata, **kwargs)

    zarr_file = (
        f"{stem}-{ANNDATA_ZARR_SUFFIX}"
        if not stem.endswith("-" + os.path.splitext(ANNDATA_ZARR_SUFFIX)[0])
        else f"{stem}{os.path.splitext(ANNDATA_ZARR_SUFFIX)[1]}"
    )

    if not batch_processing:
        # matrix sparse to dense
        if isinstance(adata.X, spmatrix):
            # use toarray() as it generates a ndarray, instead of todense() which generates a matrix
            adata.X = adata.X.toarray()

        adata.write_zarr(zarr_file, [adata.shape[0], chunk_size])
    elif path and batch_processing:
        adata.write_zarr(zarr_file)

        m = len(adata.obs)
        n = len(adata.var)

        with h5py.File(path, "r") as f:
            if isinstance(f["X"], h5py.Group) and "indptr" in f["X"].keys():
                if len(f["X"]["indptr"]) - 1 == m:
                    logging.info("Batch processing sparse CSR matrix...")
                    batch_process_sparse(path, zarr_file, m, n, batch_size, chunk_size)
                elif len(f["X"]["indptr"]) - 1 == n:
                    logging.info("Batch processing sparse CSC matrix...")
                    batch_process_sparse(
                        path, zarr_file, m, n, batch_size, chunk_size, is_csc=True
                    )
                else:
                    raise SystemError("Error identifying sparse matrix format")
            else:
                logging.info("Batch processing dense matrix...")
                batch_process_array(path, zarr_file, m, n, batch_size, chunk_size)

    if consolidate_metadata:
        zarr.consolidate_metadata(zarr_file)

    return zarr_file


def reindex_anndata_obs(adata: ad.AnnData) -> ad.AnnData:
    # check if index is numerical, if not reindex
    if not adata.obs.index.is_integer() and not (
        adata.obs.index.is_object() and all(adata.obs.index.str.isnumeric())
    ):
        IDX_NAME = "label_id"
        if IDX_NAME in adata.obs:
            adata.obs.rename(columns={IDX_NAME: f"_{IDX_NAME}"})
        adata.obs = adata.obs.reset_index(names=IDX_NAME)
        adata.obs.index = (
            pd.Categorical(adata.obs[IDX_NAME]).codes + 1
        )  # avoid 0's for possible label images
        adata.uns["webatlas_reindexed"] = True
    adata.obs.index = adata.obs.index.astype(str)

    return adata


def subset_anndata(
    adata: ad.AnnData,
    obs_subset: tuple[str, T.Any] = None,
    var_subset: tuple[str, T.Any] = None,
) -> ad.AnnData:
    # Subset adata by obs
    if obs_subset:
        obs_subset[1] = (
            [obs_subset[1]]
            if not isinstance(obs_subset[1], (list, tuple))
            else obs_subset[1]
        )
        adata = adata[adata.obs[obs_subset[0]].isin(obs_subset[1])]

    # Subset adata by var
    if var_subset:
        var_subset[1] = (
            [var_subset[1]]
            if not isinstance(var_subset[1], (list, tuple))
            else var_subset[1]
        )
        adata = adata[:, adata.var[var_subset[0]].isin(var_subset[1])]

    return adata


def preprocess_anndata(
    adata: ad.AnnData,
    compute_embeddings: bool = False,
    var_index: str = None,
    obs_subset: tuple[str, T.Any] = None,
    var_subset: tuple[str, T.Any] = None,
    **kwargs,
):
    """This function preprocesses an AnnData object, ensuring correct dtypes for zarr conversion

    Args:
        adata (AnnData): AnnData object to preprocess.
        compute_embeddings (bool, optional): If `X_umap` and `X_pca` embeddings will be computed.
            Defaults to False.
        var_index (str, optional): Alternative `var` column name with `var` names
            to be used in the visualization. Defaults to None.
        obs_subset (tuple(str, T.Any), optional): Tuple containing an `obs` column name and one or more values
            to use to subset the AnnData object. Defaults to None.
        var_subset (tuple(str, T.Any), optional): Tuple containing a `var` column name and one or more values
            to use to subset the AnnData object. Defaults to None.
    """

    adata = subset_anndata(adata, obs_subset=obs_subset, var_subset=var_subset)

    # reindex var with a specified column
    if var_index and var_index in adata.var:
        try:
            adata.var.reset_index(inplace=True)
        except ValueError:
            logging.warning(
                "Column already exists when trying to reset var index. Dropping index."
            )
            adata.var.reset_index(inplace=True, drop=True)
        adata.var.set_index(var_index, inplace=True)
        adata.var.index = adata.var.index.astype(str)
    adata.var_names_make_unique()

    adata = reindex_anndata_obs(adata)

    # turn obsm into a numpy array
    for k in adata.obsm_keys():
        adata.obsm[k] = np.array(adata.obsm[k])

    # compute embeddings if not already stored in object
    if compute_embeddings:
        if "X_pca" not in adata.obsm:
            sc.tl.pca(adata)
        if "X_umap" not in adata.obsm:
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)

    # ensure data types for obs
    for col in adata.obs:
        if adata.obs[col].dtype in ["int8", "int64"]:
            adata.obs[col] = adata.obs[col].astype("int32")
        if adata.obs[col].dtype == "bool":
            adata.obs[col] = adata.obs[col].astype(str).astype("category")

    # ensure data types for obsm
    for col in adata.obsm:
        if type(adata.obsm[col]).__name__ in ["DataFrame", "Series"]:
            adata.obsm[col] = adata.obsm[col].to_numpy()
        if adata.obsm[col].dtype in ["int8", "int64"] or col == "spatial":
            adata.obsm[col] = adata.obsm[col].astype("int32")

    # remove unnecessary data
    del adata.raw

    return adata


def batch_process_sparse(
    file: str,
    zarr_file: str,
    m: int,
    n: int,
    batch_size: int,
    chunk_size: int,
    is_csc: bool = False,
) -> None:
    """Function to incrementally load and write a sparse matrix to Zarr

    Args:
        file (str): Path to h5ad file
        zarr_file (str): Path to output Zarr file
        m (int): Number of rows in the matrix
        n (int): Number of columns in the matrix
        batch_size (int): Number of rows/columns to load and write at a time
        chunk_size (int): Output Zarr column chunk size
        is_csc (bool, optional): If matrix is in CSC format instead of CSR format.
            Defaults to False.
    """
    l = n if is_csc else m
    z = zarr.open(zarr_file, mode="a")
    z["X"] = zarr.empty(
        (m if is_csc else 0, 0 if is_csc else n),
        chunks=(m, chunk_size),
        dtype="float32",
    )
    with h5py.File(file, "r") as f:
        indptr = f["X"]["indptr"][:]
        batch_size = l if batch_size > l else batch_size
        for i in range(l // batch_size + 1):
            j = i * batch_size
            if j + batch_size > l:
                batch_size = l - j
            k = j + batch_size

            indices = f["X"]["indices"][indptr[j] : indptr[k]]
            data = f["X"]["data"][indptr[j] : indptr[k]]

            if is_csc:
                matrix = csc_matrix(
                    (data, indices, indptr[j : k + 1] - indptr[j]),
                    shape=(m, 1 * batch_size),
                )
            else:
                matrix = csr_matrix(
                    (data, indices, indptr[j : k + 1] - indptr[j]),
                    shape=(1 * batch_size, n),
                )

            z["X"].append(matrix.toarray(), axis=(1 * is_csc))
    return


def batch_process_array(
    file: str, zarr_file: str, m: int, n: int, batch_size: int, chunk_size: int
) -> None:
    """Function to incrementally load and write a dense matrix to Zarr

    Args:
        file (str): Path to h5ad file
        zarr_file (str): Path to output Zarr file
        m (int): Number of rows in the matrix
        n (int): Number of columns in the matrix
        batch_size (int): Number of columns to load and write at a time
        chunk_size (int): Output Zarr column chunk size
    """
    z = zarr.open(zarr_file, mode="a")
    z["X"] = zarr.empty((m, 0), chunks=(m, chunk_size), dtype="float32")

    with h5py.File(file, "r") as f:
        for i in range(n // batch_size + 1):
            j = i * batch_size
            if j + batch_size > n:
                batch_size = n - j
            k = j + batch_size

            matrix = f["X"][:, j:k]

            z["X"].append(matrix, axis=1)
    return


if __name__ == "__main__":
    fire.Fire(h5ad_to_zarr)
