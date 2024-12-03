#!/usr/bin/env python3

from typing import Union
import typing as T
import os
import gc
import fire
import zarr
import h5py
import logging
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import spmatrix, hstack, csr_matrix, csc_matrix
from process_h5ad import h5ad_to_zarr
from pathlib import Path


def reindex_and_concat(
    path: str, offset: int, features: str = None, args: dict[str, T.Any] = {}, **kwargs
):
    adata = read_anndata(path)

    adata = reindex_anndata(adata, offset, **args, **kwargs)
    if features:
        adata = concat_features(adata, features, **args, **kwargs)

    out_filename = "reindexed-concat-{}".format(
        os.path.splitext(os.path.basename(path))[0]
    )
    write_anndata(adata, out_filename, **args, **kwargs)

    return


def reindex_anndata(
    data: Union[ad.AnnData, str],
    offset: int,
    no_save: bool = True,
    out_filename: str = None,
    **kwargs,
):
    if isinstance(data, ad.AnnData):
        adata = data
    else:
        adata = read_anndata(data)
        out_filename = out_filename or "reindexed-{}".format(
            os.path.splitext(os.path.basename(data))[0]
        )

    adata.obs.index = (adata.obs.index.astype(int) + offset).astype(str)

    if no_save:
        return adata
    else:
        write_anndata(adata, out_filename, **kwargs)
        return


def concat_features(
    data: Union[ad.AnnData, str],
    features: str,
    no_save: bool = True,
    out_filename: str = None,
    **kwargs,
):
    if isinstance(data, ad.AnnData):
        adata = data
    else:
        adata = read_anndata(data)
        out_filename = out_filename or "concat-{}".format(
            os.path.splitext(os.path.basename(data))[0]
        )

    if features.endswith(".h5ad") and os.path.isfile(features):
        adata = concat_matrix_from_cell2location(adata, features, **kwargs)
    elif features.startswith("obs/"):
        adata = concat_matrix_from_obs(adata, features.split("/")[1], **kwargs)
    elif features.startswith("obsm/"):
        adata = concat_matrix_from_obsm(adata, features.split("/")[1], **kwargs)

    if no_save:
        return adata
    else:
        write_anndata(adata, out_filename, **kwargs)
        return


def intersect_features(*paths, **kwargs):
    var_intersect = get_feature_intersection(*paths)

    for path in paths:
        adata = read_anndata(path)

        adata = adata[:, var_intersect]

        out_filename = "intersect-{}".format(
            os.path.splitext(os.path.basename(path))[0]
        )

        write_anndata(adata, out_filename, **kwargs)

        del adata
        gc.collect()

    return


def concat_matrix_from_obs(
    data: Union[ad.AnnData, str],
    obs: str = "celltype",
    feature_name: str = "gene",
    concat_feature_name: str = None,
):
    if isinstance(data, ad.AnnData):
        adata = data
    else:
        adata = read_anndata(data)

    ext_matrix = pd.get_dummies(adata.obs[obs], dtype="float32")

    return concat_matrices(adata, ext_matrix, feature_name, concat_feature_name or obs)


def concat_matrix_from_obsm(
    data: Union[ad.AnnData, str],
    obsm: str = "celltype",
    feature_name: str = "gene",
    concat_feature_name: str = None,
):
    if isinstance(data, ad.AnnData):
        adata = data
    else:
        adata = read_anndata(data)

    return concat_matrices(
        adata, adata.obsm[obsm], feature_name, concat_feature_name or obsm
    )


def concat_matrix_from_cell2location(
    data: Union[ad.AnnData, str],
    c2l_file: str,
    q: str = "q05_cell_abundance_w_sf",
    sample: tuple[str, str] = None,
    feature_name: str = "gene",
    concat_feature_name: str = "celltype",
    sort: bool = True,
    sort_index: str = None,
    fill_missing: bool = False,
    **kwargs,
):
    sort = sort or sort_index is not None
    if isinstance(data, ad.AnnData):
        adata = data
    else:
        adata = read_anndata(data)

    with h5py.File(c2l_file) as f:
        c2l_adata = ad.AnnData(
            obs=ad._io.h5ad.read_elem(f["obs"]) if "obs" in f else None,
            var=ad._io.h5ad.read_elem(f["var"]) if "var" in f else None,
            obsm=ad._io.h5ad.read_elem(f["obsm"]) if "obsm" in f else None,
        )

    if sample:
        c2l_adata = c2l_adata[c2l_adata.obs[sample[0]] == sample[1]]

    if sort:
        if not sort_index and adata.uns.get("webatlas_reindexed"):
            sort_index = "label_id"
        if sort_index:
            data_idx = adata.obs[sort_index]
        else:
            data_idx = adata.obs.index
        idx = c2l_adata.obs.index.get_indexer(data_idx.tolist())
        if -1 in idx:  # Indices do not match
            logging.error(
                "Values do not match between AnnData object's"
                f" `{sort_index or 'index'}`"
                " and cell2location output index."
            )

            logging.info("Attempting to match indices as substrings")
            try:
                data_idx = match_substring_indices(c2l_adata.obs.index, data_idx)
                if not data_idx.is_unique:
                    raise Exception(
                        "Found non-unique matches between indices as substrings."
                    )
                idx = c2l_adata.obs.index.get_indexer(data_idx.tolist())
                if -1 in idx:
                    if not fill_missing:
                        raise Exception("Non-matching indices present.")
                    else:
                        logging.info(
                            "Filling missing indices in cell2location output with NaN values."
                        )
                        c2l_adata, idx = fill_missing_indices(
                            idx, data_idx, adata, c2l_adata, sort_index
                        )
            except Exception:
                raise SystemError(
                    "Failed to find a match between indices as substrings."
                )

        c2l_adata = c2l_adata[idx,]

    c2l_df = pd.DataFrame(
        c2l_adata.obsm[q].to_numpy(),
        index=c2l_adata.obs.index,
        columns=c2l_adata.obsm[q].columns.str.replace(
            q.split("_")[0] + "cell_abundance_w_sf_", ""
        ),
        dtype="float32",
    )

    return concat_matrices(adata, c2l_df, feature_name, concat_feature_name, **kwargs)


def concat_matrices(
    adata: ad.AnnData,
    ext_df: pd.DataFrame,
    feature_name: str = "gene",
    concat_feature_name: str = "celltype",
):
    assert adata.shape[0] == ext_df.shape[0]

    prev_features_bool = "is_{}".format(feature_name)
    new_features_bool = "is_{}".format(concat_feature_name)

    if isinstance(adata.X, spmatrix):
        adata_concat = ad.AnnData(
            hstack(
                (
                    adata.X,
                    (
                        csr_matrix(ext_df.values)
                        if isinstance(adata.X, csr_matrix)
                        else csc_matrix(ext_df.values)
                    ),
                )
            ),
            obs=adata.obs,
            var=pd.concat(
                [
                    adata.var.assign(**{prev_features_bool: True}),
                    ext_df.columns.to_frame(concat_feature_name)
                    .drop(columns=0)
                    .assign(**{new_features_bool: True}),
                ]
            ),
            obsm=adata.obsm,
            uns=adata.uns,
        )
    else:
        adata_concat = ad.AnnData(
            np.hstack((adata.X, ext_df.values)),
            obs=adata.obs,
            var=pd.concat(
                [
                    adata.var.assign(**{prev_features_bool: True}),
                    ext_df.columns.to_frame(concat_feature_name)
                    .drop(columns=0)
                    .assign(**{new_features_bool: True}),
                ]
            ),
            obsm=adata.obsm,
            uns=adata.uns,
        )

    adata_concat.var[prev_features_bool] = adata_concat.var[prev_features_bool].fillna(
        False
    )
    adata_concat.var[new_features_bool] = adata_concat.var[new_features_bool].fillna(
        False
    )

    for col in [col for col in adata.var_keys() if adata.var[col].dtype == bool]:
        adata_concat.var[col] = adata_concat.var[col].fillna(False)

    return adata_concat


def get_feature_intersection(*paths):
    var_indices = []
    for path in paths:
        is_zarr = path.split(".")[-1] == "zarr"
        if is_zarr:
            z = zarr.open(path, "r")
            var_idx = z.var.attrs["_index"] if "_index" in z.var.attrs else "_index"
            var_indices.append(pd.Index(z.var[var_idx][:]).to_series())
        else:
            with h5py.File(path, "r") as f:
                var_indices.append(ad._io.h5ad.read_elem(f["var"]).index.to_series())

    var_intersect = pd.concat(var_indices, axis=1, join="inner").index
    logging.info(f"Got intersection of {len(var_intersect)} features")

    return var_intersect


def read_anndata(path: str):
    is_zarr = os.path.splitext(path)[-1] == ".zarr"

    if is_zarr:
        z = zarr.open(path)
        adata = ad.read_zarr(z.store)
    else:
        adata = ad.read(path)

    return adata


def write_anndata(
    adata: ad.AnnData, out_filename: str, save_h5ad: bool = False, **kwargs
):
    if save_h5ad:
        adata.write_h5ad(f"{out_filename}.h5ad")

    h5ad_to_zarr(adata=adata, stem=Path(out_filename).stem, **kwargs)

    return


def match_substring_indices(fullstring_idx, substring_idx):
    return pd.Series(substring_idx).apply(
        lambda x: (
            lambda y=fullstring_idx[fullstring_idx.str.contains(x)]: (
                y.values[0] if len(y.values) else x
            )
        )()
    )


def fill_missing_indices(idx, data_idx, adata, c2l_adata, sort_index):
    adata_missing = ad.AnnData(obs=pd.DataFrame(adata.obs.iloc[idx == -1][sort_index]))
    adata_missing.obs.set_index(sort_index, inplace=True)
    c2l_adata_w_missing = ad.concat([c2l_adata, adata_missing], join="outer")
    idx = c2l_adata_w_missing.obs.index.get_indexer(data_idx.tolist())
    return c2l_adata_w_missing, idx


if __name__ == "__main__":
    fire.Fire()
