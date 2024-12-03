#!/usr/bin/env python3
"""
process_spaceranger.py
====================================
Processes SpaceRanger output
"""

from __future__ import annotations
import os
import fire
import shutil
import typing as T
import numpy as np
import scanpy as sc
import pandas as pd
import tifffile as tf
from pathlib import Path
from skimage.draw import disk
from process_h5ad import h5ad_to_zarr, reindex_anndata_obs, subset_anndata


def spaceranger_to_anndata(
    path: str,
    load_clusters: bool = True,
    load_embeddings: bool = True,
    load_raw: bool = False,
) -> sc.AnnData:
    """Function to create an AnnData object from a SpaceRanger output directory.

    Args:
        path (str): Path to a SpaceRanger output directory
        load_clusters (bool, optional): If cluster files should be included in the
            AnnData object. Defaults to True.
        load_embeddings (bool, optional): If embedding coordinates files should be included
            in the AnnData object. Defaults to True.
        load_raw (bool, optional): If the raw matrix count file should be loaded
            instead of the filtered matrix. Defaults to False.

    Returns:
        AnnData: AnnData object created from the SpaceRanger output data
    """

    p = Path(path)

    # Temporary fix to support spacerangerr v1.2.0 and v2.0.
    # until scanpy 1.10.0 is released and scanpy.read_visium can handle it
    if (p / "spatial" / "tissue_positions.csv").is_file():
        shutil.copyfile(
            p / "spatial" / "tissue_positions.csv",
            p / "spatial" / "tissue_positions_list.csv",
        )

    adata = sc.read_visium(
        p,
        count_file=(
            "raw_feature_bc_matrix.h5" if load_raw else "filtered_feature_bc_matrix.h5"
        ),
    )

    if load_clusters:
        for cluster in [
            d for d in (p / "analysis" / "clustering").iterdir() if d.is_dir()
        ]:
            cluster_name = cluster.name.replace("gene_expression_", "")
            cluster_df = pd.read_csv(
                cluster / "clusters.csv",
                index_col="Barcode",
            )

            clusters = cluster_df.reindex(adata.obs.index)
            adata.obs[cluster_name] = pd.Categorical(clusters["Cluster"])

    if load_embeddings:
        embeddings = [
            ("umap", "2_components"),
            ("tsne", "2_components"),
            ("pca", "10_components"),
        ]
        for embedding, components in embeddings:
            components_name = (
                components
                if (p / "analysis" / embedding / components).exists()
                else f"gene_expression_{components}"
            )
            embedding_df = pd.read_csv(
                (p / "analysis" / embedding / components_name / "projection.csv"),
                index_col="Barcode",
            )

            emb = embedding_df.reindex(adata.obs.index)
            adata.obsm[f"X_{embedding}"] = emb.values

    return adata


def spaceranger_to_zarr(
    path: str,
    stem: str,
    load_clusters: bool = True,
    load_embeddings: bool = True,
    load_raw: bool = False,
    save_h5ad: bool = False,
    **kwargs,
) -> str:
    """Function to write to Zarr an AnnData object created from SpaceRanger output data

    Args:
        path (str): Path to a SpaceRanger output directory
        stem (str): Prefix for the output Zarr filename
        load_clusters (bool, optional): If cluster files should be included in the
            AnnData object. Defaults to True.
        load_embeddings (bool, optional): If embedding coordinates files should be included
            in the AnnData object. Defaults to True.
        load_raw (bool, optional): If the raw matrix count file should be loaded
            instead of the filtered matrix. Defaults to False.
        save_h5ad (bool, optional): If the AnnData object should also be written to an h5ad file.
            Defaults to False.

    Returns:
        str: Output Zarr filename
    """

    adata = spaceranger_to_anndata(path, load_clusters, load_embeddings, load_raw)
    if save_h5ad:
        adata.write_h5ad(f"tmp-{stem}.h5ad")
    zarr_file = h5ad_to_zarr(adata=adata, stem=stem, **kwargs)

    return zarr_file


def visium_label(
    stem: str,
    file_path: str,
    shape: tuple[int, int] = None,
    obs_subset: tuple[str, T.Any] = None,
    sample_id: str = None,
    relative_size: str = None,
) -> None:
    """This function writes a label image tif file with drawn labels according to an
    Anndata object with necessary metadata stored within `uns["spatial"]`.

    Args:
        stem (str): Prefix for the output image filename.
        file_path (str): Path to the h5ad file or spaceranger output directory.
        shape (tuple[int, int], optional): Output image shape. Defaults to None.
        obs_subset (tuple(str, T.Any), optional): Tuple containing an `obs` column name and one or more values
            to use to subset the AnnData object. Defaults to None.
        sample_id (str, optional): Sample ID string within the Anndata object. Defaults to None.
        relative_size (str, optional): Optional numerical `obs` column name that holds
            a multiplier for the spot diameter. Only useful for data that has been
            processed to merge spots. Defaults to None.
    """
    # sample_id = sample_id or Path(file_path).stem

    if os.path.isdir(file_path):
        adata = spaceranger_to_anndata(file_path)
    else:
        adata = sc.read(file_path)

    sample_id = sample_id or list(adata.uns["spatial"].keys())[0]

    if not shape:
        hires_shape = adata.uns["spatial"][sample_id]["images"]["hires"].shape
        scalef = adata.uns["spatial"][sample_id]["scalefactors"]["tissue_hires_scalef"]
        shape = [int(hires_shape[0] / scalef), int(hires_shape[1] / scalef)]

    adata = subset_anndata(adata, obs_subset=obs_subset)

    adata = reindex_anndata_obs(adata)

    # turn obsm into a numpy array
    for k in adata.obsm_keys():
        adata.obsm[k] = np.array(adata.obsm[k])

    spot_diameter_fullres = adata.uns["spatial"][sample_id]["scalefactors"][
        "spot_diameter_fullres"
    ]
    spot_coords = adata.obsm["spatial"]
    assert adata.obs.shape[0] == spot_coords.shape[0]

    label_img = np.zeros(
        (shape[0], shape[1]),
        dtype=np.min_scalar_type(adata.obs.index.astype(int).max()),
    )

    if relative_size and relative_size in adata.obs:
        for spId, (y, x), m in zip(
            adata.obs.index, spot_coords, adata.obs[relative_size]
        ):
            label_img[disk((int(x), int(y)), (spot_diameter_fullres / 2) * m)] = int(
                spId
            )
    else:
        for spId, (y, x) in zip(adata.obs.index, spot_coords):
            label_img[
                disk(
                    (
                        int(x),
                        int(y),
                    ),
                    spot_diameter_fullres / 2,
                    shape=shape,
                )
            ] = int(spId)

    tf.imwrite(f"{stem}-label.tif", label_img)

    return


if __name__ == "__main__":
    fire.Fire(spaceranger_to_zarr)
