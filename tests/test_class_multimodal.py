import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from bin.integrate_anndata import (
    concat_features,
    concat_matrix_from_obs,
    concat_matrix_from_obsm,
)


class TestClass:
    @pytest.fixture(scope="class")
    def anndata_with_celltype_obs(self, tmp_path_factory):
        adata = ad.AnnData(
            np.array([[100.0] * 4] * 3),
            obs=pd.DataFrame(
                index=["obs1", "obs2", "obs3"],
                data={"obs1": ["celltype1", "celltype2", "celltype3"]},
            ),
            var=pd.DataFrame(index=["var1", "var2", "var3", "var4"]),
            dtype="float32",
        )
        fn = tmp_path_factory.mktemp("data") / "anndata.h5ad"
        adata.write_h5ad(fn)
        return fn

    @pytest.fixture(scope="class")
    def anndata_with_celltype_obsm(self, tmp_path_factory):
        adata = ad.AnnData(
            np.array([[100.0] * 4] * 3),
            obs=pd.DataFrame(
                index=["obs1", "obs2", "obs3"],
            ),
            var=pd.DataFrame(index=["var1", "var2", "var3", "var4"]),
            dtype="float32",
        )
        adata.obsm["obsm1"] = pd.DataFrame(
            index=["obs1", "obs2", "obs3"],
            data={
                "celltype1": [0.1, 0.2, 0.3],
                "celltype2": [0.4, 0.5, 0.6],
                "celltype3": [0.7, 0.8, 0.9],
            },
            dtype="float32",
        )
        fn = tmp_path_factory.mktemp("data") / "anndata.h5ad"
        adata.write_h5ad(fn)
        return fn

    def test_concat_matrix_from_obs(self, monkeypatch, anndata_with_celltype_obs):
        monkeypatch.chdir(os.path.dirname(anndata_with_celltype_obs))
        adata = ad.read(anndata_with_celltype_obs)
        adata_concat = concat_matrix_from_obs(adata, "obs1")
        assert adata_concat.X.shape == (3, 7)
        assert np.array_equal(
            adata_concat.X,
            np.hstack(
                (
                    np.array([[100.0] * 4] * 3),
                    np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]),
                ),
            ),
        )
        assert all([x in adata_concat.var.columns for x in ["is_gene", "is_obs1"]])
        assert adata_concat.var["is_gene"].tolist() == [True] * 4 + [False] * 3
        assert adata_concat.var["is_obs1"].tolist() == [False] * 4 + [True] * 3
        assert all(adata_concat.var["is_gene"] + adata_concat.var["is_obs1"] == 1)

    def test_concat_features_from_obs(self, monkeypatch, anndata_with_celltype_obs):
        monkeypatch.chdir(os.path.dirname(anndata_with_celltype_obs))
        adata = ad.read(anndata_with_celltype_obs)
        adata_concat_1 = concat_matrix_from_obs(adata, "obs1")
        adata_concat_2 = concat_features(adata, "obs/obs1")
        pd.testing.assert_frame_equal(adata_concat_1.obs, adata_concat_2.obs)
        pd.testing.assert_frame_equal(adata_concat_1.var, adata_concat_2.var)
        assert np.array_equal(adata_concat_1.X, adata_concat_2.X)

    def test_concat_matrix_from_obs_with_concat_name(
        self, monkeypatch, anndata_with_celltype_obs
    ):
        monkeypatch.chdir(os.path.dirname(anndata_with_celltype_obs))
        CONCAT_FEATURE_NAME = "celltype"
        adata = ad.read(anndata_with_celltype_obs)
        adata_concat = concat_matrix_from_obs(
            adata, "obs1", concat_feature_name=CONCAT_FEATURE_NAME
        )
        assert adata_concat.X.shape == (3, 7)
        assert np.array_equal(
            adata_concat.X,
            np.hstack(
                (
                    np.array([[100.0] * 4] * 3),
                    np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]),
                ),
            ),
        )
        assert all(
            [
                x in adata_concat.var.columns
                for x in ["is_gene", f"is_{CONCAT_FEATURE_NAME}"]
            ]
        )
        assert adata_concat.var["is_gene"].tolist() == [True] * 4 + [False] * 3
        assert (
            adata_concat.var[f"is_{CONCAT_FEATURE_NAME}"].tolist()
            == [False] * 4 + [True] * 3
        )
        assert all(
            adata_concat.var["is_gene"] + adata_concat.var[f"is_{CONCAT_FEATURE_NAME}"]
            == 1
        )

    def test_concat_features_from_obs_with_concat_name(
        self, monkeypatch, anndata_with_celltype_obs
    ):
        monkeypatch.chdir(os.path.dirname(anndata_with_celltype_obs))
        CONCAT_FEATURE_NAME = "celltype"
        adata = ad.read(anndata_with_celltype_obs)
        adata_concat_1 = concat_matrix_from_obs(
            adata, "obs1", concat_feature_name=CONCAT_FEATURE_NAME
        )
        adata_concat_2 = concat_features(
            adata, "obs/obs1", concat_feature_name=CONCAT_FEATURE_NAME
        )
        pd.testing.assert_frame_equal(adata_concat_1.obs, adata_concat_2.obs)
        pd.testing.assert_frame_equal(adata_concat_1.var, adata_concat_2.var)
        assert np.array_equal(adata_concat_1.X, adata_concat_2.X)

    def test_concat_matrix_from_obsm(self, monkeypatch, anndata_with_celltype_obsm):
        monkeypatch.chdir(os.path.dirname(anndata_with_celltype_obsm))
        adata = ad.read(anndata_with_celltype_obsm)
        adata_concat = concat_matrix_from_obsm(adata, "obsm1")
        assert adata_concat.X.shape == (3, 7)
        assert np.array_equal(
            adata_concat.X,
            np.hstack(
                (
                    np.array([[100.0] * 4] * 3),
                    np.array([[0.1, 0.4, 0.7], [0.2, 0.5, 0.8], [0.3, 0.6, 0.9]]),
                ),
                dtype="float32",
            ),
        )
        assert all([x in adata_concat.var.columns for x in ["is_gene", "is_obsm1"]])
        assert adata_concat.var["is_gene"].tolist() == [True] * 4 + [False] * 3
        assert adata_concat.var["is_obsm1"].tolist() == [False] * 4 + [True] * 3
        assert all(adata_concat.var["is_gene"] + adata_concat.var["is_obsm1"] == 1)

    def test_concat_features_from_obsm(self, monkeypatch, anndata_with_celltype_obsm):
        monkeypatch.chdir(os.path.dirname(anndata_with_celltype_obsm))
        adata = ad.read(anndata_with_celltype_obsm)
        adata_concat_1 = concat_matrix_from_obsm(adata, "obsm1")
        adata_concat_2 = concat_features(adata, "obsm/obsm1")
        pd.testing.assert_frame_equal(adata_concat_1.obs, adata_concat_2.obs)
        pd.testing.assert_frame_equal(adata_concat_1.var, adata_concat_2.var)
        assert np.array_equal(adata_concat_1.X, adata_concat_2.X)

    def test_concat_matrix_from_obsm_with_concat_name(
        self, monkeypatch, anndata_with_celltype_obsm
    ):
        monkeypatch.chdir(os.path.dirname(anndata_with_celltype_obsm))
        CONCAT_FEATURE_NAME = "celltype"
        adata = ad.read(anndata_with_celltype_obsm)
        adata_concat = concat_matrix_from_obsm(
            adata, "obsm1", concat_feature_name=CONCAT_FEATURE_NAME
        )
        assert adata_concat.X.shape == (3, 7)
        assert np.array_equal(
            adata_concat.X,
            np.hstack(
                (
                    np.array([[100.0] * 4] * 3),
                    np.array([[0.1, 0.4, 0.7], [0.2, 0.5, 0.8], [0.3, 0.6, 0.9]]),
                ),
                dtype="float32",
            ),
        )
        assert all(
            [
                x in adata_concat.var.columns
                for x in ["is_gene", f"is_{CONCAT_FEATURE_NAME}"]
            ]
        )
        assert adata_concat.var["is_gene"].tolist() == [True] * 4 + [False] * 3
        assert (
            adata_concat.var[f"is_{CONCAT_FEATURE_NAME}"].tolist()
            == [False] * 4 + [True] * 3
        )
        assert all(
            adata_concat.var["is_gene"] + adata_concat.var[f"is_{CONCAT_FEATURE_NAME}"]
            == 1
        )

    def test_concat_features_from_obsm_with_concat_name(
        self, monkeypatch, anndata_with_celltype_obsm
    ):
        monkeypatch.chdir(os.path.dirname(anndata_with_celltype_obsm))
        CONCAT_FEATURE_NAME = "celltype"
        adata = ad.read(anndata_with_celltype_obsm)
        adata_concat_1 = concat_matrix_from_obsm(
            adata, "obsm1", concat_feature_name=CONCAT_FEATURE_NAME
        )
        adata_concat_2 = concat_features(
            adata, "obsm/obsm1", concat_feature_name=CONCAT_FEATURE_NAME
        )
        pd.testing.assert_frame_equal(adata_concat_1.obs, adata_concat_2.obs)
        pd.testing.assert_frame_equal(adata_concat_1.var, adata_concat_2.var)
        assert np.array_equal(adata_concat_1.X, adata_concat_2.X)
