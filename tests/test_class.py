import csv
import json
import os
from xml.etree.ElementTree import ElementTree

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import xmlschema
import zarr
from scipy.sparse import csc_matrix, csr_matrix

from bin.consolidate_md import consolidate
from bin.generate_image import create_img
from bin.ome_zarr_metadata import get_metadata
from bin.process_h5ad import h5ad_to_zarr
from bin.process_molecules import tsv_to_json
from bin.router import process

# from bin.build_config_multimodal import write_json as write_json_multimodal
# from bin.build_config import write_json


class TestClass:
    @pytest.fixture(scope="class")
    def anndata_h5ad_file(self, tmp_path_factory):
        adata = ad.AnnData(
            np.array([[100.0] * 4] * 3),
            obs=pd.DataFrame(index=["obs1", "obs2", "obs3"]),
            var=pd.DataFrame(index=["var1", "var2", "var3", "var4"]),
            dtype="float32",
        )
        adata.uns["spatial"] = {
            "anndata": {"scalefactors": {"spot_diameter_fullres": 10}}
        }
        adata.obsm["spatial"] = np.array([[20.0, 20.0], [40.0, 40.0], [60.0, 60.0]])
        fn = tmp_path_factory.mktemp("data") / "anndata.h5ad"
        adata.write_h5ad(fn)
        return fn

    @pytest.fixture(scope="class")
    def anndata_csr_h5ad_file(self, tmp_path_factory):
        adata = ad.AnnData(
            csr_matrix(np.array([[100.0] * 4] * 3)),
            obs=pd.DataFrame(index=["obs1", "obs2", "obs3"]),
            var=pd.DataFrame(index=["var1", "var2", "var3", "var4"]),
            dtype="float32",
        )
        adata.uns["spatial"] = {
            "anndata": {"scalefactors": {"spot_diameter_fullres": 10}}
        }
        adata.obsm["spatial"] = np.array([[20.0, 20.0], [40.0, 40.0], [60.0, 60.0]])
        fn = tmp_path_factory.mktemp("data") / "anndata.h5ad"
        adata.write_h5ad(fn)
        return fn

    @pytest.fixture(scope="class")
    def anndata_csc_h5ad_file(self, tmp_path_factory):
        adata = ad.AnnData(
            csc_matrix(np.array([[100.0] * 4] * 3)),
            obs=pd.DataFrame(index=["obs1", "obs2", "obs3"]),
            var=pd.DataFrame(index=["var1", "var2", "var3", "var4"]),
            dtype="float32",
        )
        adata.uns["spatial"] = {
            "anndata": {"scalefactors": {"spot_diameter_fullres": 10}}
        }
        adata.obsm["spatial"] = np.array([[20.0, 20.0], [40.0, 40.0], [60.0, 60.0]])
        fn = tmp_path_factory.mktemp("data") / "anndata.h5ad"
        adata.write_h5ad(fn)
        return fn

    @pytest.fixture(scope="class")
    def molecules_tsv_file(self, tmp_path_factory):
        fn = tmp_path_factory.mktemp("data") / "molecules.tsv"
        with open(fn, "w") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter="\t")
            csvwriter.writerow(["Name", "x_int", "y_int"])
            csvwriter.writerow(["background", "0", "0"])
        return fn

    @pytest.fixture(scope="class")
    def molecules_json_file(self, tmp_path_factory):
        fn = tmp_path_factory.mktemp("data") / "molecules.json"
        molecules_json = {"background": [[0.0, 0.0]]}
        with open(fn, "w") as out_file:
            json.dump(molecules_json, out_file)
        return fn

    @pytest.fixture(scope="class")
    def zarr_file(self, tmp_path_factory):
        fn = tmp_path_factory.mktemp("data") / "dummy.zarr"
        zarr.open(fn, mode="w", shape=(100, 100), chunks=(10, 10), dtype="i4")
        return fn

    @pytest.fixture(scope="class")
    def ome_xml_file(self, tmp_path_factory):
        xs = xmlschema.XMLSchema(
            "https://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd"
        )
        xml_data = {
            "@xmlns": "http://www.openmicroscopy.org/Schemas/OME/2016-06",
            "@xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "@xsi:schemaLocation": "http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd",
            "Image": [
                {
                    "@ID": "Image:0",
                    "Pixels": {
                        "@DimensionOrder": "XYZCT",
                        "@ID": "Pixels:0",
                        "@Type": "uint32",
                        "@SizeC": 1,
                        "@SizeT": 1,
                        "@SizeX": 100,
                        "@SizeY": 100,
                        "@SizeZ": 1,
                        "Channel": [{"@ID": "Channel:0:0", "@Name": "Channel_0"}],
                        "TiffData": [],
                    },
                }
            ],
        }
        xml = xmlschema.from_json(
            json.dumps(xml_data),
            schema=xs,
            preserve_root=True,
            namespaces={"": "http://www.openmicroscopy.org/Schemas/OME/2016-06"},
            path="OME",
        )
        xs = xmlschema.XMLSchema(
            "https://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd"
        )
        fn = tmp_path_factory.mktemp("data") / "METADATA.ome.xml"
        ElementTree(xml).write(fn)
        return fn

    def test_h5ad_to_zarr(self, monkeypatch, anndata_h5ad_file):
        monkeypatch.chdir(os.path.dirname(anndata_h5ad_file))
        stem = "test"
        out_file = h5ad_to_zarr(anndata_h5ad_file, stem)
        assert os.path.exists(out_file)
        z = zarr.open(out_file, mode="r")
        assert "X" in z and isinstance(z["X"], zarr.Array)
        assert "obs" in z
        assert "var" in z
        assert "obsm" in z and "spatial" in z["obsm"]
        assert "uns" in z and "spatial" in z["uns"]
        assert out_file == stem + "-anndata.zarr"

    def test_batch_h5ad_to_zarr(self, monkeypatch, anndata_h5ad_file):
        monkeypatch.chdir(os.path.dirname(anndata_h5ad_file))
        stem = "batch_test"
        out_batch_file = h5ad_to_zarr(
            anndata_h5ad_file, stem, batch_processing=True, batch_size=2
        )
        assert os.path.exists(out_batch_file)
        assert out_batch_file == stem + "-anndata.zarr"
        z_batch = zarr.open(out_batch_file, mode="r")
        assert "X" in z_batch and isinstance(z_batch["X"], zarr.Array)
        assert "obs" in z_batch
        assert "var" in z_batch
        assert "obsm" in z_batch and "spatial" in z_batch["obsm"]
        assert "uns" in z_batch and "spatial" in z_batch["uns"]
        stem = "test"
        out_file = h5ad_to_zarr(anndata_h5ad_file, stem)
        z = zarr.open(out_file, mode="r")
        assert np.array_equal(z_batch["X"], z["X"])

    def test_csc_h5ad_to_zarr(self, monkeypatch, anndata_csc_h5ad_file):
        monkeypatch.chdir(os.path.dirname(anndata_csc_h5ad_file))
        stem = "test"
        out_file = h5ad_to_zarr(anndata_csc_h5ad_file, stem)
        assert os.path.exists(out_file)
        z = zarr.open(out_file, mode="r")
        assert "X" in z and isinstance(z["X"], zarr.Array)
        assert "obs" in z
        assert "var" in z
        assert "obsm" in z and "spatial" in z["obsm"]
        assert "uns" in z and "spatial" in z["uns"]
        assert out_file == stem + "-anndata.zarr"

    def test_batch_csc_h5ad_to_zarr(self, monkeypatch, anndata_csc_h5ad_file):
        monkeypatch.chdir(os.path.dirname(anndata_csc_h5ad_file))
        stem = "batch_test"
        out_batch_file = h5ad_to_zarr(
            anndata_csc_h5ad_file, stem, batch_processing=True, batch_size=4
        )
        assert os.path.exists(out_batch_file)
        assert out_batch_file == stem + "-anndata.zarr"
        z_batch = zarr.open(out_batch_file, mode="r")
        assert "X" in z_batch and isinstance(z_batch["X"], zarr.Array)
        assert "obs" in z_batch
        assert "var" in z_batch
        assert "obsm" in z_batch and "spatial" in z_batch["obsm"]
        assert "uns" in z_batch and "spatial" in z_batch["uns"]
        stem = "test"
        out_file = h5ad_to_zarr(anndata_csc_h5ad_file, stem)
        z = zarr.open(out_file, mode="r")
        assert np.array_equal(z_batch["X"], z["X"])

    def test_csr_h5ad_to_zarr(self, monkeypatch, anndata_csr_h5ad_file):
        monkeypatch.chdir(os.path.dirname(anndata_csr_h5ad_file))
        stem = "test"
        out_file = h5ad_to_zarr(anndata_csr_h5ad_file, stem)
        assert os.path.exists(out_file)
        z = zarr.open(out_file, mode="r")
        assert "X" in z and isinstance(z["X"], zarr.Array)
        assert "obs" in z
        assert "var" in z
        assert "obsm" in z and "spatial" in z["obsm"]
        assert "uns" in z and "spatial" in z["uns"]
        assert out_file == stem + "-anndata.zarr"

    def test_batch_csr_h5ad_to_zarr(self, monkeypatch, anndata_csr_h5ad_file):
        monkeypatch.chdir(os.path.dirname(anndata_csr_h5ad_file))
        stem = "batch_test"
        out_batch_file = h5ad_to_zarr(
            anndata_csr_h5ad_file, stem, batch_processing=True, batch_size=2
        )
        assert os.path.exists(out_batch_file)
        assert out_batch_file == stem + "-anndata.zarr"
        z_batch = zarr.open(out_batch_file, mode="r")
        assert "X" in z_batch and isinstance(z_batch["X"], zarr.Array)
        assert "obs" in z_batch
        assert "var" in z_batch
        assert "obsm" in z_batch and "spatial" in z_batch["obsm"]
        assert "uns" in z_batch and "spatial" in z_batch["uns"]
        stem = "test"
        out_file = h5ad_to_zarr(anndata_csr_h5ad_file, stem)
        z = zarr.open(out_file, mode="r")
        assert np.array_equal(z_batch["X"], z["X"])

    def test_tsv_to_json(self, monkeypatch, molecules_tsv_file, molecules_json_file):
        monkeypatch.chdir(os.path.dirname(molecules_tsv_file))
        stem = "test"
        out_file = tsv_to_json(molecules_tsv_file, stem)
        assert os.path.exists(out_file)
        assert out_file == stem + "-molecules.json"
        with open(out_file, "r") as jsonfile:
            output_json = json.load(jsonfile)
        with open(molecules_json_file, "r") as jsonfile:
            expected_json = json.load(jsonfile)
        assert output_json == expected_json

    def test_zarr(self, zarr_file):
        consolidate(zarr_file)
        assert os.path.exists(os.path.join(zarr_file, ".zmetadata"))

    def test_route_h5ad(self, monkeypatch, anndata_h5ad_file):
        monkeypatch.chdir(os.path.dirname(anndata_h5ad_file))
        stem = "test"
        out_file = process("h5ad", anndata_h5ad_file, stem, {})
        assert os.path.exists(out_file)
        assert out_file == stem + "-anndata.zarr"

    def test_route_molecules(self, monkeypatch, molecules_tsv_file):
        monkeypatch.chdir(os.path.dirname(molecules_tsv_file))
        stem = "test"
        out_file = process("molecules", molecules_tsv_file, stem, {})
        assert os.path.exists(out_file)
        assert out_file == stem + "-molecules.json"

    def test_ome_metadata(self, monkeypatch, ome_xml_file):
        monkeypatch.chdir(os.path.dirname(ome_xml_file))
        out_json = json.loads(get_metadata(ome_xml_file))
        md = {
            "dimOrder": "XYZCT",
            "channel_names": ["Channel_0"],
            "X": "100",
            "Y": "100",
            "Z": "1",
            "C": "1",
            "T": "1",
        }
        assert out_json == md

    def test_generate_label(self, monkeypatch, anndata_h5ad_file):
        monkeypatch.chdir(os.path.dirname(anndata_h5ad_file))
        stem = "test"
        args = {"shape": [100, 100]}
        create_img(stem, "label", "visium", anndata_h5ad_file, args=args)
        assert os.path.exists(stem + "-label.tif")

    # def test_build_config(self, request, tmp_path_factory):
    #     input_dir = os.path.dirname(request.fspath) # to get files in tests dir
    #     fn = tmp_path_factory.mktemp("data")
    #     for config_type in ["minimal", "simple", "advanced", "custom"]:
    #         input_file = os.path.join(input_dir, "input", config_type + "_config.json")
    #         with open(input_file) as jsonfile:
    #             input_json = json.load(jsonfile)
    #         input_json["outdir"] = fn
    #         expected_file = os.path.join(input_dir, "expected_output", config_type + "_test_config.json")
    #         with open(expected_file) as jsonfile:
    #             expected_json = json.load(jsonfile)
    #         write_json(**input_json)
    #         output_file = os.path.join(fn, config_type + "_test_config.json")
    #         with open(output_file) as jsonfile:
    #             output_json = json.load(jsonfile)
    #         assert output_json == expected_json

    # def test_build_config_multimodal(self):
    #     input = {
    #         "project": "test_project",
    #         "datasets": {
    #             "dataset_1": {
    #                 "file_paths": ["test_project-dataset_1-anndata.zarr"],
    #                 "images": {
    #                     "raw": [
    #                         {
    #                             "path": "/path/to/raw/image.zarr",
    #                             "md": {
    #                                 "dimOrder": "XYZT",
    #                                 "channel_names": ["Channel_1"],
    #                                 "X": 10,
    #                                 "Y": 10,
    #                                 "Z": 1,
    #                                 "C": 1,
    #                                 "T": 0,
    #                             },
    #                         }
    #                     ],
    #                     "label": [
    #                         {
    #                             "path": "/path/to/label/image.zarr",
    #                             "md": {
    #                                 "dimOrder": "XYZT",
    #                                 "channel_names": ["Channel_1"],
    #                                 "X": 10,
    #                                 "Y": 10,
    #                                 "Z": 1,
    #                                 "C": 1,
    #                                 "T": 0,
    #                             },
    #                         }
    #                     ],
    #                 },
    #                 "options": {
    #                     "matrix": "X",
    #                     "factors": ["obs/sample"],
    #                     "mappings": {"obsm/X_umap": [0, 1]},
    #                     "sets": ["obs/cluster"],
    #                     "spatial": {"xy": "obsm/spatial"},
    #                 },
    #                 "extended_features": "celltype",
    #                 "obs_type": "cell",
    #                 "is_spatial": True,
    #             }
    #         },
    #         "config_filename_suffix": "config.json",
    #     }

    #     write_json_multimodal(**input)

    #     assert os.path.exists(
    #         f"{input['project']}-multimodal-{input['config_filename_suffix']}"
    #     )
