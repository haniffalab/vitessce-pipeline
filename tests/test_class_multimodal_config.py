import operator
import os
from functools import reduce

from bin.build_config_multimodal import write_json as write_json_multimodal


def iss_dataset(name="iss_dataset"):
    dataset = {
        f"{name}": {
            "file_paths": [f"test_project-{name}-anndata.zarr"],
            "images": {
                "raw": [
                    {
                        "path": f"/path/to/iss/{name}-raw-image.zarr",
                        "md": {
                            "dimOrder": "XYZT",
                            "channel_names": ["Channel_1"],
                            "X": 10,
                            "Y": 10,
                            "Z": 1,
                            "C": 1,
                            "T": 0,
                        },
                    }
                ],
                "label": [
                    {
                        "path": f"/path/to/iss/{name}-label-image.zarr",
                        "md": {
                            "dimOrder": "XYZT",
                            "channel_names": ["Channel_1"],
                            "X": 10,
                            "Y": 10,
                            "Z": 1,
                            "C": 1,
                            "T": 0,
                        },
                    }
                ],
            },
            "options": {
                "matrix": "X",
                "factors": ["obs/sample"],
                "mappings": {"obsm/X_umap": [0, 1]},
                "sets": ["obs/cluster", "obs/celltype"],
                "spatial": {"xy": "obsm/spatial"},
            },
            "obs_type": "cell",
            "is_spatial": True,
        }
    }
    return dataset


def visium_dataset(name="visium_dataset"):
    dataset = {
        f"{name}": {
            "file_paths": [f"test_project-{name}-anndata.zarr"],
            "images": {
                "raw": [
                    {
                        "path": f"/path/to/visium/{name}-raw-image.zarr",
                        "md": {
                            "dimOrder": "XYZT",
                            "channel_names": ["Channel_1"],
                            "X": 10,
                            "Y": 10,
                            "Z": 1,
                            "C": 1,
                            "T": 0,
                        },
                    }
                ],
                "label": [
                    {
                        "path": f"/path/to/visium/{name}-label-image.zarr",
                        "md": {
                            "dimOrder": "XYZT",
                            "channel_names": ["Channel_1"],
                            "X": 10,
                            "Y": 10,
                            "Z": 1,
                            "C": 1,
                            "T": 0,
                        },
                    }
                ],
            },
            "options": {
                "matrix": "X",
                "factors": ["obs/sample"],
                "mappings": {"obsm/X_umap": [0, 1]},
                "spatial": {"xy": "obsm/spatial"},
            },
            "obs_type": "spot",
            "is_spatial": True,
        }
    }
    return dataset


def scrnaseq_dataset(name="scrnaseq_dataset"):
    dataset = {
        f"{name}": {
            "file_paths": [f"test_project-{name}-anndata.zarr"],
            "options": {
                "matrix": "X",
                "factors": ["obs/sample"],
                "sets": ["obs/celltype"],
                "mappings": {"obsm/X_umap": [0, 1]},
                "spatial": {"xy": "obsm/spatial"},
            },
            "obs_type": "cell",
            "is_spatial": False,
        }
    }
    return dataset


class TestClass:
    def test_build_config_multimodal(
        self,
    ):
        tests = [
            (
                "test-iss_visium_sc",
                [iss_dataset(), visium_dataset(), scrnaseq_dataset()],
            ),
            (
                "test-visium_visium",
                [
                    visium_dataset("visium_dataset_1"),
                    visium_dataset("visium_dataset_2"),
                ],
            ),
            (
                "test-sc_sc",
                [scrnaseq_dataset("sc_dataset_1"), scrnaseq_dataset("sc_dataset_2")],
            ),
            (
                "test-iss_iss_visium",
                [
                    iss_dataset("iss_dataset_1"),
                    iss_dataset("iss_dataset_2"),
                    visium_dataset(),
                ],
            ),
            ("test-sc", [scrnaseq_dataset()]),
            ("test-iss", [iss_dataset()]),
            ("test-visium", [visium_dataset()]),
        ]

        for test in tests:
            input = {
                "project": test[0],
                "extended_features": "celltype",
                "url": "http://localhost/",
                "config_filename_suffix": "config.json",
                "datasets": reduce(operator.ior, test[1], {}),
            }

            write_json_multimodal(**input)

            assert os.path.exists(
                f"{input['project']}-multimodal-{input['config_filename_suffix']}"
            )
