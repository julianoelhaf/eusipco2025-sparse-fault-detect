"""Import smoke test for the sparse_fault_detect package.

Importing every module catches packaging regressions and stale imports that
would otherwise only surface at runtime.
"""

import importlib

import pytest

MODULES = [
    "sparse_fault_detect",
    "sparse_fault_detect.constants",
    "sparse_fault_detect.data.preprocessing",
    "sparse_fault_detect.data.create_windows",
    "sparse_fault_detect.models.data_sparsity",
    "sparse_fault_detect.models.run_model",
    "sparse_fault_detect.visualization.create_plots",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)
