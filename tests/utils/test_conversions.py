from typing import Any

import pytest
import xarray as xr
from earthkit.data.wrappers.xarray import XArrayDatasetWrapper

from earthkit.climate.utils.conversions import (
    to_xarray_dataset,
    to_earthkit_field,
)


class DummyWrapper:
    """Simple wrapper stand-in to validate get_wrapper usage without importing earthkit.data."""

    def __init__(self, obj: Any):
        self.obj = obj

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DummyWrapper) and other.obj.equals(self.obj)


@pytest.mark.parametrize(
    "xr_input",
    [
        xr.Dataset({"tas": ("time", [1.0, 2.0])}, coords={"time": [0, 1]}),
        xr.DataArray([1.0, 2.0], dims=["time"], name="tas"),
    ],
)
def test_to_xarray_dataset_accepts_xarray_and_propagates_metadata(xr_input: Any) -> None:
    """
    Test that `to_xarray_dataset` correctly handles both xarray.Dataset and xarray.DataArray inputs,
    returning a Dataset and propagating the correct metadata about the input type.
    """
    ds, meta = to_xarray_dataset(xr_input, {"a": 1})
    assert isinstance(ds, xr.Dataset)
    assert meta["earthkit_internal"]["input_type"] in {"xarray.Dataset", "xarray.DataArray"}


def test_to_xarray_dataset_uses_to_xarray_on_wrapped_object() -> None:
    """
    Test that `to_xarray_dataset` calls `to_xarray()` method when provided with an object
    implementing that method, and correctly extracts the Dataset and metadata.
    """
    class HasToXarray:
        def to_xarray(self) -> xr.Dataset:
            return xr.Dataset({"pr": ("time", [0.1, 0.2])}, coords={"time": [0, 1]})

    ds, meta = to_xarray_dataset(HasToXarray(), {})
    assert set(ds.data_vars) == {"pr"}
    assert meta["earthkit_internal"]["input_type"].endswith("HasToXarray")


def test_to_xarray_dataset_rejects_invalid_to_xarray_return() -> None:
    """
    Test that `to_xarray_dataset` raises a TypeError when an object's `to_xarray()` method
    does not return a valid xarray object.
    """
    class BadToXarray:
        def to_xarray(self) -> int:
            return 123

    with pytest.raises(TypeError):
        to_xarray_dataset(BadToXarray(), {})


def test_to_earthkit_field_wraps_dataset_and_attaches_provenance() -> None:
    """
    Test that `to_earthkit_field` wraps an xarray.DataArray inside a DummyWrapper
    and attaches provenance metadata correctly.
    """
    da = xr.DataArray([1, 2, 3], dims=["time"], name="tas")
    ek_obj = to_earthkit_field(da, {"provenance": {"step": "x"}})
    assert isinstance(ek_obj, XArrayDatasetWrapper)
    assert "earthkit_provenance" in ek_obj.to_xarray().attrs