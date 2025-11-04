"""Utilities to bridge Earthkit data objects and xarray structures."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import xarray as xr
from earthkit.data import FieldList, Field

EarthkitData = FieldList | Field
MetadataDict = Dict[str, Any]

__all__ = ["EarthkitData", "MetadataDict", "to_xarray_dataset", "to_earthkit_field"]


def to_xarray_dataset(
    earthkit_input: EarthkitData,
    metadata: Mapping[str, Any] | None = None,
) -> Tuple[xr.Dataset, MetadataDict]:
    """
    Convert Earthkit-like data to an ``xarray.Dataset`` and gather metadata.

    Parameters
    ----------
    earthkit_input : EarthkitData
        Input data in any supported Earthkit or xarray representation.
    metadata : Mapping[str, Any], optional
        Existing metadata to propagate and enrich during the conversion.

    Returns
    -------
    tuple[xr.Dataset, dict[str, Any]]
        Dataset ready to be consumed by xclim and an updated metadata mapping.

    Raises
    ------
    TypeError
        If the input cannot be converted to an ``xarray.Dataset``.
    """
    meta: MetadataDict = dict(metadata or {})
    earthkit_internal = dict(meta.get("earthkit_internal", {}))
    earthkit_internal["input_type"] = _describe_type(earthkit_input)

    if isinstance(earthkit_input, xr.Dataset):
        dataset = earthkit_input
    elif isinstance(earthkit_input, xr.DataArray):
        variable_name = earthkit_input.name or "variable"
        dataset = earthkit_input.to_dataset(name=variable_name)
        earthkit_internal["dataarray_name"] = variable_name
    elif hasattr(earthkit_input, "to_xarray"):
        dataset = earthkit_input.to_xarray()
        if isinstance(dataset, xr.DataArray):
            variable_name = dataset.name or "variable"
            dataset = dataset.to_dataset(name=variable_name)
            earthkit_internal["dataarray_name"] = variable_name
        elif not isinstance(dataset, xr.Dataset):
            raise TypeError(
                "The object returned by 'to_xarray' is not an xarray.Dataset instance."
            )
    else:
        raise TypeError(
            "Unsupported input type for conversion to xarray. "
            "Expected an xarray object or an Earthkit field exposing 'to_xarray'."
        )

    meta["earthkit_internal"] = earthkit_internal
    return dataset, meta


def to_earthkit_field(
    output: xr.Dataset | xr.DataArray,
    metadata: Mapping[str, Any] | None = None,
) -> EarthkitData:
    """
    Convert a xarray result back into an Earthkit representation.

    Parameters
    ----------
    output : xarray.Dataset or xarray.DataArray
        Resulting data returned by an xclim indicator.
    metadata : Mapping[str, Any], optional
        Provenance metadata gathered during the conversion and call workflow.

    Returns
    -------
    EarthkitData
        The indicator output converted to the closest possible Earthkit type.
    """
    meta: MetadataDict = dict(metadata or {})
    earthkit_internal = dict(meta.pop("earthkit_internal", {}))
    dataset: xr.Dataset
    if isinstance(output, xr.DataArray):
        dataset = output.to_dataset(name=output.name or "variable")
    else:
        dataset = output

    dataset = dataset.copy()
    provenance = dict(meta)
    if provenance:
        dataset.attrs.setdefault("earthkit_provenance", provenance)

    input_type = earthkit_internal.get("input_type")

    if input_type == "xarray.Dataset":
        return dataset
    if input_type == "xarray.DataArray":
        variable_name = earthkit_internal.get("dataarray_name")
        if variable_name and variable_name in dataset:
            return dataset[variable_name]
        return dataset.to_array().squeeze(drop=True)

    try:
        earthkit_data = _xarray_to_earthkit(dataset)
    except ModuleNotFoundError:
        return dataset
    except Exception:
        return dataset
    return earthkit_data


def _describe_type(obj: Any) -> str:
    """
    Return a human-readable description of an object's type, with special handling for xarray objects.

    Parameters
    ----------
    obj : Any
        The object to describe.

    Returns
    -------
    str
        A string describing the object type.
    """
    if isinstance(obj, xr.Dataset):
        return "xarray.Dataset"
    if isinstance(obj, xr.DataArray):
        return "xarray.DataArray"
    obj_type = type(obj)
    return f"{obj_type.__module__}.{obj_type.__qualname__}"


def _xarray_to_earthkit(dataset: xr.Dataset) -> EarthkitData:
    """
    Convert an `xarray.Dataset` into an `EarthkitData` object using available Earthkit helpers.

    The function attempts to use `earthkit.data.from_xarray` first, and if that fails or is
    unavailable, it tries `earthkit.data.wrap_xarray`. If neither works, a `TypeError` is raised.

    Parameters
    ----------
    dataset : xarray.Dataset
        The xarray dataset to convert.

    Returns
    -------
    EarthkitData
        The converted Earthkit data object.
    """
    try:
        import earthkit.data  # type: ignore
    except ModuleNotFoundError:
        raise

    conversion_candidates = (
        getattr(earthkit.data, "from_xarray", None),
        getattr(earthkit.data, "wrap_xarray", None),
    )

    for converter in conversion_candidates:
        if callable(converter):
            try:
                return converter(dataset)
            except Exception:
                continue

    raise TypeError(
        "Unable to convert xarray.Dataset back to an Earthkit object using the available helpers."
    )