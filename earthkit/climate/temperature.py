"""Temperature-based climate indices."""
from __future__ import annotations

from typing import Any

import xarray as xr
import xclim.indices

from .conversions import EarthkitData, MetadataDict, to_earthkit_field, to_xarray_dataset
from .provenance import add_indicator_provenance


__all__ = ["daily_temperature_range"]


def daily_temperature_range(
    tasmax: EarthkitData,
    tasmin: EarthkitData,
    **kwargs: Any,
) -> EarthkitData:
    """
    Compute the daily temperature range (DTR) using the xclim indices module.

    Parameters
    ----------
    tasmax : EarthkitData
        Input data containing maximum daily temperature values.
    tasmin : EarthkitData
        Input data containing minimum daily temperature values.
    **kwargs : Any
        Additional keyword arguments forwarded to
        :func:`xclim.indices.daily_temperature_range`.

    Returns
    -------
    EarthkitData
        The computed daily temperature range converted back to an Earthkit-compatible type.

    Raises
    ------
    ModuleNotFoundError
        If :mod:`xclim` is not installed.
    TypeError
        If the input data cannot be converted to an ``xarray.DataArray``.
    """
    # Convert both inputs to xarray objects
    metadata: MetadataDict = {}
    tasmax_da, metadata = to_xarray_dataset(tasmax, metadata)
    tasmin_da, metadata = to_xarray_dataset(tasmin, metadata)

    kwargs = dict(kwargs)

    # Compute the DTR index
    result = xclim.indices.daily_temperature_range(tasmax_da, tasmin_da, **kwargs)
    output_dataset = result.to_dataset(name=result.name or "dtr")

    # Add provenance metadata
    add_indicator_provenance(metadata, xclim.indices.daily_temperature_range, output_dataset, **kwargs)

    # Convert back to Earthkit format
    return to_earthkit_field(output_dataset, metadata)


def warm_spell_duration_index(
    daily_max: xr.DataArray,
    threshold: float,
    *,
    min_duration: int = 6,
) -> xr.DataArray:
    """Return the warm spell duration index (WSDI)."""
    raise NotImplementedError


def heating_degree_days(
    daily_mean: xr.DataArray,
    *,
    base_temperature: float = 18.0,
    frequencies: Iterable[str] | None = None,
) -> xr.DataArray:
    """Return heating degree days (HDD) accumulated over the requested period(s)."""
    raise NotImplementedError


def _extract_temperature_variables(
    dataset: xr.Dataset,
    max_variable: str,
    min_variable: str,
) -> Tuple[xr.DataArray, xr.DataArray]:
    if max_variable not in dataset:
        raise KeyError(
            f"The dataset must contain a '{max_variable}' variable representing daily maximum temperature."
        )
    if min_variable not in dataset:
        raise KeyError(
            f"The dataset must contain a '{min_variable}' variable representing daily minimum temperature."
        )

    return dataset[max_variable], dataset[min_variable]
