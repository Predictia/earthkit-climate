"""Temperature-based climate indices."""
from __future__ import annotations

from typing import Any, Tuple, Iterable

import xarray as xr
import xclim.indices
import xclim.indicators.atmos

from .conversions import EarthkitData, MetadataDict, to_earthkit_field, to_xarray_dataset
from .provenance import add_indicator_provenance


__all__ = [
    "daily_temperature_range",
    "warm_spell_duration_index",
    "heating_degree_days",
]



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
    tasmax: EarthkitData,
    tasmax_per: EarthkitData,
    **kwargs: Any,
) -> EarthkitData:
    """
    Compute the Warm Spell Duration Index (WSDI) using the xclim indices module.

    Parameters
    ----------
    tasmax : EarthkitData
        Daily maximum temperature data.
    tasmax_per : EarthkitData
        Reference daily maximum temperature percentiles (usually 90th percentile over a reference period).
    **kwargs : Any
        Additional keyword arguments forwarded to :func:`xclim.indices.warm_spell_duration_index`.
        Common arguments include:
        - `freq` : str, optional, frequency of resampling (e.g. "YS" for yearly)
        - `window` : int, optional, minimum spell length (default 6)
        - `thresh` : float or xarray.DataArray, optional, threshold percentile (default 90)

    Returns
    -------
    EarthkitData
        The computed WSDI index as an Earthkit-compatible field.
    """
    # Convert inputs to xarray DataArray
    metadata: MetadataDict = {}
    tasmax_da, metadata = to_xarray_dataset(tasmax, metadata)
    tasmax_per_da, metadata = to_xarray_dataset(tasmax_per, metadata)

    kwargs = dict(kwargs)

    # Compute WSDI
    result = xclim.indicators.atmos.warm_spell_duration_index(tasmax_da, tasmax_per_da, **kwargs)
    output_dataset = result.to_dataset(name=result.name or "wsdi")

    # Add provenance
    add_indicator_provenance(metadata, xclim.indicators.atmos.warm_spell_duration_index, output_dataset, **kwargs)

    # Convert back to EarthkitData
    return to_earthkit_field(output_dataset, metadata)


def heating_degree_days(
    tas: EarthkitData,
    **kwargs: Any,
) -> EarthkitData:
    """
    Compute the Heating Degree Days (HDD) using the xclim indicators module.

    Parameters
    ----------
    tas : EarthkitData
        Daily mean temperature data.
    **kwargs : Any
        Additional keyword arguments forwarded to
        :func:`xclim.indicators.atmos.heating_degree_days`.

        Common arguments include:
        - `thresh` : str, default "17.0 degC"
            Base temperature threshold for heating.
        - `freq` : str, default "YS"
            Frequency for accumulation (e.g., "YS" = yearly sum).

    Returns
    -------
    EarthkitData
        The computed Heating Degree Days (HDD) converted back to an Earthkit-compatible type.
    """
    metadata: MetadataDict = {}
    tas_da, metadata = to_xarray_dataset(tas, metadata)
    kwargs = dict(kwargs)

    # Compute HDD index
    result = xclim.indicators.atmos.heating_degree_days(tas=tas_da, **kwargs)
    output_dataset = result.to_dataset(name=result.name or "hdd")

    # Add provenance
    add_indicator_provenance(metadata, xclim.indicators.atmos.heating_degree_days, output_dataset, **kwargs)

    return to_earthkit_field(output_dataset, metadata)
