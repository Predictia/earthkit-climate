"""Temperature-based climate indices."""
from __future__ import annotations

from typing import Any

import xclim.indicators.atmos
from xclim.core.calendar import percentile_doy

from earthkit.climate.utils.conversions import (
    EarthkitData,
    MetadataDict,
    to_earthkit_field,
    to_xarray_dataset,
)
from earthkit.climate.utils.percentile import get_percentile
from earthkit.climate.utils.provenance import add_indicator_provenance

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
    result = xclim.indices.daily_temperature_range(tasmax_da["tasmax"], tasmin_da["tasmin"], **kwargs)
    output_dataset = result.to_dataset(name=result.name or "dtr")

    # Add provenance metadata
    add_indicator_provenance(metadata, xclim.indices.daily_temperature_range, output_dataset, **kwargs)

    # Convert back to Earthkit format
    return to_earthkit_field(output_dataset, metadata)


def warm_spell_duration_index(
    tasmax: EarthkitData,
    tasmax_hist: EarthkitData,
    freq: str = "YS",
    window: int = 6,
    **kwargs: Any,
) -> EarthkitData:
    """
    Compute the Warm Spell Duration Index (WSDI) using the xclim indices module.
    The 90th percentile threshold is computed internally from the historical period.

    Parameters
    ----------
    tasmax : EarthkitData
        Daily maximum temperature data for the target period.
    tasmax_hist : EarthkitData
        Historical daily maximum temperature data used to compute the 90th percentile threshold.
    freq : str, optional, default "YS"
        Frequency of resampling (e.g. yearly).
    window : int, optional, default 6
        Minimum number of consecutive days above the threshold.
    **kwargs : Any
        Additional arguments forwarded to :func:`xclim.indicators.atmos.warm_spell_duration_index`.

    Returns
    -------
    EarthkitData
        The computed WSDI index as an Earthkit-compatible field.
    """
    metadata: MetadataDict = {}
    tasmax_ds, metadata = to_xarray_dataset(tasmax, metadata)
    hist_ds, _ = to_xarray_dataset(tasmax_hist, metadata)

    # Get 90th percentile over time (regular type, not doy)
    tasmax_per = percentile_doy(hist_ds, per=90)

    # Compute WSDI with xclim
    result = xclim.indicators.atmos.warm_spell_duration_index(
        tasmax=tasmax_ds["tasmax"],
        tasmax_per=tasmax_per,
        freq=freq,
        window=window,
        **kwargs,
    )

    output_dataset = result.to_dataset(name=result.name or "wsdi")

    # Add provenance
    add_indicator_provenance(
        metadata,
        xclim.indicators.atmos.warm_spell_duration_index,
        output_dataset,
        freq=freq,
        window=window,
        **kwargs,
    )

    return to_earthkit_field(output_dataset, metadata)


def heating_degree_days(
    tasmax: EarthkitData,
    tasmin: EarthkitData,
    tas: EarthkitData,
    **kwargs: Any,
) -> EarthkitData:
    """
    Compute the Heating Degree Days (HDD) using the approximation method
    from the xclim indicators module.

    This version uses both daily maximum and minimum temperatures, following
    the approach used in :func:`xclim.indicators.atmos.heating_degree_days_approximation`.

    Parameters
    ----------
    tasmax : EarthkitData
        Daily maximum temperature data.
    tasmin : EarthkitData
        Daily minimum temperature data.
    tas : EarthkitData
        Daily mean temperature data.
    **kwargs : Any
        Additional keyword arguments forwarded to
        :func:`xclim.indicators.atmos.heating_degree_days_approximation`.

        Common arguments include:
        - `thresh` : str, default "18.0 degC"
            Base temperature threshold for heating.
        - `freq` : str, default "YS"
            Frequency for accumulation (e.g., "YS" = yearly sum).

    Returns
    -------
    EarthkitData
        The computed Heating Degree Days (HDD) converted back to an Earthkit-compatible type.
    """
    metadata: MetadataDict = {}

    # Convert inputs to xarray
    tasmax_ds, metadata = to_xarray_dataset(tasmax, metadata)
    tasmin_ds, _ = to_xarray_dataset(tasmin, metadata)
    tas_ds, _ = to_xarray_dataset(tas, metadata)

    kwargs = dict(kwargs)

    # Compute HDD index using approximation
    result = xclim.indicators.atmos.heating_degree_days_approximation(
        tasmax=tasmax_ds["tasmax"],
        tasmin=tasmin_ds["tasmin"],
        tas=tas_ds["tas"],
        **kwargs,
    )

    output_dataset = result.to_dataset(name=result.name or "hdd")

    # Add provenance
    add_indicator_provenance(
        metadata,
        xclim.indicators.atmos.heating_degree_days_approximation,
        output_dataset,
        **kwargs,
    )

    return to_earthkit_field(output_dataset, metadata)
