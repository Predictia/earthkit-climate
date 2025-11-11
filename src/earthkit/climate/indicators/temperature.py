"""Temperature-based climate indices."""
from __future__ import annotations

from typing import Any

import xarray
import xclim.indicators.atmos
from xclim.core.calendar import percentile_doy

from earthkit.climate.utils.conversions import (
    EarthkitData,
    MetadataDict,
    to_earthkit_field,
    to_xarray_dataset
)
from earthkit.climate.utils.provenance import add_indicator_provenance
from earthkit.climate.utils.units import ensure_units


def daily_temperature_range(
    tasmax: EarthkitData | xarray.Dataset,
    tasmin: EarthkitData | xarray.Dataset,
    **kwargs: Any,
) -> EarthkitData:
    """
    Compute the daily temperature range (DTR) using the xclim indices module.

    Parameters
    ----------
    tasmax : EarthkitData | xarray.Dataset
        Input data containing maximum daily temperature values.
    tasmin : EarthkitData | xarray.Dataset
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
    tasmax_ds, metadata = to_xarray_dataset(tasmax, metadata)
    tasmin_ds, metadata = to_xarray_dataset(tasmin, metadata)

    # Ensure correct units for precipitation
    tasmax_ds = ensure_units(tasmax_ds, "tasmax", "degC", strict=False)
    tasmin_ds = ensure_units(tasmin_ds, "tasmin", "degC", strict=False)

    # Compute the DTR index
    result = xclim.indices.daily_temperature_range(tasmax_ds["tasmax"], tasmin_ds["tasmin"], **kwargs)
    output_dataset = result.to_dataset(name=result.name or "dtr")

    # Add provenance metadata
    add_indicator_provenance(metadata, xclim.indices.daily_temperature_range, output_dataset, **kwargs)

    # Convert back to Earthkit format
    return to_earthkit_field(output_dataset, metadata)


def warm_spell_duration_index(
    tasmax: EarthkitData | xarray.Dataset,
    tasmax_hist: EarthkitData | xarray.Dataset,
    freq: str = "YS",
    window: int = 6,
    **kwargs: Any,
) -> EarthkitData:
    """
    Compute the Warm Spell Duration Index (WSDI) using the xclim indices module.
    The 90th percentile threshold is computed internally from the historical period.

    Parameters
    ----------
    tasmax : EarthkitData | xarray.Dataset
        Daily maximum temperature data for the target period.
    tasmax_hist : EarthkitData | xarray.Dataset
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

    # Ensure correct units for precipitation
    tasmax_ds = ensure_units(tasmax_ds, "tasmax", "degC", strict=False)
    hist_ds = ensure_units(hist_ds, "tasmax", "degC", strict=False)


    # Get 90th percentile over time
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
    metadata = add_indicator_provenance(
        metadata,
        xclim.indicators.atmos.warm_spell_duration_index,
        output_dataset,
        freq=freq,
        window=window,
        **kwargs,
    )

    return to_earthkit_field(output_dataset, metadata)


def heating_degree_days(
    tasmax: EarthkitData | xarray.Dataset,
    tasmin: EarthkitData | xarray.Dataset,
    tas: EarthkitData | xarray.Dataset,
    **kwargs: Any,
) -> EarthkitData:
    """
    Compute the Heating Degree Days (HDD) using the approximation method
    from the xclim indicators module.

    This version uses both daily maximum and minimum temperatures, following
    the approach used in :func:`xclim.indicators.atmos.heating_degree_days_approximation`.

    Parameters
    ----------
    tasmax : EarthkitData | xarray.Dataset
        Daily maximum temperature data.
    tasmin : EarthkitData | xarray.Dataset
        Daily minimum temperature data.
    tas : EarthkitData | xarray.Dataset
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

    # Ensure correct units for precipitation
    tasmax_ds = ensure_units(tasmax_ds, "tasmax", "degC", strict=False)
    tasmin_ds = ensure_units(tasmin_ds, "tasmin", "degC", strict=False)
    tas_ds = ensure_units(tas_ds, "tas", "degC", strict=False)
    # Compute HDD index using approximation
    result = xclim.indicators.atmos.heating_degree_days_approximation(
        tasmax=tasmax_ds["tasmax"],
        tasmin=tasmin_ds["tasmin"],
        tas=tas_ds["tas"],
        **kwargs,
    )

    output_dataset = result.to_dataset(name=result.name or "hdd")

    # Add provenance
    metadata = add_indicator_provenance(
        metadata,
        xclim.indicators.atmos.heating_degree_days_approximation,
        output_dataset,
        **kwargs,
    )

    return to_earthkit_field(output_dataset, metadata)
