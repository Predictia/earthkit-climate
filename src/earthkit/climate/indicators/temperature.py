"""Temperature-based climate indices."""

from __future__ import annotations

from typing import Any

import xarray
import xclim.indicators.atmos
from xclim.core.calendar import percentile_doy

import earthkit.climate.utils.conversions as conversions
import earthkit.climate.utils.provenance as provenance
import earthkit.climate.utils.units as units


def daily_temperature_range(
    tasmax: conversions.EarthkitData | xarray.Dataset,
    tasmin: conversions.EarthkitData | xarray.Dataset,
    **kwargs: Any,
) -> conversions.EarthkitData:
    """
    Compute the daily temperature range (DTR) using the xclim indices module.

    Parameters
    ----------
    tasmax : conversions.EarthkitData | xarray.Dataset
        Input data containing maximum daily temperature values.
    tasmin : conversions.EarthkitData | xarray.Dataset
        Input data containing minimum daily temperature values.
    **kwargs : Any
        Additional keyword arguments forwarded to
        :func:`xclim.indices.daily_temperature_range`.

    Returns
    -------
    conversions.EarthkitData
        The computed daily temperature range converted back to an Earthkit-compatible type.

    """
    # Convert both inputs to xarray objects
    metadata: conversions.MetadataDict = {}
    tasmax_ds, metadata = conversions.to_xarray_dataset(tasmax, metadata)
    tasmin_ds, metadata = conversions.to_xarray_dataset(tasmin, metadata)

    # Ensure correct units for precipitation
    tasmax_ds = units.ensure_units(tasmax_ds, "tasmax", "degC", strict=False)
    tasmin_ds = units.ensure_units(tasmin_ds, "tasmin", "degC", strict=False)

    # Compute the DTR index
    result = xclim.indicators.atmos.daily_temperature_range(
        tasmax_ds["tasmax"], tasmin_ds["tasmin"], **kwargs
    )
    output_dataset = result.to_dataset(name=result.name or "dtr")

    # Add provenance metadata
    metadata = provenance.add_indicator_provenance(
        metadata, xclim.indicators.atmos.daily_temperature_range, output_dataset, **kwargs
    )

    # Convert back to Earthkit format
    return conversions.to_earthkit_field(output_dataset, metadata)


def warm_spell_duration_index(
    tasmax: conversions.EarthkitData | xarray.Dataset,
    tasmax_hist: conversions.EarthkitData | xarray.Dataset,
    freq: str = "YS",
    window: int = 6,
    **kwargs: Any,
) -> conversions.EarthkitData:
    """
    Compute the Warm Spell Duration Index (WSDI) using the xclim indices module.
    The 90th percentile threshold is computed internally from the historical period.

    Parameters
    ----------
    tasmax : conversions.EarthkitData | xarray.Dataset
        Daily maximum temperature data for the target period.
    tasmax_hist : conversions.EarthkitData | xarray.Dataset
        Historical daily maximum temperature data used to compute the 90th percentile threshold.
    freq : str, optional, default "YS"
        Frequency of resampling (e.g. yearly).
    window : int, optional, default 6
        Minimum number of consecutive days above the threshold.
    **kwargs : Any
        Additional arguments forwarded to :func:`xclim.indicators.atmos.warm_spell_duration_index`.

    Returns
    -------
    conversions.EarthkitData
        The computed WSDI index as an Earthkit-compatible field.
    """
    metadata: conversions.MetadataDict = {}
    tasmax_ds, metadata = conversions.to_xarray_dataset(tasmax, metadata)
    hist_ds, _ = conversions.to_xarray_dataset(tasmax_hist, metadata)

    # Ensure correct units for precipitation
    tasmax_ds = units.ensure_units(tasmax_ds, "tasmax", "degC", strict=False)
    hist_ds = units.ensure_units(hist_ds, "tasmax", "degC", strict=False)

    # Get 90th percentile over time
    tasmax_per = percentile_doy(hist_ds["tasmax"], per=90)

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
    metadata = provenance.add_indicator_provenance(
        metadata,
        xclim.indicators.atmos.warm_spell_duration_index,
        output_dataset,
        freq=freq,
        window=window,
        **kwargs,
    )

    return conversions.to_earthkit_field(output_dataset, metadata)


def heating_degree_days(
    tasmax: conversions.EarthkitData | xarray.Dataset,
    tasmin: conversions.EarthkitData | xarray.Dataset,
    tas: conversions.EarthkitData | xarray.Dataset,
    **kwargs: Any,
) -> conversions.EarthkitData:
    """
    Compute the Heating Degree Days (HDD) using the approximation method
    from the xclim indicators module.

    This version uses both daily maximum and minimum temperatures, following
    the approach used in :func:`xclim.indicators.atmos.heating_degree_days_approximation`.

    Parameters
    ----------
    tasmax : conversions.EarthkitData | xarray.Dataset
        Daily maximum temperature data.
    tasmin : conversions.EarthkitData | xarray.Dataset
        Daily minimum temperature data.
    tas : conversions.EarthkitData | xarray.Dataset
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
    conversions.EarthkitData
        The computed Heating Degree Days (HDD) converted back to an Earthkit-compatible type.
    """
    metadata: conversions.MetadataDict = {}

    # Convert inputs to xarray
    tasmax_ds, metadata = conversions.to_xarray_dataset(tasmax, metadata)
    tasmin_ds, _ = conversions.to_xarray_dataset(tasmin, metadata)
    tas_ds, _ = conversions.to_xarray_dataset(tas, metadata)

    # Ensure correct units for precipitation
    tasmax_ds = units.ensure_units(tasmax_ds, "tasmax", "degC", strict=False)
    tasmin_ds = units.ensure_units(tasmin_ds, "tasmin", "degC", strict=False)
    tas_ds = units.ensure_units(tas_ds, "tas", "degC", strict=False)
    # Compute HDD index using approximation
    result = xclim.indicators.atmos.heating_degree_days_approximation(
        tasmax=tasmax_ds["tasmax"],
        tasmin=tasmin_ds["tasmin"],
        tas=tas_ds["tas"],
        **kwargs,
    )

    output_dataset = result.to_dataset(name=result.name or "hdd")

    # Add provenance
    metadata = provenance.add_indicator_provenance(
        metadata,
        xclim.indicators.atmos.heating_degree_days_approximation,
        output_dataset,
        **kwargs,
    )

    return conversions.to_earthkit_field(output_dataset, metadata)
