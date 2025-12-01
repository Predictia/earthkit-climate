"""Temperature-based climate indices."""

from typing import Any

import xarray
import xclim

import earthkit.climate.utils.conversions as conversions
import earthkit.climate.utils.units as units
from earthkit.climate.api.wrapper import wrap_xclim_indicator


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

    # Ensure correct units
    tasmax_ds = units.ensure_units(tasmax_ds, "tasmax", "degC", strict=False)
    tasmin_ds = units.ensure_units(tasmin_ds, "tasmin", "degC", strict=False)

    # Merge into a single dataset for the wrapper
    ds = xarray.merge([tasmax_ds, tasmin_ds])

    # Create wrapper inside the function
    wrapper = wrap_xclim_indicator(xclim.indicators.atmos.daily_temperature_range)
    return wrapper(ds, **kwargs)


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

    # Ensure correct units
    tasmax_ds = units.ensure_units(tasmax_ds, "tasmax", "degC", strict=False)
    tasmin_ds = units.ensure_units(tasmin_ds, "tasmin", "degC", strict=False)
    tas_ds = units.ensure_units(tas_ds, "tas", "degC", strict=False)

    # Merge
    ds = xarray.merge([tasmax_ds, tasmin_ds, tas_ds])

    # Create wrapper inside the function
    wrapper = wrap_xclim_indicator(xclim.indicators.atmos.heating_degree_days)
    return wrapper(ds, **kwargs)


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
    # Create wrapper inside the function
    wrapper = wrap_xclim_indicator(xclim.indicators.atmos.warm_spell_duration_index)

    # The wrapper handles reference_data for percentile calculation.
    # We map tasmax_hist to reference_data.
    return wrapper(earthkit_input=tasmax, reference_data=tasmax_hist, freq=freq, window=window, **kwargs)
