"""Precipitation-based climate indices."""

from typing import Any

import xarray
import xclim

import earthkit.climate.utils.conversions as conversions
import earthkit.climate.utils.units as units
from earthkit.climate.api.wrapper import wrap_xclim_indicator


def daily_precipitation_intensity(
    pr: conversions.EarthkitData | xarray.Dataset,
    thresh: str = "1 mm/day",
    freq: str = "YS",
    **kwargs: Any,
) -> conversions.EarthkitData:
    """
    Compute the Daily Precipitation Intensity (SDII) using the xclim indices module.

    Parameters
    ----------
    pr : conversions.EarthkitData | xarray.Dataset
        Daily precipitation flux.
    thresh : str, optional, default "1 mm/day"
        Threshold for wet days.
    freq : str, optional, default "YS"
        Frequency of resampling (e.g. yearly).
    **kwargs : Any
        Additional keyword arguments forwarded to
        :func:`xclim.indices.daily_pr_intensity`.

    Returns
    -------
    conversions.EarthkitData
        The computed Daily Precipitation Intensity as an Earthkit-compatible field.
    """
    # Convert input to xarray
    metadata: conversions.MetadataDict = {}
    pr_ds, metadata = conversions.to_xarray_dataset(pr, metadata)

    # Ensure correct units
    pr_ds = units.ensure_units(pr_ds, "pr", "mm/day", strict=False)

    # Create wrapper inside the function
    wrapper = wrap_xclim_indicator(xclim.indicators.atmos.daily_pr_intensity)
    return wrapper(pr_ds, thresh=thresh, freq=freq, **kwargs)


def maximum_consecutive_wet_days(
    pr: conversions.EarthkitData | xarray.Dataset,
    thresh: str = "1 mm/day",
    freq: str = "YS",
    **kwargs: Any,
) -> conversions.EarthkitData:
    """
    Compute the Maximum Consecutive Wet Days (CWD) using the xclim indices module.

    Parameters
    ----------
    pr : conversions.EarthkitData | xarray.Dataset
        Daily precipitation flux.
    thresh : str, optional, default "1 mm/day"
        Threshold for wet days.
    freq : str, optional, default "YS"
        Frequency of resampling (e.g. yearly).
    **kwargs : Any
        Additional keyword arguments forwarded to
        :func:`xclim.indices.maximum_consecutive_wet_days`.

    Returns
    -------
    conversions.EarthkitData
        The computed Maximum Consecutive Wet Days as an Earthkit-compatible field.
    """
    # Convert input to xarray
    metadata: conversions.MetadataDict = {}
    pr_ds, metadata = conversions.to_xarray_dataset(pr, metadata)

    # Ensure correct units
    pr_ds = units.ensure_units(pr_ds, "pr", "mm/day", strict=False)

    # Create wrapper inside the function
    wrapper = wrap_xclim_indicator(xclim.indicators.atmos.maximum_consecutive_wet_days)
    return wrapper(pr_ds, thresh=thresh, freq=freq, **kwargs)
