"""Precipitation-based climate indices."""

from __future__ import annotations

from typing import Any

import xarray as xr
import xclim.indicators.atmos

from earthkit.climate.utils.conversions import (
    EarthkitData,
    MetadataDict,
    to_earthkit_field,
    to_xarray_dataset,
)
from earthkit.climate.utils.provenance import add_indicator_provenance

__all__ = [
    "maximum_consecutive_wet_days",
    "daily_precipitation_intensity",
]


def maximum_consecutive_wet_days(
    earthkit_input: EarthkitData,
    *,
    wet_day_threshold: float | str = 1.0,
    **kwargs: Any,
) -> EarthkitData:
    """
    Compute the maximum number of consecutive wet days (CWD).

    Parameters
    ----------
    earthkit_input : EarthkitData
        Input precipitation data. Supported inputs include ``xarray.Dataset``,
        ``xarray.DataArray`` and, if ``earthkit-data`` is installed, any object
        exposing a ``to_xarray`` method (for example ``Field`` or ``FieldList``).
    wet_day_threshold : float or str, default: 1.0
        Wet-day threshold forwarded to the xclim indicator. When a float is
        provided it is assumed to be expressed in ``mm/day``. Strings are
        forwarded unchanged (for example ``"1 mm/day"``).
    **kwargs : Any
        Additional keyword arguments forwarded directly to
        :func:`xclim.indicators.atmos.maximum_consecutive_wet_days`.

    Returns
    -------
    EarthkitData
        Indicator results converted back to the closest possible Earthkit
        representation (same type as the input when feasible).
    """
    metadata: MetadataDict = {}
    dataset, metadata = to_xarray_dataset(earthkit_input, metadata)

    kwargs = dict(kwargs)
    kwargs.setdefault("thresh", _format_precipitation_threshold(wet_day_threshold))

    # Call the xclim indicator
    output_dataset: xr.Dataset = xclim.indicators.atmos.maximum_consecutive_wet_days(ds=dataset, **kwargs)

    # Add provenance
    metadata = add_indicator_provenance(
        metadata, xclim.indicators.atmos.maximum_consecutive_wet_days, dataset, **kwargs
    )

    return to_earthkit_field(output_dataset, metadata)


def daily_precipitation_intensity(
    earthkit_input: EarthkitData,
    *,
    wet_day_threshold: float | str | None = None,
    frequency: str | None = None,
    **kwargs: Any,
) -> EarthkitData:
    """
    Compute the Simple Daily Intensity Index (SDII).

    Parameters
    ----------
    earthkit_input : EarthkitData
        Input precipitation data. Supported inputs include ``xarray.Dataset``,
        ``xarray.DataArray`` and, if ``earthkit-data`` is installed, any object
        exposing a ``to_xarray`` method (for example ``Field`` or ``FieldList``).
    wet_day_threshold : float or str, optional
        Wet-day threshold forwarded to the xclim indicator. Floats are assumed
        to be expressed in ``mm/day`` while strings are forwarded unchanged.
    frequency : str, optional
        Resampling frequency forwarded to :func:`xclim.indicators.atmos.daily_pr_intensity`.
    **kwargs : Any
        Additional keyword arguments forwarded directly to
        :func:`xclim.indicators.atmos.daily_pr_intensity`.

    Returns
    -------
    EarthkitData
        Indicator results converted back to the closest possible Earthkit
        representation (same type as the input when feasible).
    """
    metadata: MetadataDict = {}
    dataset, metadata = to_xarray_dataset(earthkit_input, metadata)

    kwargs = dict(kwargs)

    if wet_day_threshold is not None:
        kwargs.setdefault("thresh", _format_precipitation_threshold(wet_day_threshold))
    if frequency is not None:
        kwargs.setdefault("freq", frequency)

    # Call the xclim indicator
    output_dataset: xr.Dataset = xclim.indicators.atmos.daily_pr_intensity(ds=dataset, **kwargs)

    # Add provenance
    metadata = add_indicator_provenance(
        metadata, xclim.indicators.atmos.daily_pr_intensity, dataset, **kwargs
    )

    return to_earthkit_field(output_dataset, metadata)


def _format_precipitation_threshold(threshold: float | str) -> float | str:
    """
    Format a precipitation threshold for use in xclim indicators.

    Parameters
    ----------
    threshold : float or str
        Wet-day threshold to format.

    Returns
    -------
    float or str
        If a numeric value is provided, it is formatted as a string with units
        in ``mm/day``. If a string is provided, it is returned unchanged.
    """
    if isinstance(threshold, (int, float)):
        return f"{threshold} mm/day"
    return threshold
