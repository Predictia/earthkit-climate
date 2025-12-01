"""Temperature-based climate indices."""

from typing import Any, Optional, Tuple

import xarray
import xclim.indicators.atmos

import earthkit.climate.utils.conversions as conversions
from earthkit.climate.api.wrapper import wrap_xclim_indicator


def daily_temperature_range(
    ds: conversions.EarthkitData | xarray.Dataset,
    **kwargs: Any,
) -> conversions.EarthkitData:
    """
    Compute the daily temperature range (DTR) using the xclim indices module.

    Parameters
    ----------
    ds : conversions.EarthkitData | xarray.Dataset
        Input data containing maximum and minimum daily temperature values.
    **kwargs : Any
        Additional keyword arguments forwarded to
        :func:`xclim.indices.daily_temperature_range`.

    Returns
    -------
    conversions.EarthkitData
        The computed daily temperature range converted back to an Earthkit-compatible type.

    """

    # Create wrapper inside the function
    wrapper = wrap_xclim_indicator(xclim.indicators.atmos.daily_temperature_range)
    return wrapper(ds, **kwargs)


def heating_degree_days(
    ds: conversions.EarthkitData | xarray.Dataset,
    **kwargs: Any,
) -> conversions.EarthkitData:
    """
    Compute the Heating Degree Days (HDD) using the approximation method
    from the xclim indicators module.

    This version uses both daily maximum and minimum temperatures, following
    the approach used in :func:`xclim.indicators.atmos.heating_degree_days_approximation`.

    Parameters
    ----------
    ds : conversions.EarthkitData | xarray.Dataset
        Daily maximum, minimum and mean temperature data.
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

    # Create wrapper inside the function
    wrapper = wrap_xclim_indicator(xclim.indicators.atmos.heating_degree_days)
    return wrapper(ds, **kwargs)


def warm_spell_duration_index(
    ds: conversions.EarthkitData | xarray.Dataset,
    ds_hist: Optional[conversions.EarthkitData | xarray.Dataset],
    reference_period: Optional[Tuple[str, str]] = None,
    window: int = 6,
    **kwargs: Any,
) -> conversions.EarthkitData:
    """
    Compute the Warm Spell Duration Index (WSDI) using the xclim indices module.
    The 90th percentile threshold is computed internally from the historical period.

    Parameters
    ----------
    ds : conversions.EarthkitData | xarray.Dataset
        Daily maximum temperature data for the target period.
    ds_hist : conversions.EarthkitData | xarray.Dataset, default None
        Historical daily maximum temperature data used to compute the 90th percentile threshold.
    reference_period : tuple, optional, default None
        The time period to use as a reference (start, end), by default None.
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
    return wrapper(
        earthkit_input=ds,
        reference_data=ds_hist,
        reference_period=reference_period,
        percentile_val=90,
        window=window,
        **kwargs
    )
