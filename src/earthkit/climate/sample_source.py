# SPDX-FileCopyrightText: 2025 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

"""Source plugin for earthkit-data to access sample data of earthkit-climate."""

import warnings

import numpy as np
import xarray as xr
from earthkit.data.sources import Source, _from_source_internal

# ---------------------------------------------------------------------------
# Dataset URLs
# ---------------------------------------------------------------------------
_SITES_URL = "https://sites.ecmwf.int/repository/earthkit-climate"

_SAMPLE_DATA_URLS: dict[str, str] = {
    # Main descriptive keys
    "tasmax_ACCESS-CM2_historical_reference": f"{_SITES_URL}/tasmax_ACCESS-CM2_historical_reference.nc",
    "tasmin_ACCESS-CM2_historical_reference": f"{_SITES_URL}/tasmin_ACCESS-CM2_historical_reference.nc",
    "tasmax_ACCESS-CM2_ssp585_far_future": f"{_SITES_URL}/tasmax_ACCESS-CM2_ssp585_far_future.nc",
    "tasmin_ACCESS-CM2_ssp585_far_future": f"{_SITES_URL}/tasmin_ACCESS-CM2_ssp585_far_future.nc",
    "pr_ACCESS-CM2_historical_reference": f"{_SITES_URL}/pr_ACCESS-CM2_historical_reference.nc",
    "pr_ACCESS-CM2_ssp585_far_future": f"{_SITES_URL}/pr_ACCESS-CM2_ssp585_far_future.nc",
}


def generate_sample_dataset(
    start_date: str,
    end_date: str,
    *,
    tas_value: float = 20.0,
    hurs_value: float = 70.0,
    pr_value: float = 0.0,
) -> xr.Dataset:
    """Generate a daily sample dataset for climate-indicator examples.

    Parameters
    ----------
    start_date : str
        First date in the dataset, in a format accepted by
        :func:`xarray.date_range`.
    end_date : str
        Last date in the dataset, in a format accepted by
        :func:`xarray.date_range`. The date is included in the dataset.
    tas_value : float, default: 20.0
        Constant daily mean near-surface air temperature, in degrees Celsius.
    hurs_value : float, default: 70.0
        Constant daily mean relative humidity, in percent.
    pr_value : float, default: 0.0
        Constant daily precipitation rate, in millimetres per day.

    Returns
    -------
    xarray.Dataset
        Dataset containing daily ``tas``, ``hurs``, and ``pr`` variables with
        units, standard names, and cell methods.
    """
    time = xr.date_range(start=start_date, end=end_date, freq="D")
    shape = time.size

    tas = xr.DataArray(
        np.full(shape, tas_value),
        coords={"time": time},
        dims="time",
        name="tas",
        attrs={
            "units": "degC",
            "standard_name": "air_temperature",
            "cell_methods": "time: mean within days",
        },
    )
    hurs = xr.DataArray(
        np.full(shape, hurs_value),
        coords={"time": time},
        dims="time",
        name="hurs",
        attrs={
            "units": "%",
            "standard_name": "relative_humidity",
            "cell_methods": "time: mean within days",
        },
    )
    pr = xr.DataArray(
        np.full(shape, pr_value),
        coords={"time": time},
        dims="time",
        name="pr",
        attrs={
            "units": "mm/day",
            "standard_name": "precipitation_flux",
            "cell_methods": "time: mean within days",
        },
    )

    return xr.Dataset({"tas": tas, "hurs": hurs, "pr": pr})


class SampleSource(Source):
    # Notify the user to not rely on these datasets once
    __has_notified = False

    def __init__(self, name, **kwargs):
        super().__init__()
        self._kwargs = kwargs

        if name not in _SAMPLE_DATA_URLS:
            raise ValueError(f"Unknown sample dataset: {name!r}")
        self._name = name

        if not self.__has_notified:
            self.__class__.__has_notified = True
            warnings.warn(
                "earthkit-climate-sample datasets are made available for demonstration purposes only. "
                "Files are not guaranteed to be available long-term and may change over time. "
                "Please use official channels to obtain the contained datasets reliably for other purposes."
            )

    def mutate(self):
        return _from_source_internal("url", _SAMPLE_DATA_URLS[self._name], **self._kwargs)


source = SampleSource
