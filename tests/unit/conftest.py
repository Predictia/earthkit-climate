import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def dummy_precip_ds() -> xr.Dataset:
    """Simple constant precipitation dataset."""
    time = xr.cftime_range(
        start="2001-01-01", end="2001-01-10", freq="D", calendar="noleap"
    ).to_datetimeindex()
    ds = xr.Dataset(
        {"pr": ("time", [1.0] * len(time))},
        coords={"time": time},
    )
    ds["pr"].attrs["units"] = "kg m-2 s-1"
    return ds


@pytest.fixture
def daily_temperature_ds() -> xr.Dataset:
    """Synthetic daily temperature dataset for percentile and grouping tests."""
    rng = np.random.default_rng(0)
    time = xr.cftime_range(
        start="2000-01-01", end="2001-12-31", freq="D", calendar="noleap"
    ).to_datetimeindex()
    data = rng.normal(loc=10.0, scale=2.0, size=time.size)
    ds = xr.Dataset({"tas": ("time", data)}, coords={"time": time})
    return ds
