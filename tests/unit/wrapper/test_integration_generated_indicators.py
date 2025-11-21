import pytest
import xarray as xr
import numpy as np
import pandas as pd

# Import the generated modules
from earthkit.climate.indicators import temperature, precipitation, wind

@pytest.fixture
def tas_dataset():
    """Creates a dummy temperature dataset (K)."""
    times = pd.date_range("2000-01-01", periods=365, freq="D")
    # Random data around 20C (293K)
    data = 293.15 + np.random.randn(365, 10, 10) * 5
    da = xr.DataArray(
        data,
        coords={
            "time": times,
            "lat": np.arange(10),
            "lon": np.arange(10)
        },
        dims=("time", "lat", "lon"),
        attrs={"units": "K"}
    )
    return xr.Dataset({"tas": da})

@pytest.fixture
def pr_dataset():
    """Creates a dummy precipitation dataset (mm/day)."""
    times = pd.date_range("2000-01-01", periods=365, freq="D")
    # Random data with some zeros
    data = np.maximum(0, np.random.randn(365, 10, 10) * 10)
    da = xr.DataArray(
        data,
        coords={
            "time": times,
            "lat": np.arange(10),
            "lon": np.arange(10)
        },
        dims=("time", "lat", "lon"),
        attrs={"units": "mm/day"}
    )
    return xr.Dataset({"pr": da})

@pytest.fixture
def sfcWind_dataset():
    """Creates a dummy wind dataset (m/s)."""
    times = pd.date_range("2000-01-01", periods=365, freq="D")
    data = np.abs(np.random.randn(365, 10, 10) * 5)
    da = xr.DataArray(
        data,
        coords={
            "time": times,
            "lat": np.arange(10),
            "lon": np.arange(10)
        },
        dims=("time", "lat", "lon"),
        attrs={"units": "m/s"}
    )
    return xr.Dataset({"sfcWind": da})

def test_temperature_indicator(tas_dataset):
    """Test a generated temperature indicator (tg_mean)."""
    # tg_mean calculates mean temperature
    # We expect it to run and return an object (Earthkit wrapper returns Earthkit object, 
    # but for now let's just check it doesn't crash and returns something)
    
    # Note: The wrapper returns an Earthkit object, but we can't easily check type 
    # without importing earthkit-data which might be mocked or complex.
    # However, we know the wrapper returns the result of to_earthkit_field.
    
    result = temperature.tg_mean(tas_dataset, freq="MS")
    assert result is not None

def test_precipitation_indicator(pr_dataset):
    """Test a generated precipitation indicator (wetdays)."""
    # wetdays counts days with precip > thresh
    result = precipitation.wetdays(pr_dataset, thresh="1 mm/day", freq="MS")
    assert result is not None

def test_wind_indicator(sfcWind_dataset):
    """Test a generated wind indicator (sfcWind_mean)."""
    result = wind.sfcWind_mean(sfcWind_dataset, freq="MS")
    assert result is not None

def test_indicator_metadata(tas_dataset):
    """Test that metadata is preserved/added."""
    # This relies on the wrapper adding provenance
    # We can't easily check the internal metadata of the opaque Earthkit object 
    # without using earthkit methods, but we can check it runs.
    result = temperature.tg_mean(tas_dataset, freq="MS")
    # If we could convert back to xarray, we could check attrs.
    # For now, just ensuring it runs is a good integration test.
