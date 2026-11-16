import numpy as np
import pytest
import xarray as xr

from earthkit.climate.indicators import (
    apparent_temperature,
    heat_index,
    humidex,
    net,
    normal_effective_temperature,
    universal_thermal_climate_index,
    utci,
    wbgt,
    wet_bulb_globe_temperature,
    wind_chill,
)


@pytest.fixture
def sample_thermal_dataset() -> xr.Dataset:
    """
    Create a synthetic dataset for testing thermal comfort indicators.

    Returns
    -------
    xr.Dataset
        Dataset containing tas, sfc_wind, mrt, hurs, and tdps.
    """
    time = np.array(["2020-07-01", "2020-07-02", "2020-07-03"], dtype="datetime64[D]")
    tas = xr.DataArray([298.15, 303.15, 308.15], dims=["time"], coords={"time": time}, attrs={"units": "K"})
    sfc_wind = xr.DataArray([1.5, 3.0, 5.0], dims=["time"], coords={"time": time}, attrs={"units": "m s-1"})
    mrt = xr.DataArray([305.15, 315.15, 320.15], dims=["time"], coords={"time": time}, attrs={"units": "K"})
    hurs = xr.DataArray([50.0, 65.0, 75.0], dims=["time"], coords={"time": time}, attrs={"units": "%"})
    tdps = xr.DataArray([287.15, 296.15, 303.15], dims=["time"], coords={"time": time}, attrs={"units": "K"})
    rsds = xr.DataArray([400.0, 600.0, 800.0], dims=["time"], coords={"time": time}, attrs={"units": "W m-2"})

    return xr.Dataset({
        "tas": tas,
        "sfc_wind": sfc_wind,
        "mrt": mrt,
        "hurs": hurs,
        "tdps": tdps,
        "rsds": rsds,
    })


def test_utci_indicator(sample_thermal_dataset: xr.Dataset) -> None:
    """
    Test Universal Thermal Climate Index calculation and alias.

    Parameters
    ----------
    sample_thermal_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    res = universal_thermal_climate_index(ds=sample_thermal_dataset)
    assert isinstance(res, xr.DataArray)
    assert res.attrs["units"] == "K"
    assert res.attrs["standard_name"] == "universal_thermal_climate_index"
    assert len(res) == 3
    # Check that values are in realistic range for hot summer conditions (~295 to 325 K)
    assert np.all(res.values > 280.0)
    assert np.all(res.values < 340.0)

    # Test alias
    res_alias = utci(ds=sample_thermal_dataset)
    np.testing.assert_allclose(res.values, res_alias.values)

    # Test degC output
    res_c = utci(ds=sample_thermal_dataset, output_units="degC")
    assert res_c.attrs["units"] == "degC"
    np.testing.assert_allclose(res_c.values, res.values - 273.15)


def test_apparent_temperature_indicator(sample_thermal_dataset: xr.Dataset) -> None:
    """
    Test Apparent Temperature with and without solar radiation.

    Parameters
    ----------
    sample_thermal_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    res_no_rad = apparent_temperature(ds=sample_thermal_dataset)
    assert res_no_rad.attrs["units"] == "K"
    assert res_no_rad.attrs["standard_name"] == "apparent_temperature"

    res_rad = apparent_temperature(
        tas="tas",
        sfc_wind="sfc_wind",
        hurs="hurs",
        rsds="rsds",
        ds=sample_thermal_dataset,
    )
    # Solar radiation should increase perceived apparent temperature
    assert np.all(res_rad.values >= res_no_rad.values)


def test_humidex_indicator(sample_thermal_dataset: xr.Dataset) -> None:
    """
    Test Humidex calculation using dew point and relative humidity.

    Parameters
    ----------
    sample_thermal_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    res_td = humidex(tas="tas", tdps="tdps", ds=sample_thermal_dataset)
    assert res_td.attrs["units"] == "K"
    assert res_td.attrs["standard_name"] == "humidex"

    res_rh = humidex(tas="tas", hurs="hurs", ds=sample_thermal_dataset)
    assert len(res_rh) == 3
    assert np.all(res_rh.values > 280.0)


def test_wind_chill_indicator(sample_thermal_dataset: xr.Dataset) -> None:
    """
    Test Wind Chill index calculation.

    Parameters
    ----------
    sample_thermal_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    res = wind_chill(ds=sample_thermal_dataset)
    assert res.attrs["units"] == "K"
    assert res.attrs["standard_name"] == "wind_chill"
    assert len(res) == 3


def test_heat_index_indicator(sample_thermal_dataset: xr.Dataset) -> None:
    """
    Test NWS Heat Index calculation.

    Parameters
    ----------
    sample_thermal_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    res = heat_index(tas="tas", hurs="hurs", ds=sample_thermal_dataset)
    assert res.attrs["units"] == "K"
    assert res.attrs["standard_name"] == "heat_index"
    assert len(res) == 3


def test_wbgt_indicator(sample_thermal_dataset: xr.Dataset) -> None:
    """
    Test simplified Wet Bulb Globe Temperature and alias.

    Parameters
    ----------
    sample_thermal_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    res = wet_bulb_globe_temperature(ds=sample_thermal_dataset)
    assert res.attrs["units"] == "K"
    assert res.attrs["standard_name"] == "wet_bulb_globe_temperature"

    res_alias = wbgt(ds=sample_thermal_dataset)
    np.testing.assert_allclose(res.values, res_alias.values)


def test_normal_effective_temperature_indicator(sample_thermal_dataset: xr.Dataset) -> None:
    """
    Test Normal Effective Temperature and alias.

    Parameters
    ----------
    sample_thermal_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    res = normal_effective_temperature(ds=sample_thermal_dataset)
    assert res.attrs["units"] == "K"
    assert res.attrs["standard_name"] == "normal_effective_temperature"

    res_alias = net(ds=sample_thermal_dataset)
    np.testing.assert_allclose(res.values, res_alias.values)


def test_dask_parallelization(sample_thermal_dataset: xr.Dataset) -> None:
    """
    Verify that calculations respect Dask chunking and lazy execution.

    Parameters
    ----------
    sample_thermal_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    ds_chunked = sample_thermal_dataset.chunk({"time": 1})
    res_lazy = utci(ds=ds_chunked)
    # Result must remain a dask array until computed
    assert hasattr(res_lazy.data, "dask")
    computed = res_lazy.compute()
    assert isinstance(computed, xr.DataArray)
    assert not hasattr(computed.data, "dask")

