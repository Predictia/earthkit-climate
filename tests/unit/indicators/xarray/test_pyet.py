import numpy as np
import pytest
import xarray as xr

from earthkit.climate.indicators import (
    pet_fao56,
    pet_hargreaves,
    pet_makkink,
    pet_penman,
    pet_priestley_taylor,
    pet_turc,
    potential_evapotranspiration_fao56,
    potential_evapotranspiration_hargreaves,
    potential_evapotranspiration_makkink,
    potential_evapotranspiration_penman,
    potential_evapotranspiration_priestley_taylor,
    potential_evapotranspiration_turc,
)


@pytest.fixture
def sample_evap_dataset() -> xr.Dataset:
    """
    Create a synthetic dataset for testing potential evapotranspiration indicators.

    Returns
    -------
    xr.Dataset
        Dataset containing tas, tasmax, tasmin, sfc_wind, rsds, and hurs.
    """
    time = np.array(["2020-07-01", "2020-07-02", "2020-07-03"], dtype="datetime64[D]")
    lats = np.array([40.0, 45.0])
    lons = np.array([0.0, 5.0])

    shape = (3, 2, 2)
    tas = xr.DataArray(
        np.ones(shape) * 298.15,
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lats, "lon": lons},
        attrs={"units": "K"},
    )
    tasmax = xr.DataArray(
        np.ones(shape) * 303.15,
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lats, "lon": lons},
        attrs={"units": "K"},
    )
    tasmin = xr.DataArray(
        np.ones(shape) * 293.15,
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lats, "lon": lons},
        attrs={"units": "K"},
    )
    sfc_wind = xr.DataArray(
        np.ones(shape) * 2.0,
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lats, "lon": lons},
        attrs={"units": "m s-1"},
    )
    rsds = xr.DataArray(
        np.ones(shape) * 250.0,  # 250 W m-2 (~21.6 MJ m-2 d-1)
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lats, "lon": lons},
        attrs={"units": "W m-2"},
    )
    hurs = xr.DataArray(
        np.ones(shape) * 50.0,
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lats, "lon": lons},
        attrs={"units": "%"},
    )

    return xr.Dataset({
        "tas": tas,
        "tasmax": tasmax,
        "tasmin": tasmin,
        "sfc_wind": sfc_wind,
        "rsds": rsds,
        "hurs": hurs,
    })


def test_fao56_indicator(sample_evap_dataset: xr.Dataset) -> None:
    """
    Test FAO-56 Penman-Monteith potential evapotranspiration and alias.

    Parameters
    ----------
    sample_evap_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    res = potential_evapotranspiration_fao56(ds=sample_evap_dataset)
    assert isinstance(res, xr.DataArray)
    assert res.attrs["units"] == "mm d-1"
    assert res.attrs["standard_name"] == "water_potential_evaporation_flux"
    assert res.shape == (3, 2, 2)
    # Summer ET0 values should typically be between 2 and 10 mm/day
    assert np.all(res.values > 1.0)
    assert np.all(res.values < 15.0)

    # Test alias
    res_alias = pet_fao56(ds=sample_evap_dataset)
    np.testing.assert_allclose(res.values, res_alias.values)


def test_priestley_taylor_indicator(sample_evap_dataset: xr.Dataset) -> None:
    """
    Test Priestley-Taylor potential evapotranspiration and alias.

    Parameters
    ----------
    sample_evap_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    res = potential_evapotranspiration_priestley_taylor(ds=sample_evap_dataset)
    assert res.attrs["units"] == "mm d-1"
    assert res.attrs["standard_name"] == "water_potential_evaporation_flux"
    assert np.all(res.values > 1.0)

    res_alias = pet_priestley_taylor(ds=sample_evap_dataset)
    np.testing.assert_allclose(res.values, res_alias.values)


def test_hargreaves_indicator(sample_evap_dataset: xr.Dataset) -> None:
    """
    Test Hargreaves-Samani potential evapotranspiration on gridded and 1D data.

    Parameters
    ----------
    sample_evap_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    res = potential_evapotranspiration_hargreaves(ds=sample_evap_dataset)
    assert res.attrs["units"] == "mm d-1"
    assert res.attrs["standard_name"] == "water_potential_evaporation_flux"
    assert res.shape == (3, 2, 2)
    assert np.all(res.values > 1.0)

    res_alias = pet_hargreaves(ds=sample_evap_dataset)
    np.testing.assert_allclose(res.values, res_alias.values)

    # Test 1D timeseries
    ds_1d = sample_evap_dataset.isel(lat=0, lon=0)
    res_1d = pet_hargreaves(ds=ds_1d)
    assert res_1d.shape == (3,)
    assert np.all(res_1d.values > 1.0)


def test_turc_indicator(sample_evap_dataset: xr.Dataset) -> None:
    """
    Test Turc potential evapotranspiration and alias.

    Parameters
    ----------
    sample_evap_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    res = potential_evapotranspiration_turc(ds=sample_evap_dataset)
    assert res.attrs["units"] == "mm d-1"
    assert res.attrs["standard_name"] == "water_potential_evaporation_flux"
    assert np.all(res.values > 1.0)

    res_alias = pet_turc(ds=sample_evap_dataset)
    np.testing.assert_allclose(res.values, res_alias.values)


def test_makkink_indicator(sample_evap_dataset: xr.Dataset) -> None:
    """
    Test Makkink potential evapotranspiration and alias.

    Parameters
    ----------
    sample_evap_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    res = potential_evapotranspiration_makkink(ds=sample_evap_dataset)
    assert res.attrs["units"] == "mm d-1"
    assert res.attrs["standard_name"] == "water_potential_evaporation_flux"
    assert np.all(res.values > 1.0)

    res_alias = pet_makkink(ds=sample_evap_dataset)
    np.testing.assert_allclose(res.values, res_alias.values)


def test_penman_indicator(sample_evap_dataset: xr.Dataset) -> None:
    """
    Test Penman open-water potential evaporation and alias.

    Parameters
    ----------
    sample_evap_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    res = potential_evapotranspiration_penman(ds=sample_evap_dataset)
    assert res.attrs["units"] == "mm d-1"
    assert res.attrs["standard_name"] == "water_potential_evaporation_flux"
    assert np.all(res.values > 1.0)

    res_alias = pet_penman(ds=sample_evap_dataset)
    np.testing.assert_allclose(res.values, res_alias.values)


def test_dask_parallelization(sample_evap_dataset: xr.Dataset) -> None:
    """
    Verify that calculations respect Dask chunking and lazy execution.

    Parameters
    ----------
    sample_evap_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    ds_chunked = sample_evap_dataset.chunk({"time": 1, "lat": 1, "lon": 1})
    res_lazy = pet_fao56(ds=ds_chunked)
    assert hasattr(res_lazy.data, "dask")
    computed = res_lazy.compute()
    assert isinstance(computed, xr.DataArray)
    assert not hasattr(computed.data, "dask")


def test_regression_fao56_resolves_tmax_tmin_from_dataset(sample_evap_dataset: xr.Dataset) -> None:
    """
    Verify that FAO-56 resolves tasmax and tasmin from dataset if not explicitly passed.

    Parameters
    ----------
    sample_evap_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    res_implicit = potential_evapotranspiration_fao56(ds=sample_evap_dataset)
    res_explicit = potential_evapotranspiration_fao56(
        ds=sample_evap_dataset,
        tasmax="tasmax",
        tasmin="tasmin",
    )
    np.testing.assert_allclose(res_implicit.values, res_explicit.values)


def test_regression_equatorial_latitude_conversion(sample_evap_dataset: xr.Dataset) -> None:
    """
    Verify that equatorial latitudes near zero in degrees are correctly converted to radians.

    Parameters
    ----------
    sample_evap_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    ds_eq = sample_evap_dataset.copy(deep=True)
    ds_eq = ds_eq.assign_coords(lat=np.array([0.5, 1.0]))
    res = potential_evapotranspiration_fao56(ds=ds_eq)
    assert not np.isnan(res.values).any()
    assert np.all(res.values > 0.0)


def test_regression_numpy_array_latitude(sample_evap_dataset: xr.Dataset) -> None:
    """
    Verify that passing a raw numpy array as lat does not raise AttributeError.

    Parameters
    ----------
    sample_evap_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    lat_arr = np.array([40.0, 45.0])
    res = potential_evapotranspiration_fao56(ds=sample_evap_dataset, lat=lat_arr)
    assert not np.isnan(res.values).any()


def test_regression_radiation_unicode_units_conversion(sample_evap_dataset: xr.Dataset) -> None:
    """
    Verify that radiation with unicode units W/m² is converted to MJ m-2 d-1.

    Parameters
    ----------
    sample_evap_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    ds_unicode = sample_evap_dataset.copy(deep=True)
    ds_unicode["rsds"].attrs["units"] = "W/m²"
    res_unicode = potential_evapotranspiration_fao56(ds=ds_unicode)

    ds_std = sample_evap_dataset.copy(deep=True)
    ds_std["rsds"].attrs["units"] = "W m-2"
    res_std = potential_evapotranspiration_fao56(ds=ds_std)

    np.testing.assert_allclose(res_unicode.values, res_std.values)


def test_regression_hargreaves_negative_trange(sample_evap_dataset: xr.Dataset) -> None:
    """
    Verify that inverted tasmax < tasmin does not produce NaNs in Hargreaves.

    Parameters
    ----------
    sample_evap_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    ds_inverted = sample_evap_dataset.copy(deep=True)
    # Set tasmax lower than tasmin
    ds_inverted["tasmax"] = ds_inverted["tasmin"] - 2.0
    res = potential_evapotranspiration_hargreaves(ds=ds_inverted)
    assert not np.isnan(res.values).any()
    assert np.all(res.values == 0.0)


def test_regression_geopotential_height_elevation(sample_evap_dataset: xr.Dataset) -> None:
    """
    Verify that geopotential z in m2 s-2 is converted to elevation in metres.

    Parameters
    ----------
    sample_evap_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    ds_z = sample_evap_dataset.copy(deep=True)
    # 980.665 m2 s-2 corresponds to ~100 m elevation
    ds_z["z"] = xr.DataArray(
        np.full((2, 2), 980.665),
        dims=["lat", "lon"],
        coords={"lat": ds_z.lat, "lon": ds_z.lon},
        attrs={"units": "m2 s-2"},
    )
    res_z = potential_evapotranspiration_fao56(ds=ds_z)
    assert not np.isnan(res_z.values).any()
