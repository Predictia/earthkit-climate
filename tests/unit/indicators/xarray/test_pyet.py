import numpy as np
import pandas as pd
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


def test_pyet_direct_dataarray_inputs(sample_evap_dataset: xr.Dataset) -> None:
    """
    Test calling all pyet indicator functions with direct DataArray inputs instead of a Dataset.

    Parameters
    ----------
    sample_evap_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    ds = sample_evap_dataset
    tas = ds["tas"]
    tasmax = ds["tasmax"]
    tasmin = ds["tasmin"]
    wind = ds["sfc_wind"]
    rsds = ds["rsds"]
    hurs = ds["hurs"]

    res_fao = pet_fao56(tas=tas, tasmax=tasmax, tasmin=tasmin, sfc_wind=wind, rsds=rsds, hurs=hurs)
    assert isinstance(res_fao, xr.DataArray)
    assert res_fao.shape == (3, 2, 2)

    res_harg = pet_hargreaves(tas=tas, tasmax=tasmax, tasmin=tasmin)
    assert isinstance(res_harg, xr.DataArray)

    res_pt = pet_priestley_taylor(tas=tas, rsds=rsds, hurs=hurs)
    assert isinstance(res_pt, xr.DataArray)

    res_mak = pet_makkink(tas=tas, rsds=rsds)
    assert isinstance(res_mak, xr.DataArray)

    res_turc = pet_turc(tas=tas, rsds=rsds, hurs=hurs)
    assert isinstance(res_turc, xr.DataArray)

    res_pen = pet_penman(tas=tas, sfc_wind=wind, rsds=rsds, hurs=hurs)
    assert isinstance(res_pen, xr.DataArray)


def test_pyet_unit_conversions_celsius_and_fraction_rh(sample_evap_dataset: xr.Dataset) -> None:
    """
    Verify PET calculations with temperature in Celsius and humidity as fraction.

    Parameters
    ----------
    sample_evap_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    ds_celsius = sample_evap_dataset.copy(deep=True)
    ds_celsius["tas"] = ds_celsius["tas"] - 273.15
    ds_celsius["tas"].attrs["units"] = "degC"
    ds_celsius["tasmax"] = ds_celsius["tasmax"] - 273.15
    ds_celsius["tasmax"].attrs["units"] = "°C"
    ds_celsius["tasmin"] = ds_celsius["tasmin"] - 273.15
    ds_celsius["tasmin"].attrs["units"] = "C"

    # Humidity as ratio 0.0 - 1.0
    ds_celsius["hurs"] = ds_celsius["hurs"] / 100.0
    ds_celsius["hurs"].attrs["units"] = "1"

    # Radiation directly in MJ m-2 d-1
    ds_celsius["rsds"] = ds_celsius["rsds"] * 0.0864
    ds_celsius["rsds"].attrs["units"] = "MJ m-2 d-1"

    res_celsius = pet_fao56(ds=ds_celsius)
    res_orig = pet_fao56(ds=sample_evap_dataset)

    np.testing.assert_allclose(res_celsius.values, res_orig.values, rtol=1e-4)


def test_pyet_elevation_options(sample_evap_dataset: xr.Dataset) -> None:
    """
    Verify PET calculations with explicit elevation and surface elevation parameters.

    Parameters
    ----------
    sample_evap_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    # Explicit elevation
    res_elev = pet_fao56(ds=sample_evap_dataset, elevation=1200.0)
    assert isinstance(res_elev, xr.DataArray)
    assert np.all(res_elev.values > 0.0)

    # Elevation as dataset variable
    ds_with_elev = sample_evap_dataset.copy(deep=True)
    ds_with_elev["elevation"] = xr.DataArray(
        np.array([[500.0, 600.0], [700.0, 800.0]]),
        dims=["lat", "lon"],
        coords={"lat": ds_with_elev.lat, "lon": ds_with_elev.lon},
        attrs={"units": "m"},
    )
    res_ds_elev = pet_fao56(ds=ds_with_elev)
    assert isinstance(res_ds_elev, xr.DataArray)

    # Alternative elevation variable name 'orog'
    ds_with_orog = sample_evap_dataset.copy(deep=True)
    ds_with_orog["orog"] = xr.DataArray(
        np.full((2, 2), 300.0),
        dims=["lat", "lon"],
        coords={"lat": ds_with_orog.lat, "lon": ds_with_orog.lon},
        attrs={"units": "m"},
    )
    res_orog = pet_fao56(ds=ds_with_orog)
    assert isinstance(res_orog, xr.DataArray)


def test_pyet_latitude_variations_and_coordinate_inference(sample_evap_dataset: xr.Dataset) -> None:
    """
    Verify latitude coordinate resolution from alternative coordinate names and units.

    Parameters
    ----------
    sample_evap_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    # Coordinate named 'latitude'
    ds_latitude = sample_evap_dataset.rename({"lat": "latitude"})
    res_latitude = pet_fao56(ds=ds_latitude)
    assert isinstance(res_latitude, xr.DataArray)

    # Latitude in radians
    ds_rad = sample_evap_dataset.copy(deep=True)
    ds_rad = ds_rad.assign_coords(lat=np.deg2rad(ds_rad.lat))
    ds_rad.lat.attrs["units"] = "rad"
    res_rad = pet_fao56(ds=ds_rad)
    res_deg = pet_fao56(ds=sample_evap_dataset)
    np.testing.assert_allclose(res_rad.values, res_deg.values, rtol=1e-4)

    # Scalar float latitude
    res_scalar_lat = pet_fao56(ds=sample_evap_dataset, lat=45.0)
    assert isinstance(res_scalar_lat, xr.DataArray)


def test_pyet_numerical_consistency_with_upstream_library() -> None:
    """
    Verify exact numerical match against native pyet library execution.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    import pyet

    time = np.array(["2020-07-01", "2020-07-02", "2020-07-03"], dtype="datetime64[D]")
    lat_val = 40.0
    lat_rad = np.deg2rad(lat_val)

    ds_1d = xr.Dataset(
        {
            "tas": (["time"], [298.15, 298.15, 298.15], {"units": "K"}),
            "tasmax": (["time"], [303.15, 303.15, 303.15], {"units": "K"}),
            "tasmin": (["time"], [293.15, 293.15, 293.15], {"units": "K"}),
            "sfc_wind": (["time"], [2.0, 2.0, 2.0], {"units": "m s-1"}),
            "rsds": (["time"], [250.0, 250.0, 250.0], {"units": "W m-2"}),
            "hurs": (["time"], [50.0, 50.0, 50.0], {"units": "%"}),
        },
        coords={"time": time, "lat": lat_val},
    )

    tmean_s = pd.Series([25.0, 25.0, 25.0], index=pd.to_datetime(time))
    tmax_s = pd.Series([30.0, 30.0, 30.0], index=pd.to_datetime(time))
    tmin_s = pd.Series([20.0, 20.0, 20.0], index=pd.to_datetime(time))
    wind_s = pd.Series([2.0, 2.0, 2.0], index=pd.to_datetime(time))
    rs_s = pd.Series([250.0 * 0.0864, 250.0 * 0.0864, 250.0 * 0.0864], index=pd.to_datetime(time))
    rh_s = pd.Series([50.0, 50.0, 50.0], index=pd.to_datetime(time))

    res_upstream = pyet.pm_fao56(
        tmean=tmean_s,
        wind=wind_s,
        rs=rs_s,
        elevation=0.0,
        lat=lat_rad,
        tmax=tmax_s,
        tmin=tmin_s,
        rh=rh_s,
    )

    res_wrapped = pet_fao56(ds=ds_1d)
    np.testing.assert_allclose(res_wrapped.values, res_upstream.values, rtol=1e-5)


def test_pyet_error_handling(sample_evap_dataset: xr.Dataset) -> None:
    """
    Verify error handling for missing variables, invalid latitude, and missing dataset.

    Parameters
    ----------
    sample_evap_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    # Missing variable in dataset
    with pytest.raises(KeyError, match="Dataset has no variable"):
        pet_fao56(ds=sample_evap_dataset, sfc_wind="nonexistent_variable")

    # Variable name passed without dataset
    with pytest.raises(ValueError, match="is a variable name, so ds must be supplied"):
        pet_fao56(tas="tas", ds=None)

    # Missing latitude when coordinates lack latitude
    ds_no_lat = sample_evap_dataset.drop_vars("lat")
    with pytest.raises(ValueError, match="Latitude is required"):
        pet_fao56(ds=ds_no_lat)
