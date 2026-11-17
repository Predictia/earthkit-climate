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


def test_thermofeel_direct_dataarray_inputs(sample_thermal_dataset: xr.Dataset) -> None:
    """
    Test calling all thermofeel indicators with standalone DataArray inputs instead of a Dataset.

    Parameters
    ----------
    sample_thermal_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    ds = sample_thermal_dataset
    tas = ds["tas"]
    wind = ds["sfc_wind"]
    mrt = ds["mrt"]
    hurs = ds["hurs"]
    tdps = ds["tdps"]

    res_utci = utci(tas=tas, sfc_wind=wind, mrt=mrt, hurs=hurs)
    assert isinstance(res_utci, xr.DataArray)
    assert res_utci.shape == (3,)

    res_chill = wind_chill(tas=tas, sfc_wind=wind)
    assert isinstance(res_chill, xr.DataArray)

    res_hi = heat_index(tas=tas, hurs=hurs)
    assert isinstance(res_hi, xr.DataArray)

    res_hum = humidex(tas=tas, tdps=tdps)
    assert isinstance(res_hum, xr.DataArray)

    res_app = apparent_temperature(tas=tas, sfc_wind=wind, hurs=hurs)
    assert isinstance(res_app, xr.DataArray)

    res_wbgt = wbgt(tas=tas, hurs=hurs)
    assert isinstance(res_wbgt, xr.DataArray)

    res_net = net(tas=tas, sfc_wind=wind, hurs=hurs)
    assert isinstance(res_net, xr.DataArray)


def test_thermofeel_spatial_multidimensional_grid() -> None:
    """
    Test thermofeel indicators on a multi-dimensional spatial grid (time, lat, lon).

    Verifies coordinate preservation, dimension alignment, and spatial execution.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    time = np.array(["2020-07-01", "2020-07-02"], dtype="datetime64[D]")
    lats = np.array([30.0, 40.0])
    lons = np.array([0.0, 10.0])
    shape = (2, 2, 2)

    ds_3d = xr.Dataset(
        {
            "tas": xr.DataArray(np.full(shape, 300.0), dims=["time", "lat", "lon"], attrs={"units": "K"}),
            "sfc_wind": xr.DataArray(np.full(shape, 2.0), dims=["time", "lat", "lon"], attrs={"units": "m s-1"}),
            "mrt": xr.DataArray(np.full(shape, 310.0), dims=["time", "lat", "lon"], attrs={"units": "K"}),
            "hurs": xr.DataArray(np.full(shape, 60.0), dims=["time", "lat", "lon"], attrs={"units": "%"}),
        },
        coords={"time": time, "lat": lats, "lon": lons},
    )

    res = utci(ds=ds_3d)
    assert res.dims == ("time", "lat", "lon")
    assert res.shape == (2, 2, 2)
    np.testing.assert_allclose(res.lat.values, lats)
    np.testing.assert_allclose(res.lon.values, lons)


def test_thermofeel_unit_conversions_celsius_and_fraction_rh(sample_thermal_dataset: xr.Dataset) -> None:
    """
    Verify calculation invariance when input temperature is in Celsius and humidity is a fraction.

    Parameters
    ----------
    sample_thermal_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    ds_c = sample_thermal_dataset.copy(deep=True)
    ds_c["tas"] = ds_c["tas"] - 273.15
    ds_c["tas"].attrs["units"] = "degC"
    ds_c["mrt"] = ds_c["mrt"] - 273.15
    ds_c["mrt"].attrs["units"] = "°C"
    ds_c["tdps"] = ds_c["tdps"] - 273.15
    ds_c["tdps"].attrs["units"] = "C"

    # Relative humidity as ratio 0.0 - 1.0
    ds_c["hurs"] = ds_c["hurs"] / 100.0
    ds_c["hurs"].attrs["units"] = "1"

    res_c = utci(ds=ds_c)
    res_orig = utci(ds=sample_thermal_dataset)
    np.testing.assert_allclose(res_c.values, res_orig.values, rtol=1e-5)


def test_thermofeel_output_units_degc_across_all_indicators(sample_thermal_dataset: xr.Dataset) -> None:
    """
    Verify output_units='degC' across all thermal comfort indicators.

    Parameters
    ----------
    sample_thermal_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    ds = sample_thermal_dataset
    funcs = [
        (utci, "universal_thermal_climate_index"),
        (apparent_temperature, "apparent_temperature"),
        (heat_index, "heat_index"),
        (humidex, "humidex"),
        (wind_chill, "wind_chill"),
        (wbgt, "wet_bulb_globe_temperature"),
        (net, "normal_effective_temperature"),
    ]

    for func, name in funcs:
        res_k = func(ds=ds, output_units="K")
        res_c = func(ds=ds, output_units="degC")

        assert res_c.attrs["units"] == "degC"
        assert res_c.attrs["standard_name"] == name
        np.testing.assert_allclose(res_c.values, res_k.values - 273.15, rtol=1e-5)


def test_thermofeel_physical_monotonicity_checks() -> None:
    """
    Verify physical behavior and monotonicity of thermal indices.

    Tests that wind chill decreases with increasing wind speed, heat index increases
    with increasing humidity, and apparent temperature increases with solar radiation.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    time = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")

    # Wind chill: colder when wind increases from 2 m/s to 10 m/s
    cold_tas = xr.DataArray([265.15, 265.15], dims=["time"], coords={"time": time}, attrs={"units": "K"})
    wind_calm = xr.DataArray([2.0, 2.0], dims=["time"], coords={"time": time}, attrs={"units": "m s-1"})
    wind_strong = xr.DataArray([10.0, 10.0], dims=["time"], coords={"time": time}, attrs={"units": "m s-1"})

    wc_calm = wind_chill(tas=cold_tas, sfc_wind=wind_calm)
    wc_strong = wind_chill(tas=cold_tas, sfc_wind=wind_strong)
    assert np.all(wc_strong.values < wc_calm.values)

    # Heat index: warmer perceived temperature when relative humidity increases
    hot_tas = xr.DataArray([305.15, 305.15], dims=["time"], coords={"time": time}, attrs={"units": "K"})
    rh_low = xr.DataArray([40.0, 40.0], dims=["time"], coords={"time": time}, attrs={"units": "%"})
    rh_high = xr.DataArray([80.0, 80.0], dims=["time"], coords={"time": time}, attrs={"units": "%"})

    hi_low = heat_index(tas=hot_tas, hurs=rh_low)
    hi_high = heat_index(tas=hot_tas, hurs=rh_high)
    assert np.all(hi_high.values > hi_low.values)

    # UTCI: warmer perceived temperature when mean radiant temperature increases (direct sun)
    mild_tas = xr.DataArray([295.15, 295.15], dims=["time"], coords={"time": time}, attrs={"units": "K"})
    wind_med = xr.DataArray([2.0, 2.0], dims=["time"], coords={"time": time}, attrs={"units": "m s-1"})
    rh_med = xr.DataArray([50.0, 50.0], dims=["time"], coords={"time": time}, attrs={"units": "%"})
    mrt_shade = xr.DataArray([295.15, 295.15], dims=["time"], coords={"time": time}, attrs={"units": "K"})
    mrt_sun = xr.DataArray([325.15, 325.15], dims=["time"], coords={"time": time}, attrs={"units": "K"})

    utci_shade = utci(tas=mild_tas, sfc_wind=wind_med, mrt=mrt_shade, hurs=rh_med)
    utci_sun = utci(tas=mild_tas, sfc_wind=wind_med, mrt=mrt_sun, hurs=rh_med)
    assert np.all(utci_sun.values > utci_shade.values)


def test_thermofeel_numerical_consistency_with_upstream() -> None:
    """
    Verify exact numerical match against native thermofeel library functions.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    import thermofeel as tf

    t_val = 300.15
    wind_val = 2.0
    mrt_val = 310.15
    rh_val = 60.0

    td_val = tf.calculate_dew_point_from_relative_humidity(rh_val, t_val)
    tf_utci = tf.calculate_utci(t_val, wind_val, mrt_val, td_val)

    ds_1d = xr.Dataset({
        "tas": xr.DataArray([t_val], dims=["time"], coords={"time": ["2020-07-01"]}, attrs={"units": "K"}),
        "sfc_wind": xr.DataArray([wind_val], dims=["time"], coords={"time": ["2020-07-01"]}, attrs={"units": "m s-1"}),
        "mrt": xr.DataArray([mrt_val], dims=["time"], coords={"time": ["2020-07-01"]}, attrs={"units": "K"}),
        "hurs": xr.DataArray([rh_val], dims=["time"], coords={"time": ["2020-07-01"]}, attrs={"units": "%"}),
    })

    ek_utci = utci(ds=ds_1d)
    np.testing.assert_allclose(ek_utci.values[0], tf_utci, rtol=1e-5)

    # Wind chill comparison
    t_cold = 263.15
    wind_cold = 8.0
    tf_wc = tf.calculate_wind_chill(t_cold, wind_cold)
    ek_wc = wind_chill(
        tas=xr.DataArray([t_cold], dims=["time"], coords={"time": ["2020-01-01"]}, attrs={"units": "K"}),
        sfc_wind=xr.DataArray([wind_cold], dims=["time"], coords={"time": ["2020-01-01"]}, attrs={"units": "m s-1"}),
    )
    np.testing.assert_allclose(ek_wc.values[0], tf_wc, rtol=1e-5)


def test_thermofeel_error_handling(sample_thermal_dataset: xr.Dataset) -> None:
    """
    Verify error handling for missing variables, missing dataset, and invalid units.

    Parameters
    ----------
    sample_thermal_dataset : xr.Dataset
        Sample meteorological dataset.

    Returns
    -------
    None
    """
    # Missing variable in dataset
    with pytest.raises(KeyError, match="Dataset has no variable"):
        utci(ds=sample_thermal_dataset, sfc_wind="nonexistent_wind")

    # Variable name string passed without dataset
    with pytest.raises(ValueError, match="is a variable name, so ds must be supplied"):
        utci(tas="tas", ds=None)

    # Invalid output_units
    with pytest.raises(ValueError, match="Unsupported output_units"):
        utci(ds=sample_thermal_dataset, output_units="Rankine")

    # Neither tdps nor hurs provided
    ds_no_humidity = sample_thermal_dataset.drop_vars(["hurs", "tdps"])
    with pytest.raises(ValueError, match="Either 'tdps' .* or 'hurs' .* must be provided"):
        utci(ds=ds_no_humidity)
