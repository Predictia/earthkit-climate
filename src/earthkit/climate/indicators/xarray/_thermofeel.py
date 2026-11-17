import xarray as xr
from earthkit.utils.units import convert_units

from earthkit.climate.utils.decorators import wrap_indicator

try:
    import thermofeel as tf
except ImportError:
    tf = None


def _check_thermofeel_installed() -> None:
    """
    Check if the thermofeel package is installed.

    Raises
    ------
    ImportError
        If thermofeel is not installed.
    """
    if tf is None:
        raise ImportError(
            "thermofeel is required to calculate thermal comfort indicators. "
            "Install it with 'uv add thermofeel' or 'pip install thermofeel'."
        )


def _resolve_dataarray(
    value: xr.DataArray | str,
    ds: xr.Dataset | None,
    argument: str,
) -> xr.DataArray:
    """
    Resolve a DataArray from an explicit DataArray or a variable name in a Dataset.

    Parameters
    ----------
    value : xr.DataArray | str
        Input DataArray or string naming a variable in ``ds``.
    ds : xr.Dataset | None
        Dataset containing variables when string names are passed.
    argument : str
        Name of the argument being resolved, for error reporting.

    Returns
    -------
    xr.DataArray
        The resolved DataArray.
    """
    if isinstance(value, xr.DataArray):
        return value
    if ds is None:
        raise ValueError(f"{argument!r} is a variable name, so ds must be supplied")
    try:
        return ds[value]
    except KeyError as exc:
        raise KeyError(f"Dataset has no variable {value!r} for {argument!r}") from exc


def _ensure_kelvin(da: xr.DataArray) -> xr.DataArray:
    """
    Ensure temperature DataArray is in Kelvin for thermofeel calculations.

    Parameters
    ----------
    da : xr.DataArray
        Input temperature DataArray.

    Returns
    -------
    xr.DataArray
        Temperature DataArray in Kelvin.
    """
    units = str(da.attrs.get("units", "")).strip().lower()
    if units in {"degc", "celsius", "°c", "c"}:
        out = da + 273.15
        out.attrs["units"] = "K"
        return out
    if units in {"k", "kelvin", "degk"}:
        return da
    try:
        return convert_units(da, "K")
    except Exception:
        return da


def _ensure_percent(da: xr.DataArray) -> xr.DataArray:
    """
    Ensure relative humidity DataArray is expressed in percentage (0 to 100).

    Parameters
    ----------
    da : xr.DataArray
        Input relative humidity DataArray.

    Returns
    -------
    xr.DataArray
        Relative humidity DataArray in percentage.
    """
    units = str(da.attrs.get("units", "")).strip().lower()
    if units in {"1", "fraction", "ratio"} or (units == "" and float(da.max()) <= 1.0):
        out = da * 100.0
        out.attrs["units"] = "%"
        return out
    return da


def _convert_temperature_output(da: xr.DataArray, target_units: str) -> xr.DataArray:
    """
    Convert a temperature result from Kelvin to target units.

    Parameters
    ----------
    da : xr.DataArray
        Calculated temperature DataArray in Kelvin.
    target_units : str
        Target output units, typically 'K' or 'degC'.

    Returns
    -------
    xr.DataArray
        DataArray converted to target units with updated metadata.
    """
    clean_target = target_units.strip().lower()
    if clean_target in {"degc", "celsius", "°c", "c"}:
        out = da - 273.15
        out.attrs["units"] = "degC"
        return out
    if clean_target in {"k", "kelvin", "degk"}:
        out = da.copy()
        out.attrs["units"] = "K"
        return out
    raise ValueError(f"Unsupported output_units {target_units!r}. Supported units: 'K', 'degC'.")


@wrap_indicator()
def universal_thermal_climate_index(
    tas: xr.DataArray | str = "tas",
    sfc_wind: xr.DataArray | str = "sfc_wind",
    mrt: xr.DataArray | str = "mrt",
    hurs: xr.DataArray | str | None = None,
    tdps: xr.DataArray | str | None = None,
    ds: xr.Dataset | None = None,
    output_units: str = "K",
) -> xr.DataArray:
    """
    Calculate the Universal Thermal Climate Index (UTCI).

    The Universal Thermal Climate Index assesses outdoor thermal comfort by
    modelling the human physiological response based on 2m temperature, 10m
    wind speed, mean radiant temperature, and ambient humidity.

    Parameters
    ----------
    tas : xr.DataArray | str, default="tas"
        2-metre air temperature.
    sfc_wind : xr.DataArray | str, default="sfc_wind"
        10-metre wind speed in m/s.
    mrt : xr.DataArray | str, default="mrt"
        Mean radiant temperature in Kelvin.
    hurs : xr.DataArray | str | None, default=None
        Relative humidity in percentage [0, 100].
    tdps : xr.DataArray | str | None, default=None
        Dew point temperature in Kelvin.
    ds : xr.Dataset | None, default=None
        Dataset containing variables if strings are passed.
    output_units : str, default="K"
        Units of the returned DataArray ('K' or 'degC').

    Returns
    -------
    xr.DataArray
        Universal Thermal Climate Index.
    """
    _check_thermofeel_installed()
    t2_da = _ensure_kelvin(_resolve_dataarray(tas, ds, "tas"))
    va_da = _resolve_dataarray(sfc_wind, ds, "sfc_wind")
    mrt_da = _ensure_kelvin(_resolve_dataarray(mrt, ds, "mrt"))

    if tdps is not None:
        td_da = _ensure_kelvin(_resolve_dataarray(tdps, ds, "tdps"))
    elif hurs is not None:
        rh_da = _ensure_percent(_resolve_dataarray(hurs, ds, "hurs"))
        td_da = xr.apply_ufunc(
            tf.calculate_dew_point_from_relative_humidity,
            rh_da,
            t2_da,
            dask="parallelized",
            output_dtypes=[float],
        )
    elif ds is not None and "tdps" in ds:
        td_da = _ensure_kelvin(ds["tdps"])
    elif ds is not None and "hurs" in ds:
        rh_da = _ensure_percent(ds["hurs"])
        td_da = xr.apply_ufunc(
            tf.calculate_dew_point_from_relative_humidity,
            rh_da,
            t2_da,
            dask="parallelized",
            output_dtypes=[float],
        )
    else:
        raise ValueError("Either 'tdps' (dew point) or 'hurs' (relative humidity) must be provided.")

    res = xr.apply_ufunc(
        tf.calculate_utci,
        t2_da,
        va_da,
        mrt_da,
        td_da,
        dask="parallelized",
        output_dtypes=[float],
    )
    res = _convert_temperature_output(res, output_units)
    res.attrs["standard_name"] = "universal_thermal_climate_index"
    res.attrs["long_name"] = "Universal Thermal Climate Index (UTCI)"
    return res


utci = universal_thermal_climate_index


@wrap_indicator()
def apparent_temperature(
    tas: xr.DataArray | str = "tas",
    sfc_wind: xr.DataArray | str = "sfc_wind",
    hurs: xr.DataArray | str = "hurs",
    rsds: xr.DataArray | str | None = None,
    ds: xr.Dataset | None = None,
    output_units: str = "K",
) -> xr.DataArray:
    """
    Calculate the Apparent Temperature (Steadman's model).

    Apparent Temperature combines air temperature, wind speed, relative humidity,
    and optionally net solar radiation to measure the perceived temperature.

    Parameters
    ----------
    tas : xr.DataArray | str, default="tas"
        2-metre air temperature.
    sfc_wind : xr.DataArray | str, default="sfc_wind"
        10-metre wind speed in m/s.
    hurs : xr.DataArray | str, default="hurs"
        Relative humidity in percentage.
    rsds : xr.DataArray | str | None, default=None
        Optional downwelling shortwave solar radiation in W/m2 for radiation version.
    ds : xr.Dataset | None, default=None
        Dataset containing variables if strings are passed.
    output_units : str, default="K"
        Units of the returned DataArray ('K' or 'degC').

    Returns
    -------
    xr.DataArray
        Apparent temperature.
    """
    _check_thermofeel_installed()
    t2_da = _ensure_kelvin(_resolve_dataarray(tas, ds, "tas"))
    va_da = _resolve_dataarray(sfc_wind, ds, "sfc_wind")
    rh_da = _ensure_percent(_resolve_dataarray(hurs, ds, "hurs"))

    if rsds is not None:
        q_da = _resolve_dataarray(rsds, ds, "rsds")
        res = xr.apply_ufunc(
            tf.calculate_apparent_temperature_radiation,
            t2_da,
            va_da,
            rh_da,
            q_da,
            dask="parallelized",
            output_dtypes=[float],
        )
    else:
        res = xr.apply_ufunc(
            tf.calculate_apparent_temperature,
            t2_da,
            va_da,
            rh_da,
            dask="parallelized",
            output_dtypes=[float],
        )

    res = _convert_temperature_output(res, output_units)
    res.attrs["standard_name"] = "apparent_temperature"
    res.attrs["long_name"] = "Apparent Temperature"
    return res


@wrap_indicator()
def humidex(
    tas: xr.DataArray | str = "tas",
    tdps: xr.DataArray | str | None = None,
    hurs: xr.DataArray | str | None = None,
    ds: xr.Dataset | None = None,
    output_units: str = "K",
) -> xr.DataArray:
    """
    Calculate the Humidex rating.

    Humidex is an index used by Environment Canada to describe how hot the
    weather feels to the average person by combining heat and humidity.

    Parameters
    ----------
    tas : xr.DataArray | str, default="tas"
        2-metre air temperature.
    tdps : xr.DataArray | str | None, default=None
        Dew point temperature.
    hurs : xr.DataArray | str | None, default=None
        Relative humidity in percentage (used to calculate dew point if tdps is not given).
    ds : xr.Dataset | None, default=None
        Dataset containing variables if strings are passed.
    output_units : str, default="K"
        Units of the returned DataArray ('K' or 'degC').

    Returns
    -------
    xr.DataArray
        Humidex index.
    """
    _check_thermofeel_installed()
    t2_da = _ensure_kelvin(_resolve_dataarray(tas, ds, "tas"))

    if tdps is not None:
        td_da = _ensure_kelvin(_resolve_dataarray(tdps, ds, "tdps"))
    elif hurs is not None:
        rh_da = _ensure_percent(_resolve_dataarray(hurs, ds, "hurs"))
        td_da = xr.apply_ufunc(
            tf.calculate_dew_point_from_relative_humidity,
            rh_da,
            t2_da,
            dask="parallelized",
            output_dtypes=[float],
        )
    elif ds is not None and "tdps" in ds:
        td_da = _ensure_kelvin(ds["tdps"])
    elif ds is not None and "hurs" in ds:
        rh_da = _ensure_percent(ds["hurs"])
        td_da = xr.apply_ufunc(
            tf.calculate_dew_point_from_relative_humidity,
            rh_da,
            t2_da,
            dask="parallelized",
            output_dtypes=[float],
        )
    else:
        raise ValueError("Either 'tdps' (dew point) or 'hurs' (relative humidity) must be provided.")

    res = xr.apply_ufunc(
        tf.calculate_humidex,
        t2_da,
        td_da,
        dask="parallelized",
        output_dtypes=[float],
    )
    res = _convert_temperature_output(res, output_units)
    res.attrs["standard_name"] = "humidex"
    res.attrs["long_name"] = "Humidex"
    return res


@wrap_indicator()
def wind_chill(
    tas: xr.DataArray | str = "tas",
    sfc_wind: xr.DataArray | str = "sfc_wind",
    ds: xr.Dataset | None = None,
    output_units: str = "K",
) -> xr.DataArray:
    """
    Calculate the Wind Chill temperature index.

    Wind Chill models the cooling effect of wind on exposed human skin at
    low air temperatures.

    Parameters
    ----------
    tas : xr.DataArray | str, default="tas"
        2-metre air temperature.
    sfc_wind : xr.DataArray | str, default="sfc_wind"
        10-metre wind speed in m/s.
    ds : xr.Dataset | None, default=None
        Dataset containing variables if strings are passed.
    output_units : str, default="K"
        Units of the returned DataArray ('K' or 'degC').

    Returns
    -------
    xr.DataArray
        Wind Chill index.
    """
    _check_thermofeel_installed()
    t2_da = _ensure_kelvin(_resolve_dataarray(tas, ds, "tas"))
    va_da = _resolve_dataarray(sfc_wind, ds, "sfc_wind")

    res = xr.apply_ufunc(
        tf.calculate_wind_chill,
        t2_da,
        va_da,
        dask="parallelized",
        output_dtypes=[float],
    )
    res = _convert_temperature_output(res, output_units)
    res.attrs["standard_name"] = "wind_chill"
    res.attrs["long_name"] = "Wind Chill Index"
    return res


@wrap_indicator()
def heat_index(
    tas: xr.DataArray | str = "tas",
    tdps: xr.DataArray | str | None = None,
    hurs: xr.DataArray | str | None = None,
    ds: xr.Dataset | None = None,
    output_units: str = "K",
) -> xr.DataArray:
    """
    Calculate the NWS Heat Index (adjusted).

    The Heat Index measures perceived temperature during hot conditions by
    combining ambient air temperature and humidity.

    Parameters
    ----------
    tas : xr.DataArray | str, default="tas"
        2-metre air temperature.
    tdps : xr.DataArray | str | None, default=None
        Dew point temperature.
    hurs : xr.DataArray | str | None, default=None
        Relative humidity in percentage.
    ds : xr.Dataset | None, default=None
        Dataset containing variables if strings are passed.
    output_units : str, default="K"
        Units of the returned DataArray ('K' or 'degC').

    Returns
    -------
    xr.DataArray
        Heat Index.
    """
    _check_thermofeel_installed()
    t2_da = _ensure_kelvin(_resolve_dataarray(tas, ds, "tas"))

    if tdps is not None:
        td_da = _ensure_kelvin(_resolve_dataarray(tdps, ds, "tdps"))
    elif hurs is not None:
        rh_da = _ensure_percent(_resolve_dataarray(hurs, ds, "hurs"))
        td_da = xr.apply_ufunc(
            tf.calculate_dew_point_from_relative_humidity,
            rh_da,
            t2_da,
            dask="parallelized",
            output_dtypes=[float],
        )
    elif ds is not None and "tdps" in ds:
        td_da = _ensure_kelvin(ds["tdps"])
    elif ds is not None and "hurs" in ds:
        rh_da = _ensure_percent(ds["hurs"])
        td_da = xr.apply_ufunc(
            tf.calculate_dew_point_from_relative_humidity,
            rh_da,
            t2_da,
            dask="parallelized",
            output_dtypes=[float],
        )
    else:
        raise ValueError("Either 'tdps' (dew point) or 'hurs' (relative humidity) must be provided.")

    res = xr.apply_ufunc(
        tf.calculate_heat_index_adjusted,
        t2_da,
        td_da,
        dask="parallelized",
        output_dtypes=[float],
    )
    res = _convert_temperature_output(res, output_units)
    res.attrs["standard_name"] = "heat_index"
    res.attrs["long_name"] = "Heat Index"
    return res


@wrap_indicator()
def wet_bulb_globe_temperature(
    tas: xr.DataArray | str = "tas",
    hurs: xr.DataArray | str = "hurs",
    ds: xr.Dataset | None = None,
    output_units: str = "K",
) -> xr.DataArray:
    """
    Calculate the Wet Bulb Globe Temperature (WBGT) simplified estimate.

    The Wet Bulb Globe Temperature provides a measure of heat stress in direct
    sunlight, accounting for temperature and relative humidity.

    Parameters
    ----------
    tas : xr.DataArray | str, default="tas"
        2-metre air temperature.
    hurs : xr.DataArray | str, default="hurs"
        Relative humidity in percentage.
    ds : xr.Dataset | None, default=None
        Dataset containing variables if strings are passed.
    output_units : str, default="K"
        Units of the returned DataArray ('K' or 'degC').

    Returns
    -------
    xr.DataArray
        Wet Bulb Globe Temperature.
    """
    _check_thermofeel_installed()
    t2_da = _ensure_kelvin(_resolve_dataarray(tas, ds, "tas"))
    rh_da = _ensure_percent(_resolve_dataarray(hurs, ds, "hurs"))

    res = xr.apply_ufunc(
        tf.calculate_wbgt_simple,
        t2_da,
        rh_da,
        dask="parallelized",
        output_dtypes=[float],
    )
    res = _convert_temperature_output(res, output_units)
    res.attrs["standard_name"] = "wet_bulb_globe_temperature"
    res.attrs["long_name"] = "Wet Bulb Globe Temperature (simplified)"
    return res


wbgt = wet_bulb_globe_temperature


@wrap_indicator()
def normal_effective_temperature(
    tas: xr.DataArray | str = "tas",
    sfc_wind: xr.DataArray | str = "sfc_wind",
    hurs: xr.DataArray | str = "hurs",
    ds: xr.Dataset | None = None,
    output_units: str = "K",
) -> xr.DataArray:
    """
    Calculate the Normal Effective Temperature (NET).

    NET evaluates environmental thermal sensation incorporating temperature,
    wind speed, and relative humidity for clothed persons in moderate conditions.

    Parameters
    ----------
    tas : xr.DataArray | str, default="tas"
        2-metre air temperature.
    sfc_wind : xr.DataArray | str, default="sfc_wind"
        10-metre wind speed in m/s.
    hurs : xr.DataArray | str, default="hurs"
        Relative humidity in percentage.
    ds : xr.Dataset | None, default=None
        Dataset containing variables if strings are passed.
    output_units : str, default="K"
        Units of the returned DataArray ('K' or 'degC').

    Returns
    -------
    xr.DataArray
        Normal Effective Temperature.
    """
    _check_thermofeel_installed()
    t2_da = _ensure_kelvin(_resolve_dataarray(tas, ds, "tas"))
    va_da = _resolve_dataarray(sfc_wind, ds, "sfc_wind")
    rh_da = _ensure_percent(_resolve_dataarray(hurs, ds, "hurs"))

    res = xr.apply_ufunc(
        tf.calculate_normal_effective_temperature,
        t2_da,
        va_da,
        rh_da,
        dask="parallelized",
        output_dtypes=[float],
    )
    res = _convert_temperature_output(res, output_units)
    res.attrs["standard_name"] = "normal_effective_temperature"
    res.attrs["long_name"] = "Normal Effective Temperature"
    return res


net = normal_effective_temperature
