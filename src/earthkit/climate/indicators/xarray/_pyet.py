import numpy as np
import xarray as xr
from earthkit.utils.units import convert_units

from earthkit.climate.utils.decorators import wrap_indicator

try:
    import pyet
except ImportError:
    pyet = None


def _check_pyet_installed() -> None:
    """
    Check if the pyet package is installed.

    Raises
    ------
    ImportError
        If pyet is not installed.
    """
    if pyet is None:
        raise ImportError(
            "pyet is required to calculate evapotranspiration indicators. "
            "Install it with 'uv add pyet' or 'pip install pyet'."
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


def _ensure_celsius(da: xr.DataArray) -> xr.DataArray:
    """
    Ensure temperature DataArray is in Celsius for pyet calculations.

    Parameters
    ----------
    da : xr.DataArray
        Input temperature DataArray.

    Returns
    -------
    xr.DataArray
        Temperature DataArray in Celsius.
    """
    units = str(da.attrs.get("units", "")).strip().lower()
    if units in {"k", "kelvin", "degk"}:
        out = da - 273.15
        out.attrs["units"] = "degC"
        return out
    if units in {"degc", "celsius", "°c", "c"}:
        return da
    try:
        return convert_units(da, "degC")
    except Exception:
        if not hasattr(da.data, "dask") and float(da.mean()) > 150.0:
            out = da - 273.15
            out.attrs["units"] = "degC"
            return out
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
    if units in {"1", "fraction", "ratio"}:
        out = da * 100.0
        out.attrs["units"] = "%"
        return out
    if units in {"%", "percent", "percentage"}:
        return da
    if units == "" and not hasattr(da.data, "dask") and float(da.max()) <= 1.0:
        out = da * 100.0
        out.attrs["units"] = "%"
        return out
    return da


def _ensure_radiation_mj(da: xr.DataArray) -> xr.DataArray:
    """
    Ensure surface solar radiation DataArray is in MJ m-2 d-1 for pyet.

    Parameters
    ----------
    da : xr.DataArray
        Input surface solar radiation DataArray.

    Returns
    -------
    xr.DataArray
        Radiation DataArray in MJ m-2 d-1.
    """
    units = str(da.attrs.get("units", "")).strip().lower()
    if units in {"mj m-2 d-1", "mj m-2 day-1", "mj/m2/d", "mj/m2/day"}:
        return da
    try:
        return convert_units(da, "MJ m-2 d-1")
    except Exception:
        u_clean = units.replace("²", "2").replace("^2", "2").replace("**2", "2").replace("/", " ")
        if "w" in u_clean:
            out = da * 0.0864
            out.attrs["units"] = "MJ m-2 d-1"
            return out
        if "j" in u_clean and "mj" not in u_clean:
            out = da * 1e-6
            out.attrs["units"] = "MJ m-2 d-1"
            return out
        return da


def _resolve_latitude_rad(
    da: xr.DataArray,
    ds: xr.Dataset | None,
    lat: float | xr.DataArray | np.ndarray | str | None,
) -> float | xr.DataArray:
    """
    Resolve latitude coordinate and convert to radians if given in degrees.

    Parameters
    ----------
    da : xr.DataArray
        Primary temperature DataArray.
    ds : xr.Dataset | None
        Optional Dataset containing coordinates or variables.
    lat : float | xr.DataArray | numpy.ndarray | str | None
        Explicit latitude value, coordinate name, or None to infer.

    Returns
    -------
    float | xr.DataArray
        Latitude in radians.
    """
    lat_val: float | xr.DataArray | np.ndarray
    if lat is not None:
        if isinstance(lat, str):
            lat_val = _resolve_dataarray(lat, ds, "lat")
        else:
            lat_val = lat
    elif "lat" in da.coords:
        lat_val = da.coords["lat"]
    elif "latitude" in da.coords:
        lat_val = da.coords["latitude"]
    elif ds is not None and "lat" in ds:
        lat_val = ds["lat"]
    elif ds is not None and "latitude" in ds:
        lat_val = ds["latitude"]
    else:
        raise ValueError("Latitude is required. Provide 'lat' or ensure a latitude coordinate exists.")

    units = str(getattr(lat_val, "attrs", {}).get("units", "")).lower()

    if isinstance(lat_val, (int, float)) or (isinstance(lat_val, (np.ndarray, xr.DataArray)) and lat_val.ndim == 0):
        if "rad" in units:
            return float(lat_val)
        return float(np.deg2rad(lat_val))

    if isinstance(lat_val, np.ndarray):
        lat_rad = np.deg2rad(lat_val)
        if "lat" in da.coords and lat_rad.size == da.coords["lat"].size:
            return xr.DataArray(lat_rad, dims=["lat"], coords={"lat": da.coords["lat"]})
        if "latitude" in da.coords and lat_rad.size == da.coords["latitude"].size:
            return xr.DataArray(lat_rad, dims=["latitude"], coords={"latitude": da.coords["latitude"]})
        if lat_rad.shape == da.shape:
            return xr.DataArray(lat_rad, dims=da.dims, coords=da.coords)
        return xr.DataArray(lat_rad)

    if isinstance(lat_val, xr.DataArray) and lat_val.ndim > 0:
        if "lat" in da.coords and "lat" not in lat_val.dims and lat_val.size == da.coords["lat"].size:
            lat_val = xr.DataArray(lat_val.values, dims=["lat"], coords={"lat": da.coords["lat"]}, attrs=lat_val.attrs)
        elif "latitude" in da.coords and "latitude" not in lat_val.dims and lat_val.size == da.coords["latitude"].size:
            lat_val = xr.DataArray(
                lat_val.values, dims=["latitude"], coords={"latitude": da.coords["latitude"]}, attrs=lat_val.attrs
            )

    if "rad" in units:
        return lat_val
    return np.deg2rad(lat_val)


def _resolve_elevation(
    ds: xr.Dataset | None,
    elevation: float | xr.DataArray | str | None,
) -> float | xr.DataArray:
    """
    Resolve elevation in metres, defaulting to 0.0 if not provided.

    Parameters
    ----------
    ds : xr.Dataset | None
        Optional Dataset to inspect for topography coordinates.
    elevation : float | xr.DataArray | str | None
        Explicit elevation or None.

    Returns
    -------
    float | xr.DataArray
        Elevation in metres above sea level.
    """
    if elevation is not None:
        if isinstance(elevation, str):
            return _resolve_dataarray(elevation, ds, "elevation")
        return elevation
    if ds is not None:
        for z_name in ("elevation", "orog", "altitude"):
            if z_name in ds:
                return ds[z_name]
        if "z" in ds:
            z_var = ds["z"]
            z_units = str(getattr(z_var, "attrs", {}).get("units", "")).lower()
            if "m2" in z_units or "m**2" in z_units or "s-2" in z_units or "s**-2" in z_units:
                return z_var / 9.80665
            return z_var
    return 0.0


@wrap_indicator()
def potential_evapotranspiration_fao56(
    tas: xr.DataArray | str = "tas",
    sfc_wind: xr.DataArray | str = "sfc_wind",
    rsds: xr.DataArray | str = "rsds",
    tasmax: xr.DataArray | str | None = None,
    tasmin: xr.DataArray | str | None = None,
    hurs: xr.DataArray | str | None = None,
    lat: float | xr.DataArray | str | None = None,
    elevation: float | xr.DataArray | str | None = None,
    ds: xr.Dataset | None = None,
) -> xr.DataArray:
    """
    Calculate reference crop evapotranspiration using FAO-56 Penman-Monteith.

    The FAO-56 Penman-Monteith method is the international standard reference
    for computing potential evapotranspiration from a hypothetical grass reference
    surface with specific canopy resistance and albedo.

    Parameters
    ----------
    tas : xr.DataArray | str, default="tas"
        Daily mean 2-metre air temperature.
    sfc_wind : xr.DataArray | str, default="sfc_wind"
        Wind speed at 2m height in m/s.
    rsds : xr.DataArray | str, default="rsds"
        Surface solar radiation downwards.
    tasmax : xr.DataArray | str | None, default=None
        Daily maximum 2-metre air temperature.
    tasmin : xr.DataArray | str | None, default=None
        Daily minimum 2-metre air temperature.
    hurs : xr.DataArray | str | None, default=None
        Relative humidity in percentage.
    lat : float | xr.DataArray | str | None, default=None
        Site latitude in degrees or radians. Inferred from coordinates if omitted.
    elevation : float | xr.DataArray | str | None, default=None
        Elevation above sea level in metres. Default is 0.0.
    ds : xr.Dataset | None, default=None
        Dataset containing variables when strings are passed.

    Returns
    -------
    xr.DataArray
        Reference crop evapotranspiration in mm d-1.
    """
    _check_pyet_installed()
    t_da = _ensure_celsius(_resolve_dataarray(tas, ds, "tas"))
    wind_da = _resolve_dataarray(sfc_wind, ds, "sfc_wind")
    rs_da = _ensure_radiation_mj(_resolve_dataarray(rsds, ds, "rsds"))
    lat_rad = _resolve_latitude_rad(t_da, ds, lat)
    elev_val = _resolve_elevation(ds, elevation)

    tmax_da: xr.DataArray | None
    if tasmax is not None:
        tmax_da = _ensure_celsius(_resolve_dataarray(tasmax, ds, "tasmax"))
    elif ds is not None and "tasmax" in ds:
        tmax_da = _ensure_celsius(ds["tasmax"])
    else:
        tmax_da = None

    tmin_da: xr.DataArray | None
    if tasmin is not None:
        tmin_da = _ensure_celsius(_resolve_dataarray(tasmin, ds, "tasmin"))
    elif ds is not None and "tasmin" in ds:
        tmin_da = _ensure_celsius(ds["tasmin"])
    else:
        tmin_da = None

    rh_da: xr.DataArray | None
    if hurs is not None:
        rh_da = _ensure_percent(_resolve_dataarray(hurs, ds, "hurs"))
    elif ds is not None and "hurs" in ds:
        rh_da = _ensure_percent(ds["hurs"])
    else:
        rh_da = None

    res = pyet.pm_fao56(
        t_da,
        wind_da,
        rs=rs_da,
        tmax=tmax_da,
        tmin=tmin_da,
        rh=rh_da,
        lat=lat_rad,
        elevation=elev_val,
    )
    res.attrs["units"] = "mm d-1"
    res.attrs["standard_name"] = "water_potential_evaporation_flux"
    res.attrs["long_name"] = "FAO-56 Penman-Monteith Reference Crop Evapotranspiration"
    return res


pet_fao56 = potential_evapotranspiration_fao56


@wrap_indicator()
def potential_evapotranspiration_priestley_taylor(
    tas: xr.DataArray | str = "tas",
    rsds: xr.DataArray | str = "rsds",
    tasmax: xr.DataArray | str | None = None,
    tasmin: xr.DataArray | str | None = None,
    hurs: xr.DataArray | str | None = None,
    lat: float | xr.DataArray | str | None = None,
    elevation: float | xr.DataArray | str | None = None,
    alpha: float = 1.26,
    ds: xr.Dataset | None = None,
) -> xr.DataArray:
    """
    Calculate potential evapotranspiration using the Priestley-Taylor method.

    The Priestley-Taylor method estimates potential evapotranspiration under
    wet surface conditions using available energy and an empirical advection factor.

    Parameters
    ----------
    tas : xr.DataArray | str, default="tas"
        Daily mean 2-metre air temperature.
    rsds : xr.DataArray | str, default="rsds"
        Surface solar radiation downwards.
    tasmax : xr.DataArray | str | None, default=None
        Daily maximum 2-metre air temperature.
    tasmin : xr.DataArray | str | None, default=None
        Daily minimum 2-metre air temperature.
    hurs : xr.DataArray | str | None, default=None
        Relative humidity in percentage.
    lat : float | xr.DataArray | str | None, default=None
        Site latitude in degrees or radians.
    elevation : float | xr.DataArray | str | None, default=None
        Elevation above sea level in metres. Default is 0.0.
    alpha : float, default=1.26
        Priestley-Taylor advection coefficient.
    ds : xr.Dataset | None, default=None
        Dataset containing variables when strings are passed.

    Returns
    -------
    xr.DataArray
        Priestley-Taylor potential evapotranspiration in mm d-1.
    """
    _check_pyet_installed()
    t_da = _ensure_celsius(_resolve_dataarray(tas, ds, "tas"))
    rs_da = _ensure_radiation_mj(_resolve_dataarray(rsds, ds, "rsds"))
    lat_rad = _resolve_latitude_rad(t_da, ds, lat)
    elev_val = _resolve_elevation(ds, elevation)

    tmax_da: xr.DataArray | None
    if tasmax is not None:
        tmax_da = _ensure_celsius(_resolve_dataarray(tasmax, ds, "tasmax"))
    elif ds is not None and "tasmax" in ds:
        tmax_da = _ensure_celsius(ds["tasmax"])
    else:
        tmax_da = None

    tmin_da: xr.DataArray | None
    if tasmin is not None:
        tmin_da = _ensure_celsius(_resolve_dataarray(tasmin, ds, "tasmin"))
    elif ds is not None and "tasmin" in ds:
        tmin_da = _ensure_celsius(ds["tasmin"])
    else:
        tmin_da = None

    rh_da: xr.DataArray | None
    if hurs is not None:
        rh_da = _ensure_percent(_resolve_dataarray(hurs, ds, "hurs"))
    elif ds is not None and "hurs" in ds:
        rh_da = _ensure_percent(ds["hurs"])
    else:
        rh_da = None

    res = pyet.priestley_taylor(
        t_da,
        rs=rs_da,
        tmax=tmax_da,
        tmin=tmin_da,
        rh=rh_da,
        lat=lat_rad,
        elevation=elev_val,
        alpha=alpha,
    )
    res.attrs["units"] = "mm d-1"
    res.attrs["standard_name"] = "water_potential_evaporation_flux"
    res.attrs["long_name"] = "Priestley-Taylor Potential Evapotranspiration"
    return res


pet_priestley_taylor = potential_evapotranspiration_priestley_taylor


@wrap_indicator()
def potential_evapotranspiration_hargreaves(
    tas: xr.DataArray | str = "tas",
    tasmax: xr.DataArray | str = "tasmax",
    tasmin: xr.DataArray | str = "tasmin",
    lat: float | xr.DataArray | str | None = None,
    ds: xr.Dataset | None = None,
) -> xr.DataArray:
    """
    Calculate potential evapotranspiration using the Hargreaves-Samani method.

    The Hargreaves-Samani method computes potential evapotranspiration based on
    daily mean, maximum, and minimum air temperatures and extraterrestrial radiation.

    Parameters
    ----------
    tas : xr.DataArray | str, default="tas"
        Daily mean 2-metre air temperature.
    tasmax : xr.DataArray | str, default="tasmax"
        Daily maximum 2-metre air temperature.
    tasmin : xr.DataArray | str, default="tasmin"
        Daily minimum 2-metre air temperature.
    lat : float | xr.DataArray | str | None, default=None
        Site latitude in degrees or radians. Inferred from coordinates if omitted.
    ds : xr.Dataset | None, default=None
        Dataset containing variables when strings are passed.

    Returns
    -------
    xr.DataArray
        Hargreaves potential evapotranspiration in mm d-1.
    """
    _check_pyet_installed()
    t_da = _ensure_celsius(_resolve_dataarray(tas, ds, "tas"))
    tmax_da = _ensure_celsius(_resolve_dataarray(tasmax, ds, "tasmax"))
    tmin_da = _ensure_celsius(_resolve_dataarray(tasmin, ds, "tasmin"))
    lat_rad = _resolve_latitude_rad(t_da, ds, lat)

    lambd = pyet.calc_lambda(t_da)
    index = pyet.utils.get_index(t_da)
    ra = pyet.extraterrestrial_r(index, lat_rad)
    if not isinstance(ra, xr.DataArray):
        ra = xr.DataArray(ra, dims=["time"], coords={"time": t_da.time})

    trange = np.maximum(tmax_da - tmin_da, 0.0)
    res = 0.0023 * (t_da + 17.8) * np.sqrt(trange) * (ra / lambd)
    res = res.where(res >= 0.0, 0.0)

    res.attrs["units"] = "mm d-1"
    res.attrs["standard_name"] = "water_potential_evaporation_flux"
    res.attrs["long_name"] = "Hargreaves-Samani Potential Evapotranspiration"
    return res


pet_hargreaves = potential_evapotranspiration_hargreaves


@wrap_indicator()
def potential_evapotranspiration_turc(
    tas: xr.DataArray | str = "tas",
    rsds: xr.DataArray | str = "rsds",
    hurs: xr.DataArray | str = "hurs",
    ds: xr.Dataset | None = None,
) -> xr.DataArray:
    """
    Calculate potential evapotranspiration using the Turc method.

    The Turc method estimates potential evapotranspiration using daily mean
    air temperature, incoming solar radiation, and relative humidity.

    Parameters
    ----------
    tas : xr.DataArray | str, default="tas"
        Daily mean 2-metre air temperature.
    rsds : xr.DataArray | str, default="rsds"
        Surface solar radiation downwards.
    hurs : xr.DataArray | str, default="hurs"
        Relative humidity in percentage.
    ds : xr.Dataset | None, default=None
        Dataset containing variables when strings are passed.

    Returns
    -------
    xr.DataArray
        Turc potential evapotranspiration in mm d-1.
    """
    _check_pyet_installed()
    t_da = _ensure_celsius(_resolve_dataarray(tas, ds, "tas"))
    rs_da = _ensure_radiation_mj(_resolve_dataarray(rsds, ds, "rsds"))
    rh_da = _ensure_percent(_resolve_dataarray(hurs, ds, "hurs"))

    res = pyet.turc(t_da, rs=rs_da, rh=rh_da)
    res.attrs["units"] = "mm d-1"
    res.attrs["standard_name"] = "water_potential_evaporation_flux"
    res.attrs["long_name"] = "Turc Potential Evapotranspiration"
    return res


pet_turc = potential_evapotranspiration_turc


@wrap_indicator()
def potential_evapotranspiration_makkink(
    tas: xr.DataArray | str = "tas",
    rsds: xr.DataArray | str = "rsds",
    elevation: float | xr.DataArray | str | None = None,
    ds: xr.Dataset | None = None,
) -> xr.DataArray:
    """
    Calculate potential evapotranspiration using the Makkink radiation method.

    The Makkink method is an empirical radiation-based formulation widely
    used in Western Europe to estimate grassland evapotranspiration.

    Parameters
    ----------
    tas : xr.DataArray | str, default="tas"
        Daily mean 2-metre air temperature.
    rsds : xr.DataArray | str, default="rsds"
        Surface solar radiation downwards.
    elevation : float | xr.DataArray | str | None, default=None
        Elevation above sea level in metres. Default is 0.0.
    ds : xr.Dataset | None, default=None
        Dataset containing variables when strings are passed.

    Returns
    -------
    xr.DataArray
        Makkink potential evapotranspiration in mm d-1.
    """
    _check_pyet_installed()
    t_da = _ensure_celsius(_resolve_dataarray(tas, ds, "tas"))
    rs_da = _ensure_radiation_mj(_resolve_dataarray(rsds, ds, "rsds"))
    elev_val = _resolve_elevation(ds, elevation)

    res = pyet.makkink(t_da, rs=rs_da, elevation=elev_val)
    res.attrs["units"] = "mm d-1"
    res.attrs["standard_name"] = "water_potential_evaporation_flux"
    res.attrs["long_name"] = "Makkink Potential Evapotranspiration"
    return res


pet_makkink = potential_evapotranspiration_makkink


@wrap_indicator()
def potential_evapotranspiration_penman(
    tas: xr.DataArray | str = "tas",
    sfc_wind: xr.DataArray | str = "sfc_wind",
    rsds: xr.DataArray | str = "rsds",
    tasmax: xr.DataArray | str | None = None,
    tasmin: xr.DataArray | str | None = None,
    hurs: xr.DataArray | str | None = None,
    lat: float | xr.DataArray | str | None = None,
    elevation: float | xr.DataArray | str | None = None,
    ds: xr.Dataset | None = None,
) -> xr.DataArray:
    """
    Calculate potential open-water evaporation using the original Penman equation.

    The Penman (1948) method combines an energy balance with an aerodynamic
    transport component to estimate evaporation from open water surfaces.

    Parameters
    ----------
    tas : xr.DataArray | str, default="tas"
        Daily mean 2-metre air temperature.
    sfc_wind : xr.DataArray | str, default="sfc_wind"
        Wind speed at 2m height in m/s.
    rsds : xr.DataArray | str, default="rsds"
        Surface solar radiation downwards.
    tasmax : xr.DataArray | str | None, default=None
        Daily maximum 2-metre air temperature.
    tasmin : xr.DataArray | str | None, default=None
        Daily minimum 2-metre air temperature.
    hurs : xr.DataArray | str | None, default=None
        Relative humidity in percentage.
    lat : float | xr.DataArray | str | None, default=None
        Site latitude in degrees or radians. Inferred from coordinates if omitted.
    elevation : float | xr.DataArray | str | None, default=None
        Elevation above sea level in metres. Default is 0.0.
    ds : xr.Dataset | None, default=None
        Dataset containing variables when strings are passed.

    Returns
    -------
    xr.DataArray
        Penman open-water potential evaporation in mm d-1.
    """
    _check_pyet_installed()
    t_da = _ensure_celsius(_resolve_dataarray(tas, ds, "tas"))
    wind_da = _resolve_dataarray(sfc_wind, ds, "sfc_wind")
    rs_da = _ensure_radiation_mj(_resolve_dataarray(rsds, ds, "rsds"))
    lat_rad = _resolve_latitude_rad(t_da, ds, lat)
    elev_val = _resolve_elevation(ds, elevation)

    tmax_da: xr.DataArray | None
    if tasmax is not None:
        tmax_da = _ensure_celsius(_resolve_dataarray(tasmax, ds, "tasmax"))
    elif ds is not None and "tasmax" in ds:
        tmax_da = _ensure_celsius(ds["tasmax"])
    else:
        tmax_da = None

    tmin_da: xr.DataArray | None
    if tasmin is not None:
        tmin_da = _ensure_celsius(_resolve_dataarray(tasmin, ds, "tasmin"))
    elif ds is not None and "tasmin" in ds:
        tmin_da = _ensure_celsius(ds["tasmin"])
    else:
        tmin_da = None

    rh_da: xr.DataArray | None
    if hurs is not None:
        rh_da = _ensure_percent(_resolve_dataarray(hurs, ds, "hurs"))
    elif ds is not None and "hurs" in ds:
        rh_da = _ensure_percent(ds["hurs"])
    else:
        rh_da = None

    res = pyet.penman(
        t_da,
        wind_da,
        rs=rs_da,
        tmax=tmax_da,
        tmin=tmin_da,
        rh=rh_da,
        lat=lat_rad,
        elevation=elev_val,
    )
    res.attrs["units"] = "mm d-1"
    res.attrs["standard_name"] = "water_potential_evaporation_flux"
    res.attrs["long_name"] = "Penman Open-Water Potential Evaporation"
    return res


pet_penman = potential_evapotranspiration_penman
