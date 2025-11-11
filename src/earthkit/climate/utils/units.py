import warnings

import xarray
from xclim.core.units import convert_units_to


def ensure_units(
        ds: xarray.Dataset, var: str, expected_units: str, strict: bool = False
) -> xarray.Dataset:
    """
    Ensure that a variable in the dataset has the expected units for xclim indicators.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the variable.
    var : str
        Variable name (e.g. "tas", "pr").
    expected_units : str
        Units expected by xclim (e.g. "degC", "mm/day").
    strict : bool, default False
        If True, attempt to physically convert units using pint.
        If False, only overwrite the unit attribute and issue a warning.

    Returns
    -------
    xarray.Dataset
        The dataset with corrected units.
    """
    current_units = ds[var].attrs.get("units")

    if current_units != expected_units:
        if strict:
            try:
                ds[var] = convert_units_to(ds[var], expected_units)
                ds[var].attrs["units"] = expected_units
                warnings.warn(
                    f"Variable '{var}' converted from {current_units} to {expected_units}.",
                    UserWarning,
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to convert {var} from {current_units} to {expected_units}: {e}"
                )
        else:
            warnings.warn(
                f"Variable '{var}' has units '{current_units}', expected '{expected_units}'. "
                f"Overwriting without conversion.",
                UserWarning,
            )
            ds[var].attrs["units"] = expected_units

    return ds
