from functools import wraps
import xarray as xr
from typing import Callable, Any

from earthkit.climate.utils import conversions, provenance, units


def wrap_xclim_indicator(xclim_fn: Callable) -> Callable:
    """
    Wrap an xclim indicator function so it can be called using Earthkit data structures.

    Parameters
    ----------
    xclim_fn : Callable
        The xclim indicator function to wrap.

    Returns
    -------
    Callable
        A wrapped function that handles Earthkit → xarray conversion, unit
        normalization, indicator execution, and xarray → Earthkit conversion.
    """

    @wraps(xclim_fn)
    def wrapper(earthkit_input: conversions.EarthkitData | xr.Dataset, *args, **kwargs) -> conversions.EarthkitData:
        """
        Execute the wrapped xclim indicator using Earthkit inputs.

        Parameters
        ----------
        earthkit_input : conversions.EarthkitData or xarray.Dataset
            Input data to be converted into an xarray Dataset before applying
            the indicator.
        *args : tuple
            Positional arguments forwarded to the xclim indicator.
        **kwargs : dict
            Keyword arguments forwarded to the xclim indicator.

        Returns
        -------
        conversions.EarthkitData
            The indicator output converted back into an Earthkit-compatible object.
        """
        metadata: dict = {}

        # 1) Convert Earthkit → xarray Dataset
        dataset, metadata = conversions.to_xarray_dataset(earthkit_input, metadata)

        # 2) Ensure correct units
        if "tas" in dataset:
            dataset = units.ensure_units(dataset, "tas", "K", strict=False)
        if "pr" in dataset:
            dataset = units.ensure_units(dataset, "pr", "mm/day", strict=False)

        # 3) Call the xclim indicator
        output_dataset: xr.Dataset = xclim_fn(ds=dataset, *args, **kwargs)

        # 4) Add provenance
        metadata = provenance.add_indicator_provenance(
            metadata,
            xclim_fn,
            dataset,
            **kwargs,
        )

        # 5) Convert back xarray → Earthkit output
        return conversions.to_earthkit_field(output_dataset, metadata)

    return wrapper
