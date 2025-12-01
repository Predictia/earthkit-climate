import inspect
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, Union

import xarray as xr
from xclim.core.calendar import percentile_doy

from earthkit.climate.utils import conversions, provenance, units


def wrap_xclim_indicator(xclim_fn: Callable) -> Callable:
    """
    Wraps an xclim indicator to handle Earthkit inputs, unit alignment,
    and on-the-fly percentile calculation using a reference period.

    Parameters
    ----------
    xclim_fn : Callable
        The xclim indicator function to be wrapped.

    Returns
    -------
    Callable
        The wrapped function which accepts Earthkit inputs.
    """

    # @wraps preserves the original docstring and name of the xclim function
    # allowing help() and IDEs to see the original documentation.
    @wraps(xclim_fn)
    def wrapper(
        earthkit_input: Union[conversions.EarthkitData, xr.Dataset],
        reference_data: Optional[Union[conversions.EarthkitData, xr.Dataset]] = None,
        reference_period: Optional[Tuple[str, str]] = None,
        percentile_val: Optional[int] = None,
        window: Optional[int] = None,
        *args,
        **kwargs,
    ) -> conversions.EarthkitData:
        """
        Wrapper function that processes Earthkit inputs and calls the xclim indicator.

        Parameters
        ----------
        earthkit_input : Union[conversions.EarthkitData, xr.Dataset]
            The input data, either as an Earthkit object or an xarray Dataset.
        reference_data : Optional[Union[conversions.EarthkitData, xr.Dataset]], optional
            Reference data for percentile calculations, by default None.
        reference_period : Optional[Tuple[str, str]], optional
            The time period to use as a reference (start, end), by default None.
        percentile_val : Optional[int], optional
            The percentile value to calculate (e.g., 90 for 90th percentile), by default None.
        window : Optional[int], optional
            The window size for rolling calculations, by default None.
        *args
            Variable length argument list passed to the xclim indicator.
        **kwargs
            Arbitrary keyword arguments passed to the xclim indicator.

        Returns
        -------
        conversions.EarthkitData
            The result of the indicator calculation wrapped as an Earthkit object.
        """
        metadata: Dict[str, Any] = {}

        # --- STEP 1: Load & Standardize Main Data ---
        # Convert Earthkit object to xarray Dataset
        dataset, metadata = conversions.to_xarray_dataset(earthkit_input, metadata)

        # Standardize units for common variables to Kelvin
        for var in ["tas", "tasmin", "tasmax"]:
            if var in dataset:
                dataset = units.ensure_units(dataset, var, "K", strict=False)
        if "pr" in dataset:
            dataset = units.ensure_units(dataset, "pr", "mm/day", strict=False)

        # --- STEP 2: Logic for Percentile-Based Indicators ---
        # We detect if the user is trying to calculate a percentile index
        # Only trigger calculation if a threshold is required AND reference data is provided

        if reference_data is not None or reference_period is not None:
            ref_ds = None

            # Scenario A: Reference is a separate dataset (e.g., distinct historical file)
            if reference_data is not None:
                ref_ds, _ = conversions.to_xarray_dataset(reference_data, {})
                # Standardize units for reference data too
                for var in ["tas", "tasmin", "tasmax"]:
                    if var in ref_ds:
                        ref_ds = units.ensure_units(ref_ds, var, "K", strict=False)
                if "pr" in ref_ds:
                    ref_ds = units.ensure_units(ref_ds, "pr", "mm/day", strict=False)

            # Scenario B: Reference is a slice of the main dataset (e.g., 1981-2010)
            elif reference_period is not None:
                ref_ds = dataset.sel(time=slice(*reference_period))

            if ref_ds is not None:
                # Inspect the indicator signature to find percentile arguments
                sig = inspect.signature(xclim_fn)

                for param_name in sig.parameters:
                    if param_name.endswith("_per") and param_name not in kwargs:
                        # Found a missing percentile argument, e.g. 'tasmin_per'
                        # Find corresponding data argument, e.g. 'tasmin'
                        target_arg_name = param_name.replace("_per", "")

                        # Determine the variable name in the dataset
                        # If the user mapped it (e.g. tasmin='my_min'), it's in kwargs.
                        # Otherwise, assume the variable name matches the argument name.
                        target_var_name = kwargs.get(target_arg_name, target_arg_name)

                        # Handle DataArray passed explicitly (get its name)
                        if isinstance(target_var_name, xr.DataArray) and target_var_name.name:
                            target_var_name = target_var_name.name

                        if isinstance(target_var_name, str) and target_var_name in ref_ds:
                            # Compute the Percentile DOY (Climatology)
                            per_doy = percentile_doy(
                                ref_ds[target_var_name], window=window, per=percentile_val
                            )

                            # Rename to match the expected argument name or a unique name
                            threshold_name = f"{target_var_name}_per"
                            per_doy = per_doy.rename(threshold_name)

                            # Merge into main dataset
                            dataset = xr.merge([dataset, per_doy])

                            # Update kwargs to tell xclim to use this variable
                            kwargs[param_name] = threshold_name

        # --- STEP 3: Execution ---
        # We pass the single merged dataset (ds) and the variable name mappings
        output_dataset: xr.Dataset = xclim_fn(ds=dataset, *args, **kwargs)

        # --- STEP 4: Provenance & Output ---
        metadata = provenance.add_indicator_provenance(metadata, xclim_fn, dataset, **kwargs)

        return conversions.to_earthkit_field(output_dataset, metadata)

    return wrapper
