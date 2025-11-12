import xarray as xr
import numpy as np
from xarray import DataArray
from xsdba.nbutils import quantile


def get_percentile(
    baseline_dataset: xr.Dataset,
    varname: str,
    percentile: float,
    freq: str = "YS",
) -> xr.Dataset:
    """
    Compute a regular percentile (e.g. 90th) of a variable over time,
    grouped by a temporal component (month, season, or year),
    and expand the result to have one value per day of the input period.
    The final dataset uses 'dayofyear' as the main temporal dimension.

    Parameters
    ----------
    baseline_dataset : xr.Dataset
        Dataset containing the variable to analyze.
    varname : str
        Name of the variable within the dataset.
    percentile : float
        Percentile value (e.g., 90 for 90th percentile).
    freq : str, optional
        Frequency for grouping (e.g. 'YS', 'MS', 'QS'). Default is yearly.

    Returns
    -------
    xr.Dataset
        Dataset with the computed percentile values expanded to daily resolution,
        using 'dayofyear' (1–365) as the main coordinate.
    """
    da = baseline_dataset[varname]
    q = percentile / 100.0

    def custom_percentile(group: DataArray) -> DataArray:
        return quantile(group, q=np.array([q]), dim="time")

    time_component = pandas_offset2time_component(freq)

    # Compute percentile by period
    if time_component == "year":
        ds_percentile = custom_percentile(da).to_dataset(name=varname)
    else:
        ds_percentile = (
            da.groupby(f"time.{time_component}")
            .map(custom_percentile)
            .to_dataset(name=varname)
        )

    ds_percentile = ds_percentile.squeeze().drop_vars("quantiles", errors="ignore")

    # --- Expand to daily resolution: assign same percentile to all days of the same period
    if time_component == "year":
        expanded = xr.full_like(da, ds_percentile[varname].item())
    else:
        expanded = (
            da.groupby(f"time.{time_component}")
            .map(
                lambda group: xr.full_like(
                    group,
                    ds_percentile[varname].sel(
                        {time_component: getattr(group.time.dt, time_component)[
                            0].item()}
                    ),
                )
            )
        )

    # Add dayofyear coordinate and drop time
    expanded_ds = expanded.to_dataset(name=varname)
    expanded_ds = expanded_ds.assign_coords(dayofyear=expanded_ds["time"].dt.dayofyear)
    expanded_ds = expanded_ds.swap_dims({"time": "dayofyear"}).drop_vars("time")

    return expanded_ds

def pandas_offset2time_component(aggregation: str) -> str:
    """
    Map a pandas-style frequency string to a corresponding time component.

    Parameters
    ----------
    aggregation : str
        Frequency alias following pandas conventions (e.g., 'YS', 'QS-DEC', 'MS').

    Returns
    -------
    str
        Time component corresponding to the given frequency:
        - 'YS' → 'year'
        - 'QS-DEC' → 'season'
        - 'MS' → 'month'

    Raises
    ------
    NotImplementedError
        If the provided frequency alias is not supported.
    """
    if aggregation == "YS":
        resolution = "year"
    elif aggregation == "QS-DEC":
        resolution = "season"
    elif aggregation == "MS":
        resolution = "month"
    else:
        raise NotImplementedError(f"Unsupported aggregation: {aggregation}")
    return resolution
