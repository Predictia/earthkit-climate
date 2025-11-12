import numpy as np
import pytest
import xarray as xr
from pytest_mock import MockerFixture


@pytest.fixture
def dummy_precip_ds() -> xr.Dataset:
    """Simple constant precipitation dataset."""
    time = xr.cftime_range(
        start="2001-01-01", end="2001-01-10", freq="D", calendar="noleap"
    ).to_datetimeindex()
    ds = xr.Dataset(
        {"pr": ("time", [1.0] * len(time))},
        coords={"time": time},
    )
    ds["pr"].attrs["units"] = "kg m-2 s-1"
    return ds


@pytest.fixture
def daily_temperature_ds() -> xr.Dataset:
    """Synthetic daily temperature dataset for percentile and grouping tests."""
    rng = np.random.default_rng(0)
    time = xr.cftime_range(
        start="2000-01-01", end="2001-12-31", freq="D", calendar="noleap"
    ).to_datetimeindex()
    data = rng.normal(loc=10.0, scale=2.0, size=time.size)
    ds = xr.Dataset({"tas": ("time", data)}, coords={"time": time})
    return ds


@pytest.fixture
def common_mocks(mocker: MockerFixture, dummy_precip_ds: xr.Dataset) -> dict:
    """
    Fixture that sets up common mocks used across precipitation indicator tests.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest-mock fixture used to create and manage mocks.
    dummy_precip_ds : xr.Dataset
        The dummy precipitation dataset fixture.

    Returns
    -------
    dict[str, Any]
        Dictionary with references to key mock objects for assertions.
    """
    sentinel_ek = object()

    mock_to_xr = mocker.patch(
        "earthkit.climate.utils.conversions.to_xarray_dataset",
        return_value=(dummy_precip_ds, {"earthkit_internal": {}}),
    )

    mock_ensure_units = mocker.patch(
        "earthkit.climate.utils.units.ensure_units",
        side_effect=lambda ds, var, units, strict=False: ds.assign_attrs({"ensured": True}),
    )

    mock_add_prov = mocker.patch(
        "earthkit.climate.utils.provenance.add_indicator_provenance",
        side_effect=lambda md, *a, **k: {**md, "prov": True},
    )

    mock_to_ek = mocker.patch(
        "earthkit.climate.utils.conversions.to_earthkit_field",
        return_value=sentinel_ek,
    )

    return {
        "mock_to_xr": mock_to_xr,
        "mock_ensure_units": mock_ensure_units,
        "mock_add_prov": mock_add_prov,
        "mock_to_ek": mock_to_ek,
        "sentinel_ek": sentinel_ek,
    }
