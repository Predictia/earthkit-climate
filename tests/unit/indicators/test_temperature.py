import pytest
import xarray as xr
from pytest_mock import MockerFixture

from earthkit.climate.indicators import temperature


class MockEarthkitData:
    """Mock object for Earthkit input."""

    pass


def test_daily_temperature_range(mocker: MockerFixture, common_mocks):
    """Test daily_temperature_range calls wrapper with merged dataset."""
    # Setup mocks
    mock_to_xr = common_mocks["mock_to_xr"]
    # We need to return different datasets for tasmax and tasmin to verify merging
    ds_tasmax = xr.Dataset({"tasmax": (("time"), [20])}, coords={"time": [0]})
    ds_tasmin = xr.Dataset({"tasmin": (("time"), [10])}, coords={"time": [0]})

    # side_effect to return different datasets based on input
    # We can't easily check input equality with MockEarthkitData instances in side_effect
    # without keeping references.
    # Instead, we can just return a sequence of results.
    mock_to_xr.side_effect = [(ds_tasmax, {}), (ds_tasmin, {})]

    # Mock the wrapper creator and the wrapped function
    mock_wrapper_factory = mocker.patch("earthkit.climate.indicators.temperature.wrap_xclim_indicator")
    mock_wrapped_fn = mocker.MagicMock()
    mock_wrapper_factory.return_value = mock_wrapped_fn

    # Call function
    tasmax_in = MockEarthkitData()
    tasmin_in = MockEarthkitData()
    temperature.daily_temperature_range(tasmax_in, tasmin_in, arg="val")

    # Verify to_xarray_dataset called twice
    assert mock_to_xr.call_count == 2

    # Verify wrapper created with correct xclim function
    import xclim.indicators.atmos

    mock_wrapper_factory.assert_called_once_with(xclim.indicators.atmos.daily_temperature_range)

    # Verify wrapped function called with merged dataset
    # We check if the first arg to wrapped_fn is a dataset containing both vars
    call_args = mock_wrapped_fn.call_args
    assert call_args is not None
    ds_arg = call_args[0][0]
    assert isinstance(ds_arg, xr.Dataset)
    assert "tasmax" in ds_arg
    assert "tasmin" in ds_arg
    assert call_args.kwargs["arg"] == "val"


def test_heating_degree_days(mocker: MockerFixture, common_mocks):
    """Test heating_degree_days calls wrapper with merged dataset."""
    mock_to_xr = common_mocks["mock_to_xr"]

    ds_tasmax = xr.Dataset({"tasmax": (("time"), [20])}, coords={"time": [0]})
    ds_tasmin = xr.Dataset({"tasmin": (("time"), [10])}, coords={"time": [0]})
    ds_tas = xr.Dataset({"tas": (("time"), [15])}, coords={"time": [0]})

    mock_to_xr.side_effect = [(ds_tasmax, {}), (ds_tasmin, {}), (ds_tas, {})]

    mock_wrapper_factory = mocker.patch("earthkit.climate.indicators.temperature.wrap_xclim_indicator")
    mock_wrapped_fn = mocker.MagicMock()
    mock_wrapper_factory.return_value = mock_wrapped_fn

    tasmax_in = MockEarthkitData()
    tasmin_in = MockEarthkitData()
    tas_in = MockEarthkitData()

    temperature.heating_degree_days(tasmax_in, tasmin_in, tas_in, thresh="18 degC")

    assert mock_to_xr.call_count == 3

    import xclim.indicators.atmos

    mock_wrapper_factory.assert_called_once_with(xclim.indicators.atmos.heating_degree_days)

    call_args = mock_wrapped_fn.call_args
    ds_arg = call_args[0][0]
    assert "tasmax" in ds_arg
    assert "tasmin" in ds_arg
    assert "tas" in ds_arg
    assert call_args.kwargs["thresh"] == "18 degC"


def test_warm_spell_duration_index(mocker: MockerFixture, common_mocks):
    """Test warm_spell_duration_index maps history to reference_data."""
    # Mock wrapper factory
    mock_wrapper_factory = mocker.patch("earthkit.climate.indicators.temperature.wrap_xclim_indicator")
    mock_wrapped_fn = mocker.MagicMock()
    mock_wrapper_factory.return_value = mock_wrapped_fn

    tasmax_in = MockEarthkitData()
    tasmax_hist_in = MockEarthkitData()

    temperature.warm_spell_duration_index(tasmax_in, tasmax_hist_in, window=10)

    import xclim.indicators.atmos

    mock_wrapper_factory.assert_called_once_with(xclim.indicators.atmos.warm_spell_duration_index)

    # Verify call args
    mock_wrapped_fn.assert_called_once()
    call_kwargs = mock_wrapped_fn.call_args.kwargs

    assert call_kwargs["earthkit_input"] is tasmax_in
    assert call_kwargs["reference_data"] is tasmax_hist_in
    assert call_kwargs["window"] == 10
