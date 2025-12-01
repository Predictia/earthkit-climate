from pytest_mock import MockerFixture

from earthkit.climate.indicators import temperature


class MockEarthkitData:
    """Mock object for Earthkit input."""

    pass


def test_daily_temperature_range(mocker: MockerFixture, common_mocks):
    """Test daily_temperature_range calls wrapper with merged dataset."""
    # Mock the wrapper creator and the wrapped function
    mock_wrapper_factory = mocker.patch("earthkit.climate.indicators.temperature.wrap_xclim_indicator")
    mock_wrapped_fn = mocker.MagicMock()
    mock_wrapper_factory.return_value = mock_wrapped_fn

    # Call function with single dataset
    ds_in = MockEarthkitData()
    temperature.daily_temperature_range(ds_in, arg="val")

    # Verify wrapper created with correct xclim function
    import xclim.indicators.atmos

    mock_wrapper_factory.assert_called_once_with(xclim.indicators.atmos.daily_temperature_range)

    # Verify wrapped function called with the dataset
    call_args = mock_wrapped_fn.call_args
    assert call_args is not None
    ds_arg = call_args[0][0]
    # The wrapper receives the raw input, conversion happens inside the wrapper (which is mocked)
    assert ds_arg is ds_in
    assert call_args.kwargs["arg"] == "val"


def test_heating_degree_days(mocker: MockerFixture, common_mocks):
    """Test heating_degree_days calls wrapper with merged dataset."""
    mock_wrapper_factory = mocker.patch("earthkit.climate.indicators.temperature.wrap_xclim_indicator")
    mock_wrapped_fn = mocker.MagicMock()
    mock_wrapper_factory.return_value = mock_wrapped_fn

    ds_in = MockEarthkitData()

    temperature.heating_degree_days(ds_in, thresh="18 degC")

    import xclim.indicators.atmos

    mock_wrapper_factory.assert_called_once_with(xclim.indicators.atmos.heating_degree_days)

    call_args = mock_wrapped_fn.call_args
    ds_arg = call_args[0][0]
    assert ds_arg is ds_in
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
