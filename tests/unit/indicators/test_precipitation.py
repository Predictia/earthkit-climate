from pytest_mock import MockerFixture

from earthkit.climate.indicators import precipitation


class MockEarthkitData:
    """Mock object for Earthkit input."""

    pass


def test_maximum_consecutive_wet_days(mocker: MockerFixture, common_mocks):
    """Test maximum_consecutive_wet_days calls wrapper correctly."""
    mock_wrapper_factory = mocker.patch("earthkit.climate.indicators.precipitation.wrap_xclim_indicator")
    mock_wrapped_fn = mocker.MagicMock()
    mock_wrapper_factory.return_value = mock_wrapped_fn

    pr_in = MockEarthkitData()
    precipitation.maximum_consecutive_wet_days(pr_in, thresh="2 mm/day", freq="MS")

    import xclim.indicators.atmos

    mock_wrapper_factory.assert_called_once_with(xclim.indicators.atmos.maximum_consecutive_wet_days)

    mock_wrapped_fn.assert_called_once()
    call_args = mock_wrapped_fn.call_args
    ds_arg = call_args[0][0]
    assert ds_arg is pr_in
    assert call_args.kwargs["thresh"] == "2 mm/day"
    assert call_args.kwargs["freq"] == "MS"


def test_daily_precipitation_intensity(mocker: MockerFixture, common_mocks):
    """Test daily_precipitation_intensity calls wrapper correctly."""
    mock_wrapper_factory = mocker.patch("earthkit.climate.indicators.precipitation.wrap_xclim_indicator")
    mock_wrapped_fn = mocker.MagicMock()
    mock_wrapper_factory.return_value = mock_wrapped_fn

    pr_in = MockEarthkitData()
    precipitation.daily_precipitation_intensity(pr_in, thresh="2 mm/day", freq="MS")

    import xclim.indicators.atmos

    mock_wrapper_factory.assert_called_once_with(xclim.indicators.atmos.daily_pr_intensity)

    mock_wrapped_fn.assert_called_once()
    call_args = mock_wrapped_fn.call_args
    ds_arg = call_args[0][0]
    assert ds_arg is pr_in
    assert call_args.kwargs["thresh"] == "2 mm/day"
    assert call_args.kwargs["freq"] == "MS"
