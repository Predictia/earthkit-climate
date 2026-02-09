# (C) Copyright 2025 - ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

from pytest_mock import MockerFixture

from earthkit.climate.indicators import wind


class MockEarthkitData:
    """Mock object for Earthkit input."""

    pass


def test_calm_days(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.calm_days")

    ds_in = MockEarthkitData()
    wind.calm_days(ds_in, thresh="2 m s-1")

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted
    assert mock_fn.call_args.kwargs["thresh"] == "2 m s-1"


def test_sfcWind_max(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.sfcWind_max")

    ds_in = MockEarthkitData()
    wind.sfcWind_max(ds_in, freq="MS")

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted
    assert mock_fn.call_args.kwargs["freq"] == "MS"


def test_sfcWind_mean(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.sfcWind_mean")

    ds_in = MockEarthkitData()
    wind.sfcWind_mean(ds_in, freq="YS")

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted
    assert mock_fn.call_args.kwargs["freq"] == "YS"


def test_sfcWind_min(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.sfcWind_min")

    ds_in = MockEarthkitData()
    wind.sfcWind_min(ds_in, freq="MS")

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted
    assert mock_fn.call_args.kwargs["freq"] == "MS"


def test_sfcWindmax_max(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.sfcWindmax_max")

    ds_in = MockEarthkitData()
    wind.sfcWindmax_max(ds_in, freq="MS")

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted
    assert mock_fn.call_args.kwargs["freq"] == "MS"


def test_sfcWindmax_mean(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.sfcWindmax_mean")

    ds_in = MockEarthkitData()
    wind.sfcWindmax_mean(ds_in, freq="MS")

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted
    assert mock_fn.call_args.kwargs["freq"] == "MS"


def test_sfcWindmax_min(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.sfcWindmax_min")

    ds_in = MockEarthkitData()
    wind.sfcWindmax_min(ds_in, freq="MS")

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted
    assert mock_fn.call_args.kwargs["freq"] == "MS"


def test_windy_days(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.windy_days")

    ds_in = MockEarthkitData()
    wind.windy_days(ds_in, thresh="10 m s-1")

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted
    assert mock_fn.call_args.kwargs["thresh"] == "10 m s-1"
