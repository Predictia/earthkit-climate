# (C) Copyright 2025 - ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

from pytest_mock import MockerFixture

from earthkit.climate.indicators import temperature


class MockEarthkitData:
    """Mock object for Earthkit input."""

    pass


def test_daily_temperature_range(mocker: MockerFixture, common_mocks):
    """Test daily_temperature_range calls wrapper with merged dataset."""
    # Mock the underlying xclim function
    mock_fn = mocker.patch("xclim.indicators.atmos.daily_temperature_range")

    # Call function with single dataset
    ds_in = MockEarthkitData()
    temperature.daily_temperature_range(ds_in, arg="val")

    # Verify conversions were called (handled by common_mocks)
    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    # Verify wrapped function called with the dataset
    call_args = mock_fn.call_args
    assert call_args is not None
    # The first argument to the xclim function should be the converted dataset
    assert call_args.kwargs["ds"] is ds_converted
    assert call_args.kwargs["arg"] == "val"


def test_heating_degree_days(mocker: MockerFixture, common_mocks):
    """Test heating_degree_days calls wrapper with merged dataset."""
    mock_fn = mocker.patch("xclim.indicators.atmos.heating_degree_days")

    ds_in = MockEarthkitData()

    temperature.heating_degree_days(ds_in, thresh="18 degC")

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted
    assert mock_fn.call_args.kwargs["thresh"] == "18 degC"


def test_warm_spell_duration_index(mocker: MockerFixture, common_mocks):
    """Test warm_spell_duration_index passes merged dataset (tasmax + tasmax_per)."""
    # Mock wrapper factory
    mock_fn = mocker.patch("xclim.indicators.atmos.warm_spell_duration_index")

    # Create a dummy input that represents a merged dataset
    ds_merged_in = MockEarthkitData()

    # Call with single merged input
    temperature.warm_spell_duration_index(ds_merged_in, window=10)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_merged_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    # Verify call args
    mock_fn.assert_called_once()
    call_kwargs = mock_fn.call_args.kwargs

    assert call_kwargs["ds"] is ds_converted
    assert call_kwargs["window"] == 10
    # Ensure reference_data is NOT passed
    assert "reference_data" not in call_kwargs
