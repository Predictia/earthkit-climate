# (C) Copyright 2025 - ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

from pytest_mock import MockerFixture

from earthkit.climate.indicators import synoptic


class MockEarthkitData:
    """Mock object for Earthkit input."""

    pass


def test_jetstream_metric_woollings(mocker: MockerFixture, common_mocks):
    mock_metric = mocker.patch("xclim.indicators.atmos.jetstream_metric_woollings")

    ds_in = MockEarthkitData()
    synoptic.jetstream_metric_woollings(ds_in, freq="MS")

    # Verify conversions were called (handled by common_mocks)
    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})

    # Verify xclim indicator was called with the converted dataset (which is common_mocks['dummy_precip_ds'])
    # The first element of the return value of mock_to_xr is the dataset
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_metric.assert_called_once()
    call_args = mock_metric.call_args
    assert call_args.kwargs["ds"] is ds_converted
    assert call_args.kwargs["freq"] == "MS"

    # Verify result conversion
    common_mocks["mock_to_ek"].assert_called_once()
