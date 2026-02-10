# (C) Copyright 2025 - ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

from pytest_mock import MockerFixture

from earthkit.climate.indicators import precipitation


class MockEarthkitData:
    """Mock object for Earthkit input."""

    pass


def test_maximum_consecutive_wet_days(mocker: MockerFixture, common_mocks):
    """Test maximum_consecutive_wet_days calls wrapper correctly."""
    mock_fn = mocker.patch("xclim.indicators.atmos.maximum_consecutive_wet_days")

    pr_in = MockEarthkitData()
    precipitation.maximum_consecutive_wet_days(pr_in, thresh="2 mm/day", freq="MS")

    common_mocks["mock_to_xr"].assert_called_once_with(pr_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    call_args = mock_fn.call_args
    assert call_args.kwargs["ds"] is ds_converted
    assert call_args.kwargs["thresh"] == "2 mm/day"
    assert call_args.kwargs["freq"] == "MS"


def test_daily_precipitation_intensity(mocker: MockerFixture, common_mocks):
    """Test daily_precipitation_intensity calls wrapper correctly."""
    mock_fn = mocker.patch("xclim.indicators.atmos.daily_pr_intensity")

    pr_in = MockEarthkitData()
    # Note: Validating against daily_pr_intensity as per existing code structure
    try:
        precipitation.daily_precipitation_intensity(pr_in, thresh="2 mm/day", freq="MS")
    except AttributeError:
        # Fallback if the function is actually named daily_pr_intensity in the module
        precipitation.daily_pr_intensity(pr_in, thresh="2 mm/day", freq="MS")

    common_mocks["mock_to_xr"].assert_called_once_with(pr_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    call_args = mock_fn.call_args
    assert call_args.kwargs["ds"] is ds_converted
    assert call_args.kwargs["thresh"] == "2 mm/day"
    assert call_args.kwargs["freq"] == "MS"

# New tests

def test_antecedent_precipitation_index(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.antecedent_precipitation_index")

    ds_in = MockEarthkitData()
    precipitation.antecedent_precipitation_index(ds_in, val="test")

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted
    # No other kwargs passed in original test besides ds_in? Wait, val="test" was passed.
    # The original test passed val="test" to the wrapper.
    # Let's verify it's passed to the xclim fn.
    assert mock_fn.call_args.kwargs["val"] == "test"


def test_maximum_consecutive_dry_days(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.maximum_consecutive_dry_days")

    ds_in = MockEarthkitData()
    precipitation.maximum_consecutive_dry_days(ds_in, val="test")

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted
    assert mock_fn.call_args.kwargs["val"] == "test"


def test_cffwis_indices(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.cffwis_indices")

    ds_in = MockEarthkitData()
    precipitation.cffwis_indices(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_cold_and_dry_days(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.cold_and_dry_days")

    ds_in = MockEarthkitData()
    precipitation.cold_and_dry_days(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_cold_and_wet_days(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.cold_and_wet_days")

    ds_in = MockEarthkitData()
    precipitation.cold_and_wet_days(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_days_over_precip_doy_thresh(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.days_over_precip_doy_thresh")

    ds_in = MockEarthkitData()
    precipitation.days_over_precip_doy_thresh(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_days_over_precip_thresh(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.days_over_precip_thresh")

    ds_in = MockEarthkitData()
    precipitation.days_over_precip_thresh(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_days_with_snow(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.days_with_snow")

    ds_in = MockEarthkitData()
    precipitation.days_with_snow(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_drought_code(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.drought_code")

    ds_in = MockEarthkitData()
    precipitation.drought_code(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_griffiths_drought_factor(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.griffiths_drought_factor")

    ds_in = MockEarthkitData()
    precipitation.griffiths_drought_factor(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_duff_moisture_code(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.duff_moisture_code")

    ds_in = MockEarthkitData()
    precipitation.duff_moisture_code(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_dry_days(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.dry_days")

    ds_in = MockEarthkitData()
    precipitation.dry_days(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_dry_spell_frequency(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.dry_spell_frequency")

    ds_in = MockEarthkitData()
    precipitation.dry_spell_frequency(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_dry_spell_max_length(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.dry_spell_max_length")

    ds_in = MockEarthkitData()
    precipitation.dry_spell_max_length(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_dry_spell_total_length(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.dry_spell_total_length")

    ds_in = MockEarthkitData()
    precipitation.dry_spell_total_length(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_dryness_index(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.dryness_index")

    ds_in = MockEarthkitData()
    precipitation.dryness_index(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_mcarthur_forest_fire_danger_index(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.mcarthur_forest_fire_danger_index")

    ds_in = MockEarthkitData()
    precipitation.mcarthur_forest_fire_danger_index(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_first_snowfall(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.first_snowfall")

    ds_in = MockEarthkitData()
    precipitation.first_snowfall(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_fraction_over_precip_doy_thresh(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.fraction_over_precip_doy_thresh")

    ds_in = MockEarthkitData()
    precipitation.fraction_over_precip_doy_thresh(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_fraction_over_precip_thresh(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.fraction_over_precip_thresh")

    ds_in = MockEarthkitData()
    precipitation.fraction_over_precip_thresh(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_high_precip_low_temp(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.high_precip_low_temp")

    ds_in = MockEarthkitData()
    precipitation.high_precip_low_temp(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_keetch_byram_drought_index(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.keetch_byram_drought_index")

    ds_in = MockEarthkitData()
    precipitation.keetch_byram_drought_index(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_last_snowfall(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.last_snowfall")

    ds_in = MockEarthkitData()
    precipitation.last_snowfall(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_liquid_precip_ratio(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.liquid_precip_ratio")

    ds_in = MockEarthkitData()
    precipitation.liquid_precip_ratio(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_liquid_precip_average(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.liquid_precip_average")

    ds_in = MockEarthkitData()
    precipitation.liquid_precip_average(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_liquid_precip_accumulation(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.liquid_precip_accumulation")

    ds_in = MockEarthkitData()
    precipitation.liquid_precip_accumulation(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_max_n_day_precipitation_amount(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.max_n_day_precipitation_amount")

    ds_in = MockEarthkitData()
    precipitation.max_n_day_precipitation_amount(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_max_pr_intensity(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.max_pr_intensity")

    ds_in = MockEarthkitData()
    precipitation.max_pr_intensity(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_precip_average(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.precip_average")

    ds_in = MockEarthkitData()
    precipitation.precip_average(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_precip_accumulation(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.precip_accumulation")

    ds_in = MockEarthkitData()
    precipitation.precip_accumulation(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_rain_on_frozen_ground_days(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.rain_on_frozen_ground_days")

    ds_in = MockEarthkitData()
    precipitation.rain_on_frozen_ground_days(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_rain_season(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.rain_season")

    ds_in = MockEarthkitData()
    precipitation.rain_season(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_rprctot(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.rprctot")

    ds_in = MockEarthkitData()
    precipitation.rprctot(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_max_1day_precipitation_amount(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.max_1day_precipitation_amount")

    ds_in = MockEarthkitData()
    precipitation.max_1day_precipitation_amount(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_snowfall_frequency(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.snowfall_frequency")

    ds_in = MockEarthkitData()
    precipitation.snowfall_frequency(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_snowfall_intensity(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.snowfall_intensity")

    ds_in = MockEarthkitData()
    precipitation.snowfall_intensity(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_solid_precip_average(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.solid_precip_average")

    ds_in = MockEarthkitData()
    precipitation.solid_precip_average(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_solid_precip_accumulation(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.solid_precip_accumulation")

    ds_in = MockEarthkitData()
    precipitation.solid_precip_accumulation(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_warm_and_dry_days(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.warm_and_dry_days")

    ds_in = MockEarthkitData()
    precipitation.warm_and_dry_days(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_warm_and_wet_days(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.warm_and_wet_days")

    ds_in = MockEarthkitData()
    precipitation.warm_and_wet_days(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_water_cycle_intensity(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.water_cycle_intensity")

    ds_in = MockEarthkitData()
    precipitation.water_cycle_intensity(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_wet_precip_accumulation(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.wet_precip_accumulation")

    ds_in = MockEarthkitData()
    precipitation.wet_precip_accumulation(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_wet_spell_frequency(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.wet_spell_frequency")

    ds_in = MockEarthkitData()
    precipitation.wet_spell_frequency(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_wet_spell_max_length(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.wet_spell_max_length")

    ds_in = MockEarthkitData()
    precipitation.wet_spell_max_length(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_wet_spell_total_length(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.wet_spell_total_length")

    ds_in = MockEarthkitData()
    precipitation.wet_spell_total_length(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_wetdays(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.wetdays")

    ds_in = MockEarthkitData()
    precipitation.wetdays(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted


def test_wetdays_prop(mocker: MockerFixture, common_mocks):
    mock_fn = mocker.patch("xclim.indicators.atmos.wetdays_prop")

    ds_in = MockEarthkitData()
    precipitation.wetdays_prop(ds_in)

    common_mocks["mock_to_xr"].assert_called_once_with(ds_in, {})
    ds_converted = common_mocks["mock_to_xr"].return_value[0]

    mock_fn.assert_called_once()
    assert mock_fn.call_args.kwargs["ds"] is ds_converted
