import pytest
import xarray as xr
from pytest_mock import MockerFixture

from earthkit.climate.indicators.precipitation import (
    maximum_consecutive_wet_days,
    daily_precipitation_intensity,
)


def test_cwd_end_to_end_returns_earthkit_object(
    mocker: MockerFixture,
    dummy_precip_ds: xr.Dataset,
    common_mocks: dict,
) -> None:
    """
    End-to-end test for maximum_consecutive_wet_days ensuring orchestration works
    and dependencies are called correctly.
    """
    mock_to_xr = common_mocks["mock_to_xr"]
    mock_add_prov = common_mocks["mock_add_prov"]
    mock_to_ek = common_mocks["mock_to_ek"]
    sentinel_ek = common_mocks["sentinel_ek"]

    dummy_out = xr.Dataset({"cwd": ("time", [5])})
    mock_xclim = mocker.patch(
        "xclim.indicators.atmos.maximum_consecutive_wet_days",
        return_value=dummy_out,
    )

    result = maximum_consecutive_wet_days(dummy_precip_ds, wet_day_threshold=2.0)

    assert result is sentinel_ek
    mock_to_xr.assert_called_once()
    mock_xclim.assert_called_once()
    mock_add_prov.assert_called_once()
    mock_to_ek.assert_called_once_with(dummy_out, {"earthkit_internal": {}, "prov": True})


def test_sdii_with_frequency_end_to_end(
    mocker: MockerFixture,
    dummy_precip_ds: xr.Dataset,
    common_mocks: dict,
) -> None:
    """
    Test daily_precipitation_intensity end-to-end behavior, verifying that
    the frequency argument is forwarded correctly to xclim and metadata flows properly.
    """
    mock_to_xr = common_mocks["mock_to_xr"]
    mock_add_prov = common_mocks["mock_add_prov"]
    mock_to_ek = common_mocks["mock_to_ek"]
    sentinel_ek = common_mocks["sentinel_ek"]

    dummy_out = xr.Dataset({"sdii": ("time", [1.23])})
    mock_xclim = mocker.patch(
        "xclim.indicators.atmos.daily_pr_intensity",
        return_value=dummy_out,
    )

    res = daily_precipitation_intensity(dummy_precip_ds, frequency="MS")

    assert res is sentinel_ek
    mock_to_xr.assert_called_once()
    assert mock_xclim.call_args.kwargs["freq"] == "MS"
    assert mock_xclim.call_args.kwargs["ds"].attrs.get("ensured") is True
    mock_add_prov.assert_called_once()
    mock_to_ek.assert_called_once_with(dummy_out, {"earthkit_internal": {}, "prov": True})


def test_threshold_numeric_formats_with_units(
    mocker: MockerFixture,
    dummy_precip_ds: xr.Dataset,
    common_mocks: dict,
) -> None:
    """
    Test that numeric thresholds are automatically formatted with units (mm/day).
    """
    mock_xclim = mocker.patch(
        "xclim.indicators.atmos.maximum_consecutive_wet_days",
        return_value=xr.Dataset(),
    )

    maximum_consecutive_wet_days(dummy_precip_ds, wet_day_threshold=3)
    assert mock_xclim.call_args.kwargs["thresh"] == "3 mm/day"


def test_threshold_string_forwarded_unchanged(
    mocker: MockerFixture,
    dummy_precip_ds: xr.Dataset,
    common_mocks: dict,
) -> None:
    """
    Test that string thresholds (e.g., '1 mm/day') are passed unchanged to xclim.
    """
    mock_xclim = mocker.patch(
        "xclim.indicators.atmos.maximum_consecutive_wet_days",
        return_value=xr.Dataset(),
    )

    maximum_consecutive_wet_days(dummy_precip_ds, wet_day_threshold="1 mm/day")
    assert mock_xclim.call_args.kwargs["thresh"] == "1 mm/day"


def test_ensure_units_non_strict_warns_and_overwrites(
    mocker: MockerFixture,
    dummy_precip_ds: xr.Dataset,
    common_mocks: dict,
) -> None:
    """
    Test that ensure_units is called with strict=False and overwrites units as expected.
    """
    def _ensure_units_side_effect(ds, var, units, strict=False):
        ds[var].attrs["units"] = units
        return ds

    ensure_mock = mocker.patch(
        "earthkit.climate.utils.units.ensure_units",
        side_effect=_ensure_units_side_effect,
    )
    mocker.patch("xclim.indicators.atmos.daily_pr_intensity", return_value=xr.Dataset())

    daily_precipitation_intensity(dummy_precip_ds)

    args, kwargs = ensure_mock.call_args
    assert args[1] == "pr"
    assert args[2] == "mm/day"
    assert kwargs.get("strict", False) is False
