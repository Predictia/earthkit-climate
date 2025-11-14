import pytest
import xarray as xr
from pytest_mock import MockerFixture

from earthkit.climate.indicators.temperature import daily_temperature_range, \
    warm_spell_duration_index, heating_degree_days


def test_dtr_end_to_end_returns_earthkit_object(
    mocker: MockerFixture, dummy_temp_ds: xr.Dataset, common_mocks: dict
) -> None:
    """
    Ensure daily_temperature_range computes successfully and returns an Earthkit object.
    """
    mock_to_xr = common_mocks["mock_to_xr"]
    mock_ensure = common_mocks["mock_ensure_units"]
    mock_add_prov = common_mocks["mock_add_prov"]
    mock_to_ek = common_mocks["mock_to_ek"]
    object_ek = common_mocks["object_ek"]

    mock_to_xr.side_effect = [
        (dummy_temp_ds[["tasmax"]], {"earthkit_internal": {}}),
        (dummy_temp_ds[["tasmin"]], {"earthkit_internal": {}}),
    ]

    dtr_da = xr.DataArray([10.0], name="dtr")
    mock_xclim = mocker.patch("xclim.indicators.atmos.daily_temperature_range", return_value=dtr_da)

    res = daily_temperature_range(dummy_temp_ds[["tasmax"]], dummy_temp_ds[["tasmin"]])

    assert res is object_ek
    assert mock_to_xr.call_count == 2
    assert mock_ensure.call_count == 2
    mock_xclim.assert_called_once()
    mock_add_prov.assert_called_once()
    mock_to_ek.assert_called_once_with(dtr_da.to_dataset(name="dtr"), {"earthkit_internal": {}, "prov": True})


def test_wsdi_end_to_end_computes_correctly(
    mocker: MockerFixture, dummy_temp_ds: xr.Dataset, common_mocks: dict
) -> None:
    """
    Ensure warm_spell_duration_index orchestrates correctly and metadata flows as expected.
    """
    mock_to_xr = common_mocks["mock_to_xr"]
    mock_ensure = common_mocks["mock_ensure_units"]
    mock_add_prov = common_mocks["mock_add_prov"]
    mock_to_ek = common_mocks["mock_to_ek"]
    object_ek = common_mocks["object_ek"]

    mock_to_xr.side_effect = [
        (dummy_temp_ds[["tasmax"]], {"earthkit_internal": {}}),
        (dummy_temp_ds[["tasmax"]], {"earthkit_internal": {}}),
    ]

    mocker.patch("earthkit.climate.indicators.temperature.percentile_doy", return_value=xr.DataArray([25.0]))

    wsdi_da = xr.DataArray([5.0], name="wsdi")
    mock_xclim = mocker.patch(
        "xclim.indicators.atmos.warm_spell_duration_index",
        return_value=wsdi_da,
    )

    res = warm_spell_duration_index(dummy_temp_ds[["tasmax"]], dummy_temp_ds[["tasmax"]], freq="YS", window=6)

    assert res is object_ek
    assert mock_to_xr.call_count == 2
    assert mock_ensure.call_count == 2
    mock_xclim.assert_called_once()
    mock_add_prov.assert_called_once()
    mock_to_ek.assert_called_once_with(wsdi_da.to_dataset(name="wsdi"), {"earthkit_internal": {}, "prov": True})


def test_hdd_end_to_end_returns_earthkit_object(
    mocker: MockerFixture, dummy_temp_ds: xr.Dataset, common_mocks: dict
) -> None:
    """
    Ensure heating_degree_days computes correctly and returns the proper Earthkit object.
    """
    mock_to_xr = common_mocks["mock_to_xr"]
    mock_ensure = common_mocks["mock_ensure_units"]
    mock_add_prov = common_mocks["mock_add_prov"]
    mock_to_ek = common_mocks["mock_to_ek"]
    object_ek = common_mocks["object_ek"]

    mock_to_xr.side_effect = [
        (dummy_temp_ds[["tasmax"]], {"earthkit_internal": {}}),
        (dummy_temp_ds[["tasmin"]], {"earthkit_internal": {}}),
        (dummy_temp_ds[["tas"]], {"earthkit_internal": {}}),
    ]

    hdd_da = xr.DataArray([50.0], name="hdd")
    mock_xclim = mocker.patch(
        "xclim.indicators.atmos.heating_degree_days_approximation",
        return_value=hdd_da,
    )

    res = heating_degree_days(dummy_temp_ds[["tasmax"]], dummy_temp_ds[["tasmin"]], dummy_temp_ds[["tas"]])

    assert res is object_ek
    assert mock_to_xr.call_count == 3
    assert mock_ensure.call_count == 3
    mock_xclim.assert_called_once()
    mock_add_prov.assert_called_once()
    mock_to_ek.assert_called_once_with(hdd_da.to_dataset(name="hdd"), {"earthkit_internal": {}, "prov": True})
