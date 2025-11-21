import os
import pytest
from pytest_mock import MockerFixture
from earthkit.climate.generate_wrappers import generate_module

@pytest.fixture
def mock_xclim_module(mocker: MockerFixture):
    """Mocks an xclim module with some indicators."""
    mock_mod = mocker.MagicMock()
    mock_mod.__all__ = ["indicator1", "indicator2", "other_func"]
    # Setup attributes
    mock_mod.indicator1 = "indicator1_fn"
    mock_mod.indicator2 = "indicator2_fn"
    mock_mod.other_func = "other_func_fn"
    return mock_mod

def test_generate_module_content(tmp_path, mock_xclim_module, mocker: MockerFixture):
    """Test that the generated file contains expected content."""
    output_file = tmp_path / "test_indicators.py"
    
    mocker.patch("importlib.import_module", return_value=mock_xclim_module)
    generate_module("xclim.indicators.test", str(output_file))
        
    assert output_file.exists()
    content = output_file.read_text()
    
    # Check imports
    assert "import xclim" in content
    assert "from earthkit.climate.api.wrapper import wrap_xclim_indicator" in content
    
    # Check wrapper calls
    assert "indicator1 = wrap_xclim_indicator(xclim.indicators.test.indicator1)" in content
    assert "indicator2 = wrap_xclim_indicator(xclim.indicators.test.indicator2)" in content
    assert "other_func = wrap_xclim_indicator(xclim.indicators.test.other_func)" in content

def test_generate_module_filter(tmp_path, mock_xclim_module, mocker: MockerFixture):
    """Test that the filter function works."""
    output_file = tmp_path / "test_filtered.py"
    
    def filter_func(name):
        return "indicator" in name
        
    mocker.patch("importlib.import_module", return_value=mock_xclim_module)
    generate_module("xclim.indicators.test", str(output_file), filter_func)
        
    content = output_file.read_text()
    
    # Should contain indicator1 and indicator2
    assert "indicator1 =" in content
    assert "indicator2 =" in content
    # Should NOT contain other_func
    assert "other_func =" not in content

def test_generate_module_docstring(tmp_path, mock_xclim_module, mocker: MockerFixture):
    """Test that the module docstring is generated."""
    output_file = tmp_path / "test_docstring.py"
    
    mocker.patch("importlib.import_module", return_value=mock_xclim_module)
    generate_module("xclim.indicators.test", str(output_file))
        
    content = output_file.read_text()
    assert '"""' in content
    assert "Wrapper module for xclim.indicators.test" in content
    assert "auto-generated" in content
