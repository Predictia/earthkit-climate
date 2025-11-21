"""
Module for generating wrapper modules for xclim indicators.
"""
import importlib
import os
from typing import Callable, Optional

# We import this to ensure it's available, though not strictly used in generation logic below
# pylint: disable=unused-import
from earthkit.climate.api.wrapper import wrap_xclim_indicator  # noqa: F401


def generate_module(
    module_name: str,
    output_file: str,
    filter_func: Optional[Callable[[str], bool]] = None
) -> None:
    """
    Generate a wrapper module for the given xclim module.

    Parameters
    ----------
    module_name : str
        The name of the xclim module to wrap (e.g., "xclim.indicators.atmos").
    output_file : str
        The path to the output file to generate.
    filter_func : Callable[[str], bool], optional
        A function to filter which indicators to include. If None, all indicators are included.
    """
    module = importlib.import_module(module_name)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    lines = [
        '"""',
        f'Wrapper module for {module_name}.',
        '',
        'This module is auto-generated. Do not edit directly.',
        '"""',
        "import xclim",
        "from earthkit.climate.api.wrapper import wrap_xclim_indicator",
        "",
    ]

    for name in module.__all__:
        if filter_func and not filter_func(name):
            continue

        lines.append(
            f"{name} = wrap_xclim_indicator({module_name}.{name})"
        )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


if __name__ == "__main__":
    def is_precip(name: str) -> bool:
        return any(x in name.lower() for x in ['precip', 'pr_', 'snow', 'rain', 'drought', 'dry', 'wet'])

    def is_wind(name: str) -> bool:
        return 'wind' in name.lower()

    def is_temp(name: str) -> bool:
        return not is_precip(name) and not is_wind(name)

    # Temperature indicators
    generate_module(
        "xclim.indicators.atmos",
        "src/earthkit/climate/indicators/temperature.py",
        is_temp
    )
    # Precipitation indicators
    generate_module(
        "xclim.indicators.atmos",
        "src/earthkit/climate/indicators/precipitation.py",
        is_precip
    )
    # Wind indicators
    generate_module(
        "xclim.indicators.atmos",
        "src/earthkit/climate/indicators/wind.py",
        is_wind
    )
