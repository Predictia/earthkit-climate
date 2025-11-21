import os
import sys
from setuptools.command.build_py import build_py

from earthkit.climate.generate_wrappers import generate_module


class GenerateIndicators(build_py):
    """
    Custom build command to generate indicator wrappers.

    This command generates the indicator wrapper modules for temperature,
    precipitation, and wind indicators from xclim.
    """

    def run(self) -> None:
        """
        Run the generation process.
        """
        def is_precip(name: str) -> bool:
            return any(x in name.lower() for x in ['precip', 'pr_', 'snow', 'rain', 'drought', 'dry', 'wet'])

        def is_wind(name: str) -> bool:
            return 'wind' in name.lower()

        def is_temp(name: str) -> bool:
            return not is_precip(name) and not is_wind(name)

        generate_module(
            "xclim.indicators.atmos",
            "src/earthkit/climate/indicators/temperature.py",
            is_temp
        )
        generate_module(
            "xclim.indicators.atmos",
            "src/earthkit/climate/indicators/precipitation.py",
            is_precip
        )
        generate_module(
            "xclim.indicators.atmos",
            "src/earthkit/climate/indicators/wind.py",
            is_wind
        )
        super().run()
