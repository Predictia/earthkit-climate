# earthkit-climate

**A toolkit for statistical analysis and processing of climate and geospatial data.**

`earthkit-climate` provides tools to compute and analyze **climate indicators** (e.g., precipitation, temperature) and perform **unit conversions, percentiles, and provenance tracking**.
It is part of the **Earthkit ecosystem** and designed for reproducible, modular workflows.

______________________________________________________________________

## Disclaimer

This project is currently in **BETA** and **experimental**.
Interfaces, structure, and functionality are subject to change without notice.
Do **not** use this software in any operational or production system.

______________________________________________________________________

## Quick Start

Install the package in editable mode:

```bash
pip install -e .
```

Example usage:

```python
from earthkit.climate.indicators import precipitation, temperature
from earthkit.climate.utils import conversions

# Example: compute a precipitation index
pr = precipitation.simple_daily_intensity(precip_data, freq="monthly")
```

______________________________________________________________________

## Documentation

For full documentation, including API reference and example notebooks, visit the
[earthkit-climate ReadTheDocs page](https://earthkit-climate.readthedocs.io)

______________________________________________________________________

## Development & Contribution Workflow

### 1. Setup environment (with Pixi)

This project uses [**Pixi**](https://pixi.sh) for dependency and environment management.
It provides fast, reproducible environments and replaces Conda-based workflows.

Install Pixi following the [official instructions](https://pixi.sh/latest/#installation), then run:

```bash
pixi install --locked
pixi shell
```

This command installs all dependencies as defined in `pyproject.toml` and `pixi.lock`.

### 2. Install the package

Inside the Pixi environment:

```bash
pip install -e .
```

### 3. Quality checks and tests

Before pushing to GitHub, run:

```bash
make qa
make unit-tests
```

You can also perform type checking and integration tests:

```bash
make type-check
make integration-tests
```

### 4. Documentation

To build the documentation locally (using Sphinx):

```bash
make docs-build
```

### 5. Optional: Sync with ECMWF template

```bash
make template-update
```

______________________________________________________________________

## Project Structure

```
earthkit-climate/
├── src/earthkit/
│   ├── climate/
│   │   ├── indicators/        # Climate indices (precipitation, temperature, etc.)
│   │   └── utils/             # Type conversions, percentiles, provenance
│   └── __init__.py
├── tests/
│   ├── integration/           # Integration tests
│   └── unit/                  # Unit tests for indicators and utils
├── docs/                      # Sphinx documentation
├── ci/                        # Continuous integration configs
├── .github/workflows/         # GitHub Actions (push/release)
├── .pixi/                     # Pixi configuration
├── pixi.lock                  # Locked dependency versions
├── Dockerfile                 # Pixi-based container
├── pyproject.toml             # Project configuration
├── Makefile                   # Developer utilities (Pixi integrated)
└── README.md
```

______________________________________________________________________

## License

```
Copyright 2022,
European Centre for Medium Range Weather Forecasts (ECMWF)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

    http://www.apache.org/licenses/LICENSE-2.0
```
