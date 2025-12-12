Development
===========

Development & Contribution Workflow
-----------------------------------

1. Setup environment (with Pixi)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This project uses `Pixi <https://pixi.sh>`_ for dependency and environment management.
It provides fast, reproducible environments and replaces Conda-based workflows.

Install Pixi following the `official instructions <https://pixi.sh/latest/#installation>`_, then run:

.. code-block:: bash

   pixi install

This command installs all dependencies as defined in ``pyproject.toml`` and ``pixi.lock``.

2. Common Tasks
~~~~~~~~~~~~~~~

This project uses ``pixi`` tasks to manage development workflows, replacing the legacy ``Makefile``.

- **Quality Assurance**: Run pre-commit hooks to ensure code quality.

  .. code-block:: bash

     pixi run qa

- **Unit Tests**: Run the test suite using pytest.

  .. code-block:: bash

     pixi run unit-tests

- **Type Checking**: Run static type analysis with mypy.

  .. code-block:: bash

     pixi run type-check

- **Build Documentation**: Build the Sphinx documentation. Note that this task runs in the ``docs`` environment.

  .. code-block:: bash

     pixi run -e docs docs-build

- **Docker**: Build and run the docker container.

  .. code-block:: bash

     pixi run docker-build
     pixi run docker-run

- **Sync with ECMWF template**:

  .. code-block:: bash

     pixi run template-update
