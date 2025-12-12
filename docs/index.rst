Welcome to Earthkit-climate's documentation!
============================================

**earthkit-climate** is the package responsible for the climate index calculation within the earthkit ecosystem. It includes a wrapper prototype that allows the use of the ``xclim`` python package to compute a large amount of pre-defined climate indices used by the climate science community, and to define new ones.

``xclim`` relies heavily on the ``xarray`` python library and the ``numpy`` & ``scipy`` ecosystem. Its main elements are:

- **Climate indices**: available to be directly computed with python functions. The input and output units are defined in these functions by using a decorator and are validated during runtime.
- **Climate indicators**: climate indices wrapped in an object that provides more metadata and validation facilities (health checks) of the input. it includes attributes for CF metadata (cell methods), references, keywords, and more.
- **Lower level process functions**: these include aggregation, computation spell length and counting, optimized computation of reference percentiles, bias correction methods and ensemble statistics. These functions are used by the implemented indices and can also be used to build new indices not included in the library.

.. toctree::
   :caption: EXAMPLES
   :maxdepth: 2

   tutorials
   gallery

.. toctree::
   :caption: DOCUMENTATION
   :maxdepth: 2

   user-guide
   API Reference <_api/index>
   development

.. toctree::
   :caption: INSTALLATION
   :maxdepth: 2

   installation
   release-notes
   license

.. toctree::
   :caption: PROJECTS
   :maxdepth: 2

   earthkit <https://earthkit.readthedocs.io/en/latest/>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
