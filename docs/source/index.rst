data-dashboard
**************

Introduction
============

``data_dashboard`` library allows you to build HTML Dashboard visualizing not only the data and relationships between
features but also automatically search for the best 'baseline' sklearn compatible Model.

.. image:: screenshots/dashboard.gif

You can install ``data_dashboard`` with pip::

   pip install data-dashboard

.. note::
   Please keep in mind that package name is data-dashboard (with hyphen: '-') whereas module to import from is called
   data_dashboard (with underscore: '_').

To create a Dashboard you need the data: ``X``, ``y`` and the ``output_directory`` where the HTML files will be placed.
You can use toy datasets from :doc:`examples` (e.g. ``iris`` dataset) included in the library as well::

   from data_dashboard import Dashboard
   from data_dashboard.examples import iris
   output_directory = "your_path/dashboard_output"
   X, y, descriptions = iris()  # descriptions is additional argument described further in docs

   dsh = Dashboard(X, y, output_directory, descriptions)
   dsh.create_dashboard()

.. note::
   Depending on the size of your data, fitting process might take some time. Please be patient!

Created HTML Dashboard will contain 3 subpages for you to investigate:

    * ``Overview`` with summary statistics of the data;
    * ``Features`` where you can dig deeper into each feature in the data;
    * ``Models`` showing search results and models performances.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   dashboard
   views
   examples
   faq
   license
   help


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
