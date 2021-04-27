.. data-dashboard documentation master file, created by
   sphinx-quickstart on Tue Apr 27 23:11:41 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

data-dashboard
**************

Introduction
============

``data_dashboard`` is a library for creating dashboards visualizing the data and searching for the best scoring
Model for a given problem.

To create a dashboard you need the data: X, y and the output directory where the HTML wil be placed. You can use
toy datasets examples included in the library as well:
   from data_dashboard import Dashboard
   from data_dashboard.examples import iris
   import os

   output_directory = os.path.join(os.getcwd(), "dashboard_output")
   X, y, descriptions = iris()

   dsh = Dashboard(X, y, output_directory, descriptions)
   dsh.create_dashboard()



Welcome to data-dashboard's documentation!
==========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
