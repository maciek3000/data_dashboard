Examples
********

.. _library_examples:

Library examples
================

Library comes with few defined *toy datasets* that you can use if you don't have any of your data handy.

::

    from data_dashboard.examples import iris, boston, diabetes, digits, wine, breast_cancer
    X, y, descriptions = iris()  # multiclass
    X, y, descriptions = boston()  # regression
    X, y, descriptions = diabetes()  # regression
    X, y, descriptions = digits()  # multiclass
    X, y, descriptions = wine()  # multiclass
    X, y, descriptions = breast_cancer()  # classification

Example Dashboard
=================

Deployed Dashboard example can be found `here <https://example-data-dashboard.herokuapp.com/>`_.

Documentation
=============

.. automodule:: data_dashboard.examples.examples
   :members:
   :member-order: bysource
