HTML Subpages
*************

Overview
========

.. image:: screenshots/overview.*

``Overview`` subpage gives general information on the features - their summary statistics, number of missing values,
etc. (similar to ``describe`` method you can call on ``pandas.DataFrame`` object). First few rows of the
data are also shown for your convenience (transposed).


If the number of features isn't too big, `seaborn PairPlot <https://seaborn.pydata.org/generated/seaborn.pairplot.html>`_
is also included.

.. note::

    Feature names in all tables are 'hoverable' - upon hovering, corresponding description and mapping will be shown.

    .. image:: screenshots/hover.*

Features View
=============

.. image:: screenshots/features.*

``Features`` subpage allows you to dive deeper into each Feature present in your data - not only to see summary
statistics and distribution, but also how it is transformed in the process and how it affects other features
so that engineering new ones from it might be possible. To change displayed feature, you can click on the 'burger'
button in the upper left corner and choose another one from the opened menu.

.. image:: screenshots/features-menu.*

.. note::

    Depending on the number of features, refreshing the page upon feature selection change might take a while -
    please be patient!

Content in ``Features`` subpage is divided into subsections that you can easily expand or hide (in case the page gets
too cluttered). First two sections give information on a chosen feature (both original and transformed), whereas the
third section shows correlations between **all** features (normalized and original), regardless of selection.

.. note::

    When Numerical Feature is selected, Transformations sections will show plots of Normal Transformations so you can
    make a conscious decision which of the Transformer is the best in that case (per
    `sklearn guidance <https://scikit-learn.org/stable/auto_examples/preprocessing/plot_map_data_to_normal.html>`_).

    .. image:: screenshots/normal-transformations.*

*ScatterPlot Grid* is included in the last section - Bokeh-driven visualization that plots every feature against
another as scatter plots in a manner similar to seaborn's PairPlot, but with a twist - every feature is also used
as a hue (coloring). The idea behind it is that this visualization might help you with manual feature engineering if
there is a connection between 3 features (instead of 2 as is the case in regular scatter plots).

.. note::

    Row with a chosen feature on X axis and coloring by the same feature is greyed out to minimize confusion (ideally
    separate colored groups would be present). Greying out was included as an alternative to removing that row, as it
    was technically difficult to not break the whole structure of the Grid in the meantime.

.. image:: screenshots/scatterplotgrid.*

Models
======

.. image:: screenshots/models.*

``Models`` subpage gives information on provided Models performance. Provided results and visualization will always
correspond to the 3 best scoring Models chosen during dashboard creation (search methods). Even though provided plots
will be different based on a problem type, scoring table in the upper left corner (with different scoring) and
predictions table at the bottom stay the same.

Classification
--------------

``Models`` in *classification* problem will provide you with Plots of performance curves (``ROC curve``,
``precision-recall curve`` and ``detection error tradeoff curve``) and Confusion Matrices for every model (see sklearn
`roc <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html>`_,
`precision-recall <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html>`_,
`det <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.det_curve.html>`_.


.. image:: screenshots/classification.*

.. note::

    Legend is interactive - you can mute lines by clicking at a particular legend entry.


Regression
----------

``Models`` in *regression* provide two plots for every Model: ``prediction errors`` and ``residuals``.

* ``prediction errors`` plot shows Actual Target on X axis and Predicted Target on Y Axis
* ``residuals`` plot shows Actual Target on X axis and the difference between Predicted and Actual on Y Axis

.. image:: screenshots/regression.*

Multiclass
----------

``Models`` in *multiclass* problem shows you Confusion Matrices for every Model assesed.

.. image:: screenshots/multiclass.*

