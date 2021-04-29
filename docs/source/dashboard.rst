Dashboard
****************

Creating Dashboard instance
===========================

Features Descriptions Dictionary
--------------------------------

``Dashboard`` instance can be created with just ``X``, ``y`` and ``output_directory`` arguments, but it doesn't mean
it cannot be customized. The most notable optional argument that can augment the Dashboard is
``feature_descriptions_dict`` dictionary-like object. Structure of the dict should be::

    feature_descriptions_dict = {

        "Feature1": {

            "description": "description of feature1, what does it represent in a human friendly way",
            "category": "cat" OR "num",
            "mapping": {
                "value1": "better name or explanation for value1",
                "value2": "better name or explanation for value2"
            }
        },

        "Feature2": {

            (...)

        }

    }


* ``"description"`` describes what the feature is about - whats the story behind numbers and values.
* ``"category"`` defines what kind of a data you want the feature to be treated as

    * ``"num"`` stands for numerical values
    * ``"cat"`` is for categorical variables

* ``"mapping"`` represents what do every number mean in your data - e.g. 0: "didn't buy a product", 1: "bought a product".

This external information is fed to the Dashboard and it will be included in HTML files (where appropriate)
for your convenience. Last but not least, by providing a specific ``"category"`` you are forcing the Dashboard to
interpret a given feature the way you want, e.g.: you could provide ``"num"`` to a binary variable (only 0 and 1s) and
Dashboard will treat that feature as Numerical (which means that, for example, Normal Transformations will be applied).

.. note::
    Please keep in mind that every argument in ``feature_descriptions_dict`` is optional: you can provide all Features
    or only one, only ``"category"`` for few of the Features and ``"description"`` for others, etc.


Other Dashboard instance arguments
----------------------------------

``already_transformed_columns`` can be a list of features that are already transformed and won't need additional
transformations from the Dashboard::

    Dashboard(X, y, output_directory,
              already_transformed_columns=["Feature1", "pre-transformed Feature4"]
              )

``classification_pos_label`` forces the Dashboard to treat provided label as a positive (1) label::

    Dashboard(X, y, output_directory,
              classification_pos_label="Not Survived"
              )

``force_classification_pos_label_multiclass`` is a ``bool`` flag useful when you provide classification_pos_label in a
multiclass problem (essentially turning it into classification problem) - without it classification_pos_label will be
ignored::

    Dashboard(X, y, output_directory,
              classification_pos_label="Iris-Setosa",
              force_classification_pos_label_multiclass=True
              )

``random_state`` can be provided for reproducibility::

    Dashboard(X, y, output_directory,
              random_state=13,
              )

Example
------------

::

    dsh = Dashboard(
        X=your_X,
        y=your_y,
        output_directory="path/output",
        features_descriptions_dict={"petal width (cm)": {"description": "width of petal (in cm)"}},
        already_transformed_columns=["sepal length (cm)"],
        classification_pos_label=1,
        force_classification_pos_label=True,
        random_state=10
        )

Creating HTML Dashboard
=======================

To create HTML Dashboard from Dashboard instance, you need to call ``create_dashboard()`` method::

    dsh.create_dashboard()

You can customize the process further by providing appropriate arguments to the method (see below).

Models
------

``models`` can be provided in different ways

* list of Models instances
* dict of Model class: param_grid attributes to do GridSearch on
* None - in which case default set will be used

::

    # list of Models
    models = [DecisionTreeClassifier(), SVC(C=100.0), LogisticRegression()]

    # dict for GridSearch
    models = {
        DecisionTreeClassifier: {"max_depth": [1, 5, 10], "criterion": ["gini", "entropy"]},
        SVC: {"C": [10, 100, 1000]}
    }

    # None
    models = None

Scoring
-------

``scoring`` should be an sklearn scoring function appropriate for a given X, y problem. It can also be None, in which
case default scoring for a given problem will be used::

    scoring = precision_score

.. note::
    Some functions might not work right for some type of problems (e.g. roc_auc_score for multiclass)

.. create-dashboard-mode:

Mode
----

``mode`` should be provided as either ``"quick"`` or ``"detailed"`` string literal. Argument is useful only
when *models=None*.

* if ``"quick"``, then the initial search is done only on default instances of Models (similar to ``LazyPredict`` library) and top scoring Models are GridSearched
* if ``"detailed"``, then all available combinations of default Models are GridSearched

.. create-dashboard-logging:

Logging
-------

``logging`` is a ``bool`` flag indicating if you want to have .csv files (search logs) included in your output
directory in logs subdirectory.


Disabling Pairplots
-------------------

Both Seaborn PairPlot in Overview subpage and bokeh-driven ScatterPlot Grid in Features subpage were identified to be
the biggest bottlenecks in creating HTML Dashboard. If you feel like speeding up the process, set
``disable_pairplots=True``.

.. note::
    Pairplots are disabled by default when the number of features in your data crosses certain threshold.
    See also :ref:`forcing-pairplots`.

.. _forcing-pairplots:

Forcing Pairplots
-----------------

When number of features in X and y crosses a certain threshold, creation of both Seaborn PairPlot and bokeh-driven
ScatterPlot Grid is disabled. This was a conscious decision, as not only it extremely slows down the process (and
might even lead to raising Exceptions), PairPlots are getting so enormous that the insight gained from them is
minuscule.

If you know what you're doing, provide ``force_pairplot=True``.

.. note::
    If ``disable_pairplots=True`` and ``force_pairplot=True`` are both provided, ``disable_pairplots``
    takes precedence and pairplots **will be disabled**.

.. _create-dashboard-example:

Example
-------

::

    dsh.create_dashboard(
        models=None,
        scoring=sklearn.metrics.precision_score,
        mode="detailed",
        logging=True,
        disable_pairplots=False,
        force_pairplots=True
    )

Setting Custom Preprocessors in Dashboard
=========================================

``set_custom_transformers`` is a method to provide your own Transformers to Dashboard pipeline. Dashboard's
preprocessing is simple, so you are free to change it to your liking. There are 3 arguments (all optional):

* *categorical_transformers*
* *numerical_transformers*
* *y_transformer*

Both ``categorical_transformers`` and ``numerical_transformers`` should be a list-like object of instantiated
Transformers. As names suggest, ``categorical_transformers`` will be used to transform Categorical features, whereas
``numerical_transformers`` will transform Numerical features.

``y_transformer`` should be a **single Transformer**.

::

    dsh.set_custom_transformers(
        categorical_transformers=[SimpleImputer(strategy="most_frequent")],
        numerical_transformers=[StandardScaler()],
        y_transformer=LabelEncoder()
    )

.. note::
    Keep in mind that in *regression* problems, Dashboard already wraps the target in TransformedTargetRegressor object
    (with QuantileTransformer as a transformer). See also `sklearn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html>`_.


Using Dashboard as sklearn pipeline
===================================

Dashboard can also be used as a simpler version of sklearn.pipelines - methods such as transform, predict, etc. are
expoded and available. Please refer to :ref:`dashboard-documentation` for more information.


.. _dashboard-documentation:

Documentation
=============

.. autoclass:: data_dashboard.dashboard.Dashboard
   :members: create_dashboard, search_and_fit, set_and_fit, predict, transform, best_model, set_custom_transformers
   :member-order: bysource

   .. automethod:: __init__
      :noindex:

