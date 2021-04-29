Questions and Answers
=====================

* *What types of Machine Learning problems does it work on?*

    Currently *classification*, *regression* and *multiclass* problems are viable. *Multilabel* problem is
    **not available**.

* *How are the Model results calculated?*

    Provided data is split into train and tests split. Every time any Models search is initiated, Models are fit on
    train portion of the data, but the scores are calculated on test splits. However, the chosen Model (or the one set
    by you with ``set_and_fit`` method) is fit on **all** data.

* *What are the default models used for searches?*

    You can inspect default models used `here <https://github.com/maciek3000/data_dashboard/blob/master/data_dashboard/models.py>`_.

* *What are the default Transformers used?*

    For Categorical Features:

    * ``SimpleImputer(strategy="most_frequent")``
    * ``OneHotEncoder(handle_unknown="ignore")``

    For Numerical Features:

    * ``SimpleImputer(strategy="median")``
    * ``QuantileTransformer(output_distribution="normal")``
    * ``StandardScaler()``

* *Can I provide my own default models to GridSearch?*

    Yes! You can do it with ``models`` argument in ``create_dashboard`` method - you just need to provide your own
    dictionary object with ``Model class: param_grid`` pairs.

* *What to do when I am working on a NLP-related problem?*

    At the time of building this library, I wasn't experienced enough to tackle the NLP problem in any simplified,
    automated way. The best way to approach it would be to transform the data on your end and then provide transformed
    ``X`` and ``y`` to the ``Dashboard`` (with ``already_transformed_columns`` argument as needed).

* *Why did you decide on HTML static files instead of server-client architecture (e.g. with bokeh server)?*

    Two main reasons:

    * I wanted to have a lightweight HTML output that doesn't rely on any server process (either localhost or regular);
    * I have done something similar in my `other project <https://github.com/maciek3000/GnuCash-Expenses-Vis>`_
      and I wanted to try something new.

Known Issues / Problems
-----------------------

    * *Multilabel* problem type is currently not supported.
    * ``Features`` subpage might get laggy when trying to change feature selection and when number of all features in
      the data is high.
    * CSS/HTML might get *wonky* sometimes

