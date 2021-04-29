import time
import warnings
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import make_scorer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, det_curve
from sklearn.exceptions import NotFittedError
from .models import classifiers, regressors
from .functions import reverse_sorting_order, obj_name


class ModelNotSetError(ValueError):
    """Model is not set."""
    pass


class ModelsNotSearchedError(ValueError):
    """Search for Models was not done."""
    pass


class WrappedModelRegression:
    """Wrapper for Models in Regression problems.

    Models get wrapped with TransformedTargetRegressor to transform y target before predictions on X features take
    place. Wrapper additionally customizes __name__, __class__ and __str__ methods/attributes to return those values
    from main Model (not TransformedTargetRegressor).

    Attributes:
        clf (sklearn.compose.TransformedTargetRegressor): Wrapped model for regression problems
    """
    def __init__(self, regressor, transformer):
        """Create WrappedModelRegression object.

        Override __name__ and __class__ attributes with appropriate attributes from regressor.

        Args:
            regressor (sklearn.Model): Model used to predict regression target
            transformer (sklearn.Transformer): Transformer used to transform y (target)
        """
        self.clf = TransformedTargetRegressor(regressor=regressor, transformer=transformer)
        self.__name__ = self.clf.regressor.__class__.__name__
        self.__class__ = self.clf.regressor.__class__

    def fit(self, *args, **kwargs):
        """Fit Model in clf attribute with provided arguments.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            self
        """
        self.clf.fit(*args, **kwargs)
        return self

    def predict(self, *args, **kwargs):
        """Predict provided arguments with Model in clf attribute.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            numpy.ndarray: predictions
        """
        return self.clf.predict(*args, **kwargs)

    def get_params(self, *args, **kwargs):
        """Return params of regressor inside wrapped clf Model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            dict: params of regressor
        """
        return self.clf.regressor.get_params(*args, **kwargs)

    def __str__(self):
        """Return __str__method of regressor inside wrapped clf Model.

        Returns:
            str: __str__ method of regressor
        """
        return self.clf.regressor.__str__()

    def __class__(self, *args, **kwargs):
        """Return new object of regressor class instantiated with *args and **kwargs arguments.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            regressor: new regressor object

        """
        return self.clf.regressor.__class__(*args, **kwargs)


class ModelFinder:
    """ModelFinder used to search for Models with best scores.

    ModelFinder works in two ways - as a search engine for Models (either with GridSearch or by simply comparing
    scores in a way similar to LazyPredict package) and as a container for the chosen Model to be used for fit/predict
    methods, etc.

    search is bread and butter method used to compare performance of different Models on provided X and y data. As also
    explained in search, Models can be either predefined (as already created instances or Model: param_grid pairs) or
    not, in which case default collection of Models is used in search.

    Train splits are used to fit Models, but assessment of how well they perform given a specific scoring function is
    performed on test splits.

    Note:
        provided X and y needs to be pre-transformed.

    Attributes:
        X (pandas.DataFrame, numpy.ndarray, scipy.csr_matrix): X feature space (transformed)
        y (pandas.Series, numpy.ndarray): target variable
        X_train (pandas.DataFrame, numpy.ndarray, scipy.csr_matrix): train split of X features (transformed)
        X_test (pandas.DataFrame, numpy.ndarray, scipy.csr_matrix): test split of X features (transformed)
        y_train (pandas.Series, numpy.ndarray): train split of target variable
        y_test (pandas.Series, numpy.ndarray): test split of target variable
        problem (str): string representing type of a problem, one of _classification, _regression or _multiclass
            attributes
        random_state (int, None): integer for reproducibility on fitting and transformations, defaults to None if not
            provided during __init__
        scoring_functions (list): scoring functions specific to a given problem type
        default_models (dict): dictionary of 'Model class': param_grid dict pairs specific to a given problem type
        default_scoring (function): default scoring to be used specific to a given problem type
        target_transformer (sklearn.Transformer): Transformer used to transform y target variable in regression
    """
    _classification = "classification"
    _regression = "regression"
    _multiclass = "multiclass"
    _quicksearch_limit = 3   # how many Models are chosen in quicksearch

    _scoring_classification = [accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score]
    _scoring_regression = [mean_squared_error, mean_absolute_error, explained_variance_score, r2_score]
    _scoring_multiclass_parametrized = [
        (f1_score, {"average": "micro"}, "f1_score_micro"),
        (f1_score, {"average": "weighted"}, "f1_score_weighted"),
        (precision_score, {"average": "weighted"}, "precision_score_weighted"),
        (recall_score, {"average": "weighted"}, "recall_score_weighted"),
    ]
    _scoring_multiclass = [accuracy_score, balanced_accuracy_score]

    # hardcoded strings and parameters
    _model_name = "model"
    _fit_time_name = "fit_time"
    _params_name = "params"
    _transformed_target_name = "TransformedTargetRegressor__transformer"

    _mode_quick = "quick"
    _mode_detailed = "detailed"
    _modes = [_mode_quick, _mode_detailed]
    _target_categorical = "categorical"
    _target_numerical = "numerical"
    _target_categories = [_target_categorical, _target_numerical]
    _probas_functions = ["roc_auc_score"]

    def __init__(self, X, y, X_train, X_test, y_train, y_test, target_type, random_state=None):
        """Create ModelFinder object with provided X and y arguments.

        Set default values of attributes and create dummy model depending on target_type.

        Args:
            X (pandas.DataFrame, numpy.ndarray, scipy.csr_matrix): X feature space (transformed)
            y (pandas.Series, numpy.ndarray): target variable
            X_train (pandas.DataFrame, numpy.ndarray, scipy.csr_matrix): train split of X features (transformed)
            X_test (pandas.DataFrame, numpy.ndarray, scipy.csr_matrix): test split of X features (transformed)
            y_train (pandas.Series, numpy.ndarray): train split of target variable
            y_test (pandas.Series, numpy.ndarray): test split of target variable
            target_type (str): string representing type of a problem to which Models should be created
            random_state (int, optional): integer for reproducibility on fitting and transformations, defaults to None

        Raises:
            ValueError: if target_type is not one of _classification, _regression or _multiclass attributes
        """
        self.random_state = random_state

        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        if target_type in self._target_categories:
            self._set_problem(target_type)
        else:
            raise ValueError("Expected one of the categories: {categories}; got {category}".format(
                categories=", ".join(self._target_categories), category=target_type
            ))

        self._chosen_model = None
        self._chosen_model_params = None
        self._chosen_model_scores = None

        self._search_results = None
        self._search_results_dataframe = None

        self._quicksearch_results = None
        self._gridsearch_results = None

        self._dummy_model, self._dummy_model_scores = self._create_dummy_model()

    ####################################################################
    # =============== Exposed Search and Fit functions =============== #
    ####################################################################

    def search_and_fit(self, models=None, scoring=None, mode=_mode_quick):
        """Search for the Model and set it as the chosen Model.

        Refer to specific functions for further details.

        Args:
            models (list, dict, optional): list or 'model class': param_grid dict pairs, defaults to None
            scoring (func, optional): scoring function with which performance assessment will be made, defaults to None
            mode (str, optional): mode of search to be done, defaults to _mode_quick class attribute

        Returns:
            sklearn.Model: Model with best performance, fit to X and y data
        """
        model = self.search(models, scoring, mode)
        self.set_model(model)
        self.fit()
        return self._chosen_model

    def set_model_and_fit(self, model):
        """Set provided model as a chosen Model of ModelFinder and fit it to X and y data.

        Args:
            model (sklearn.Model): instantiated Model
        """
        self.set_model(model)
        self.fit()

    def search(self, models=None, scoring=None, mode=_mode_quick):
        """Search for Models that have the best performance with provided arguments.

        Models can be:
            - list of initialized models
            - dict of 'Model Class': param_grid of a given model to do the GridSearch on
            - None - default Models collection will be used

        scoring should be a sklearn scoring function. If None is provided, default scoring function will be used.

        mode can be:
            - "quick": search is initially done on all models but with no parameter tuning after which top
                _quick_search_limit Models are chosen and GridSearched with their param_grids
            - "detailed": GridSearch is done on all default models and their params
        Provided mode doesn't matter when models are explicitly provided (not None).

        After initial quicksearch/gridsearch, best model from each class is chosen and have its assessment performed.
        Results of this are set in _search_results and _search_results_dataframe attributes.

        Note:
            random_state of Models will be overriden with random_state attribute

        Args:
            models (list, dict, optional): list of Models or 'Model class': param_grid dict pairs, defaults to None
            scoring (func, optional): sklearn scoring function, defaults to None
            mode ("quick", "detailed", optional): either "quick" or "detailed" string, defaults to "quick"

        Returns:
            sklearn.Model: new (not fitted) instance of the best scoring Model

        Raises:
            ValueError: mode is not "quick" or "detailed"; type of models is not dict, list-like or None
        """
        if scoring is None:
            scoring = self.default_scoring

        if mode not in self._modes:
            raise ValueError("Expected one of the modes: {modes}; got {mode}".format(
                modes=", ".join(self._modes), mode=mode
            ))

        if isinstance(models, dict) or models is None:
            initiated_models = self._search_for_models(models, mode, scoring)
        else:
            try:
                # in case of Str as it doesn't raise TypeError with iter() function
                if isinstance(models, str):
                    raise TypeError
                iter(models)
                initiated_models = [self._wrap_model(single_model) for single_model in models]
            except TypeError:
                raise ValueError("models should be Dict, List-like or None, got {models}".format(models=models))

        scored_and_fitted_models, search_results = self._assess_models_performance(initiated_models, scoring)
        self._search_results = scored_and_fitted_models
        self._search_results_dataframe = self._create_search_results_dataframe(search_results, scoring)

        sorting_order = reverse_sorting_order(obj_name(scoring))
        scored_and_fitted_models.sort(key=lambda x: x[1], reverse=sorting_order)

        best_model = scored_and_fitted_models[0][0]  # first item in the list, first item in the (model, score) tuple
        new_model = best_model.__class__(**best_model.get_params())

        return new_model

    def set_model(self, model):
        """Set model as a chosen ModelFinder model.

        Create additional copy of the model and calculate it's scores with scoring functions specific to a given
        problem.

        Args:
            model (sklearn.Model): Instantiated Model
        """
        model = self._wrap_model(model)
        params = model.get_params()
        copy_for_scoring = model.__class__(**params).fit(self.X_train, self.y_train)
        self._chosen_model = model
        self._chosen_model_params = params
        self._chosen_model_scores = self._score_model(copy_for_scoring, self.default_scoring)

    def best_model(self):
        """Return _chosen_model attribute (chosen Model).

        Returns:
            sklearn.Model
        """
        return self._chosen_model

    def fit(self):
        """Fit chosen Model from _chosen_model attribute on X and y data.

        Raises:
            ModelNotSetError: when no Model was set as a chosen Model
        """
        if self._chosen_model is None:
            raise ModelNotSetError(
                "Model needs to be set before fitting. Call 'set_model' or 'search' for a model before trying to fit."
            )
        self._chosen_model.fit(self.X, self.y)

    def predict(self, X):
        """Predict target variable from provided X features.

        Returns:
            numpy.ndarray: predicted target values from X

        Raises:
            ModelNotSetError: when no Model was set as a chosen Model
        """
        if self._chosen_model is None:
            raise ModelNotSetError(
                "Model needs to be set and fitted before prediction. Call 'set_model' or 'search' for a model before."
            )
        return self._chosen_model.predict(X)

    def quicksearch_results(self):
        """Return quicksearch results from _quicksearch_results attribute.

        Returns:
            pandas.DataFrame
        """
        return self._quicksearch_results

    def gridsearch_results(self):
        """Return gridsearch results from _gridsearch_results attribute.

        Returns:
            pandas.DataFrame
        """
        return self._gridsearch_results

    #########################################################################
    # =============== Visualization Data for View functions =============== #
    #########################################################################

    def search_results(self, model_limit):
        """Return detailed search results DataFrame from _search_results_dataframe.

        model_limit restricts the number of Models and their results to be returned. Number of rows in the DataFrame
        is always model_limit + 1, as results from Dummy Model are being appended at the end.

        Args:
            model_limit (int): number of rows to be returned

        Returns:
            pandas.DataFrame: search results

        Raises:
            ModelsNotSearchError: when no search and performance assessment between Models happened
        """
        if self._search_results_dataframe is None:
            raise ModelsNotSearchedError("Search Results is not available. Call 'search' to obtain comparison models.")

        # dummy is always included, regardless of model limit
        models = self._search_results_dataframe.iloc[:model_limit]
        dummy = self._dummy_model_results()

        df = pd.concat([models, dummy], axis=0).set_index(self._model_name)
        return df

    def dataframe_params_name(self):
        """Return params column name.

        Returns:
            str
        """
        return self._params_name

    def test_target_proportion(self):
        """Calculate and return proportion of positive label (1) in target variable when compared with all observations.

        Note:
            Meaningless results are returned when used in regression or multiclass problems.

        Returns:
            float: proportion of positive label in target variable
        """
        # only meaningful in binary classification
        return self.y_test.sum() / self.y_test.shape[0]

    def roc_curves(self, model_limit):
        """Return data for ROC curves for model_limit # of Models from search results.

        Note:
            Useful only in classification.

        Args:
            model_limit (int): number of Models from search results

        Returns:
            list: list of tuples - (model, data for curve)
        """
        return self._plot_curves(roc_curve, model_limit)

    def precision_recall_curves(self, model_limit):
        """Return data for Precision-Recall curves for model_limit # of Models from search results.

        Args:
            model_limit (int): number of Models from search results

        Note:
            Useful only in classification.

        Returns:
            list: list of tuples - (model, data for curve)
        """
        return self._plot_curves(precision_recall_curve, model_limit)

    def det_curves(self, model_limit):
        """Return data for Detection-Error Tradeoff curves for model_limit # of Models from search results.

        Note:
            Useful only in classification.

        Args:
            model_limit (int): number of Models from search results

        Returns:
            list: list of tuples - (model, data for curve)
        """
        return self._plot_curves(det_curve, model_limit)

    def confusion_matrices(self, model_limit):
        """Return data for Confusion Matrices for model_limit # of Models from search results.

        Note:
            Useful only in classification/multiclass.

        Args:
            model_limit (int): number of Models from search results

        Returns:
            list: list of tuples - (model, data for confusion matrix)

        Raises:
            ModelsNotSearchedError: when no search and performance assessment between Models happened
        """
        if self._search_results_dataframe is None:
            raise ModelsNotSearchedError("Search Results is not available. Call 'search' to obtain comparison models.")

        models = [tp[0] for tp in self._search_results[:model_limit]]

        _ = []
        for model in models:
            _.append((model, confusion_matrix(self.y_test, model.predict(self.X_test))))
        return _

    def prediction_errors(self, model_limit):
        """Return data for Prediction Errors for model_limit # of Models from search results.

        Note:
            Useful only in regression.

        Args:
            model_limit (int): number of Models from search results

        Returns:
            list: list of tuples - (model, prediction errors)

        Raises:
            ModelsNotSearchedError: when no search and performance assessment between Models happened
        """
        if self._search_results_dataframe is None:
            raise ModelsNotSearchedError("Search Results is not available. Call 'search' to obtain comparison models.")

        models = [tp[0] for tp in self._search_results[:model_limit]]
        _ = []
        for model in models:
            _.append((model, (self.y_test, model.predict(self.X_test))))

        return _

    def residuals(self, model_limit):
        """Return data for Residuals for model_limit # of Models from search results.

        Note:
            Useful only in regression.

        Args:
            model_limit (int): number of Models from search results

        Returns:
            list: list of tuples - (model, residuals data)

        Raises:
            ModelsNotSearchedError: when no search and performance assessment between Models happened
        """
        if self._search_results_dataframe is None:
            raise ModelsNotSearchedError("Search Results is not available. Call 'search' to obtain comparison models.")

        models = [tp[0] for tp in self._search_results[:model_limit]]
        _ = []
        for model in models:
            predictions = model.predict(self.X_test)
            _.append((model, (predictions, predictions - self.y_test)))

        return _

    def predictions_X_test(self, model_limit):
        """Return prediction data of X_test split for model_limit # of Models from search results.

        Args:
            model_limit (int): number of Models from search results

        Returns:
            list: list of tuples - (model, predictions)

        Raises:
            ModelsNotSearchedError: when no search and performance assessment between Models happened
        """
        if self._search_results_dataframe is None:
            raise ModelsNotSearchedError("Search Results is not available. Call 'search' to obtain comparison models.")

        models = [tp[0] for tp in self._search_results[:model_limit]]
        _ = []
        for model in models:
            _.append((model, model.predict(self.X_test)))

        return _

    ######################################################
    # =============== Problem assessment =============== #
    ######################################################

    def _set_problem(self, problem_type):
        """Set different instance attributes depending on the problem_type provided.

        Args:
            problem_type (str): string representing type of a problem
        """
        if problem_type == self._target_categorical:

            # multiclass
            if len(np.unique(self.y)) > 2:
                self.problem = self._multiclass
                self.scoring_functions = self._create_scoring_multiclass()
                self.default_models = classifiers
                self.default_scoring = self.scoring_functions[0]

            # binary classification
            else:
                self.problem = self._classification
                self.scoring_functions = self._scoring_classification
                self.default_models = classifiers
                self.default_scoring = roc_auc_score

        # regression
        elif problem_type == self._target_numerical:
            self.problem = self._regression
            self.scoring_functions = self._scoring_regression
            self.default_models = regressors
            self.default_scoring = mean_squared_error
            self.target_transformer = QuantileTransformer(output_distribution="normal", random_state=self.random_state)

    ####################################################
    # =============== Search Functions =============== #
    ####################################################

    def _search_for_models(self, models, mode, scoring):
        """Assess Models in regard to their performance with provided scoring function.

        If models are provided as 'Model class': param_grid dict pairs in a dictionary, do gridsearch on provided
        parameters. If models is None, default models will be used depending on provided mode.

        "quick" mode makes the assessment happen first on Models initiated with 'default' params and some number of
        the best performing Models classes are returned and gridsearched. "detailed" mode makes the search happen across
        all default Models and their stated param_grids.

        Args:
            models (dict, None): dictionary of 'Model class': param_grid dict pairs or None
            mode ("quick", "detailed"): mode determining the manner in which search should happen, only relevant when
                models is None or dict
            scoring (function): sklearn scoring function

        Returns:
            list: Models from Models classes instanced with parameters that got the best results in search.

        Raises:
            ValueError: mode is not "quick" or "detailed"; type of models is not dict, list-like or None
        """
        if isinstance(models, dict):
            gridsearch_models = self._gridsearch(models, scoring)

        elif models is None:
            # filtering warnings from gridsearch/quicksearch (e.g. models not converging)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if mode == self._mode_quick:
                    chosen_models = self._quicksearch(self.default_models.keys(), scoring)
                    param_grid = {clf: self.default_models[clf] for clf in chosen_models}
                    gridsearch_models = self._gridsearch(param_grid, scoring)
                elif mode == self._mode_detailed:
                    gridsearch_models = self._gridsearch(self.default_models, scoring)
                else:
                    # this branch shouldn't be possible without explicit class/object properties manipulation
                    raise ValueError("models should be Dict or None, got {models}".format(models=models))
        else:
            raise ValueError("Expected one of the modes: {modes}; got {mode}".format(
                modes=", ".join(self._modes), mode=mode
            ))

        initiated_models = [self._wrap_model(model(**params)) for model, params in gridsearch_models]

        return initiated_models

    def _gridsearch(self, models_param_grid, scoring):
        """Perform gridsearch and update _gridsearch_results attribute with results DataFrame.

        Args:
            models_param_grid (dict): 'Model class': param_grid dict pairs
            scoring (function): sklearn scoring function

        Returns:
            list: 2-element tuples (Model class, params to instance the Model from Model Class) sorted with the best
                performer in the beginning
        """
        chosen_models, all_results = self._perform_gridsearch(models_param_grid, scoring)
        self._gridsearch_results = self._wrap_results_dataframe(self._create_gridsearch_results_dataframe(all_results))
        return chosen_models

    def _perform_gridsearch(self, models_param_grid, scoring, cv=5):
        """Perform gridsearch on provided models_param_grid and return results.

        Gridsearch is performed on provided Models space ('Models class': param_grid dict pairs) with scoring used
        as a sole metric to decide which Models perform well and which perform poor. HalvingGridSearch is used
        instead of a regular GridSearch to possibly save time.

        Note:
            GridSearch might fail with NotFittedError - All Models failed to fit. This might happen sometimes when
            parameters for the same type of Model are provided wrongly (e.g. DecisionTreeClassifier instanced with
            criterion: "mae" which is used in DecisionTreeRegressor. Changing param_grid solves this issue.

        Args:
            models_param_grid (dict): 'Model class': param_grid dict pairs
            scoring (function): sklearn scoring function
            cv (int, optional): number of folds, defaults to 5

        Returns:
            tuple: (
                list of (Model class, best params for the Model class) tuples
                dict of 'Model class': cv_results_ from GridSearch object
                )

        Raises:
            NotFittedError: when GridSearch object raises NotFittedError
        """
        all_results = {}
        best_of_their_class = []

        for model, params in models_param_grid.items():

            # not every model requires random_state, e.g. KNeighborsClassifier
            if "random_state" in model().get_params().keys():
                params["random_state"] = [self.random_state]

            # https://scikit-learn.org/stable/modules/model_evaluation.html#defining-your-scoring-strategy-from-metric-functions
            sorting_order = reverse_sorting_order(obj_name(scoring))

            created_model = self._wrap_model(model())

            # GridSearch will fail with NotFittedError("All estimators failed to fit") when argument provided
            # in the param grid is incorrect for a given model (even one combination will trigger it).
            clf = HalvingGridSearchCV(
                created_model,
                params,
                scoring=make_scorer(scoring, greater_is_better=sorting_order),
                cv=cv,
                error_score=0,  # to ignore errors that might happen,
                random_state=self.random_state
            )
            try:
                clf.fit(self.X_train, self.y_train)
            except NotFittedError:
                # printing out warning for user as the NotFittedError might be misleading in this case
                params_str = ["{}: {}".format(key, item) for key, item in params.items()]
                warn_msg = "WARNING: {} might potentially have incorrect params provided: {}".format(
                    model, params_str
                )
                warnings.warn(warn_msg)
                raise

            all_results[model] = clf.cv_results_
            best_of_their_class.append((model, clf.best_params_))

        return best_of_their_class, all_results,

    def _create_gridsearch_results_dataframe(self, cv_results):
        """Create DataFrame with GridSearch results from cv_results dictionary of results.

        Args:
            cv_results (dict): 'Model class': results dict pairs

        Returns:
            pandas.DataFrame: DataFrame with gridsearch results
        """
        df = None
        for model in cv_results.keys():
            single_results = pd.DataFrame(cv_results[model])
            single_results[self._model_name] = obj_name(model)  # column with Model name
            df = pd.concat([df, single_results], axis=0)

        df = df.reset_index().drop(["index"], axis=1)
        return df

    def _quicksearch(self, models, scoring):
        """Perform quicksearch and update _quicksearch_results attribute with results DataFrame.

        Returned Models list is truncated depending on _quicksearch_limit class attribute.

        Args:
            models (list): list of Model classes
            scoring (function): sklearn scoring function

        Returns:
            list: Model class list, with the best performer in the beginning
        """
        scored_models, all_results = self._perform_quicksearch(models, scoring)
        results_df = self._create_search_results_dataframe(all_results, scoring)
        self._quicksearch_results = self._wrap_results_dataframe(results_df)

        sorting_order = reverse_sorting_order(obj_name(scoring))
        scored_models.sort(key=lambda x: x[1], reverse=sorting_order)
        return [model[0] for model in scored_models][:self._quicksearch_limit]

    def _perform_quicksearch(self, models, scoring):
        """Assess performance of Models created with default parameters and return the results.

        Quicksearch works in a similar manner to LazyPredict package - Models are created with their default parameters
        and score is calculated with sklearn scoring function. The better the score, the better the Model.

        Args:
            models (list): list of Model classes
            scoring (function): sklearn scoring function

        Returns:
            tuple: 2-element tuple of (
                list of tuples (Models Class, scoring score),
                dict of Model class: dict of quicksearch results pairs
                )
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X_train, self.y_train, random_state=self.random_state)
        all_results = {}
        scored_models = []

        for model in models:
            if "random_state" in model().get_params().keys():
                clf = model(random_state=self.random_state)
            else:
                clf = model()
            clf = self._wrap_model(clf)

            start_time = time.time()
            clf.fit(X_train, y_train)
            stop_time = time.time()

            score = self._calculate_model_score(clf, X_test, y_test, scoring)
            params = clf.get_params()

            all_results[model] = {
                self._fit_time_name: stop_time-start_time,
                obj_name(scoring): score,
                self._params_name: params
            }
            scored_models.append((model, score))

        return scored_models, all_results

    def _create_search_results_dataframe(self, results, chosen_scoring):
        """Create pandas.DataFrame from quicksearch results and return it.

        DataFrame is sorted with the best scoring Model put on top.

        Args:
            results (dict): 'Model class': results pairs
            chosen_scoring (function): sklearn scoring function

        Returns:
            pandas.DataFrame: DataFrame with quicksearch results

        """
        data = defaultdict(list)
        for model, values in results.items():
            data[self._model_name].append(obj_name(model))
            for key, val in values.items():
                data[key].append(val)

        sorting = not reverse_sorting_order(obj_name(chosen_scoring))  # reverse of reverse for DataFrame sorting
        return pd.DataFrame(data).sort_values(by=[obj_name(chosen_scoring)], ascending=sorting)

    def _assess_models_performance(self, initiated_models, chosen_scoring):
        """Assess Models performance and return them in the order based on achieved scores.

        In comparison to quicksearch or gridsearch methods, Models must be already instanced with some parameters
        to have its score calculated.

        Note:
            Models are fit with X_train and y_train splits, but are scored on X_test and y_test splits attributes.

        Args:
            initiated_models (list): instances of Models
            chosen_scoring (function): sklearn scoring function

        Returns:
            tuple: (
                list of tuples ('Model instance', scores of chosen scoring),
                dict of 'Model instances': all results pairs
                )
        """
        all_results = {}
        fitted_and_scored = []

        for model in initiated_models:
            with warnings.catch_warnings():  # ignoring warnings, esp QuantileTransformer for Wrapper in Regression
                warnings.simplefilter("ignore")
                start_time = time.time()
                model.fit(self.X_train, self.y_train)
                stop_time = time.time()

            model_results = {
                self._fit_time_name: stop_time-start_time,
                self._params_name: self._wrap_params(model.get_params())
            }
            score_results = self._score_model(model, chosen_scoring)
            model_results.update(score_results)

            all_results[model] = model_results
            fitted_and_scored.append((model, score_results[obj_name(chosen_scoring)]))

        return fitted_and_scored, all_results

    #####################################################
    # =============== Scoring Functions =============== #
    #####################################################

    def _score_model(self, fitted_model, chosen_scoring):
        """Score Model with all default scoring functions of a given problem type (+) chosen_scoring on X_test and
        y_test splits and return the result.

        Args:
            fitted_model (sklearn.Model): Model that was already fit to X_train/y_train data
            chosen_scoring (function): sklearn scoring function

        Returns:
            dict: 'scoring function': result pairs
        """
        scorings = self._get_scorings(chosen_scoring)
        scoring_results = {}
        for scoring in scorings:
            score = self._calculate_model_score(fitted_model, self.X_test, self.y_test, scoring)
            scoring_results[obj_name(scoring)] = score

        return scoring_results

    def _get_scorings(self, chosen_scoring):
        """Check if chosen_scoring is already in list of default scoring functions and add it or not to have it
        included in the list, but not duplicated.

        Args:
            chosen_scoring (function): sklearn scoring function

        Returns:
            list: functions scoring_functions attribute with chosen_scoring added if it isn't there
        """
        if chosen_scoring in self.scoring_functions:
            scorings = self.scoring_functions
        else:
            scorings = [chosen_scoring] + self.scoring_functions

        return scorings

    def _calculate_model_score(self, model, X, y_true, scoring_function):
        """Calculate Model score based on the type of scoring_function - if it requires simple '0-1' predictions
        or calculated probabilities.

        scoring_function is checked if it's defined in _probas_function class attribute - container of pre-defined
        functions that might require probabilities. If it is, then probabilities are used as predictions. Otherwise,
        simple predict call is used.

        Args:
            model (sklearn.Model): Model which performance will be assessed
            X (pandas.DataFrame, numpy.ndarray, scipy.csr_matrix): X feature space on which predictions will happen
            y_true (pandas.Series, numpy.ndarray): true target variable which will be compared with predictions
            scoring_function (function): sklearn scoring function

        Returns:
            float: score of scoring_function
        """
        # check if scoring_function is defined in _probas_function class attribute
        if any((func == obj_name(scoring_function) for func in self._probas_functions)):
            try:
                predictions = model.predict_proba(X)
                if self.problem == self._classification:
                    predictions = predictions[:, 1]
            except AttributeError:
                predictions = model.decision_function(X)  # decision_function returns 1d result
        else:
            predictions = model.predict(X)
        score = scoring_function(y_true, predictions)
        return score

    def _create_scoring_multiclass(self):
        """Create scoring functions for multiclass problem with some of the parameters pre-set.

        Some functions used for multiclass scoring require different arguments to be called with when assessing the
        score of multiclass problems. They aren't called up until the point of scoring (late in the process) so
        it's more convenient to provide the necessary arguments beforehand with creating closures.

        Returns:
            list: functions used for scoring multiclass problems (both regular and enclosed in closures)
        """
        # (roc_auc_score, {"average": "weighted", "multi_class": "ovr"})
        # multiclass roc_auc_score requires probabilities for every class in comparison to simple predictions
        # required by other metrics, not included cause it breaks program flow
        scorings = []
        for scoring, params, fname in self._scoring_multiclass_parametrized:
            def closure():
                func_scoring = scoring
                func_params = params
                func_name = fname

                def make_scoring(y_true, y_score):
                    make_scoring.__name__ = func_name  # name change for results logging
                    return func_scoring(y_true, y_score, **func_params)
                return make_scoring
            scorings.append(closure())

        scorings += self._scoring_multiclass  # appending functions that do not require specific parameters
        return scorings

    ######################################################
    # =============== Wrapping Functions =============== #
    ######################################################

    def _wrap_model(self, model):
        """Wrap Model in WrappedModelRegression if the problem type is regression, return unchanged model otherwise.

        Args:
            model (sklearn.Model): model to be wrapped

        Returns:
            WrappedModelRegression, sklearn.Model: WrappedModelRegression object if regression, model otherwise
        """
        if self.problem == self._regression and type(model) != WrappedModelRegression:
            wrapped_model = WrappedModelRegression(regressor=model, transformer=self.target_transformer)
            return wrapped_model
        else:
            return model

    def _wrap_results_dataframe(self, df):
        """Add another column to results DataFrame indicating params of TransformedTargerRegressor used in regression
        problem.

        Return unchanged df if the problem is different from regression.

        Args:
            df (pandas.DataFrame): results DataFrame

        Returns:
            pandas.DataFrame: df DataFrame with one column added in regression or unchanged df otherwise
        """
        if self.problem == self._regression:
            df[self._transformed_target_name] = str(self.target_transformer.get_params())

        return df

    def _wrap_params(self, params):
        """Update params dictionary with parameters of TransformedTargetRegressor transformer used in regression
        problem.

        Return unchanged params dictionary if the problem is different from regression.

        Args:
            params (dict): dictionary of 'param': value pairs

        Returns:
            dict: updated params dict in regression or unchanged params otherwise
        """
        if self.problem == self._regression:
            params[self._transformed_target_name] = obj_name(self.target_transformer)  # name of the Transformer used
            transformer_params = self.target_transformer.get_params()  # params of the Transformer
            new_params = {self._transformed_target_name + "__" + key: item for key, item in transformer_params.items()}
            params.update(new_params)

        return params

    ######################################################
    # =============== Visualization Data =============== #
    ######################################################

    def _plot_curves(self, plot_func, model_limit):
        """Calculate data needed to plot performance curve from plot_func.

        Functions calculating performance curves require probas instead of '0-1' predictions. As not every Model has
        predict_probas method, decision_function method is also tried to get the probabilities.

        Args:
            plot_func (function): sklearn function that calculates curve data
            model_limit (int): number of Models from search results

        Returns:
            list: tuples of (Model, calculated curve data)

        Raises:
            ModelsNotSearchedError: when no search and performance assessment between Models happened
        """
        if self._search_results is None:
            raise ModelsNotSearchedError("Search Results is not available. Call 'search' to obtain comparison models.")

        curves = []
        models = [tp[0] for tp in self._search_results[:model_limit]]
        for model in models:
            try:
                y_score = model.predict_proba(self.X_test)
                # https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/metrics/_plot/base.py#L104
                # predictions for class 1 in a binary setting - second column
                y_score = y_score[:, 1]
            except AttributeError:
                y_score = model.decision_function(self.X_test)

            result = plot_func(self.y_test, y_score)
            curves.append((model, result))

        return curves

    ###############################################
    # =============== Dummy Model =============== #
    ###############################################

    def _create_dummy_model(self):
        """Create dummy Model for a given problem type and calculate default scores for it.

        Note:
            Dummy Model is scored in the beginning with a default set of functions to prevent errors from custom
            scoring function provided that Dummy Model might not be able to handle. This is a potential improvement to
            be made during future implementations.

        Returns:
            tuple: (Dummy Model fit for X_train/y_train splits, default scoring results)

        Raises:
            ValueError: problem attribute is different than expected

        """
        if self.problem == self._classification or self.problem == self._multiclass:
            model = DummyClassifier(strategy="stratified", random_state=self.random_state)
        elif self.problem == self._regression:
            model = DummyRegressor(strategy="median")
        else:
            raise ValueError("Provided problem type: {problem}, expected one of {expected_problems}.".format(
                problem=self.problem,
                expected_problems=", ".join([self._classification, self._multiclass, self._regression])
            ))

        model.fit(self.X_train, self.y_train)
        results = self._score_model(model, self.default_scoring)
        return model, results

    def _dummy_model_results(self):
        """Create Dummy Model results DataFrame from _dummy_model_scores attribute.

        Returns:
            pandas.DataFrame: DataFrame with Dummy Model score results
        """
        _ = {
            self._model_name: obj_name(self._dummy_model),
            self._fit_time_name: np.nan,
            self._params_name: str(self._dummy_model.get_params()),
            **self._dummy_model_scores
        }
        return pd.DataFrame(_, index=[9999])
