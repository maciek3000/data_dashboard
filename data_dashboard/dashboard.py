from .analyzer import Analyzer
from .features import Features
from .output import Output
from .transformer import Transformer
from .model_finder import ModelFinder
from .descriptor import FeatureDescriptor
from .plot_design import PlotDesign
from .functions import sanitize_input, make_pandas_data, obj_name
import os
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import pandas as pd
import random
import warnings
import copy
import pathlib
import webbrowser


class Dashboard:
    """Main object of the package.

        The role of this class is to analyze the provided data (in terms of summary statistics, visualizing any
        "easy" correlations, showing any potential new features, etc.), to transform the data for Machine Learning
        models and then train and find the best model it could find, additionally providing all the metadata on it.
        The end-goal is to create a series of static HTML files, which can be accessed like a dashboard to easily
        navigate through them in search for any potential insights in the machine learning pipeline.

        Coordinator encompasses all the classes that are used in creating the analysis and finding a model of
        given data. Coordinator exposes several functions that might be called separately (if there is a need
        for just a singular task, e.g. creating a dashboard for analyzing the data), but it also exposes a
        .full_analysis() function which tries to build the whole analysis from the ground up.
    """

    _name = "data_dashboard"
    _output_created_text = "Created output at {directory}"
    _model_found_text = "Model: {name}\nScore: {score}\nParams: {params}"

    _n_features_pairplots_limit = 15

    def __init__(self, output_directory, X, y, feature_descriptions_dict=None, root_path=None, random_state=None,
                 classification_pos_label=None, force_classification_pos_label_multiclass=False,
                 already_transformed_columns=None):

        # if root_path is None:
        #     self.root_path = os.getcwd()
        # else:
        #     self.root_path = root_path

        self._create_pairplots_flag = True
        self._force_classification_pos_label_multiclass_flag = force_classification_pos_label_multiclass

        self.random_state = random_state

        self.X, self.y = self._check_provided_data(X, y)

        if classification_pos_label is not None:
            classification_pos_label = self._check_classification_pos_label(classification_pos_label)

        self.already_transformed_columns = self._check_transformed_cols(already_transformed_columns)
        self.output_directory = self._check_output_directory(output_directory)

        self.transformed_X = None
        self.transformed_y = None
        self.model_finder = None
        self.output = None

        self.plot_design = PlotDesign()
        self.features_descriptions = FeatureDescriptor(feature_descriptions_dict)
        self.features = Features(self.X, self.y, self.features_descriptions, self.already_transformed_columns)
        self.analyzer = Analyzer(self.features)

        # dropping features assessed as unusable
        self.X = self.X.drop(self.features.unused_features(), axis=1)
        self._assess_n_features(self.X)

        # excluding Transformed columns from transformations, they will be concatted at the end because of
        # remainder='passthrough' argument of ColumnTransformer
        self.transformer = Transformer(
            categorical_features=self.features.categorical_features(drop_target=True, exclude_transformed=True),
            numerical_features=self.features.numerical_features(drop_target=True, exclude_transformed=True),
            target_type=self.features[self.features.target].feature_type,
            random_state=self.random_state,
            classification_pos_label=classification_pos_label
        )

        self.transformer_eval = copy.deepcopy(
            self.transformer)  # provided to Output as it creates dashboard with splits

        # https://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics
        # Just as it is important to test a predictor on data held-out from training, preprocessing
        # (such as standardization, feature selection, etc.) and similar data transformations similarly should be
        # learnt from a training set and applied to held-out data for prediction.

        self._create_test_splits()
        self._do_transformations()
        self._initialize_model_and_output()

    def search_and_fit(self, models=None, scoring=None, mode="quick"):
        if scoring is None:
            scoring = self.model_finder.default_scoring
        clf = self.model_finder.search_and_fit(models, scoring, mode)
        return clf

    def set_and_fit(self, model):
        self.model_finder.set_model_and_fit(model)

    def predict(self, X):
        output = self.model_finder.predict(X)
        return output

    def create_dashboard(self, models=None, scoring=None, mode="quick", logging=True, disable_pairplots=False,
                         force_pairplot=False):
        """Creates several Views (Subpages) and joins them together to form an interactive WebPage/Dashboard.
        Every view is static - there is no server communication between html template and provided data. Every
        possible interaction is created with CSS/HTML or JS. This was a conscious design choice - albeit much slower
        than emulating server interaction, files can be easily shared between parties of interest.
        Keep in mind that basically all the data and the calculations are embedded into the files - if you'd wish
        to post them on the server you have to be aware if the data itself can be posted for public scrutiny."""
        # force_pairplot - useful only when n of features > limit
        # disable_pairplots - disables pairplots in the dashboard, takes precedence over force_pairplot

        if scoring is None:
            scoring = self.model_finder.default_scoring
        clf = self.model_finder.search_and_fit(models, scoring, mode)
        print("Found model: {clf}".format(clf=obj_name(clf)))
        print("Creating Dashboard...")

        if disable_pairplots:
            do_pairplots = False
        else:
            do_pairplots = self._create_pairplots_flag
            if not do_pairplots and force_pairplot:
                do_pairplots = True

        # do_transformations = not self._custom_transformer_flag
        do_logging = logging

        self.output.create_html(
            do_pairplots=do_pairplots,
            do_logs=do_logging
        )
        print(self._output_created_text.format(directory=self.output.output_directory))
        webbrowser.open_new(self.output.overview_file())

    def set_custom_transformers(self, categorical_transformers=None, numerical_transformers=None, y_transformers=None):
        for tr in [self.transformer, self.transformer_eval]:
            tr.set_custom_preprocessor_X(
                numerical_transformers=numerical_transformers,
                categorical_transformers=categorical_transformers
            )
            if y_transformers:
                tr.set_custom_preprocessor_y(y_transformers)

        self._do_transformations()
        self._initialize_model_and_output()

    # exposed method in case only transformation is needed
    def transform(self, X):
        return self.transformer.transform(X)

    def transform_predict(self, X):
        transformed = self.transform(X)
        return self.predict(transformed)

    def best_model(self):
        return self.model_finder.best_model()

    def _do_transformations(self):
        self._fit_transform_test_splits()
        self._fit_transformer()
        self.transformed_X = self.transformer.transform(self.X)
        self.transformed_y = self.transformer.transform_y(self.y)

    def _initialize_model_and_output(self):
        self.model_finder = ModelFinder(
            X=self.transformed_X,
            y=self.transformed_y,
            X_train=self.transformed_X_train,
            X_test=self.transformed_X_test,
            y_train=self.transformed_y_train,
            y_test=self.transformed_y_test,
            target_type=self.features[self.features.target].feature_type.lower(),
            random_state=self.random_state
        )

        self.output = Output(
            output_directory=self.output_directory,
            package_name=self._name,
            pre_transformed_columns=self.already_transformed_columns,
            features=self.features,
            analyzer=self.analyzer,
            transformer=self.transformer_eval,
            model_finder=self.model_finder,
            X_train=self.X_train,
            X_test=self.X_test,
            y_train=self.y_train,
            y_test=self.y_test,
            transformed_X_train=self.transformed_X_train,
            transformed_X_test=self.transformed_X_test,
            transformed_y_train=self.transformed_y_train,
            transformed_y_test=self.transformed_y_test,
            random_state=self.random_state
        )

    def _create_test_splits(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=self.random_state)

        # resetting index so it can be joined later on with test predictions
        output = []
        for d in [X_train, X_test, y_train, y_test]:
            new_d = d.reset_index(drop=True)
            output.append(new_d)

        self.X_train, self.X_test, self.y_train, self.y_test = output

    def _fit_transform_test_splits(self):
        # fitting only on train data
        self.transformer_eval.fit(self.X_train)
        self.transformer_eval.fit_y(self.y_train)

        transformed = (
            self.transformer_eval.transform(self.X_train),
            self.transformer_eval.transform(self.X_test),
            self.transformer_eval.transform_y(self.y_train),
            self.transformer_eval.transform_y(self.y_test)
        )

        self.transformed_X_train, self.transformed_X_test, self.transformed_y_train, self.transformed_y_test = transformed

    def _fit_transformer(self, X=None, y=None):
        if X is None:
            X = self.X
        if y is None:
            y = self.y

        self.transformer.fit(X)
        self.transformer.fit_y(y)

    def _check_provided_data(self, X, y):

        new_X = make_pandas_data(X, pd.DataFrame)
        new_y = make_pandas_data(y, pd.Series)

        new_X.columns = sanitize_input(new_X.columns)
        new_y.name = sanitize_input([new_y.name])[0]

        return new_X, new_y

    def _check_classification_pos_label(self, label):
        unique = set(np.unique(self.y))
        if label not in unique:
            raise ValueError(
                "label '{label}' not in unique values of y: {values}".format(label=label, values=str(unique))
            )

        if len(unique) > 2 and not self._force_classification_pos_label_multiclass_flag:
            warnings.warn("n of unique values in y is > 2, classification_pos_label will be ignored. "
                          "Provide force_classification_pos_label_multiclass=True to force classification_pos_label.")
            return None
        else:
            return label

    def _assess_n_features(self, df):
        n = df.shape[1]
        if n > self._n_features_pairplots_limit:
            self._create_pairplots_flag = False
            warnings.warn(
                "Number of features crossed their default limit - pairplots will be turned off to reduce"
                " runtime and/or to avoid MemoryErrors. To force pairplots to be created, call 'create_html' with"
                " 'force_pairplots=True' argument.")

    def _check_transformed_cols(self, transformed_columns):
        if transformed_columns is not None:
            cols_in_data = set(self.X.columns)
            transformed = set(transformed_columns)

            if not transformed <= cols_in_data:  # checking if transformed is a subset of cols_in_data
                raise ValueError("Provided transformed_columns: {transformed} are not a subset"
                                 " of columns in data: {columns}".format(
                    transformed=transformed,
                    columns=cols_in_data
                )
                )
            else:
                return sorted(transformed)
        else:
            return []

    def _check_output_directory(self, directory):
        pathlib.Path(directory).mkdir(exist_ok=True, parents=True)
        return directory
