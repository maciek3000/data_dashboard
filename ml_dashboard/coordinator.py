from .analyzer import Analyzer
from .features import Features
from .output import Output
from .transformer import Transformer
from .model_finder import ModelFinder
from .descriptor import FeatureDescriptor
from .plot_design import PlotDesign
from .functions import sanitize_input, make_pandas_data
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import warnings
import copy




class Coordinator:
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

    _name = "ml_dashboard"
    _output_created_text = "Created output at {directory}"
    _model_found_text = "Model: {name}\nScore: {score}\nParams: {params}"

    def __init__(self, X, y, output_directory, scoring=None, feature_descriptions_dict=None, root_path=None, random_state=None,
                 classification_pos_label=None):

        self.random_state = random_state
        self.output_directory = output_directory

        self.X, self.y = self._check_provided_data(X, y)

        if classification_pos_label is not None:
            classification_pos_label = self._check_classification_pos_label(classification_pos_label)

        self.scoring = scoring

        if root_path is None:
            self.root_path = os.getcwd()
        else:
            self.root_path = root_path

        self.transformed_X = None
        self.transformed_y = None
        self.model_finder = None
        self.output = None

        self.plot_design = PlotDesign()
        self.features_descriptions = FeatureDescriptor(feature_descriptions_dict)
        self.features = Features(self.X, self.y, self.features_descriptions)
        self.analyzer = Analyzer(self.features)

        self.transformer = Transformer(
            categorical_features=self.features.categorical_features(drop_target=True),
            numerical_features=self.features.numerical_features(drop_target=True),
            target_type=self.features[self.features.target].type,
            random_state=self.random_state,
            classification_pos_label=classification_pos_label
        )

        self.transformer_eval = copy.deepcopy(self.transformer)  # provided to Output as it creates dashboard with splits

        # https://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics
        # Just as it is important to test a predictor on data held-out from training, preprocessing
        # (such as standardization, feature selection, etc.) and similar data transformations similarly should be
        # learnt from a training set and applied to held-out data for prediction.

        self._create_test_split()
        self._transform_test_splits()
        # X_train, X_test, y_train, y_test = self._create_test_split()
        # transformed_X_train, transformed_X_test, transformed_y_train, transformed_y_test = self._transform_splits(
        #     X_train, X_test, y_train, y_test
        # )

        self._fit_transformer()
        self.transformed_X = self.transformer.transform(self.X)
        self.transformed_y = self.transformer.transform_y(self.y)

        self._initialize_model_and_output()

    def search_and_fit(self, models=None, scoring=None, mode="quick"):
        if scoring is None:
            scoring = self.scoring
        clf = self.model_finder.search_and_fit(models, scoring, mode)
        return clf

    def set_and_fit(self, model):
        self.model_finder.set_model_and_fit(model)

    def predict(self, X):
        transformed = self.transformer.transform(X)
        output = self.model_finder.predict(transformed)
        return output

    def create_dashboard(self):
        self.output.create_html()
        print(self._output_created_text.format(directory=self.output.output_directory))

    def set_custom_transformers(self, numerical_transformers, categorical_transformers, y_transformer):
        # TODO: set flag about custom transformer
        pass

    # exposed method in case only transformation is needed
    def transform(self, X):
        return self.transformer.transform(X)

    def _initialize_model_and_output(self):
        self.model_finder = ModelFinder(
            X=self.transformed_X,
            y=self.transformed_y,
            X_train=self.transformed_X_train,
            X_test=self.transformed_X_test,
            y_train=self.transformed_y_train,
            y_test=self.transformed_y_test,
            target_type=self.features[self.features.target].type.lower(),
            random_state=self.random_state
        )

        self.output = Output(
            root_path=self.root_path,
            output_directory=self.output_directory,
            package_name=self._name,
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
            transformed_y_test=self.transformed_y_test
        )

    def _create_test_split(self):

        # TODO: implement Stratified split in case of imbalance
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=self.random_state)

        # resetting index so it can be joined later on with test predictions
        output = []
        for d in [X_train, X_test, y_train, y_test]:
            new_d = d.reset_index(drop=True)
            output.append(new_d)

        self.X_train, self.X_test, self.y_train, self.y_test = output

    def _transform_test_splits(self):
        # fitting only on train data
        # TODO: move transformer_eval somewhere? another function?
        self.transformer_eval.fit(self.X_train)
        self.transformer_eval.fit_y(self.y_train)

        transformed = (
            self.transformer_eval.transform(self.X_train),
            self.transformer_eval.transform(self.X_test),
            self.transformer_eval.transform_y(self.y_train),
            self.transformer_eval.transform_y(self.y_test)
        )

        # TODO: rethink, as csr_matrix should stay as it for fitting
        new_tr = []
        for split in transformed:
            try:
                split_arr = split.toarray()
            except AttributeError:
                split_arr = split
            new_tr.append(split_arr)

        self.transformed_X_train, self.transformed_X_test, self.transformed_y_train, self.transformed_y_test = new_tr

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

        if len(unique) > 2:
            # TODO: I dont understand why warnings arent showing up in the console
            warnings.warn("n of unique values in y is > 2, classification_pos_label will be ignored")
            print("WARNING: n of unique values in y is > 2, classification_pos_label will be ignored")
            return None
        else:
            return label
