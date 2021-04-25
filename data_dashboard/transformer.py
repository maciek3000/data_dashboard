import numpy as np
import pandas as pd
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer, LabelEncoder, FunctionTransformer
from sklearn.preprocessing import PowerTransformer
from .functions import calculate_numerical_bins, modify_histogram_edges


class WrapperFunctionTransformer:
    """Wrapper class for FunctionTransformer - defines custom __str__ method.

    Attributes:
        str_repr (str): text used in str method
        transformer (sklearn.preprocessing.FunctionTransformer): FunctionTransformer that is used for fit/transform
    """
    def __init__(self, str_repr, func_transformer):
        """Create WrapperFunctionTransformer object.

        Args:
            str_repr (str): text used in str method
            func_transformer (sklearn.preprocessing.FunctionTransformer): FunctionTransformer that is used for
                fit/transform
        """
        self.str_repr = str_repr
        self.transformer = func_transformer

    def fit(self, *args, **kwargs):
        """Call fit on transformer attribute."""
        self.transformer.fit(*args, **kwargs)
        return self

    def fit_transform(self, *args, **kwargs):
        """Call fit_transform on transformer attribute."""
        return self.transformer.fit_transform(*args, **kwargs)

    def get_params(self, *args, **kwargs):
        """Call get_params on transformer attribute."""
        return self.transformer.get_params(*args, **kwargs)

    def inverse_transform(self, *args, **kwargs):
        """Call inverse_transformer on transformer attribute."""
        return self.transformer.inverse_transform(*args, **kwargs)

    def set_params(self, **kwargs):
        """Call set_params on transformer attribute."""
        self.transformer.set_params(**kwargs)
        return self

    def transform(self, *args, **kwargs):
        """Call transform on transformer attribute."""
        return self.transformer.transform(*args, **kwargs)

    def __str__(self):
        """Return str of str_repr attribute.

        Returns:
            str
        """
        return str(self.str_repr)


class Transformer:
    """Transformer that transforms input data based on the type of it.

    Transformer loosely follows sklearn API and it also defines fit_y, transform_y methods etc. to transform not only
    X but y as well.

    Default preprocessor for X is a ColumnTransformer that does different transformations on
    Categorical and Numerical Features. If the feature is not included in either of those two, it gets added to the
    final output unchanged.
    Default preprocessor for y is either LabelEncoder for classification or placeholder FunctionTransformer (with
    identity function) for regression - in that case Models get wrapped in RegressionWrapper during fitting.

    If transformers different from default are needed, then they can be also injected via set_custom_preprocessor
    methods.

    Note:
        Some methods might work differently when custom transformers are in place.

    Attributes:
        categorical_features (list): list of categorical features names
        numerical_features (list): list of numerical features names
        target_type (str): feature type of the target
        random_state (int): integer for reproducibility on fitting and transformations, defaults to None if not
            provided during __init__
        classification_pos_label (Any): label that is treated as positive (1) in y, defaults to None if not provided
            during __init__
        categorical_transformers (list): list of transformers for categorical features
        numerical_transformers (list): list of transformers for numerical features
        y_transformer (Transformer): transformer for target (y)
        preprocessor_X (sklearn.pipeline.pipeline): pipeline made from categorical and numerical transformers
        preprocessor_y (Transformer): attribute made for consistency, equals to y_transformer
    """
    _default_categorical_transformers = [
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    ]
    _default_numerical_transformers = [
        SimpleImputer(strategy="median"),
        QuantileTransformer(output_distribution="normal"),
        StandardScaler()
    ]

    _default_transformers_random_state = [QuantileTransformer]

    _categorical_pipeline = "categorical"
    _numerical_pipeline = "numerical"
    _one_hot_encoder_tr_name = "onehotencoder"
    _normal_transformers = [QuantileTransformer, PowerTransformer]

    _func_trans_classification_pos_label_text = "FunctionTransformer(classification_pos_label='{}')"
    _func_trans_regression_wrapper_text = "TransformedTargetRegressor(transformer=QuantileTransformer" \
                                          "(output_distribution='normal')) "

    def __init__(self,
                 categorical_features,
                 numerical_features,
                 target_type,
                 random_state=None,
                 classification_pos_label=None
                 ):
        """Create Transformer object.

        Set default transformers for X and y depending on provided categorical and numerical features lists and
        target type.

        Args:
            categorical_features (list): list of categorical features names
            numerical_features (list): list of numerical features names
            target_type (str): feature type of the target
            random_state (int, optional): integer for reproducibility on fitting and transformations, defaults to None
            classification_pos_label (Any, optional): label that is treated as positive (1) in y, defaults to None
        """
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.target_type = target_type
        self.random_state = random_state
        self.classification_pos_label = classification_pos_label

        self.categorical_transformers = self._check_random_state(self._default_categorical_transformers)
        self.numerical_transformers = self._check_random_state(self._default_numerical_transformers)
        self.y_transformer = self._create_default_transformer_y()

        self.preprocessor_X = self._create_preprocessor_X()
        self.preprocessor_y = self._create_preprocessor_y()

    def fit(self, X):
        """Fit preprocessor_X with X data.

        Args:
            X (pandas.DataFrame, numpy.ndarray, scipy.csr_matrix): feature space to fit the transformer

        Returns:
            self
        """
        self.preprocessor_X = self.preprocessor_X.fit(X)
        return self

    def transform(self, X):
        """Transform X with fitted preprocessor_X.

        Args:
            X (pandas.DataFrame, numpy.ndarray, scipy.csr_matrix): feature space to transform with the transformer

        Returns:
            numpy.ndarray, scipy.csr_matrix: transformed X
        """
        transformed = self.preprocessor_X.transform(X)
        return transformed

    def fit_transform(self, X):
        """Fit data and then transform it with preprocessor_X.

        Args:
            X (pandas.DataFrame, numpy.ndarray, scipy.csr_matrix): feature space to fit and transform the transformer

        Returns:
            numpy.ndarray, scipy.csr_matrix: transformed X
        """
        self.fit(X)
        return self.transform(X)

    def fit_y(self, y):
        """Fit preprocessor_y with y data.

        Args:
            y (pandas.Series, numpy.ndarray): feature space to fit the transformer

        Returns:
            self
        """
        self.preprocessor_y = self.preprocessor_y.fit(y)
        return self

    def transform_y(self, y):
        """Transform y with fitted preprocessor_y.

        Args:
            y (pandas.Series, numpy.ndarray): feature space to transform with the transformer

        Returns:
            numpy.ndarray: transformed y
        """
        transformed = self.preprocessor_y.transform(y)
        return transformed

    def fit_transform_y(self, y):
        """Fit data and then transform it with preprocessor_y.

        Args:
            y (pandas.Series, numpy.ndarray): feature space to fit the transformer

        Returns:
            numpy.ndarray: transformed y
        """
        self.fit_y(y)
        return self.transform_y(y)

    def transformed_columns(self):
        """Return list of names of transformed columns.

        Numerical features list is combined with categorical features list and the output is returned. If
        preprocessor_X transformers include 'one hot encoder', then special column names are extracted for every
        new feature created. Otherwise, regular categorical features list is used.

        Returns:
            list: categorical + numerical features list
        """
        # checking if there are any categorical_features at all
        if len(self.categorical_features) > 0:
            cat_transformers = self.preprocessor_X.named_transformers_[self._categorical_pipeline]
            try:
                onehot = cat_transformers[self._one_hot_encoder_tr_name]
                categorical_cols = onehot.get_feature_names(self.categorical_features).tolist()
            except KeyError:
                categorical_cols = self.categorical_features
        else:
            categorical_cols = []
        output = self.numerical_features + categorical_cols
        return output

    def transformations(self):
        """Return dictionary of transformers and transformations applied to every feature in X.

        Structure of a returned dictionary is 'feature name': 2-element tuple - transformers used to transform the
        feature and columns (array) of the result of transformations.

        Note:
            feature not included in either categorical or numerical features lists (feature with no transformations)
            is not included

        Returns:
            dict: 'feature name': (transformers, transformations) tuple
        """
        output = {}
        new_cols = self.transformed_columns()
        for feature in (self.categorical_features + self.numerical_features):
            transformers = self.transformers(feature)
            transformed_cols = [col for col in new_cols if col.startswith(feature)]
            output[feature] = (transformers, transformed_cols)

        return output

    def y_transformations(self):
        """Return 1-element list of y_transformer.

        Returns:
            list: [y_transformer]
        """
        return [self.y_transformer]

    def transformers(self, feature):
        """Return transformers that are used to transform provided feature name depending on its type (categorical
        or numerical), None otherwise.

        Args:
            feature (str): feature name

        Return:
            list, None: list of transformers or None
        """
        if feature in self.categorical_features:
            return self.categorical_transformers
        elif feature in self.numerical_features:
            return self.numerical_transformers
        else:
            return None

    def normal_transformations_histograms(self, feature_train_data, feature_test_data):
        """Return dict of 'feature name': histogram data for every transformer for every feature in train/test data,
        where histograms are calculated after different normalization methods of feature data.

        Features are normalized with different methods (QuantileTransformer, Yeo-Johnson, Box-Cox) and histograms
        are calculated from the transformed data to visualize which transformation makes the feature 'most' normal.

        Note:
            Transformers are fit on train data, whereas histograms are calculated from transformed test data.

        Args:
            feature_train_data (pandas.DataFrame): feature space on which transformers will be trained
            feature_test_data (pandas.DataFrame): feature space from which histograms will be created

        Returns:
            dict: 'feature name': list of tuples (normal transformer, (hist_data, left_edges, right_edges))
        """
        output = {}
        for feature in feature_train_data.columns:
            train_series = feature_train_data[feature].to_numpy().reshape(-1, 1)
            test_series = feature_test_data[feature].to_numpy().reshape(-1, 1)
            transformations = self.normal_transformations(train_series, test_series)

            histogram_data = []
            for transformer, transformed_series in transformations:
                series = pd.Series(transformed_series.reshape(1, -1)[0])
                bins = calculate_numerical_bins(series)
                hist, edges = np.histogram(series, density=True, bins=bins)
                left_edges, right_edges = modify_histogram_edges(edges)
                result = (transformer, (hist, left_edges, right_edges))
                histogram_data.append(result)
            output[feature] = histogram_data

        return output

    def normal_transformations(self, single_feature_train_data, single_feature_test_data):
        """Return differently normalized (transformed) feature data depending on the transformer used.

        Normalizing Transformers used are QuantileTransformer (with output_distribution='normal'), PowerTransformer for
        Yeo-Johnson method and PowerTransformer for Box-Cox method. Transformers are fit on train data and output is
        created on test data.

        Note:
            if feature includes values of 0 or less (in either train or test) then Box-Cox method is not included
            in transformations

        Args:
            single_feature_train_data (pd.Series, numpy.ndarray): numerical feature data on which transformers are
                trained
            single_feature_test_data (pd.Series, numpy.ndarray): numerical feature data which is transformed by
                fitted 'normal' Transformers

        Returns:
            list: list of tuples - (transformer used, transformed test data)
        """
        normal_transformers = [
            QuantileTransformer(output_distribution="normal", random_state=self.random_state),
            PowerTransformer(method="yeo-johnson")
        ]

        # box-cox works only with positive values, 0s excluded
        nmin = np.nanmin
        if nmin(single_feature_train_data) > 0 and nmin(single_feature_test_data) > 0:
            normal_transformers.append(PowerTransformer(method="box-cox"))

        output = []
        for transformer in normal_transformers:
            pipe = make_pipeline(
                SimpleImputer(strategy="median"),  # imputing missing values with their median
                transformer
            )
            # Catching warnings, mostly from n_quantiles from QuantileTransformer
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(single_feature_train_data)
            output.append((transformer, pipe.transform(single_feature_test_data)))

        return output

    def set_custom_preprocessor_X(self, categorical_transformers=None, numerical_transformers=None):
        """Set preprocessors for categorical and numerical features in X to be used later on in the process (e.g. with
        fit or transform calls).

        Both lists of transformers are optional - only one type of custom transformers can be set, the other one left
        will be set to default transformers. If none of the lists are provided, then the method is equal to setting
        default transformers.

        Args:
            categorical_transformers (list, optional): transformers to be used on categorical features, defaults to None
            numerical_transformers (list, optional): transformers to be used on numerical features, defaults to None
        """
        if categorical_transformers:
            self.categorical_transformers = categorical_transformers
        if numerical_transformers:
            self.numerical_transformers = numerical_transformers
        self.preprocessor_X = self._create_preprocessor_X()

    def set_custom_preprocessor_y(self, transformer):
        """Set provided transformer as preprocessor for y to be used later on in the process (e.g. with fit or
        transform calls).

        Args:
            transformer (Transformer): transformer that will be used to transform y (target)
        """
        self.y_transformer = transformer
        self.preprocessor_y = self._create_preprocessor_y()

    def y_classes(self):
        """Return classes (labels) present in preprocessor_y.

        Returns:
            numpy.ndarray: array of classes present in fitted preprocessor_y

        Raises:
            ValueError: when target_type is 'Numerical'
        """
        if self.target_type == "Numerical":
            raise ValueError("No classes present in regression problem.")

        return self.preprocessor_y.classes_

    def _create_preprocessor_X(self):
        """Create preprocessor for X features with different Transformers for Categorical and Numerical features.

        ColumnTransformer is created with categorical/numerical_features attributes as feature names and categorical/
        numerical_transformers attributes as Transformers. Any feature not included in categorical/numerical_features
        attributes is treated as 'pre-transformer' and is concatted at the end (remainder='passthrough').

        Returns:
            sklearn.compose.ColumnTransformer: ColumnTransformer for X features
        """
        # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
        # https://scikit-learn.org/stable/modules/compose.html

        numerical_features = self.numerical_features
        categorical_features = self.categorical_features

        numeric_transformer = make_pipeline(*self.numerical_transformers)
        categorical_transformer = make_pipeline(*self.categorical_transformers)

        col_transformer = ColumnTransformer(
            transformers=[
                (self._numerical_pipeline, numeric_transformer, numerical_features),
                (self._categorical_pipeline, categorical_transformer, categorical_features)
            ],
            remainder="passthrough"
        )

        return col_transformer

    def _create_default_transformer_y(self):
        """Create default transformer for y (target) depending on the target_type attribute.

        If target is Categorical then either regular LabelEncoder is created or if the classification_pos_label
        attribute is not None then the value in it is treated as positive (1), rest of values defaults to 0. If target
        is Numerical then no transformation takes place, as RegressionWrapper takes Model as an argument, not features.

        Returns:
            Any: y (target) Transformer

        Raises:
            ValueError: if target_type is not in ["Categorical", "Numerical"]
        """
        if self.target_type == "Categorical":
            if self.classification_pos_label is not None:
                func_transformer = FunctionTransformer(lambda x: np.where(x == self.classification_pos_label, 1, 0))
                text = self._func_trans_classification_pos_label_text.format(self.classification_pos_label)
                # Wrapper for custom text to be shown in Views
                transformer = WrapperFunctionTransformer(text, func_transformer)
            else:
                transformer = LabelEncoder()
        elif self.target_type == "Numerical":
            # in regression model finder wraps its object in TargetRegressor
            func_transformer = FunctionTransformer(func=None)  # identity function, x = x
            text = self._func_trans_regression_wrapper_text
            # Wrapper for custom text to be shown in Views
            transformer = WrapperFunctionTransformer(text, func_transformer)
        else:
            raise ValueError(
                "Target type set as {target_type}, should be Categorical or Numerical".format(
                    target_type=self.target_type
                )
            )

        return transformer

    def _create_preprocessor_y(self):
        """Return y_transformer attribute.

        Returns:
            Any: y_transformer attribute
        """
        return self.y_transformer

    def _check_random_state(self, transformers):
        """Check provided transformers and if any of them is included in _default_transformers_random_state class
        attribute list then add random_state instance attribute to it.

        This method shouldn't be used for custom transformers - those should have their random state already defined
        during initialization.

        Args:
            transformers (list): list of transformers

        Returns:
            list: transformers with random_state added (if appropriate)
        """
        for transformer in transformers:
            if transformer.__class__ in self._default_transformers_random_state:
                transformer.random_state = self.random_state

        return transformers
