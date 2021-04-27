import warnings
import copy
import webbrowser
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .analyzer import Analyzer
from .features import Features
from .output import Output
from .transformer import Transformer
from .model_finder import ModelFinder
from .descriptor import FeatureDescriptor
from .functions import sanitize_input, make_pandas_data, obj_name


class Dashboard:
    """Data Dashboard with Visualizations of original data, transformations and Machine Learning Models performance.

    Dashboard analyzes provided data (summary statistics, correlations), transforms it and feeds it to Machine Learning
    algorithms to search for the best scoring Model. All steps are written down into the HTML output for end-user
    experience.

    HTML output created is a set of 'static' HTML pages saved into provided output directory - there are no server -
    client interactions. Visualization are still interactive though through the use of Bokeh library.

    Note:
        As files are static, data might be embedded in HTML files. Please be aware when sharing produced HTML Dashboard.

    Dashboard object can also be used as a pipeline for transforming/fitting/predicting by using exposed methods
    loosely following sklearn API.

    Attributes:
        output_directory (str): directory where HTML Dashboard will be placed
        already_transformed_columns (list): list of feature names that are already pre-transformed
        random_state (int, None): integer for reproducibility on fitting and transformations, defaults to None if not
            provided during __init__

        features_descriptions (FeatureDescriptor): FeatureDescriptor containing external information on features
        features (Features): Features object with basic features information
        analyzer (Analyzer): Analyzer object analyzing and performing calculations on features
        transformer (Transformer): Transformer object responsible for transforming the data for ML algorithms, fit to
            all data
        transformer_eval (Transformer): Transformer object fit on train data only
        model_finder (ModelFinder): ModelFinder object responsible for searching for Models and assessing their
            performance

        X (pandas.DataFrame, numpy.ndarray, scipy.csr_matrix): data to be analyzed
        y (pandas.Series, numpy.ndarray): target variable
        X_train (pandas.DataFrame): train split of X
        X_test (pandas.DataFrame): test split of X
        y_train (pandas.Series): train split of y
        y_test (pandas.Series): test split of y

        transformed_X (numpy.ndarray, scipy.csr_matrix): all X data transformed with transformer
        transformed_y (numpy.ndarray): all y data transformed with transformer
        transformed_X_train (numpy.ndarray, scipy.csr_matrix): X train split transformed with transformer_eval
        transformed_X_test (numpy.ndarray, scipy.csr_matrix): X test split transformed with transformer_eval
        transformed_y_train (numpy.ndarray): y train split transformed with transformer_eval
        transformed_y_test (numpy.ndarray): y test split transformed with transformer_eval
    """

    _name = "data_dashboard"
    _output_created_text = "Created output at {directory}"
    _model_found_text = "Model: {name}\nScore: {score}\nParams: {params}"
    _n_features_pairplots_limit = 15

    def __init__(self,
                 X,
                 y,
                 output_directory,
                 feature_descriptions_dict=None,
                 already_transformed_columns=None,
                 classification_pos_label=None,
                 force_classification_pos_label_multiclass=False,
                 random_state=None
                 ):
        """Create Dashboard object.

        Provided X and y are checked and converted to pandas object for easier analysis and eventually split and
        transformed. classification_pos_label is checked if the label is present in y target variable. X is assessed
        for the number of features and appropriate flags are set. All necessary objects are created. Transformer
        and transformer_eval are fit to all and train data, appropriately.

        Attributes:
            X (pandas.DataFrame, numpy.ndarray, scipy.csr_matrix): data to be analyzed
            y (pandas.Series, numpy.ndarray): target variable
            output_directory (str): directory where HTML Dashboard will be placed
            feature_descriptions_dict (dict, None): dictionary of metadata on features in X and y, defaults to None
            already_transformed_columns (list): list of feature names that are already pre-transformed
            classification_pos_label (Any): value in target that will be used as positive label
            force_classification_pos_label_multiclass (bool): flag indicating if provided classification_pos_label in
                multiclass problem should be forced, de facto changing the problem to classification
            random_state (int, None): integer for reproducibility on fitting and transformations, defaults to None
        """
        self.X, self.y = self._check_provided_data(X, y)
        self.output_directory = output_directory
        self.already_transformed_columns = self._check_transformed_cols(already_transformed_columns)

        if classification_pos_label is not None:
            classification_pos_label = self._check_classification_pos_label(classification_pos_label)

        self.random_state = random_state

        # placeholders for attributes
        self.transformed_X = None
        self.transformed_y = None
        self.model_finder = None
        self.output = None

        #########################################
        # =============== Flags =============== #
        #########################################

        self._force_classification_pos_label_multiclass_flag = force_classification_pos_label_multiclass
        self._create_pairplots_flag = True  # pairplot flag is True by default

        ###########################################
        # =============== Objects =============== #
        ###########################################

        self.features_descriptions = FeatureDescriptor(feature_descriptions_dict)
        self.features = Features(self.X, self.y, self.features_descriptions, self.already_transformed_columns)
        self.analyzer = Analyzer(self.features)

        # excluding Transformed columns from transformations, they will be concatted at the end because of
        # remainder='passthrough' argument of ColumnTransformer
        self.transformer = Transformer(
            categorical_features=self.features.categorical_features(drop_target=True, exclude_transformed=True),
            numerical_features=self.features.numerical_features(drop_target=True, exclude_transformed=True),
            target_type=self.features[self.features.target].feature_type,
            random_state=self.random_state,
            classification_pos_label=classification_pos_label
        )

        self.transformer_eval = copy.deepcopy(self.transformer)  # copy to be fit only on train data

        ################################################
        # =============== X, y actions =============== #
        ################################################

        # dropping features assessed as unusable
        self.X = self.X.drop(self.features.unused_features(), axis=1)
        self._assess_n_features(self.X)

        # https://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics
        # Just as it is important to test a predictor on data held-out from training, preprocessing
        # (such as standardization, feature selection, etc.) and similar data transformations similarly should be
        # learnt from a training set and applied to held-out data for prediction.

        self._create_test_splits()
        self._do_transformations()
        self._initialize_model_and_output()

    def create_dashboard(self,
                         models=None,
                         scoring=None,
                         mode="quick",
                         logging=True,
                         disable_pairplots=False,
                         force_pairplot=False
                         ):
        """Create several Views (Subpages) and join them together to form an interactive WebPage/Dashboard.

        Models can be:
            - list of initialized models
            - dict of 'Model Class': param_grid of a given model to do the GridSearch on
            - None - default Models collection will be used

        scoring should be a sklearn scoring function. If None is provided, default scoring function will be used.

        mode can be:
            - "quick": search is initially done on all models but with no parameter tuning after which top
                Models are chosen and GridSearched with their param_grids
            - "detailed": GridSearch is done on all default models and their params

        Provided mode doesn't matter when models are explicitly provided (not None).

        Note:
            Some functions might not work as of now: e.g. roc_auc_score for multiclass problem as it requires
            probabilities for every class in comparison to regular predictions expected from other scoring functions.

        Depending on logging flag, .csv logs might be created or not in the output directory.

        force_pairplot flag forces the dashboard to create Pairplot and ScatterPlot Grid when it was assessed in the
        beginning not to plot it (as number of features in the data exceeded the limit).

        disable_pairplot flag disables creation of Pairplot and ScatterPlot Grid in the Dashboard - it takes precedence
        over force_pairplot flag.

        HTML output is created in output_directory attribute file path and opened in a web browser window.

        Args:
            models (list, dict, optional): list of Models or 'Model class': param_grid dict pairs, defaults to None
            scoring (func, optional): sklearn scoring function, defaults to None
            mode ("quick", "detailed", optional): either "quick" or "detailed" string, defaults to "quick"
            logging (bool, optional): flag indicating if .csv logs should be created, defaults to True
            disable_pairplots (bool, optional): flag indicating if Pairplot and ScatterPlot Grid in the Dashboard should
                be created or not, defaults to False
            force_pairplot (bool, optional): flag indicating if PairPlot and ScatterPlot Grid in the Dashboard should
                be created when number of features in the data crossed the internal limit, defaults to False
        """
        clf = self.search_and_fit(models, scoring, mode)
        print("Found model: {clf}".format(clf=obj_name(clf)))
        print("Creating Dashboard...")

        if disable_pairplots:
            do_pairplots = False
        else:
            do_pairplots = self._create_pairplots_flag
            if not do_pairplots and force_pairplot:
                do_pairplots = True

        do_logging = logging

        self.output.create_html(
            do_pairplots=do_pairplots,
            do_logs=do_logging
        )
        print(self._output_created_text.format(directory=self.output.output_directory))
        webbrowser.open_new(self.output.overview_file())

    def search_and_fit(self, models=None, scoring=None, mode="quick"):
        """Search for the best scoring Model, fit it with all data and return it.

        Models can be:
        - list of initialized models
        - dict of 'Model Class': param_grid of a given model to do the GridSearch on
        - None - default Models collection will be used

        scoring should be a sklearn scoring function. If None is provided, default scoring function will be used.

        mode can be:
            - "quick": search is initially done on all models but with no parameter tuning after which top
                Models are chosen and GridSearched with their param_grids
            - "detailed": GridSearch is done on all default models and their params

        Provided mode doesn't matter when models are explicitly provided (not None).

        Note:
            Some functions might not work as of now: e.g. roc_auc_score for multiclass problem as it requires
            probabilities for every class in comparison to regular predictions expected from other scoring functions.

        Args:
            models (list, dict, optional): list of Models or 'Model class': param_grid dict pairs, defaults to None
            scoring (func, optional): sklearn scoring function, defaults to None
            mode ("quick", "detailed", optional): either "quick" or "detailed" string, defaults to "quick"

        Returns:
            sklearn.Model: best scoring Model already fit to X and y data
        """
        if scoring is None:
            scoring = self.model_finder.default_scoring
        clf = self.model_finder.search_and_fit(models, scoring, mode)
        return clf

    def set_and_fit(self, model):
        """Set provided Model as a best scoring Model and fit it to all X and y data.

        Args:
            model (sklearn.Model): instance of ML Model
        """
        self.model_finder.set_model_and_fit(model)

    def transform(self, X):
        """Transform provided X data with Transformer.

        Returns:
            numpy.ndarray, scipy.csr_matrix: transformed X
        """
        return self.transformer.transform(X)

    def predict(self, transformed_X):
        """Predict target from provided X with the best scoring Model.

        Args:
            transformed_X (pandas.DataFrame, numpy.ndarray, scipy.csr_matrix): transformed X feature space to predict
                target variable from

        Returns:
              numpy.ndarray: predicted y target variable
        """
        output = self.model_finder.predict(transformed_X)
        return output

    def transform_predict(self, X):
        """Transform and then predict X data.

        Args:
            X (pandas.DataFrame, numpy.ndarray, scipy.csr_matrix): X data

        Returns:
               numpy.ndarray: predicted y target variable
        """
        transformed = self.transform(X)
        return self.predict(transformed)

    def best_model(self):
        """Return best (chosen) Model used in predictions.

        Returns:
            sklearn.Model: best scoring Model
        """
        return self.model_finder.best_model()

    def set_custom_transformers(self, categorical_transformers=None, numerical_transformers=None, y_transformer=None):
        """Set custom Transformers to be used in the problem pipeline.

        Provided arguments should be a list of Transformers to be used with given type of features. Only one type
        of transformers can be provided.

        Transformers are updated in both transformer and transformer_eval instances. ModelFinder and Output instances
        are recreated with the new transformed data and new Transformers.

        Args:
            categorical_transformers (list): list of Transformers to be used on categorical features
            numerical_transformers (list): list of Transformers to be used on numerical features
            y_transformer (sklearn.Transformer): singular Transformer to be used on target variable
        """
        for tr in [self.transformer, self.transformer_eval]:
            tr.set_custom_preprocessor_X(
                numerical_transformers=numerical_transformers,
                categorical_transformers=categorical_transformers
            )
            if y_transformer:
                tr.set_custom_preprocessor_y(y_transformer)

        self._do_transformations()
        self._initialize_model_and_output()

    def _do_transformations(self):
        """Fit transformer_eval to train data, transform train/test splits; fit transformer to all data, transform
        all data.

        transformed_X and transformed_y attributes are populated with transformed X and y data.
        """
        self._fit_transform_test_splits()
        self._fit_transformer()
        self.transformed_X = self.transformer.transform(self.X)
        self.transformed_y = self.transformer.transform_y(self.y)

    def _initialize_model_and_output(self):
        """Create ModelFinder and Output objects and assign them to model_finder and output attributes appropriately."""
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
            transformer=self.transformer_eval,  # Note: transformer_eval as train/test splits are included in HTML
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
        """Create train/test splits from X and y data.

        X_train, X_test, y_train and y_test attributes are populated with the appropriate splits.

        Note:
            Every split DataFrame has its index reset and the old index dropped just so there is consistency between
            train and test splits.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=self.random_state)

        # resetting index so it can be joined later on with test predictions
        output = []
        for d in [X_train, X_test, y_train, y_test]:
            new_d = d.reset_index(drop=True)
            output.append(new_d)

        self.X_train, self.X_test, self.y_train, self.y_test = output

    def _fit_transform_test_splits(self):
        """Fit transformer_eval with train splits and transform both train and test splits.

        Splits to be used are in X_train, X_test, y_train and y_test attributes. Created transformed splits are put
        in transformed_X_train, transformed_X_test, transformed_y_train and transformed_y_test attributes.
        """
        # fitting only on train data
        self.transformer_eval.fit(self.X_train)
        self.transformer_eval.fit_y(self.y_train)

        t = (
            self.transformer_eval.transform(self.X_train),
            self.transformer_eval.transform(self.X_test),
            self.transformer_eval.transform_y(self.y_train),
            self.transformer_eval.transform_y(self.y_test)
        )

        self.transformed_X_train, self.transformed_X_test, self.transformed_y_train, self.transformed_y_test = t

    def _fit_transformer(self, X=None, y=None):
        """Fit transformer with provided data.

        If provided X or y are None, appropriate X or y attributes are used.

        transformer is fit for both X and y.
        """
        if X is None:
            X = self.X
        if y is None:
            y = self.y

        self.transformer.fit(X)
        self.transformer.fit_y(y)

    def _check_provided_data(self, X, y):
        """Convert X and y to pandas object and change their column names to be JS friendly.

        Args:
            X (pandas.DataFrame, numpy.ndarray, scipy.csr_matrix): X data
            y (pd.Series, numpy.ndarray): target variable

        Returns:
            tuple: (converted X, converted y)
        """
        new_X = make_pandas_data(X, pd.DataFrame)
        new_y = make_pandas_data(y, pd.Series)

        new_X.columns = sanitize_input(new_X.columns)
        new_y.name = sanitize_input([new_y.name])[0]

        return new_X, new_y

    def _check_classification_pos_label(self, label):
        """Check if label is in unique target variable values.

        Label is returned if it's in unique values and problem type is classification. If the problem is multiclass,
        then _force_classification_pos_label_multiclass_flag attribute is checked - if the flag is False, label is
        changed to None (as it would change problem from multiclass to classification). If the flag is True, then the
        label is returned (forcing problem type change).

        Args:
            label (Any): label to be considered as positive (1) in classification problem

        Returns:
            Any: label if deemed correct in regard to problem, None otherwise

        Raises:
            ValueError: if label is not in unique values of y
        """
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
        """Check number of features in df and set _create_pairplots_flag accordingly.

        If the number of features is more than _n_features_pairplots_limit class attribute, then _create_pairplots_flag
        is set to False (to prevent long creation time and/or MemoryError in case of huge feature space).

        Args:
            df (pandas.DataFrame): features data
        """
        n = df.shape[1]
        if n > self._n_features_pairplots_limit:
            self._create_pairplots_flag = False
            warnings.warn(
                "Number of features crossed their default limit - pairplots will be turned off to reduce"
                " runtime and/or to avoid MemoryErrors. To force pairplots to be created, call 'create_html' with"
                " 'force_pairplots=True' argument.")

    def _check_transformed_cols(self, transformed_columns):
        """Check list of transformed columns if every column name is in X features names.

        If transformed_columns is a proper subset of X columns, then sorted list of unique transformed columns is
        returned. Otherwise, ValueError is raised.

        If transformed_columns is None, then empty list is returned.

        Note:
            y name is not included in expected column names.

        Args:
            transformed_columns (list): list of names of pre-transformed columns

        Returns:
            list: sorted pre-transformed feature names or empty list

        Raises:
            ValueError: when transformed_columns is not a subset of X columns
        """
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
                return sorted(transformed)  # sorted returns list
        else:
            return []
