import pandas as pd
from .functions import sort_strings


class FeatureNotSupported(Exception):
    """Exception for not supported Feature type."""
    pass


class BaseFeature(object):
    """Base Feature class that every other Feature Class should inherit from.
    """

    feature_type = None
    """
    feature_type: should be overrode by Class inheriting BaseFeature.
    """

    def data(self):
        """To be overrode."""
        raise NotImplemented

    def mapping(self):
        """To be overrode."""
        raise NotImplemented


class CategoricalFeature(BaseFeature):
    """Categorical Feature class. Inherits from BaseFeature.

        Categorical Features are those that have values that are limited and/or fixed in their nature.
        However, it doesn't mean that we can't analyze them in similar manner to Numerical Features - e.g.
        calculate mean, distribution and other variables. In order to do so, every unique value in the data needs to be
        assigned a unique number, which will allow calculations.

        In case of Categorical Features, they might already be represented in the data with key of some sort:
            "A": "Apple"
            "B": "Banana"
        This structure is also present in the dict descriptions that can be fed to the analysis.

        Raw mapping would associate those values with the unique numbers created during the mapping:
            "A": 1
            "B": 2

        CategoricalFeature class creates mapping between new unique numbers present in the data
        and the description (item) of the already provided mapping:
            1: "Apple"
            2: "Banana"

        This way of mapping things should ease out mental connections between what is seen in the visualizations
        (numbers mostly) and whats really represented behind those numbers (instead of their symbols).

        Class Attributes:
            feature_type: "Categorical" hardcoded string

        Attributes:
            series (pandas.Series): Series holding the data
            name (str): name of the Feature
            description (str): description of the Feature
            imputed_category (bool): flag indicating if the category of the Feature was provided or imputed
            transformed (bool): flag indicating if the Feature is pre-transformed or not
            mapping (dict): dictionary holding external mapping of values to their 'logical' counterparts
            raw_mapping (dict): mapping between value -> raw number
            mapped_series (pandas.Series): Series with it's content replaced with raw_mapping
    """
    feature_type = "Categorical"

    def __init__(self, series, name, description, imputed_category, transformed=False, mapping=None):
        """Construct new CategoricalFeature object.

        Additionally create raw_mapping and mapped_series attributes.

        Args:
            series (pandas.Series): Series holding the data (copy)
            name (str): name of the Feature
            description (str): description of the Feature
            imputed_category (bool): flag indicating if the category of the Feature was provided or imputed
            transformed (bool, Optional): flag indicating if the Feature is pre-transformed or not, defaults to False
            mapping (dict): dictionary holding external mapping of values to their 'logical' counterparts, defaults
                to None
        """
        self.series = series.copy()
        self.name = name
        self.description = description
        self.imputed_category = imputed_category  # flag to check if type of feature was provided or imputed
        self.transformed = transformed  # flag to check if the feature is already transformed
        self.original_mapping = mapping

        self.raw_mapping = self._create_raw_mapping()
        self.mapped_series = self._create_mapped_series()

        self._descriptive_mapping = None

    def data(self):
        """Return mapped_series property."""
        return self.mapped_series

    def original_data(self):
        """Return original Series."""
        return self.series

    def mapping(self):
        """Return _descriptive_mapping attribute and if it's None, create it with _create_descriptive_mapping method."""
        if not self._descriptive_mapping:
            self._descriptive_mapping = self._create_descriptive_mapping()

        return self._descriptive_mapping

    def _create_mapped_series(self):
        """Return series property with it's content replaced with raw_mapping dictionary."""
        return self.series.replace(self.raw_mapping)

    def _create_raw_mapping(self):
        """Return dictionary of 'unique value': number pairs.

        Replace every categorical value with a number starting from 1 (sorted alphabetically). Starting with 1
        to be consistent with "count" obtained with .describe() methods on dataframes.

        Returns:
            dict: 'unique value': number pairs dict.
        """
        values = sorted(self.series.unique(), key=str)
        mapped = {value: number for number, value in enumerate(values, start=1) if not pd.isna(value)}
        return mapped

    def _create_descriptive_mapping(self):
        """Create and return dictionary mapping for unique values present in series.

        Key is the "new" value provided with enumerating unique values in raw_mapping. Value is either the
        description of the category taken from original descriptions or the original value (if descriptions are None).

        Returns:
            dict: new mapping between 'unique number' -> original mapping
        """
        if self.original_mapping:
            mapp = {}
            for key, item in self.raw_mapping.items():
                new_key = item

                # try/except clause to try converting strings to integers in case descriptions are taken from json
                # files, where keys can only be str but python interprets them as int
                try:
                    new_item = self.original_mapping[key]
                except KeyError:
                    try:
                        new_item = self.original_mapping[str(key)]
                    except KeyError:
                        raise
                mapp[new_key] = new_item
        else:
            # Changed the order to reflect that the key is
            mapp = {item: key for key, item in self.raw_mapping.items()}

        return mapp


class NumericalFeature(BaseFeature):
    """Numerical Feature class. Inherits from BaseFeature.

    Simple Container Class for a Feature deemed as Numerical. Numerical Features are those that should be treated as
    numbers, which have underlying distribution, summary statistics, etc.

    Class Attributes:
            feature_type: "Numerical" hardcoded string

        Attributes:
            series (pandas.Series): Series holding the data
            name (str): name of the Feature
            description (str): description of the Feature
            imputed_category (bool): flag indicating if the category of the Feature was provided or imputed
            transformed (bool): flag indicating if the Feature is pre-transformed or not
    """

    feature_type = "Numerical"

    def __init__(self, series, name, description, imputed_category, transformed=False):
        """Construct new NumericalFeature object.

        Args:
            series (pandas.Series): Series holding the data (copy)
            name (str): name of the Feature
            description (str): description of the Feature
            imputed_category (bool): flag indicating if the category of the Feature was provided or imputed
            transformed (bool, Optional): flag indicating if the Feature is pre-transformed or not, defaults to False
        """
        self.series = series.copy()
        self.name = name
        self.description = description
        self.imputed_category = imputed_category  # flag to check if type of feature was provided or imputed
        self.transformed = transformed  # flag to check if the feature is already transformed

    def data(self):
        """Return series attribute."""
        return self.series

    def mapping(self):
        """Return None, as NumericalFeature has no mapping."""
        return None


class Features:
    """Container for Features objects in the analyzed data.

    Data comes with different types of features - numerical, categorical, date, bool, etc. Goal of this class
    is to have the data in one place, classify available features (columns) and provide any corresponding metadata
    that might be needed. This might include descriptions for every feature, respective mapping between original
    and mapped Categorical Features, etc.

    Please note, that target variable (y) also is treated as a feature - you need to explicitly state it in
    methods if you wish to exclude it.

    available_categories is a work in progress property that defines what type of features are implemented.
    max_categories defines what is a limit of unique values in a column for it to be treated as categorical
    (even if the data itself comes as a int/float type).

    Class Attributes:
        max_categories (int): maximum number of unique values in the NumericalFeature data so it can be treated as
            Categorical; if the # of unique values crosses that threshold, feature will be deemed Numerical

    Attributes:
        original_dataframe (pandas.DataFrame): copy of original DataFrame, on which all calculations and manipulations
            will happen.
        target (str): name of the target Feature (column in DataFrame)
        transformed_features (list): list of features that were pre-transformed
    """
    max_categories = 10

    _categorical = "cat"
    _numerical = "num"
    _date = "date"
    _available_categories = [_categorical, _numerical]

    _description_not_available = "Description not Available"

    def __init__(self, X, y, descriptor=None, transformed_features=None):
        """Construct Features object from passed arguments.

        Automatically analyze provided DataFrame (X + y) and assess their types.

        Args:
            X (pandas.DataFrame): DataFrame of features (columns), from which Models will learn
            y (pandas.Series): Series of target variable data
            descriptor (descriptor.FeatureDescriptor, optional): FeatureDescriptor object holding external information/
                metadata about features (e.g. their description, category, etc.), defaults to None
            transformed_features (list, optional): list of pre-transformed features, defaults to None
        """
        self.original_dataframe = pd.concat([X, y], axis=1).copy()
        self.target = y.name

        if transformed_features:
            self.transformed_features = transformed_features
        else:
            self.transformed_features = []  # empty list

        self._all_features = None
        self._categorical_features = None
        self._numerical_features = None
        self._unused_columns = []

        self._raw_dataframe = None
        self._mapped_dataframe = None

        self._mapping = None
        self._descriptions = None

        # {feature_name: feature object} dict
        self._features = self._analyze_features(descriptor)

    def _analyze_features(self, descriptor):
        """Analyze original_dataframe attribute and assess type of each column (Numerical or Categorical).

        Every column present in the original_dataframe will be checked and appropriate FeatureClass will be created
        for it. Every FeatureClass will also have mapping and description attributes specific to them.

        If descriptor holds information about a given feature, then this information takes priority and will be used
        to construct Features objects (e.g. if the category of the feature is provided as Categorical, then
        CategoricalFeatures object will be created, even though it might not meet the conditions for it).

        If there is no information about a feature in descriptor or descriptor is None, then description defaults to
        _description_not_available attribute. Mapping is described in CategoricalFeature docstring.

        Note:
            Currently only CategoricalFeature and NumericalFeature can be created. If any column cannot be assessed
            as one of those two, then it gets added to _unused_columns list. As of today, this happens with 'Date'
            variables.

        Args:
            descriptor (descriptor.FeatureDescriptor): FeatureDescriptor object holding external information/
                metadata about features (e.g. their description, category, etc.)

        Returns:
            dict: dictionary of 'feature name': FeatureObject pairs.

        Raises:
            FeatureNotSupported: if the column type cannot be assessed as either Categorical or Numerical
        """
        features = {}

        for column in self.original_dataframe.columns:
            try:
                description = None
                category = None
                mapping = None
                category_imputed = False

                # Provided keys extracted only if object was initialized
                if descriptor is not None and descriptor.initialized:  # !
                    if column in descriptor.keys():
                        description = descriptor.description(column)
                        category = descriptor.category(column)
                        mapping = descriptor.mapping(column)

                # category imputed in case it wasn't extracted from descriptor
                if (not category) or (category not in self._available_categories):
                    category = self._impute_column_type(self.original_dataframe[column])
                    category_imputed = True

                # default Description
                if description is None:
                    description = self._description_not_available

                # checking if feature is already transformed
                if column in self.transformed_features:
                    transformed = True
                else:
                    transformed = False

                if category == self._categorical:  # Categorical
                    feature = CategoricalFeature(
                        series=self.original_dataframe[column],
                        name=column,
                        description=description,
                        mapping=mapping,
                        transformed=transformed,
                        imputed_category=category_imputed
                    )

                elif category == self._numerical:  # Numerical
                    feature = NumericalFeature(
                        series=self.original_dataframe[column],
                        name=column,
                        description=description,
                        transformed=transformed,
                        imputed_category=category_imputed
                    )

                else:
                    raise FeatureNotSupported("Feature Category not supported: {}".format(category))

                features[column] = feature

            except FeatureNotSupported:
                self._unused_columns.append(column)

        return features

    def _impute_column_type(self, series):
        """Impute column type based on the data included in provided series.

        Args:
            series (pandas.Series): Series which column type is checked

        Returns:
            str: one of _categorical, _numerical_ or _date str attributes

        Raises:
            Exception: raised when all conversions fail
        """
        if series.dtype == bool:
            return self._categorical
        else:
            try:
                _ = series.astype("float64")
                if len(_.unique()) <= self.max_categories:
                    return self._categorical
                else:
                    return self._numerical
            except TypeError:
                return self._date
            except ValueError:
                return self._categorical
            except Exception:
                raise

    def features(self, drop_target=False, exclude_transformed=False):
        """Return list of features names present in _all_features attribute.

        If _all_features attribute is None, feature list is first created and assigned to that attribute.

        Args:
            drop_target (bool, optional): flag indicating if returned list should exclude target name or not, defaults
                to False
            exclude_transformed (bool, optional): flag indicating if returned list should exclude pre-transformed
                column names present in transformed_features attributes, defaults to False

        Returns:
            list: feature names list
        """
        if not self._all_features:
            self._all_features = self._create_features()

        features = self._all_features
        if drop_target:
            features = [feature for feature in features if feature != self.target]

        if exclude_transformed:
            features = [feature for feature in features if not self._features[feature].transformed]

        return features

    def categorical_features(self, drop_target=False, exclude_transformed=False):
        """Return list of categorical features names present in _categorical_features attribute.

        If _categorical_features attribute is None, categorical feature list is first created and assigned to
        that attribute.

        Args:
            drop_target (bool, optional): flag indicating if returned list should exclude target name or not, defaults
                to False
            exclude_transformed (bool, optional): flag indicating if returned list should exclude pre-transformed
                column names present in transformed_features attributes, defaults to False

        Returns:
            list: categorical feature names list
        """
        if not self._categorical_features:
            self._categorical_features = self._create_categorical_features()

        categorical_features = self._categorical_features
        if drop_target:
            categorical_features = [feature for feature in categorical_features if feature != self.target]

        if exclude_transformed:
            categorical_features = [
                feature for feature in categorical_features if not self._features[feature].transformed
            ]

        return categorical_features

    def numerical_features(self, drop_target=False, exclude_transformed=False):
        """Return list of numerical features names present in _numerical_features attribute.

        If _numerical_features attribute is None, numerical feature list is first created and assigned to
        that attribute.

        Args:
            drop_target (bool, optional): flag indicating if returned list should exclude target name or not, defaults
                to False
            exclude_transformed (bool, optional): flag indicating if returned list should exclude pre-transformed
                column names present in transformed_features attributes, defaults to False

        Returns:
            list: numerical feature names list
        """
        if not self._numerical_features:
            self._numerical_features = self._create_numerical_features()

        numerical_features = self._numerical_features
        if drop_target:
            numerical_features = [feature for feature in numerical_features if feature != self.target]

        if exclude_transformed:
            numerical_features = [feature for feature in numerical_features if not self._features[feature].transformed]

        return numerical_features

    def raw_data(self, drop_target=False, exclude_transformed=False):
        """Return pandas DataFrame present in _raw_dataframe attribute.

        If _raw_dataframe attribute is None, raw DataFrame is first created and assigned to that attribute.

        Args:
            drop_target (bool, optional): flag indicating if returned DataFrame should exclude target from columns,
                defaults to False
            exclude_transformed (bool, optional): flag indicating if returned DataFrame should exclude pre-transformed
                columns from transformed_features attributes, defaults to False

        Returns:
            pandas.DataFrame: dataframe with raw (original) data
        """
        if self._raw_dataframe is None:
            self._raw_dataframe = self._create_raw_dataframe()

        raw_df = self._raw_dataframe
        if drop_target:
            raw_df = raw_df.drop([self.target], axis=1)

        if exclude_transformed:
            raw_df = raw_df.drop(self.transformed_features, axis=1)

        return raw_df

    def data(self, drop_target=False, exclude_transformed=False):
        """Return pandas DataFrame present in _mapped_dataframe attribute.

        If _mapped_dataframe attribute is None, mapped DataFrame is first created and assigned to that attribute.

        Args:
            drop_target (bool, optional): flag indicating if returned DataFrame should exclude target from columns,
                defaults to False
            exclude_transformed (bool, optional): flag indicating if returned DataFrame should exclude pre-transformed
                columns from transformed_features attributes, defaults to False

        Returns:
            pandas.DataFrame: dataframe with mapped data (according to mapping)
        """
        if self._mapped_dataframe is None:
            self._mapped_dataframe = self._create_mapped_dataframe()

        mapped_df = self._mapped_dataframe

        if drop_target:
            mapped_df = mapped_df.drop([self.target], axis=1)

        if exclude_transformed:
            mapped_df = mapped_df.drop(self.transformed_features, axis=1)

        return mapped_df

    def mapping(self):
        """Return _mapping attribute and if it's None, create it.

        Returns:
            dict: 'feature name': mapping dict pairs
        """
        if self._mapping is None:
            self._mapping = self._create_mapping()
        return self._mapping

    def descriptions(self):
        """Return _descriptions attribute and if it's None, create it.

        Returns:
            dict: 'feature name': description pairs
        """
        if self._descriptions is None:
            self._descriptions = self._create_descriptions()
        return self._descriptions

    def unused_features(self):
        """Return _unused_columns attribute."""
        return self._unused_columns

    def _create_features(self):
        """Return list of names as taken from name attribute of every FeatureClass present in _features.

        Returns:
            list: list of features names
        """
        output = []
        for feature in self._features.values():
            output.append(feature.name)
        output = sort_strings(output)
        return output

    def _create_categorical_features(self):
        """Return list of names of features (name attribute) if a given FeatureClass is an instance of
        CategoricalFeature.

        Returns:
            list: list of categorical features names
        """
        output = []
        for feature in self._features.values():
            if isinstance(feature, CategoricalFeature):
                output.append(feature.name)
        output = sort_strings(output)
        return output

    def _create_numerical_features(self):
        """Return list of names of features (name attribute) if a given FeatureClass is an instance of
        NumericalFeature.

        Returns:
            list: list of numerical features names
        """
        output = []
        for feature in self._features.values():
            if isinstance(feature, NumericalFeature):
                output.append(feature.name)
        output = sort_strings(output)
        return output

    def _create_mapped_dataframe(self):
        """Return pandas.Dataframe made from single mapped series (where appropriate) of every FeatureClass
        (data method).

        Returns:
            pandas.DataFrame: dataframe consisting of mapped series
        """
        return pd.concat([self._features[feature].data() for feature in self._features], axis=1)

    def _create_raw_dataframe(self):
        """Return pandas.DataFrame made from original series data of every FeatureClass.

        Distinction is needed as NumericalFeature defines only data method, whereas CategoricalFeature has both
        data and original_data methods.

        Returns:
            pandas.DataFrame: original DataFrame constructed from series in every Feature
        """
        # raw data needs to call .original_data(), as the default function returns already mapped data
        numeric = [feature.data() for feature in self._features.values()
                   if feature.name in self.numerical_features()]
        cat = [feature.original_data() for feature in self._features.values()
               if feature.name in self.categorical_features()]
        df = pd.concat([*numeric, *cat], axis=1)
        return df

    def _create_mapping(self):
        """Create dictionary of 'feature name': mapping dict pairs, where mapping dict is taken from mapping method
        of every FeatureClass.

        Returns:
            dict: 'feature name': mapping dict pairs
        """
        output = {}
        for feature in self.features():
            output[feature] = self._features[feature].mapping()
        return output

    def _create_descriptions(self):
        """Create dictionary of 'feature name': description pairs, where description is taken from description attribute
        of every FeatureClass.

        Returns:
            dict: 'feature name': description pairs
        """
        output = {}
        for feature in self.features():
            output[feature] = self._features[feature].description
        return output

    def __getitem__(self, arg):
        """Return arg item from _features attribute dictionary.

        Args:
            arg (str, Hashable): str representing the name of the feature

        Returns:
            Feature: FeatureClass present in _features attribute dictionary

        Raises:
            KeyError: when arg is not in _features
        """
        if arg not in self._features:
            raise KeyError

        return self._features[arg]
