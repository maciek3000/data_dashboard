import pandas as pd


def sort_strings(l):
    return sorted(l, key=lambda x: x.upper())


class FeatureNotSupported(Exception):
    pass


class BaseFeature(object):
    """Base class for Features."""

    type = None

    def data(self):
        raise NotImplemented

    def mapping(self):
        raise NotImplemented


class CategoricalFeature(BaseFeature):
    """Categorical Feature class.

        Categorical Features are those that have values that are limited and/or fixed in their nature.
        However, it doesn't mean that we can't analyze them in similar manner to Numerical Features - e.g.
        calculate mean, distribution and other variables. To do that, we need to assign every unique value
        present in the data to the unique number. This allows calculations to happen on the data.

        In case of some Categorical Features, they might already be represented in the data with key of some sort:
            "A": "Apple"
            "B": "Banana"
        This structure is also present in the dict descriptions that can be fed to the analysis.
        Raw mapping would associate those values with the unique numbers created during the mapping:
            "A": 1
            "B": 2

        What CategoricalFeature does is that it creates mapping between new unique numbers present in the data
        and the description (item) of the already provided mapping:
            1: "Apple"
            2: "Banana"

        This way of mapping things should ease out mental connections between what is seen in the visualizations
        (numbers mostly) and whats really represented behind those numbers (instead of their symbols).
    """

    type = "Categorical"

    def __init__(self, series, name, description, imputed_category, mapping=None):

        self.series = series.copy()
        self.name = name
        self.description = description
        self.original_mapping = mapping
        self.imputed_category = imputed_category  # flag to check if type of feature was provided or imputed

        self.raw_mapping = self._create_raw_mapping()
        self.mapped_series = self._create_mapped_series()

        self._descriptive_mapping = None

    def data(self):
        return self.mapped_series

    def original_data(self):
        return self.series

    def mapping(self):
        if not self._descriptive_mapping:
            self._descriptive_mapping = self._create_descriptive_mapping()

        return self._descriptive_mapping

    def _create_mapped_series(self):
        return self.series.replace(self.raw_mapping)

    def _create_raw_mapping(self):
        # replaces every categorical value with a number starting from 1 (sorted alphabetically)
        # it starts with 1 to be consistent with "count" obtained with .describe() methods on dataframes
        values = sorted(self.series.unique(), key=str)
        mapped = {value: number for number, value in enumerate(values, start=1) if not pd.isna(value)}
        return mapped

    def _create_descriptive_mapping(self):
        # Key is the "new" value provided with enumerating unique values
        # Item is either the description of the category taken from original descriptions or the original value
        if self.original_mapping:
            mapp = {}
            for key, item in self.raw_mapping.items():
                new_key = item

                # try/except clause needed as json keys can only be str()
                # however, sometimes those keys can be treated as integers and then they won't be found
                # in the corresponding json object
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
    """Numerical Feature class.

        Numerical Feature is a feature in the data values of which are treated as numbers.
    """

    type = "Numerical"

    def __init__(self, series, name, description, imputed_category):

        self.series = series.copy()
        self.name = name
        self.description = description
        self.imputed_category = imputed_category  # flag to check if type of feature was provided or imputed
        self.normalized_series = None

    def data(self):
        return self.series

    def mapping(self):
        return None


class Features:
    """Container for Features in the analyzed data.

        Data comes with different types of features - numerical, categorical, date, bool, etc. Goal of this class
        is to have the data in one place, but also provide any corresponding metadata that might be needed.
        This might include descriptions for every feature, respective mapping between original and mapped Categorical
        Features, etc.

        Please note, that target variable (y) is treated as a feature by default - you need to explicitly states in
        methods if you wish to not include it.

        available_categories is a work in progress (?) property that defines what type of features are implemented.
        max_categories defines what is a limit of unique values in a column for it to be treated as categorical
            (even if the data itself comes as a int/float type).
    """

    categorical = "cat"
    numerical = "num"
    date = "date"
    available_categories = [categorical, numerical]

    max_categories = 10

    _description_not_available = "Description not Available"

    def __init__(self, X, y, descriptor=None):

        self.original_dataframe = pd.concat([X, y], axis=1).copy()
        self.target = y.name

        self._all_features = None
        self._categorical_features = None
        self._numerical_features = None
        self._unused_columns = []

        self._raw_dataframe = None
        self._mapped_dataframe = None

        self._mapping = None
        self._descriptions = None

        # returns {feature_name: feature object} dict
        self._features = self._analyze_features(descriptor)

    def _analyze_features(self, descriptor):
        features = {}

        for column in self.original_dataframe.columns:
            # TODO: rethink if can be split into smaller functions
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
                if (not category) or (category not in self.available_categories):
                    category = self._impute_column_type(self.original_dataframe[column])
                    category_imputed = True

                # default Description
                if description is None:
                    description = self._description_not_available

                if category == self.categorical:  # Categorical
                    feature = CategoricalFeature(
                        series=self.original_dataframe[column],
                        name=column,
                        description=description,
                        mapping=mapping,
                        imputed_category=category_imputed
                    )

                elif category == self.numerical:  # Numerical
                    feature = NumericalFeature(
                        series=self.original_dataframe[column],
                        name=column,
                        description=description,
                        imputed_category=category_imputed
                    )

                else:
                    raise FeatureNotSupported("Feature Category not supported: {}".format(category))

                features[column] = feature

            except FeatureNotSupported:
                self._unused_columns.append(column)

        return features

    def _impute_column_type(self, series):
        if series.dtype == bool:
            return self.categorical
        else:
            try:
                _ = series.astype("float64")
                if len(_.unique()) <= self.max_categories:
                    return self.categorical
                else:
                    return self.numerical
            except TypeError:
                return self.date
            except ValueError:
                return self.categorical
            except Exception:
                raise

    def features(self, drop_target=False):
        if not self._all_features:
            self._all_features = self._create_features()

        if drop_target:
            return [feature for feature in self._all_features if feature != self.target]
        else:
            return self._all_features

    def categorical_features(self, drop_target=False):
        if not self._categorical_features:
            self._categorical_features = self._create_categorical_features()

        if drop_target:
            return [feature for feature in self._categorical_features if feature != self.target]
        else:
            return self._categorical_features

    def numerical_features(self, drop_target=False):
        if not self._numerical_features:
            output = []
            for feature in self._features.values():
                if isinstance(feature, NumericalFeature):
                    output.append(feature.name)
            self._numerical_features = sort_strings(output)

        if drop_target:
            return [feature for feature in self._numerical_features if feature != self.target]
        else:
            return self._numerical_features

    def raw_data(self, drop_target=False):
        if self._raw_dataframe is None:
            self._raw_dataframe = self._create_raw_dataframe()

        if drop_target:
            return self._raw_dataframe.drop([self.target], axis=1)
        else:
            return self._raw_dataframe

    def data(self, drop_target=False):
        if self._mapped_dataframe is None:
            self._mapped_dataframe = self._create_mapped_dataframe()

        if drop_target:
            return self._mapped_dataframe.drop([self.target], axis=1)
        else:
            return self._mapped_dataframe

    def mapping(self):
        if self._mapping is None:
            self._mapping = self._create_mapping()
        return self._mapping

    def descriptions(self):
        if self._descriptions is None:
            self._descriptions = self._create_descriptions()
        return self._descriptions

    def unused_features(self):
        return self._unused_columns

    def _create_features(self):
        output = []
        for feature in self._features.values():
            output.append(feature.name)
        output = sort_strings(output)
        return output

    def _create_categorical_features(self):
        output = []
        for feature in self._features.values():
            if isinstance(feature, CategoricalFeature):
                output.append(feature.name)
        output = sort_strings(output)
        return output

    def _create_numerical_features(self):
        output = []
        for feature in self._features.values():
            if isinstance(feature, NumericalFeature):
                output.append(feature.name)
        output = sort_strings(output)
        return output

    def _create_mapped_dataframe(self):
        return pd.concat([self._features[feature].data() for feature in self._features], axis=1)

    def _create_raw_dataframe(self):
        # raw data needs to call .original_data(), as the default function returns already mapped data
        numeric = [feature.data() for feature in self._features.values()
                   if feature.name in self.numerical_features()]
        cat = [feature.original_data() for feature in self._features.values()
               if feature.name in self.categorical_features()]
        df = pd.concat([*numeric, *cat], axis=1)
        return df

    def _create_mapping(self):
        output = {}
        for feature in self.features():
            output[feature] = self._features[feature].mapping()
        return output

    def _create_descriptions(self):
        output = {}
        for feature in self.features():
            output[feature] = self._features[feature].description
        return output

    def __getitem__(self, arg):
        if arg not in self._features:
            raise KeyError

        return self._features[arg]
