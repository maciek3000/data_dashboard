import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from .features import NumericalFeature, CategoricalFeature
from .plot_design import PlotDesign
from .functions import calculate_numerical_bins, modify_histogram_edges


class Analyzer:
    """Analyze Features present in the data and provide results through different methods and properties.

    Attributes:
        features (features.Features): Features object with created FeaturesClass objects and the data underneath
        max_categories (int): limit of unique values in the data so that the feature is treated as Categorical
        default_plot_design (plot_design.PlotDesign): PlotDesign object
    """

    _categorical_suffix = "_categorical"
    _feature_description = "description"
    _feature_type = "type"
    _feature_mapping = "mapping"
    _feature_missing = "missing"

    def __init__(self, features):
        """Create Analyzer object.

        Args:
            features (features.Features): Features object with created FeaturesClass objects
        """
        self.features = features
        self.max_categories = features.max_categories
        self.default_plot_design = PlotDesign()

    def numerical_describe_df(self):
        """Return numerical DataFrame if len of Numerical features in the data is greater than 0, None otherwise.

        Returns:
            {pandas.DataFrame, None}: pandas.Dataframe when Numerical features are present, None otherwise.
        """
        if len(self.features.numerical_features()) > 0:
            return self._create_describe_df(self.features.numerical_features())
        else:
            return None

    def categorical_describe_df(self):
        """Return categorical DataFrame if len of Categorical features in the data is greater than 0, None otherwise.

        Returns:
            {pandas.DataFrame, None}: pandas.Dataframe when Categorical features are present, None otherwise.
        """
        if len(self.features.categorical_features()) > 0:
            return self._create_describe_df(self.features.categorical_features())
        else:
            return None

    def df_head(self):
        """Return transposed, first 5 rows of original DataFrame (excluding unusued features).

        Returns:
            pandas.DataFrame: head of the DataFrame, transposed
        """
        return self.features.raw_data()[self.features.features()].head().T

    def skipped_features(self):
        """Return list of unused features as calculated by Features object in features attribute.

        Returns:
            list: list of unused features
        """
        return self.features.unused_features()

    def features_pairplot_df(self):
        """Return data that will be used to create seaborn pairplot.

        Returns:
            pandas.DataFrame: DataFrame used for pairplot visualization.
        """
        df = self.features.data()[self.features.features()]
        return df

    def summary_statistics(self):
        """Return features data that will be used to create summary statistics section.

        Statistics are calculated with describe method of a DataFrame, with manual addition of the number of missing
        values in the data and all numbers being rounded to 4th decimal place. Additionally, every feature gets
        appended with it's description and type.

        Returns:
            dict: 'feature name': summary/statistics dict of results for every feature.
        """
        df = self.features.data().describe().T
        df[self._feature_missing] = np.sum(self.features.data().isna())  # .sum()

        # rounding here as JS doesn't have anything quick for nice formatting of numbers and it breaks the padding
        # later on
        d = df.T.round(4).to_dict()
        for key, feat_dict in d.items():
            feat_dict[self._feature_description] = self.features[key].description
            feat_dict[self._feature_type] = self.features[key].feature_type

        return d

    def histogram_data(self):
        """Return data on features that's required to plot Histogram visualization.

        If feature is Categorical, then the number of bins in histogram is equal to the number of unique values
        in the data. If the feature is Numerical, then the number of bins is calculated dynamically.

        Note:
            NaN values are dropped from calculations.

        Returns:
            dict: 'feature name': (histogram values, left edges, right edges) tuple pairs
        """
        all_histograms = {}

        for feature_name in self.features.features():
            feature = self.features[feature_name]
            # dropping NaN values - visualization shows untransformed data and nans are imputed only during
            # transformations
            series = feature.data().dropna()

            if isinstance(feature, CategoricalFeature):
                bins = len(series.unique())
            elif isinstance(feature, NumericalFeature):
                # if columns is numerical we calculate the number of bins manually
                bins = calculate_numerical_bins(series)
            else:
                bins = 1  # placeholder rule

            hist, edges = np.histogram(series, density=True, bins=bins)
            left_edges, right_edges = modify_histogram_edges(edges)  # modify edges by adding space between

            all_histograms[feature_name] = (hist, left_edges, right_edges)

        return all_histograms

    def correlation_data_normalized(self, random_state=None):
        """Return DataFrame with calculated correlations (Pearson) between features.

        Features are normalized using QuantileTransformer in comparison to calculating them on original (raw) data.

        Args:
            random_state (int, optional): integer for reproducibility on QuantileTransformer transformations, defaults
                to None

        Returns:
            pandas.DataFrame: dataframe with correlation between columns
        """
        df = self.features.data()
        qt = QuantileTransformer(output_distribution="normal", random_state=random_state)

        # ignoring warnings, mostly with n_quantiles being > than number of samples
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_df = qt.fit_transform(df)
        normalized_corr = pd.DataFrame(new_df, columns=df.columns).corr(method="pearson")
        return normalized_corr

    def correlation_data_raw(self):
        """Return DataFrame with calculated correlations (Pearson) between features.

        Returns:
            pandas.DataFrame: dataframe with correlation between columns
        """
        df = self.features.data()
        raw_corr = df.corr(method="pearson")
        return raw_corr

    def scatter_data(self):
        """Return data that will be used in ScatterPlot Visualizations.

        ScatterPlot Visualizations makes every feature (column) be used as a hue (coloring) of different combinations
        of features on plots. Bokeh's factor_cmap function that is used to provide coloring of CategoricalFeatures
        expects the data to be of Str type, whereas Plots expect the data to be numerical (to put them correctly on
        the axes). Therefore, every CategoricalFeature gets two copies of the data - one with float type and one
        with str type.

        Note:
            NaN values are dropped from calculations.

        Returns:
            dict: 'feature name': list of values
        """
        df = self.features.data().copy()
        for col in self.features.categorical_features():
            df[col + self._categorical_suffix] = df[col].astype(str)

        # dropping NaN values - visualization shows untransformed data and nans are imputed only during transformations
        scatter_data = df.dropna().to_dict(orient="list")
        return scatter_data

    def _create_describe_df(self, feature_list):
        """Return transposed pandas.DataFrame with summary statistics from describe method for features data in
        Features object. Additionally add percentage of missing values in the data.

        Args:
            feature_list (list): list of columns that will be extracted from Features data

        Returns:
            pandas.DataFrame: describe DataFrame
        """
        df = self.features.data()[feature_list]
        ds = df.describe().astype("float64").T
        ds[self._feature_missing] = df.isna().sum() / max(df.count())
        return ds

    def feature_list(self):
        """Return list of features as provided by Features.features method.

        Returns:
            list: list of features
        """
        return self.features.features()

    def unused_features(self):
        """Return list of unused features as provided by Features.unused_features method.

        Returns:
            list: list of unused features
        """
        return self.features.unused_features()

    def features_mapping(self):
        """Return mappings dict as provided by Features.mapping method.

        Returns:
            dict: features mappings dict
        """
        return self.features.mapping()

    def features_descriptions(self):
        """Return description dict as provided by Features.descriptions method.

        Returns:
            dict: features description dict
        """
        return self.features.descriptions()
