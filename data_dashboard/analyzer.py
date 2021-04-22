import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from .features import NumericalFeature, CategoricalFeature
from .plot_design import PlotDesign
from .functions import calculate_numerical_bins, modify_histogram_edges

import warnings


class Analyzer:
    """Analyzes Features present in the data.

        Main goal of the object is to analyze the data (features) provided and output any tables or plots
        used in the application. All calculations and transformations on "raw" features (excluding mapping) should
        happen here and Analyzer should expose methods or properties with the final output.
    """

    _categorical_suffix = "_categorical"
    _feature_description = "description"
    _feature_type = "type"
    _feature_mapping = "mapping"
    _feature_missing = "missing"

    def __init__(self, features):
        self.features = features
        self.max_categories = features.max_categories
        self.default_plot_design = PlotDesign()

    def numerical_describe_df(self):
        if len(self.features.numerical_features()) > 0:
            return self._create_describe_df(self.features.numerical_features())
        else:
            return None

    def categorical_describe_df(self):
        if len(self.features.categorical_features()) > 0:
            return self._create_describe_df(self.features.categorical_features())
        else:
            return None

    def df_head(self):
        return self.features.raw_data()[self.features.features()].head().T

    def skipped_features(self):
        return self.features.unused_features()

    def features_pairplot_df(self):
        df = self.features.data()[self.features.features()]
        return df

    def summary_statistics(self):
        df = self.features.data().describe().T
        df[self._feature_missing] = self.features.data().isna().sum()

        # rounding here as JS doesn't have anything quick for nice formatting of numbers and it breaks the padding
        # later on
        d = df.T.round(4).to_dict()
        for key, feat_dict in d.items():
            feat_dict[self._feature_description] = self.features[key].description
            feat_dict[self._feature_type] = self.features[key].feature_type

        return d

    def histogram_data(self):
        all_histograms = {}

        for feature_name in self.features.features():
            feature = self.features[feature_name]
            # dropping NaN values - visualization shows untransformed data and nans will be imputed during transformations
            series = feature.data().dropna()

            if isinstance(feature, CategoricalFeature):
                bins = len(series.unique())
            elif isinstance(feature, NumericalFeature):
                # if columns is numerical we calculate the number of bins manually
                bins = calculate_numerical_bins(series)
            else:
                bins = 1  # placeholder rule

            hist, edges = np.histogram(series, density=True, bins=bins)
            left_edges, right_edges = modify_histogram_edges(edges)

            all_histograms[feature_name] = (hist, left_edges, right_edges)

        return all_histograms

    def correlation_data_normalized(self, random_state=None):
        # when using Pearson coefficient data should follow Normal Distribution
        df = self.features.data()
        qt = QuantileTransformer(output_distribution="normal", random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_df = qt.fit_transform(df)
        normalized_corr = pd.DataFrame(new_df, columns=df.columns).corr(method="pearson")
        return normalized_corr

    def correlation_data_raw(self):
        df = self.features.data()
        raw_corr = df.corr(method="pearson")
        return raw_corr

    def scatter_data(self):
        # Every column will be treated as a hue (color) at some point, including categorical Columns
        # However, factor_cmap (function that provides coloring by categorical variables) expects color columns
        # to be Str, not Int/Float. Therefore, we need to create a copy of every categorical column
        # and cast it explicitly to Str (as they're already mapped to Ints so they can be plotted as XY coordinates)

        df = self.features.data().copy()
        for col in self.features.categorical_features():
            df[col + self._categorical_suffix] = df[col].astype(str)

        # dropping NaN values - visualization shows untransformed data and nans will be imputed during transformations
        scatter_data = df.dropna().to_dict(orient="list")
        return scatter_data

    def _create_describe_df(self, feature_list):
        df = self.features.data()[feature_list]
        ds = df.describe().astype("float64").T
        ds["missing"] = df.isna().sum() / max(df.count())
        return ds

    def feature_list(self):
        return self.features.features()

    def unused_features(self):
        return self.features.unused_features()

    def features_mapping(self):
        return self.features.mapping()

    def features_descriptions(self):
        return self.features.descriptions()
