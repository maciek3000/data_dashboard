import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from .features import NumericalFeature, CategoricalFeature
from .plots import InfoGrid, ScatterPlotGrid
from .plots import PairPlot
from .plot_design import PlotDesign

# Data Features should be created by Coordinator, and then Explainer should have functions to work on
# Categorical/Numerical variables based on DataFeatures object created


def calculate_numerical_bins(series):
    # https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    n = series.size
    iqr = series.quantile(0.75) - series.quantile(0.25)
    bins = (series.max() - series.min()) / ((2*iqr)*pow(n, (-1/3)))
    bins = int(round(bins, 0))

    # if the number of bins is greater than n, n is returned
    # this can be the case for small, skewed series
    if bins > n:
        bins = n
    return bins


def modify_histogram_edges(edges, interval_percentage=0.005):
    # Adding space between the edges to visualize the data properly
    interval = (max(edges) - min(edges)) * interval_percentage  # 0.5%
    left_edges = edges[:-1]
    right_edges = [edge - interval for edge in edges[1:]]
    return left_edges, right_edges


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
        return self._create_describe_df(self.features.numerical_features())

    def categorical_describe_df(self):
        return self._create_describe_df(self.features.categorical_features())

    def df_head(self):
        return self.features.raw_data()[self.features.features()].head().T

    def skipped_features(self):
        return self.features.unused_features()

    def features_pairplot_static(self):
        df = self.features.data()[self.features.features()]
        pairplot = PairPlot(self.default_plot_design).pairplot(df)
        return pairplot

    def infogrid(self, chosen_feature, feature_description_class):
        feature_list = self.features.features()
        infogrid = InfoGrid(
            features=feature_list,
            plot_design=self.default_plot_design,
            feature_description_class=feature_description_class
        )
        plot = infogrid.infogrid(
            summary_statistics=self._summary_statistics(),
            histogram_data=self._histogram_data(),
            correlation_data=self._correlation_data(),
            initial_feature=chosen_feature
        )
        return plot

    def scattergrid(self, chosen_feature, feature_description_class):
        feature_list = self.features.features()
        scattergrid = ScatterPlotGrid(
            features=feature_list,
            plot_design=self.default_plot_design,
            categorical_features=self.features.categorical_features(),
            feature_descriptions=self.features.descriptions(),
            feature_mapping=self.features.mapping(),
            feature_description_class=feature_description_class
        )
        plot = scattergrid.scattergrid(self._scatter_data(), chosen_feature)
        return plot

    def _summary_statistics(self):
        df = self.features.data().describe().T
        df[self._feature_missing] = self.features.data().isna().sum()

        # rounding here as JS doesn't have anything quick for nice formatting of numbers and it breaks the padding
        # later on
        d = df.T.round(4).to_dict()
        for key, feat_dict in d.items():
            feat_dict[self._feature_description] = self.features[key].description
            feat_dict[self._feature_type] = self.features[key].type

        return d

    def _histogram_data(self):
        all_histograms = {}

        for feature_name in self.features.features():
            feature = self.features[feature_name]
            # TODO: NaNs are not allowed
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

    def _correlation_data(self, random_state=None):
        # using Pearson coefficient - data should follow Normal Distribution
        df = self.features.data()
        qt = QuantileTransformer(output_distribution="normal", random_state=random_state)
        new_df = qt.fit_transform(df)
        corr = pd.DataFrame(new_df, columns=df.columns).corr(method="pearson").to_dict()
        return corr

    def _scatter_data(self):
        # Every column will be treated as a hue (color) at some point, including categorical Columns
        # However, factor_cmap (function that provides coloring by categorical variables) expects color columns
        # to be Str, not Int/Float. Therefore, we need to create a copy of every categorical column
        # and cast it explicitly to Str (as they're already mapped to Ints so they can be plotted as XY coordinates)

        df = self.features.data().copy()
        for col in self.features.categorical_features():
            df[col + self._categorical_suffix] = df[col].astype(str)

        # TODO: rethink NaNs
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

#
# #### ---------- DONT DELETE YET ------------ #####
#
# class Explainer:
#     """Analyzes Features present in the data.
#
#         Main goal of the object is to analyze the data (features) provided and output any tables or plots
#         used in the application. All calculations and transformations on "raw" features (excluding mapping) should
#         happen here and Analyzer should expose methods or properties with the final output.
#     """
#
#
#     key_cols = "columns"
#     key_cols_wo_target = "columns_without_target"
#     key_figs = "figures"
#     key_tables = "tables"
#     key_lists = "lists"
#     key_histograms = "histograms"
#     key_scatter = "scatter"
#
#     key_numerical = "numerical"
#     key_categorical = "categorical"
#     key_date = "date"
#
#
#
#     plot_text_color = "#8C8C8C"
#     plot_color = "#19529c"
#
#     def __init__(self, features):
#         sns.set_style("white", {
#             "axes.edgecolor": self.plot_text_color,
#             "axes.labelcolor": self.plot_text_color,
#             "text.color": self.plot_text_color,
#             "font.sans-serif": ["Lato"],
#             "xtick.color": self.plot_text_color,
#             "ytick.color": self.plot_text_color,
#         })
#
#         self.features = features
#         self.max_categories = features.max_categories
#
#         # self.columns = features.columns
#         #
#         # self.numerical_columns = features.numerical_columns
#         # self.categorical_columns = features.categorical_columns
#         # self.date_columns = features.date_columns
#
#         # self.transformed_df, self.mapping = self._transform_raw_df()
#
#     def analyze(self):
#         output = {
#             self.key_cols: self.features.features(),
#             self.key_cols_wo_target: self.features.features(drop_target=True),
#             self.key_numerical: self.features.numerical_features(),
#             self.key_categorical: self.features.categorical_features(),
#             self.key_figs: self._create_figures(),
#             self.key_tables: self._create_tables(),
#             self.key_lists: self._create_lists(),
#             self.key_histograms: self._create_histogram_data(),
#             self.key_scatter: self._create_scatter_data()
#         }
#         return output
#
#     # def _analyze_columns(self):
#     #     columns = [
#     #         self.key_numerical,
#     #         self.key_categorical,
#     #         self.key_date
#     #     ]
#     #     output = {key: list() for key in columns}
#     #
#     #     for col in self.X.columns:
#     #         result = self.__column_type(col)
#     #         output[result].append(col)
#     #
#     #     x_cols = output
#     #     xy_cols = copy.deepcopy(output)  # deepcopy is needed to not overwrite x_cols dict
#     #
#     #     y_col_name = self.y.name
#     #     xy_cols[self.__column_type(y_col_name)].append(y_col_name)
#     #
#     #     x_cols = {key: sorted(x_cols[key], key=str.upper) for key in x_cols}
#     #     xy_cols = {key: sorted(xy_cols[key], key=str.upper) for key in xy_cols}
#     #
#     #     return {
#     #         self.key_cols: xy_cols,
#     #         self.key_cols_wo_target: x_cols
#     #     }
#     #
#     # def __column_type(self, col):
#     #     if self.raw_df[col].dtype == "bool":
#     #         return self.key_numerical
#     #     else:
#     #         try:
#     #             _ = self.raw_df[col].astype("float64")
#     #             if len(_.unique()) <= self.max_categories:
#     #                 return self.key_categorical
#     #             else:
#     #                 return self.key_numerical
#     #         except TypeError:
#     #             return self.key_date
#     #         except ValueError:
#     #             return self.key_categorical
#     #         except Exception:
#     #             raise
#     #
#     # def _transform_raw_df(self):
#     #     mapping = self._create_categorical_mapping()
#     #
#     #     categorical_df = self.raw_df[self.categorical_columns].replace(mapping)
#     #     other_cols = self.numerical_columns + self.date_columns
#     #
#     #     new_df = pd.concat([categorical_df, self.raw_df[other_cols]], axis=1)
#     #     new_df[self.numerical_columns] = new_df[self.numerical_columns].astype("float64")
#     #     return new_df, mapping
#     #
#     # def _create_categorical_mapping(self):
#     #     _ = {}
#     #     for col in self.categorical_columns:
#     #         vals = sorted(self.raw_df[col].unique(), key=str)
#     #         mapped = {val: x for val, x in zip(vals, range(len(vals))) if not pd.isna(val)}  # count starts at 1
#     #         _[col] = mapped
#     #
#     #     return _
#
#     def _create_tables(self):
#         d = {
#             "described_numeric": self._numeric_describe(),
#             "described_categorical": self._categorical_describe(),
#             "head": self.__create_df_head()
#         }
#         return d
#
#     def _numeric_describe(self):
#         # df consists of only numerical features
#         df = self.features.data()[self.features.numerical_features()]
#
#         ds = df.describe().astype("float64").T
#         ds["missing"] = df.isna().sum() / max(df.count())
#         return ds
#
#     def _categorical_describe(self):
#         # df consists of only categorical features
#         df = self.features.data()[self.features.categorical_features()]
#
#         ds = df.describe().astype("float64").T
#         ds["missing"] = df.isna().sum() / max(df.count())
#         return ds
#
#     def __create_df_head(self):
#         # cols = sorted(self.features.raw_data)
#         return self.features.raw_data()[sorted(self.features.features())].head().T
#
#     def _create_lists(self):
#         return {
#             "unused_cols": self._unused_cols()
#         }
#
#     def _unused_cols(self):
#         # TODO: implement unused columns
#         return list()
#
#     def _create_figures(self):
#         d = {
#             "pairplot": self.__create_pairplot()
#         }
#         return d
#
#     def __create_pairplot(self):
#         # num = self.transformed_df[self.numerical_columns]
#         # cat = self.transformed_df[self.categorical_columns]
#         df = self.features.data()
#         df = df[sorted(df.columns)]
#         colors = {"color": self.plot_color}
#         p = sns.pairplot(df, plot_kws=colors, diag_kws=colors)
#         return p
#
#     def _create_histogram_data(self):
#         #TODO: separate deciding on number of bins / replacing edge values
#         #TODO: clean code
#
#         _ = {}
#
#         df = self.features.data()
#         for column in df.columns:
#             # removing NaN values
#             # TODO: this doesnt make sense
#             srs = df[column][df[column].notna()]
#
#             # if column is categorical we do not want too many bins
#             if column in self.features.categorical_features():
#                 unique_vals = len(df[column].unique())
#                 # if unique_vals > self.max_categories:
#                 #     bins = self.max_categories
#                 # else:
#                 bins = unique_vals
#
#             # if columns is numerical we calculate the number of bins manually
#             # https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
#             elif column in self.features.numerical_features():
#                 n = srs.size
#                 iqr = srs.quantile(0.75) - srs.quantile(0.25)
#                 bins = (srs.max() - srs.min()) / (2*iqr*pow(n, (-1/3)))
#                 bins = int(round(bins, 0))
#             else:
#                 # placeholder rule
#                 bins = 1
#
#             hist, edges = np.histogram(srs, density=True, bins=bins)
#
#             interval = (max(edges) - min(edges)) * 0.005  # 0.5%
#             left_edges = edges[:-1]
#             right_edges = [edge - interval for edge in edges[1:]]
#
#             # _[column] = (hist, edges)
#             _[column] = (hist, left_edges, right_edges)
#
#         return _
#
#     def _create_scatter_data(self):
#         # returning df as creating multiple rows of scatter plots will require a lot of data manipulation
#         df = self.features.data()
#         # cat_df = df[self.categorical_columns].astype("float64").fillna(-1)
#         # num_df = df.drop(self.categorical_columns, axis=1).astype("float64").fillna(-1)
#         #
#         # df = pd.concat([cat_df, num_df], axis=0)
#         # df = df.astype("float64")
#         for col in self.features.categorical_features():
#             df[col + "_categorical"] = df[col].astype(str)
#         scatter_data = df.dropna().to_dict(orient="list")
#
#         # factor_cmap expects categorical data to be Str, not Int/Float
#         # for col in self.categorical_columns:
#         #     scatter_data[col] = list(map(str, scatter_data[col]))
#
#         return scatter_data
#
#
#
# class DataExplainer:
#
#     key_cols = "columns"
#     key_cols_wo_target = "columns_without_target"
#     key_figs = "figures"
#     key_tables = "tables"
#     key_lists = "lists"
#     key_histograms = "histograms"
#     key_scatter = "scatter"
#
#     key_numerical = "numerical"
#     key_categorical = "categorical"
#     key_date = "date"
#
#     max_categories = 10
#
#     plot_text_color = "#8C8C8C"
#     plot_color = "#19529c"
#
#     def __init__(self, X, y):
#         sns.set_style("white", {
#             "axes.edgecolor": self.plot_text_color,
#             "axes.labelcolor": self.plot_text_color,
#             "text.color": self.plot_text_color,
#             "font.sans-serif": ["Lato"],
#             "xtick.color": self.plot_text_color,
#             "ytick.color": self.plot_text_color,
#         })
#
#         self.X = X
#         self.y = y
#         self.raw_df = pd.concat([self.X, self.y], axis=1)
#
#         self.columns = self._analyze_columns()
#
#         self.numerical_columns = self.columns[self.key_cols][self.key_numerical]
#         self.categorical_columns = self.columns[self.key_cols][self.key_categorical]
#         self.date_columns = self.columns[self.key_cols][self.key_date]
#
#         self.transformed_df, self.mapping = self._transform_raw_df()
#
#     def analyze(self):
#         output = {
#             self.key_cols: self.columns,
#             self.key_numerical: self.numerical_columns,
#             self.key_categorical: self.categorical_columns,
#             self.key_figs: self._create_figures(),
#             self.key_tables: self._create_tables(),
#             self.key_lists: self._create_lists(),
#             self.key_histograms: self._create_histogram_data(),
#             self.key_scatter: self._create_scatter_data()
#         }
#         return output
#
#     def _analyze_columns(self):
#         columns = [
#             self.key_numerical,
#             self.key_categorical,
#             self.key_date
#         ]
#         output = {key: list() for key in columns}
#
#         for col in self.X.columns:
#             result = self.__column_type(col)
#             output[result].append(col)
#
#         x_cols = output
#         xy_cols = copy.deepcopy(output)  # deepcopy is needed to not overwrite x_cols dict
#
#         y_col_name = self.y.name
#         xy_cols[self.__column_type(y_col_name)].append(y_col_name)
#
#         x_cols = {key: sorted(x_cols[key], key=str.upper) for key in x_cols}
#         xy_cols = {key: sorted(xy_cols[key], key=str.upper) for key in xy_cols}
#
#         return {
#             self.key_cols: xy_cols,
#             self.key_cols_wo_target: x_cols
#         }
#
#     def __column_type(self, col):
#         if self.raw_df[col].dtype == "bool":
#             return self.key_numerical
#         else:
#             try:
#                 _ = self.raw_df[col].astype("float64")
#                 if len(_.unique()) <= self.max_categories:
#                     return self.key_categorical
#                 else:
#                     return self.key_numerical
#             except TypeError:
#                 return self.key_date
#             except ValueError:
#                 return self.key_categorical
#             except Exception:
#                 raise
#
#     def _transform_raw_df(self):
#         mapping = self._create_categorical_mapping()
#
#         categorical_df = self.raw_df[self.categorical_columns].replace(mapping)
#         other_cols = self.numerical_columns + self.date_columns
#
#         new_df = pd.concat([categorical_df, self.raw_df[other_cols]], axis=1)
#         new_df[self.numerical_columns] = new_df[self.numerical_columns].astype("float64")
#         return new_df, mapping
#
#     def _create_categorical_mapping(self):
#         _ = {}
#         for col in self.categorical_columns:
#             vals = sorted(self.raw_df[col].unique(), key=str)
#             mapped = {val: x for val, x in zip(vals, range(len(vals))) if not pd.isna(val)}  # count starts at 1
#             _[col] = mapped
#
#         return _
#
#     def _create_tables(self):
#         d = {
#             "described_numeric": self._numeric_describe(),
#             "described_categorical": self._categorical_describe(),
#             "head": self.__create_df_head()
#         }
#         return d
#
#     def _numeric_describe(self):
#         ds = self.transformed_df[self.numerical_columns].describe().astype("float64").T
#         ds["missing"] = self.transformed_df.isna().sum() / max(self.transformed_df.count())
#         return ds
#
#     def _categorical_describe(self):
#         df = self.transformed_df[self.categorical_columns].describe().astype("float64").T
#         df["missing"] = self.transformed_df.isna().sum() / max(self.transformed_df.count())
#         return df
#
#     def __create_df_head(self):
#         cols = sorted(self.raw_df.columns)
#         return self.raw_df[cols].head().T
#
#     def _create_lists(self):
#         return {
#             "unused_cols": self._unused_cols()
#         }
#
#     def _unused_cols(self):
#         return self.date_columns
#
#     def _create_figures(self):
#         d = {
#             "pairplot": self.__create_pairplot()
#         }
#         return d
#
#     def __create_pairplot(self):
#         num = self.transformed_df[self.numerical_columns]
#         cat = self.transformed_df[self.categorical_columns]
#         df = pd.concat([num, cat], axis=1)
#         df = df[sorted(df.columns)]
#         colors = {"color": self.plot_color}
#         p = sns.pairplot(df, plot_kws=colors, diag_kws=colors)
#         return p
#
#     def _create_histogram_data(self):
#         #TODO: separate deciding on number of bins / replacing edge values
#         #TODO: clean code
#
#         _ = {}
#
#         for column in self.transformed_df.columns:
#             # removing NaN values
#             srs = self.transformed_df[column][self.transformed_df[column].notna()]
#
#             # if column is categorical we do not want too many bins
#             if column in self.categorical_columns:
#                 unique_vals = len(self.transformed_df[column].unique())
#                 # if unique_vals > self.max_categories:
#                 #     bins = self.max_categories
#                 # else:
#                 bins = unique_vals
#
#             # if columns is numerical we calculate the number of bins manually
#             # https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
#             elif column in self.numerical_columns:
#                 n = srs.size
#                 iqr = srs.quantile(0.75) - srs.quantile(0.25)
#                 bins = (srs.max() - srs.min()) / (2*iqr*pow(n, (-1/3)))
#                 bins = int(round(bins, 0))
#             else:
#                 # placeholder rule
#                 bins = 1
#
#             hist, edges = np.histogram(srs, density=True, bins=bins)
#
#             interval = (max(edges) - min(edges)) * 0.005  # 0.5%
#             left_edges = edges[:-1]
#             right_edges = [edge - interval for edge in edges[1:]]
#
#             # _[column] = (hist, edges)
#             _[column] = (hist, left_edges, right_edges)
#
#         return _
#
#     def _create_scatter_data(self):
#         # returning df as creating multiple rows of scatter plots will require a lot of data manipulation
#         df = self.transformed_df
#         # cat_df = df[self.categorical_columns].astype("float64").fillna(-1)
#         # num_df = df.drop(self.categorical_columns, axis=1).astype("float64").fillna(-1)
#         #
#         # df = pd.concat([cat_df, num_df], axis=0)
#         # df = df.astype("float64")
#         for col in self.categorical_columns:
#             df[col + "_categorical"] = df[col].astype(str)
#         scatter_data = df.dropna().to_dict(orient="list")
#
#         # factor_cmap expects categorical data to be Str, not Int/Float
#         # for col in self.categorical_columns:
#         #     scatter_data[col] = list(map(str, scatter_data[col]))
#
#         return scatter_data
