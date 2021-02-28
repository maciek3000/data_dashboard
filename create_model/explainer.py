import copy
import seaborn as sns
import pandas as pd
import numpy as np


class FeatureNotSupported(Exception):
    pass


class BaseFeature(object):

    type = None

    def data(self):
        raise NotImplemented

    def describe(self):
        raise NotImplemented


class CategoricalFeature(BaseFeature):

    type = "Categorical"

    def __init__(self, series, name, description, imputed_category, json_mapping=None):
        self.series = series.copy()

        self.name = name
        self.description = description
        self.json_mapping = json_mapping
        self.imputed_category = imputed_category

        self.raw_mapping = self._create_raw_mapping()
        self.mapped_series = self._create_mapped_series()

        self._descriptive_mapping = None

    def data(self):
        return self.mapped_series

    def describe(self):
        return self.mapped_series.describe()

    def original_data(self):
        return self.series

    def mapping(self):
        # will be used by Coordinator to provide mapping between human friendly and computer friendly variables
        if not self._descriptive_mapping:
            if self.json_mapping:
                mapp = {}
                for key, item in self.raw_mapping.items():
                    new_key = item
                    try:
                        new_item = self.json_mapping[key]
                    except KeyError:
                        try:
                            new_item = self.json_mapping[str(key)]
                        except KeyError:
                            raise
                    mapp[new_key] = new_item
            else:
                mapp = self.raw_mapping
            self._descriptive_mapping = mapp

        return self._descriptive_mapping

    def _create_mapped_series(self):
        return self.series.replace(self.raw_mapping)

    def _create_raw_mapping(self):
        # replaces every categorical value with a number starting from 1 (sorted alphabetically)
        values = sorted(self.series.unique(), key=str)
        # TODO: change to enumerate
        mapped = {value: x for value, x in zip(values, range(1, len(values) +1)) if not pd.isna(value)}  # count starts at 1
        return mapped


class NumericalFeature(BaseFeature):

    type = "Numerical"

    def __init__(self, series, name, description, imputed_category):
        self.series = series.copy()

        self.name = name
        self.description = description
        self.imputed_category = imputed_category
        self.normalized_series = None

    def data(self):
        return self.series

    def describe(self):
        return self.series.describe()

    def mapping(self):
        return None


class DataFeatures:

    categorical = "cat"
    numerical = "num"
    date = "date"
    available_categories = [categorical, numerical]

    max_categories = 10

    def __init__(self, original_dataframe, target, descriptions=None):

        self.original_dataframe = original_dataframe.copy()
        self.target = target

        self._features = self.analyze_features(descriptions)

        self._all_features = None
        self._all_features_wo_target = None
        self._categorical_features = None
        self._categorical_features_wo_target = None
        self._numerical_features = None
        self._numerical_features_wo_target = None
        self._unused_columns = None

        self._raw_dataframe = None
        self._mapped_dataframe = None

        self._mapping = None

    def analyze_features(self, descriptions):
        features = {}

        for column in self.original_dataframe.columns:
            description = "Description not Available"
            category = None
            mapping = None
            category_imputed = False

            # JSON keys extracted
            if column in descriptions.keys():

                try:
                    description = descriptions.description(column)
                except KeyError:
                    pass

                try:
                    category = descriptions.category(column)
                except KeyError:
                    pass

                try:
                    mapping = descriptions.mapping(column)
                except KeyError:
                    pass

            # category imputed in case it wasn't extracted from JSON
            if (not category) or (category not in self.available_categories):
                category = self._impute_column_type(self.original_dataframe[column])
                category_imputed = True

            if category == self.categorical:  # Categorical
                feature = CategoricalFeature(
                    series=self.original_dataframe[column],
                    name=column,
                    description=description,
                    json_mapping=mapping,
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

        return features

    def _impute_column_type(self, series):

        if series.dtype == "bool":
            return self.numerical
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
            output = []
            for feature in self._features.values():
                output.append(feature.name)
            self._all_features = output

        if drop_target:
            if self._all_features_wo_target is None:
                no_target = self._all_features.copy()
                no_target.remove(self.target)
                self._all_features_wo_target = no_target
            return self._all_features_wo_target
        else:
            return self._all_features

    def categorical_features(self, drop_target=False):
        if not self._categorical_features:
            output = []
            for feature in self._features.values():
                if isinstance(feature, CategoricalFeature):
                    output.append(feature.name)
            self._categorical_features = output

        if drop_target:
            if self.target in self._categorical_features:
                no_target = self._categorical_features.copy()
                no_target.remove(self.target)
                self._categorical_features_wo_target = no_target
            else:
                self._categorical_features_wo_target = self._categorical_features
            return self._categorical_features_wo_target
        else:
            return self._categorical_features

    def numerical_features(self, drop_target=False):
        if not self._numerical_features:
            output = []
            for feature in self._features.values():
                if isinstance(feature, NumericalFeature):
                    output.append(feature.name)
            self._numerical_features = output

        if drop_target:
            if self.target in self._numerical_features:
                no_target = self._numerical_features.copy()
                no_target.remove(self.target)
                self._numerical_features_wo_target = no_target
            else:
                self.__numerical_features_wo_target = self._numerical_features
            return self._numerical_features_wo_target
        else:
            return self._numerical_features

    def raw_data(self, drop_target=False):
        if self._raw_dataframe is None:
            numeric = [feature.data() for feature in self._features.values() if feature.name in self.numerical_features()]
            cat = [feature.original_data() for feature in self._features.values() if feature.name in self.categorical_features()]
            self._raw_dataframe = pd.concat([*numeric, *cat], axis=1)

        if drop_target:
            return self._raw_dataframe.drop([self.target], axis=1)
        else:
            return self._raw_dataframe

    def mapped_data(self, drop_target=False):
        if self._mapped_dataframe is None:
            self._mapped_dataframe = pd.concat([self._features[feature].data() for feature in self._features], axis=1)

        if drop_target:
            return self._mapped_dataframe.drop([self.target], axis=1)
        else:
            return self._mapped_dataframe

    def mapping(self):
        if self._mapping is None:
            output = {}
            for feature in self.features():
                output[feature] = self._features[feature].mapping()
            self._mapping = output
        return self._mapping


# Data Features should be created by Coordinator, and then Explainer should have functions to work on
# Categorical/Numerical variables based on DataFeatures object created

class DataExplainer:

    key_cols = "columns"
    key_cols_wo_target = "columns_without_target"
    key_figs = "figures"
    key_tables = "tables"
    key_lists = "lists"
    key_histograms = "histograms"
    key_scatter = "scatter"

    key_numerical = "numerical"
    key_categorical = "categorical"
    key_date = "date"

    max_categories = 10

    plot_text_color = "#8C8C8C"
    plot_color = "#19529c"

    def __init__(self, features):
        sns.set_style("white", {
            "axes.edgecolor": self.plot_text_color,
            "axes.labelcolor": self.plot_text_color,
            "text.color": self.plot_text_color,
            "font.sans-serif": ["Lato"],
            "xtick.color": self.plot_text_color,
            "ytick.color": self.plot_text_color,
        })

        self.features = features

        # self.columns = features.columns
        #
        # self.numerical_columns = features.numerical_columns
        # self.categorical_columns = features.categorical_columns
        # self.date_columns = features.date_columns

        # self.transformed_df, self.mapping = self._transform_raw_df()

    def analyze(self):
        output = {
            self.key_cols: self.features.features(),
            self.key_cols_wo_target: self.features.features(drop_target=True),
            self.key_numerical: self.features.numerical_features(),
            self.key_categorical: self.features.categorical_features(),
            self.key_figs: self._create_figures(),
            self.key_tables: self._create_tables(),
            self.key_lists: self._create_lists(),
            self.key_histograms: self._create_histogram_data(),
            self.key_scatter: self._create_scatter_data()
        }
        return output

    # def _analyze_columns(self):
    #     columns = [
    #         self.key_numerical,
    #         self.key_categorical,
    #         self.key_date
    #     ]
    #     output = {key: list() for key in columns}
    #
    #     for col in self.X.columns:
    #         result = self.__column_type(col)
    #         output[result].append(col)
    #
    #     x_cols = output
    #     xy_cols = copy.deepcopy(output)  # deepcopy is needed to not overwrite x_cols dict
    #
    #     y_col_name = self.y.name
    #     xy_cols[self.__column_type(y_col_name)].append(y_col_name)
    #
    #     x_cols = {key: sorted(x_cols[key], key=str.upper) for key in x_cols}
    #     xy_cols = {key: sorted(xy_cols[key], key=str.upper) for key in xy_cols}
    #
    #     return {
    #         self.key_cols: xy_cols,
    #         self.key_cols_wo_target: x_cols
    #     }
    #
    # def __column_type(self, col):
    #     if self.raw_df[col].dtype == "bool":
    #         return self.key_numerical
    #     else:
    #         try:
    #             _ = self.raw_df[col].astype("float64")
    #             if len(_.unique()) <= self.max_categories:
    #                 return self.key_categorical
    #             else:
    #                 return self.key_numerical
    #         except TypeError:
    #             return self.key_date
    #         except ValueError:
    #             return self.key_categorical
    #         except Exception:
    #             raise
    #
    # def _transform_raw_df(self):
    #     mapping = self._create_categorical_mapping()
    #
    #     categorical_df = self.raw_df[self.categorical_columns].replace(mapping)
    #     other_cols = self.numerical_columns + self.date_columns
    #
    #     new_df = pd.concat([categorical_df, self.raw_df[other_cols]], axis=1)
    #     new_df[self.numerical_columns] = new_df[self.numerical_columns].astype("float64")
    #     return new_df, mapping
    #
    # def _create_categorical_mapping(self):
    #     _ = {}
    #     for col in self.categorical_columns:
    #         vals = sorted(self.raw_df[col].unique(), key=str)
    #         mapped = {val: x for val, x in zip(vals, range(len(vals))) if not pd.isna(val)}  # count starts at 1
    #         _[col] = mapped
    #
    #     return _

    def _create_tables(self):
        d = {
            "described_numeric": self._numeric_describe(),
            "described_categorical": self._categorical_describe(),
            "head": self.__create_df_head()
        }
        return d

    def _numeric_describe(self):
        # df consists of only numerical features
        df = self.features.mapped_data()[self.features.numerical_features()]

        ds = df.describe().astype("float64").T
        ds["missing"] = df.isna().sum() / max(df.count())
        return ds

    def _categorical_describe(self):
        # df consists of only categorical features
        df = self.features.mapped_data()[self.features.categorical_features()]

        ds = df.describe().astype("float64").T
        ds["missing"] = df.isna().sum() / max(df.count())
        return ds

    def __create_df_head(self):
        # cols = sorted(self.features.raw_data)
        return self.features.raw_data()[sorted(self.features.features())].head().T

    def _create_lists(self):
        return {
            "unused_cols": self._unused_cols()
        }

    def _unused_cols(self):
        # TODO: implement unused columns
        return list()

    def _create_figures(self):
        d = {
            "pairplot": self.__create_pairplot()
        }
        return d

    def __create_pairplot(self):
        # num = self.transformed_df[self.numerical_columns]
        # cat = self.transformed_df[self.categorical_columns]
        df = self.features.mapped_data()
        df = df[sorted(df.columns)]
        colors = {"color": self.plot_color}
        p = sns.pairplot(df, plot_kws=colors, diag_kws=colors)
        return p

    def _create_histogram_data(self):
        #TODO: separate deciding on number of bins / replacing edge values
        #TODO: clean code

        _ = {}

        df = self.features.mapped_data()
        for column in df.columns:
            # removing NaN values
            # TODO: this doesnt make sense
            srs = df[column][df[column].notna()]

            # if column is categorical we do not want too many bins
            if column in self.features.categorical_features():
                unique_vals = len(df[column].unique())
                # if unique_vals > self.max_categories:
                #     bins = self.max_categories
                # else:
                bins = unique_vals

            # if columns is numerical we calculate the number of bins manually
            # https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
            elif column in self.features.numerical_features():
                n = srs.size
                iqr = srs.quantile(0.75) - srs.quantile(0.25)
                bins = (srs.max() - srs.min()) / (2*iqr*pow(n, (-1/3)))
                bins = int(round(bins, 0))
            else:
                # placeholder rule
                bins = 1

            hist, edges = np.histogram(srs, density=True, bins=bins)

            interval = (max(edges) - min(edges)) * 0.005  # 0.5%
            left_edges = edges[:-1]
            right_edges = [edge - interval for edge in edges[1:]]

            # _[column] = (hist, edges)
            _[column] = (hist, left_edges, right_edges)

        return _

    def _create_scatter_data(self):
        # returning df as creating multiple rows of scatter plots will require a lot of data manipulation
        df = self.features.mapped_data()
        # cat_df = df[self.categorical_columns].astype("float64").fillna(-1)
        # num_df = df.drop(self.categorical_columns, axis=1).astype("float64").fillna(-1)
        #
        # df = pd.concat([cat_df, num_df], axis=0)
        # df = df.astype("float64")
        for col in self.features.categorical_features():
            df[col + "_categorical"] = df[col].astype(str)
        scatter_data = df.dropna().to_dict(orient="list")

        # factor_cmap expects categorical data to be Str, not Int/Float
        # for col in self.categorical_columns:
        #     scatter_data[col] = list(map(str, scatter_data[col]))

        return scatter_data



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
