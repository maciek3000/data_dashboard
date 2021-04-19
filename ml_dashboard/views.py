from bs4 import BeautifulSoup
from bokeh.embed import components
import pandas as pd
from bokeh.core.validation.warnings import MISSING_RENDERERS
from bokeh.core.validation import silence

from .functions import append_description, series_to_dict, replace_duplicate_str, assess_models_names


class BaseView:
    """Parent Class for Views (Subpages) in the created HTML.

        BaseView defines several variables that are common to all HTML templates:
            - base_css name
            - created_on Div (includes date and time of creation of the whole analysis)
            - addresses (filepaths) to all subpages.

        standard_params() method should be called to get the dictionary of those HTML variables:
            jinja template id: content
    """
    _base_css = "base_css"
    _creation_date = "created_on"
    _file_suffix = "_file"

    def __init__(self):
        pass

    def standard_params(self, base_css, creation_date, hyperlinks):
        output = {
            self._base_css: base_css,
            self._creation_date: creation_date
        }

        for view, path in hyperlinks.items():
            output[(view + self._file_suffix)] = path

        return output


class Overview(BaseView):
    """First View (Subpage) of the Dashboard.

        Overview View tries to give a general feeling for the provided data - with what kind of features we're
        dealing with, what are their summary statistics and some general connections between them.
        Going from the top, Overview defines two summary statistics for both Categorical and Numerical features
        (transposed .describe() dataframes with additional customization). There is also a place to list out all the
        features that aren't going to be used.
        In the middle are 5 first rows of the data - transposed, to better see what are the contents of the features.
        Located at the bottom is seaborn's .pairplot(), which neatly visualizes not only correlations between
        the features, but also their distribution (histogram).

        Upon hovering over feature name (e.g. in the tables), small box with additional descriptions should appear
        (provided in JSON file and/or appropriate mapping of the values).
    """

    # Template Elements
    _numerical_table = "described_numeric"
    _categorical_table = "described_categorical"
    _table_head = "head"
    _unused_columns = "unused_cols"
    _pairplot_id = "pairplot"
    _pairplot_filename = "pairplot"
    _overview_css = "overview_css"
    _assets = "assets"

    # Strings for HTML
    _table_None = "Provided X, y data do not include this Table."
    _mapping_title = "Category - Original"
    _mapping_format = "{mapped} - {original}"
    _too_many_categories = "(...) Showing only first {} categories"

    def __init__(self, template, css_path, max_categories, feature_description_class):
        super().__init__()
        self.template = template
        self.css = css_path
        # self.output_directory = output_directory
        self.max_categories = max_categories
        self.feature_name_with_description_class = feature_description_class

    def render(self,
               base_css, creation_date, hyperlinks,  # base template params
               numerical_df, categorical_df, unused_features, head_df, do_pairplot_flag, pairplot_path,  # main elements of the View
               mapping, descriptions  # auxilliary dictionaries
               ):
        # TODO: add placeholders in case there are no categorical/numerical columns
        output = {}

        # Standard params
        standard = super().standard_params(base_css, creation_date, hyperlinks)
        output.update(standard)

        output[self._overview_css] = self.css

        # Tables
        tables = self._tables(numerical_df, categorical_df, head_df, mapping, descriptions)
        output.update(tables)

        # unused columns list
        unused_features_list = self._unused_features_html(unused_features)
        output[self._unused_columns] = unused_features_list

        # pairplot_img = self._pairplot(pairplot)
        if do_pairplot_flag:
            output[self._pairplot_id] = self._pairplot(pairplot_path)
        else:
            output[self._pairplot_id] = "<div>Too many Features</div>"

        return self.template.render(**output)

    def _tables(self, numerical_df, categorical_df, head_df, mapping, descriptions):
        output = {}

        tables_ids = [self._numerical_table, self._categorical_table, self._table_head]
        dataframes = [numerical_df, categorical_df, head_df]

        for table_id, dataframe in zip(tables_ids, dataframes):
            if dataframe is not None:
                raw_html = self._change_dataframe_to_html(dataframe)
                html_with_descriptions = self._stylize_html_table(raw_html, mapping, descriptions)
            else:
                html_with_descriptions = self._table_None
            output[table_id] = html_with_descriptions

        return output

    def _change_dataframe_to_html(self, dataframe):
        return dataframe.to_html(float_format="{:.2f}".format)

    def _stylize_html_table(self, html_table, mapping, descriptions):
        # TODO:
        # new HTML is created and appended via functions, but the reference to the table object is passed nonetheless
        # this is sloppy and should be changed (or can it?)
        table = BeautifulSoup(html_table, "html.parser")
        headers = table.select("table tbody tr th")

        for header in headers:
            description = descriptions[header.string]
            header_mapping = mapping[header.string]

            new_description = append_description(description, table)
            self._append_mapping(new_description, header_mapping, table)

            header.string.wrap(table.new_tag("p"))
            header.p.append(new_description)
            header.p["class"] = self.feature_name_with_description_class

        return str(table)

    def _append_mapping(self, html, mapping, parsed_html):
        # appending mappings to descriptions as long as they exist (they are not none)
        if mapping:
            html.append(parsed_html.new_tag("br"))
            html.append(parsed_html.new_tag("br"))
            html.append(parsed_html.new_string(self._mapping_title))
            i = 1
            for mapped, original in mapping.items():
                if i > self.max_categories:  # 0 indexing
                    html.append(parsed_html.new_tag("br"))
                    html.append(parsed_html.new_string(self._too_many_categories.format(i - 1)))
                    break
                html.append(parsed_html.new_tag("br"))
                html.append(parsed_html.new_string(self._mapping_format.format(mapped=mapped, original=original)))
                i += 1

    def _unused_features_html(self, unused_features):
        # Descriptions or mapping won't be appended - when the feature is unused, no calculations
        # are being done on this. Therefore, there might not be a respective description or mapping to append.

        html = "<ul>"
        for feature in unused_features:
            html += "<li>" + feature + "</li>"
        html += "</ul>"
        return html

    def _pairplot(self, pairplot_path):
        template = "<a href={path}><img src={path} title='Click to open larger version'></img></a>"

        # path = os.path.join(self.output_directory, self._assets, (self._pairplot_filename + ".png"))
        # pairplot.savefig(path)
        html = template.format(path=pairplot_path)
        return html


class FeatureView(BaseView):
    """Second View (Subpage) of the Dashboard.

        FeatureView aims to provide more detailed information on every feature. Menu with features is available in the
        left top corner (Burger Button) - clicking on the name of any feature should update all corresponding
        visualizations.
        First Subtab of the View gives detailed statistics on a chosen feature.
        Second Subtab provides rows of scatter plots - those are similar to seaborn's .pairplot(), but there are
        key differences: feature chosen in the Feature Menu is always plotted on the X axis and every other feature
        is plotted on Y axis. Additionally, Hue (color mapping) by every feature is added - in total, there is n x n
        scatter plots in the visualization (every row is another color mapping and every column is another feature
        in Y axis).
        Hue by the chosen feature is still being plotted but is greyed out in the process - please refer to
        ScatterPlotGrid for explanation.
    """

    _first_feature = "chosen_feature"
    _features_css = "features_css"
    _features_js = "features_js"
    _features_menu = "features_menu"

    _infogrid_summary_script = "bokeh_script_infogrid_summary"
    _infogrid_summary = "infogrid_summary"
    _infogrid_correlations_script = "bokeh_script_infogrid_correlations"
    _infogrid_correlations = "infogrid_correlations"

    _scatterplot_script = "bokeh_script_scatter_plot_grid"
    _scatterplot = "scatter_plot_grid"

    _transformed_feature_normal_transformations_plots_script = "bokeh_script_normal_transformations_plots"

    _transformed_feature = "transformed_feature"
    _transformed_feature_template = '<div class="{feature_class}" id="{feature_name}">{content}</div>'

    _transformed_feature_original_prefix = "Original_"
    _transformed_feature_transformed_df_title = "Applied Transformations (First 5 Rows) - Test Data"
    _transformed_feature_transformers_title = "Transformers (fitted on Train Data)"
    _transformed_feature_normal_transformations_title = "Normal Transformations applied on a Feature"

    # CSS
    _menu_single_feature_class = "single-feature"
    _menu_target_feature_class = "target-feature"
    _first_feature_transformed = "chosen-feature-transformed"
    _transformed_feature_div = "transformed-feature"
    _transformed_feature_grid = "transformed-grid"
    _transformed_feature_transformations_table = "transformations-table"
    _transformed_feature_subtitle_div = "subtitle"  # "transformed-feature-subtitle"
    _transformed_feature_transformer_list = "transformer-list"
    _transformed_feature_single_transformer = "single-transformer"
    _transformed_feature_plots_grid = "transformed-feature-plots"

    _feature_menu_header = "<div class='features-menu-title'><div>Features</div><div class='close-button'>x</div></div>"
    _feature_menu_single_feature = "<div class='{}'><span>{:03}. {}</span></div>"

    def __init__(self, template, css_path, js_path, target_name):
        super().__init__()
        self.template = template
        self.css = css_path
        self.js = js_path
        self.target_name = target_name

    def render(
            self,
            base_css,
            creation_date,
            hyperlinks,
            summary_grid,
            correlations_plot,
            do_scatterplot_flag,
            scatterplot,
            feature_list,
            numerical_features,
            test_features_df,
            test_transformed_features_df,
            X_transformations,
            y_transformations,
            normal_transformations_plots,
            initial_feature
    ):

        output = {}

        # Standard variables
        standard = super().standard_params(base_css, creation_date, hyperlinks)
        output.update(standard)

        # JS/CSS
        output[self._features_css] = self.css
        output[self._features_js] = self.js

        # First Feature
        output[self._first_feature] = initial_feature
        output[self._features_menu] = self._create_features_menu(feature_list)

        # Histogram
        infogrid_script, infogrid_div = components(summary_grid)
        output[self._infogrid_summary_script] = infogrid_script
        output[self._infogrid_summary] = infogrid_div

        # Correlations
        corr_script, corr_div = components(correlations_plot)
        output[self._infogrid_correlations_script] = corr_script
        output[self._infogrid_correlations] = corr_div

        # Scatter Plot
        if do_scatterplot_flag:
            # silencing warning from attaching legend colorbars to empty plots
            silence(MISSING_RENDERERS, True)
            scatterplot_script, scatterplot_div = components(scatterplot)
            output[self._scatterplot_script] = scatterplot_script
            output[self._scatterplot] = scatterplot_div
            silence(MISSING_RENDERERS, False)
        else:
            output[self._scatterplot] = "<div>Too many features</div>"

        # Transformed Features

        # adding target transformation to all transformations
        transformations = X_transformations
        transformations[self.target_name] = (y_transformations, self.target_name)

        # only first 5 rows are going to be shown
        df = test_features_df.head()
        transformed_df = test_transformed_features_df.head()

        # adding scripts from bokeh plots at the end so they dont break other JS
        normal_plots_scripts = []
        normal_plots_divs = {}
        for feature, plots in normal_transformations_plots.items():
            script, div = components(plots)
            normal_plots_scripts.append(script)
            normal_plots_divs[feature] = div

        output[self._transformed_feature] = self._transformed_features_divs(
            df=df,
            transformed_df=transformed_df,
            transformations=transformations,
            numerical_features=numerical_features,
            normal_plots=normal_plots_divs,
            initial_feature=initial_feature
        )
        output[self._transformed_feature_normal_transformations_plots_script] = "".join(normal_plots_scripts)

        return self.template.render(**output)

    def _create_features_menu(self, features):
        html = self._feature_menu_header
        template = self._feature_menu_single_feature
        i = 0
        for feat in features:
            cls = self._menu_single_feature_class
            if feat == self.target_name:
                cls += " " + self._menu_target_feature_class

            html += template.format(cls, i, feat)
            i += 1

        return html

    def _transformed_features_divs(self, df, transformed_df, transformations, numerical_features, normal_plots, initial_feature):

        output = ""
        for col in df.columns:
            feature_class = self._transformed_feature_div
            if col == initial_feature:
                feature_class += " " + self._first_feature_transformed

            transformers = transformations[col][0]
            new_cols = transformations[col][1]
            content = self._single_transformed_feature(df[col], transformed_df[new_cols], transformers)
            if col in numerical_features:
                plot_row = normal_plots[col]
                content += """<div class='{plots_class}'><div class='{subtitle_div}'>{title}</div><div>{plot_row}</div></div>""".format(
                    title=self._transformed_feature_normal_transformations_title,
                    subtitle_div=self._transformed_feature_subtitle_div,
                    plots_class=self._transformed_feature_plots_grid,
                    plot_row=plot_row
                )

            output += self._transformed_feature_template.format(feature_class=feature_class, title=col, content=content, feature_name=col)

        return output

    def _single_transformed_feature(self, series, transformed_output, transformers):
        transformers_html = self._transformers_html(transformers)
        df_html = self._transformed_dataframe_html(series, transformed_output)

        template = """
        <div class='{transformed_feature_grid_class}'>
            {transformers_html}
            <div class='{transformations_table_class}'>
                {df_html}
            </div>
        </div>"""

        output = template.format(
            transformed_feature_grid_class=self._transformed_feature_grid,
            transformers_html=transformers_html,
            transformations_table_class=self._transformed_feature_transformations_table,
            df_html=df_html
        )
        return output

    def _transformers_html(self, transformers):

        single_template = "<div class='{transformed_feature_single_transformer}'>{transformer}</div>"
        _ = []
        for transformer in transformers:
            _.append(single_template.format(
                    transformed_feature_single_transformer=self._transformed_feature_single_transformer,
                    transformer=str(transformer)
                )
            )

        template = """
        <div class='{transformer_list_class}'>
            <div class='{transformed_feature_subtitle_class}'>{transformers_title}</div>
            <div>{transformers}</div>
            </div>
        """

        output = template.format(
            transformer_list_class=self._transformed_feature_transformer_list,
            transformers_title=self._transformed_feature_transformers_title,
            transformed_feature_subtitle_class=self._transformed_feature_subtitle_div,
            transformers="".join(_)
        )
        return output

    def _transformed_dataframe_html(self, series, transformed_df):
        series.name = self._transformed_feature_original_prefix + series.name
        df = pd.concat([series, transformed_df], axis=1)

        output = "<div class='{transformed_feature_subtitle_class}'>{transformed_df_title}</div>".format(
            transformed_feature_subtitle_class=self._transformed_feature_subtitle_div,
            transformed_df_title=self._transformed_feature_transformed_df_title
        )
        output += df.to_html(index=False)
        return output


class ModelsView(BaseView):

    _models_css = "models_css"
    _models_js = "models_js"

    _models_table = "models_left_upper"

    _models_plot_title = "models_right_title"
    _models_plot = "models_right_plot"
    _models_plot_script = "bokeh_script_models_right"

    _models_left_bottom_title = "models_left_bottom_title"
    _models_left_bottom = "models_left_bottom"
    _models_left_bottom_script = "bokeh_script_models_left_bottom"

    _incorrect_predictions_table = "incorrect_predictions_table"
    _incorrect_predictions_table_script = "bokeh_script_incorrect_predictions_table"

    # CSS
    _first_model_class = "first-model"
    _other_model_class = "other-model"

    def __init__(self, template, css_path, js_path, params_name, model_with_description_class):
        super().__init__()
        self.template = template
        self.css = css_path
        self.js = js_path
        self.params_name = params_name
        self.model_with_description_class = model_with_description_class

    def render(self, base_css, creation_date, hyperlinks, model_results, models_right, models_left_bottom, incorrect_predictions_table):
        raise NotImplementedError

    def _base_output(self, base_css, creation_date, hyperlinks, model_results, incorrect_predictions_table):
        output = {}

        # Standard variables
        standard = super().standard_params(base_css, creation_date, hyperlinks)
        output.update(standard)

        output[self._models_css] = self.css
        output[self._models_js] = self.js
        output[self._models_table] = self._models_result_table(model_results)

        incorrect_predictions_script, incorrect_predictions_div = components(incorrect_predictions_table)
        output[self._incorrect_predictions_table_script] = incorrect_predictions_script
        output[self._incorrect_predictions_table] = incorrect_predictions_div

        return output

    def _models_result_table(self, results_dataframe):
        new_df = results_dataframe
        new_df.index = replace_duplicate_str(results_dataframe.index.tolist())
        new_params = series_to_dict(results_dataframe[self.params_name])

        df = results_dataframe.drop([self.params_name], axis=1)
        df.index.name = None
        html_table = df.to_html(float_format="{:.5f}".format)

        table = BeautifulSoup(html_table, "html.parser")
        headers = table.select("table tbody tr th")

        for header in headers:
            single_model_params = new_params[header.string]
            params_html = append_description(single_model_params, table)

            header.string.wrap(table.new_tag("p"))
            header.p.append(params_html)
            header.p["class"] = self.model_with_description_class

        rows = table.select("table tbody tr")
        rows[0]["class"] = self._first_model_class
        for row in rows[1:-1]:
            row["class"] = self._other_model_class

        output = str(table)

        return output


class ModelsViewClassification(ModelsView):

    # CSS
    _confusion_matrices_class = "confusion-matrices"
    _confusion_matrices_single_matrix = "confusion-matrix"
    _confusion_matrices_single_matrix_title = "confusion-matrix-title"
    _confusion_matrices_single_matrix_table = "confusion-matrix-table"

    _models_plot_title_text = "Result Curves Comparison"
    _models_confusion_matrix_title_text = "Confusion Matrices"

    # confusion matrices html
    _single_confusion_matrix_html_template = """
<table>
<thead>
<tr><th></th><th>Predicted Negative</th><th>Predicted Positive</th></tr>
</thead>
<tbody>
<tr><th>Actual Negative</th><td>{tn}</td><td>{fp}</td></tr>
<tr><th>Actual Positive</th><td>{fn}</td><td>{tp}</td></tr>
</tbody>
</table>
"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def render(self, base_css, creation_date, hyperlinks, model_results, models_right, models_left_bottom, incorrect_predictions_table):

        models_plot = models_right
        confusion_matrices = assess_models_names(models_left_bottom)

        output = super()._base_output(base_css, creation_date, hyperlinks, model_results, incorrect_predictions_table)

        models_plot_script, models_plot_div = components(models_plot)
        output[self._models_plot_title] = self._models_plot_title_text
        output[self._models_plot_script] = models_plot_script
        output[self._models_plot] = models_plot_div

        output[self._models_left_bottom_title] = self._models_confusion_matrix_title_text
        output[self._models_left_bottom] = self._confusion_matrices(confusion_matrices)

        return self.template.render(**output)

    def _confusion_matrices(self, models_confusion_matrices):

        output = "<div class='{}'>".format(self._confusion_matrices_class)
        i = 0

        for model, matrix in models_confusion_matrices:
            if i == 0:
                color_class = self._first_model_class
                i += 1
            else:
                color_class = self._other_model_class

            single_matrix = "<div class='{} {}'>".format(self._confusion_matrices_single_matrix, color_class)
            title = "<div class='{}'>{}</div>".format(self._confusion_matrices_single_matrix_title, model)
            table = self._single_confusion_matrix_html(matrix)
            single_matrix += title + table + "</div>"
            output += single_matrix

        output += "</div>"
        return output

    def _single_confusion_matrix_html(self, confusion_array):
        tn, fp, fn, tp = confusion_array.ravel()
        table = self._single_confusion_matrix_html_template.format(tn=tn, fp=fp, fn=fn, tp=tp).replace("\n", "")
        output = "<div class='{}'>{}</div>".format(self._confusion_matrices_single_matrix_table, table)
        return output


class ModelsViewRegression(ModelsView):

    _prediction_errors_title = "Prediction Error Plots"
    _residuals_title = "Residual Plots"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def render(self, base_css, creation_date, hyperlinks, model_results, models_right, models_left_bottom, incorrect_predictions_table):

        prediction_errors_plot = models_right
        residual_plot = models_left_bottom

        output = self._base_output(base_css, creation_date, hyperlinks, model_results, incorrect_predictions_table)

        models_right_plot_script, models_right_plot_div = components(prediction_errors_plot)
        output[self._models_plot_title] = self._prediction_errors_title
        output[self._models_plot_script] = models_right_plot_script
        output[self._models_plot] = models_right_plot_div

        models_left_bottom_plot_script, models_left_bottom_plot_div = components(residual_plot)
        output[self._models_left_bottom_title] = self._residuals_title
        output[self._models_left_bottom_script] = models_left_bottom_plot_script
        output[self._models_left_bottom] = models_left_bottom_plot_div

        return self.template.render(**output)


class ModelsViewMulticlass(ModelsView):

    _confusion_matrices_title = "Confusion Matrices"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def render(self, base_css, creation_date, hyperlinks, model_results, models_right, models_left_bottom, incorrect_predictions_table):
        confusion_matrices = models_left_bottom

        output = self._base_output(base_css, creation_date, hyperlinks, model_results, incorrect_predictions_table)
        models_left_bottom_plot_script, models_left_bottom_plot_div = components(confusion_matrices)
        output[self._models_left_bottom_title] = self._confusion_matrices_title
        output[self._models_left_bottom_script] = models_left_bottom_plot_script
        output[self._models_left_bottom] = models_left_bottom_plot_div

        return self.template.render(**output)
