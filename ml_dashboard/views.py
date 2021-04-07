import os
from bs4 import BeautifulSoup
from bokeh.embed import components
from .model_finder import obj_name


def info_symbol_html():
    # TODO: check if used anywhere
    return "<span class='info-symbol'>&#x1F6C8;</span>"


def append_description(description, parsed_html):
    # adding <span> that will hold description of a feature
    # every \n is replaced with <br> tag
    new_tag = parsed_html.new_tag("span")
    lines = description.split("\n")
    new_tag.string = lines[0]
    if len(lines) > 1:
        for line in lines[1:]:
            new_tag.append(parsed_html.new_tag("br"))
            new_tag.append(parsed_html.new_string("{}".format(line)))
    return new_tag


def series_to_dict(srs):
    params = srs.to_dict()
    dict_str = {model: str(params_str)[1:-1].replace(", ", "\n") for model, params_str in params.items()}
    return dict_str

# def df_to_html_table(df):
#     d = df.T.to_dict()
#
#     output_template = '<table><thead><tr><th></th>{columns}</tr></thead><tbody>{rows}</tbody></table>'
#     columns = ["<th>{col}</th>".format(col=col) for col in df.columns]
#     rows = ""
#     for row, content in d.items():
#         html = "<tr><th>{row}</th>".format(row=row)
#         for value in content.values():
#             html += "<td>{value}</td>".format(value=value)
#         html += "</tr>"
#         rows += html
#
#     output = output_template.format(columns="".join(columns), rows=rows)
#     return output


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
    _mapping_title = "Category - Original"
    _mapping_format = "{mapped} - {original}"
    _too_many_categories = "(...) Showing only first {} categories"

    def __init__(self, template, css_path, output_directory, max_categories, feature_description_class):
        super().__init__()
        self.template = template
        self.css = css_path
        self.output_directory = output_directory
        self.max_categories = max_categories
        self.feature_name_with_description_class = feature_description_class

    def render(self,
               base_css, creation_date, hyperlinks,  # base template params
               numerical_df, categorical_df, unused_features, head_df, pairplot,  # main elements of the View
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

        pairplot_img = self._pairplot(pairplot)
        output[self._pairplot_id] = pairplot_img

        return self.template.render(**output)

    def _tables(self, numerical_df, categorical_df, head_df, mapping, descriptions):
        output = {}

        tables_ids = [self._numerical_table, self._categorical_table, self._table_head]
        dataframes = [numerical_df, categorical_df, head_df]

        for table_id, dataframe in zip(tables_ids, dataframes):
            raw_html = self._change_dataframe_to_html(dataframe)
            html_with_descriptions = self._stylize_html_table(raw_html, mapping, descriptions)
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

    def _pairplot(self, pairplot):
        template = "<a href={path}><img src={path} title='Click to open larger version'></img></a>"

        path = os.path.join(self.output_directory, self._assets, (self._pairplot_filename + ".png"))
        pairplot.savefig(path)
        html = template.format(path=path)
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

    _feature_menu_header = "<div class='features-menu-title'><div>Features</div><div class='close-button'>x</div></div>"
    _feature_menu_single_feature = "<div class='single-feature'>{:03}. {}</div>"

    def __init__(self, template, css_path, js_path):
        super().__init__()
        self.template = template
        self.css = css_path
        self.js = js_path

    def render(self, base_css, creation_date, hyperlinks, summary_grid, correlations_plot, scatterplot, feature_list, first_feature):

        output = {}

        # Standard variables
        standard = super().standard_params(base_css, creation_date, hyperlinks)
        output.update(standard)

        # JS/CSS
        output[self._features_css] = self.css
        output[self._features_js] = self.js

        # First Feature
        output[self._first_feature] = first_feature
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
        scatterplot_script, scatterplot_div = components(scatterplot)
        output[self._scatterplot_script] = scatterplot_script
        output[self._scatterplot] = scatterplot_div

        return self.template.render(**output)

    def _create_features_menu(self, features):
        html = self._feature_menu_header
        template = self._feature_menu_single_feature
        i = 0
        for feat in features:
            html += template.format(i, feat)
            i += 1

        return html


class ModelsView(BaseView):

    _models_css = "models_css"
    _models_js = "models_js"

    _models_table = "models_left_upper"

    _models_plot_title = "models_right_title"
    _models_plot = "models_right_plot"
    _models_plot_script = "models_right_plot_script"

    _models_confusion_matrix_title = "models_left_bottom_title"
    _models_confusion_matrix = "models_left_bottom"

    # CSS
    _first_model_class = "first-model"
    _other_model_class = "other-model"

#     # CSS
#     _first_model_class = "first-model"
#     _other_model_class = "other-model"
#     _confusion_matrices_class = "confusion-matrices"
#     _confusion_matrices_single_matrix = "confusion-matrix"
#     _confusion_matrices_single_matrix_title = "confusion-matrix-title"
#     _confusion_matrices_single_matrix_table = "confusion-matrix-table"
#
#     _models_plot_title_text = "Result Curves Comparison"
#     _models_confusion_matrix_title_text = "Confusion Matrices"
#
#     # confusion matrices html
#     _single_confusion_matrix_html_template = """
# <table>
# <thead>
# <tr><th></th><th>Predicted Negative</th><th>Predicted Positive</th></tr>
# </thead>
# <tbody>
# <tr><th>Actual Negative</th><td>{tn}</td><td>{fp}</td></tr>
# <tr><th>Actual Positive</th><td>{fn}</td><td>{tp}</td></tr>
# </tbody>
# </table>
# """

    def __init__(self, template, css_path, js_path, params_name, model_with_description_class):
        super().__init__()
        self.template = template
        self.css = css_path
        self.js = js_path
        self.params_name = params_name
        self.model_with_description_class = model_with_description_class

    def render(self, base_css, creation_date, hyperlinks, model_results, models_right, models_left_bottom):
        raise NotImplementedError

    def _base_output(self, base_css, creation_date, hyperlinks, model_results):
        output = {}

        # Standard variables
        standard = super().standard_params(base_css, creation_date, hyperlinks)
        output.update(standard)

        output[self._models_css] = self.css
        output[self._models_js] = self.js
        output[self._models_table] = self._models_result_table(model_results)

        return output

    def _models_result_table(self, results_dataframe):
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

    def render(self, base_css, creation_date, hyperlinks, model_results, models_right, models_left_bottom):

        models_plot = models_right
        confusion_matrices = models_left_bottom

        output = super()._base_output(base_css, creation_date, hyperlinks, model_results)

        models_plot_script, models_plot_div = components(models_plot)
        output[self._models_plot_title] = self._models_plot_title_text
        output[self._models_plot_script] = models_plot_script
        output[self._models_plot] = models_plot_div

        output[self._models_confusion_matrix_title] = self._models_confusion_matrix_title_text
        output[self._models_confusion_matrix] = self._confusion_matrices(confusion_matrices)

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
            title = "<div class='{}'>{}</div>".format(self._confusion_matrices_single_matrix_title, obj_name(model))
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def render(self, base_css, creation_date, hyperlinks, model_results, models_right, models_left_bottom):

        prediction_errors_plot = models_right

        output = self._base_output(base_css, creation_date, hyperlinks, model_results)

        models_plot_script, models_plot_div = components(prediction_errors_plot)
        output[self._models_plot_title] = self._prediction_errors_title
        output[self._models_plot_script] = models_plot_script
        output[self._models_plot] = models_plot_div

        return self.template.render(**output)


class ModelsViewMulticlass(ModelsView):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def render(self, base_css, creation_date, hyperlinks, model_results, models_right, models_left_bottom):
        output = self._base_output(base_css, creation_date, hyperlinks, model_results)

        return self.template.render(**output)
