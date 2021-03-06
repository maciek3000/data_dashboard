import os
from bs4 import BeautifulSoup
from .plots import InfoGrid, ScatterPlotGrid
from bokeh.embed import components


class BaseView:

    base_css_id = "base_css"
    creation_date_id = "created_on"

    def __init__(self):
        pass

    def standard_params(self, base_css, creation_date):
        output = {
            self.base_css_id: base_css,
            self.creation_date_id: creation_date
        }

        return output


class Overview(BaseView):

    # TODO: change names
    numerical_table_id = "described_numeric"
    categorical_table_id = "described_categorical"
    table_head = "head"

    unused_columns = "unused_cols"

    pairplot = "pairplot"
    pairplot_filename = "pairplot"

    css_id = "overview_css"

    assets = "assets"

    def __init__(self, template, css_path, output_directory):
        super().__init__()
        self.template = template
        self.css = css_path
        self.output_directory = output_directory

    def render(self, base_css, creation_date, numerical_df, categorical_df, unused_features, head_df, pairplot, feature_list):

        output = {}

        # Standard params
        standard = super().standard_params(base_css, creation_date)
        output.update(standard)

        output[self.css_id] = self.css

        # Tables
        tables = self._tables(numerical_df, categorical_df, head_df, features)
        output.update(tables)

        # unused columns list
        unused_features_list = self._unused_features_html(unused_features)
        output[self.unused_columns] = unused_features_list

        pairplot_img = self._pairplot(pairplot)
        output[self.pairplot] = pairplot_img

        self.template.render(**output)

    def _tables(self, numerical_df, categorical_df, head_df, features):
        output = {}

        tables_ids = [self.numerical_table_id, self.categorical_table_id, self.table_head]
        dataframes = [numerical_df, categorical_df, head_df]

        for table_id, dataframe in zip(tables_ids, dataframes):
            raw_html = self._change_dataframe_to_html(dataframe)
            html_with_descriptions = self._append_descriptions_to_features(raw_html, features)
            output[table_id] = html_with_descriptions

        return output

    def _change_dataframe_to_html(self, dataframe):
        return dataframe.to_html(float_format="{:.2f}".format)

    def _append_descriptions_to_features(self, html_table, features):
        # TODO: split into smaller functions

        table = BeautifulSoup(html_table, "html.parser")
        headers = table.select("table tbody tr th")
        for header in headers:
            try:
                description = features[header.string].description
            except KeyError:
                continue

            # adding <span> that will hold description of a feature
            # every \n is replaced with <br> tag
            header.string.wrap(table.new_tag("p"))
            new_tag = table.new_tag("span")
            lines = description.split("\n")
            new_tag.string = lines[0]
            if len(lines) > 1:
                for line in lines[1:]:
                    new_tag.append(table.new_tag("br"))
                    new_tag.append(table.new_string("{}".format(line)))

            # appending mappings to descriptions as long as they exist (they are not none)
            mappings = features[header.string].mapping()
            if mappings:
                new_tag.append(table.new_tag("br"))
                new_tag.append(table.new_tag("br"))
                new_tag.append(table.new_string("Category - Original (Transformed)"))
                i = 0
                for key, val in mappings.items():
                    new_tag.append(table.new_tag("br"))
                    new_tag.append(table.new_string("{} - {}".format(key, val)))
                    if i >= 10:
                        new_tag.append(table.new_tag("br"))
                        new_tag.append(table.new_string("(...) Showing only first 10 categories"))
                        break
                    i += 1

            header.p.append(new_tag)
        return str(table)

    def _unused_features_html(self, unused_features):
        # TODO: redesign with bs4 and append descriptions

        html = "<ul>"
        for feature in unused_features:
            html += "<li>" + feature + "</li>"
        html += "</ul>"
        return html

    def _pairplot(self, pairplot):
        template = "<a href={path}><img src={path} title='Click to open larger version'></img></a>"

        path = os.path.join(self.output_directory, self.assets, (self.pairplot_filename + "png"))
        pairplot.savefig(path)
        html = template.format(path=path)
        return html

    def __create_figures(self, figures, output_directory):
        d = {}
        for name, plot in figures.items():
            path = os.path.join(output_directory, (name + ".png"))
            d[name] = "<a href={path}><img src={path} title='Click to open larger version'></img></a>".format(path=path)
            plot.savefig(path)
        return d


#
# class Overview:
#
#     def __init__(self, template, css_path, features, naive_mapping):
#         self.template = template
#         self.css = css_path
#         self.features = features
#         self.naive_mapping = naive_mapping
#
#     def render(self, base_dict, tables, lists, figures, figure_directory):
#         tables = self.__create_tables_html(tables)
#         lists = self.__create_lists_html(lists)
#         figures = self.__create_figures(figures, figure_directory)
#
#         output_dict = {}
#         for d in [tables, lists, figures, base_dict]:
#             output_dict.update(d)
#
#         # adding overview specific css
#         output_dict["overview_css"] = self.css
#         return self.template.render(**output_dict)
#
#     def __create_tables_html(self, tables):
#         params = {}
#         for key, arg in tables.items():
#             html_table = arg.to_html(float_format="{:.2f}".format)
#             html_table = self.__append_description(html_table)
#             params[key] = html_table
#
#         return params
#
#     def __append_description(self, html_table):
#         # if self.features:
#         table = BeautifulSoup(html_table, "html.parser")
#         headers = table.select("table tbody tr th")
#         for header in headers:
#             try:
#                 description = self.features[header.string].description
#             except KeyError:
#                 continue
#
#             # adding <span> that will hold description of a feature
#             # every \n is replaced with <br> tag
#             header.string.wrap(table.new_tag("p"))
#             new_tag = table.new_tag("span")
#             lines = description.split("\n")
#             new_tag.string = lines[0]
#             if len(lines) > 1:
#                 for line in lines[1:]:
#                     new_tag.append(table.new_tag("br"))
#                     new_tag.append(table.new_string("{}".format(line)))
#
#             # appending mappings to descriptions as long as they exist (they are not none)
#             mappings = self.naive_mapping[header.string]
#             #mappings = self._consolidate_mappings(header.string)
#             if mappings:
#                 new_tag.append(table.new_tag("br"))
#                 new_tag.append(table.new_tag("br"))
#                 new_tag.append(table.new_string("Category - Original (Transformed)"))
#                 i = 0
#                 for key, val in mappings.items():
#                     new_tag.append(table.new_tag("br"))
#                     new_tag.append(table.new_string("{} - {}".format(key, val)))
#                     if i >= 10:
#                         new_tag.append(table.new_tag("br"))
#                         new_tag.append(table.new_string("(...) Showing only first 10 categories"))
#                         break
#                     i += 1
#
#             header.p.append(new_tag)
#         return str(table)
#         # else:
#         #     return html_table
#
#     def _consolidate_mappings(self, feature):
#         # description mapping comes from json so all keys are strings
#         description_mapping = self.features.mapping(feature)
#         try:
#             naive_mapping = self.naive_mapping[feature]
#         except KeyError:
#             naive_mapping = None
#
#         if (description_mapping is None) and (naive_mapping is None):
#             return ""
#
#         # we're expecting naive mapping to be always present in case of categorical values
#         converted_naive_mapping = {str(key): str(val) for key, val in naive_mapping.items()}
#         if description_mapping is None:
#             # changing to strings to accomodate for json keys
#             new_pairs = converted_naive_mapping
#         else:
#             _ = {}
#             for key in converted_naive_mapping:
#                 _[description_mapping[key]] = "{} ({})".format(key, converted_naive_mapping[key])
#             new_pairs = _
#
#         # app = ""
#         # if len(new_pairs) > 10:
#         #     new_pairs = dict(list(new_pairs.items())[:10])
#         #     app = "Showing only first 10 categories"
#         #
#         # mapping_string += "<br>".join((" - ".join([str(key), str(val)]) for key, val in new_pairs.items()))
#         # mapping_string += app
#
#         return new_pairs
#
#     def __create_lists_html(self, lists):
#         d = {}
#
#         # Was thinking of redesigning it with bs4, but its a very simple structure so it would be an overkill
#         # TODO: redesign with bs4 and append descriptions
#         for key, l in lists.items():
#             _ = "<ul>"
#             for x in l:
#                 _ += "<li>" + x + "</li>"
#             _ += "</ul>"
#             d[key] = _
#         return d
#
#     def __create_figures(self, figures, output_directory):
#         d = {}
#         for name, plot in figures.items():
#             path = os.path.join(output_directory, (name + ".png"))
#             d[name] = "<a href={path}><img src={path} title='Click to open larger version'></img></a>".format(path=path)
#             plot.savefig(path)
#         return d

class FeatureView(BaseView):

    first_feature = "chosen_feature"
    features_css = "features_css"
    features_js = "features_js"
    features_menu = "features_menu"

    infogrid_script = "bokeh_script_info_grid"
    infogrid = "info_grid"

    scatterplot_script = "bokeh_script_scatter_plot_grid"
    scatterplot = "scatter_plot_grid"

    feature_menu_header = "<div class='features-menu-title'><div>Features</div><div class='close-button'>x</div></div>"
    feature_menu_single_feature = "<div class='single-feature'>{:03}. {}</div>"

    def __init__(self, template, css_path, js_path):
        super().__init__()
        self.template = template
        self.css = css_path
        self.js = js_path

    def render(self, base_css, creation_date, histogram, scatterplot, feature_list, first_feature):

        output = {}

        # Standard variables
        standard = super().standard_params(base_css, creation_date)
        output.update(standard)

        # JS/CSS
        output[self.features_css] = self.css
        output[self.features_js] = self.js

        # First Feature
        output[self.first_feature] = first_feature
        output[self.features_menu] = self._features_menu(feature_list)

        # Histogram
        infogrid_script, infogrid_div = components(histogram)
        output[self.infogrid_script] = infogrid_script
        output[self.infogrid] = infogrid_div

        # Scatter Plot
        scatterplot_script, scatterplot_div = components(scatterplot)
        output[self.scatterplot_script] = scatterplot_script
        output[self.scatterplot] = scatterplot_div

        return self.template.render(**output)

    def _features_menu(self, features):
        html = self.feature_menu_header
        template = self.feature_menu_single_feature
        i = 0
        for feat in sorted(features):
            html += template.format(i, feat)
            i += 1

        return html


#
# class FeatureView:
#
#     chosen_feature = "chosen_feature"
#     features_css = "features_css"
#     features_js = "features_js"
#
#
#     def __init__(self, template, css_path, js_path, features, naive_mapping):
#
#         self.template = template
#         self.css = css_path
#         self.js = js_path
#         self.features = features
#         self.naive_mapping = naive_mapping
#
#         self.chosen_feature = None
#
#         self.info_grid = InfoGrid(features)
#         self.scatter_plot_grid = ScatterPlotGrid(features)
#
#     def render(self, base_dict, histogram_data, scatter_data, categorical_columns):
#         self.chosen_feature = sorted(self.features.features())[0]
#         output_dict = {}
#         output_dict.update(base_dict)
#
#         output_dict["chosen_feature"] = self.chosen_feature
#         output_dict["features_css"] = self.css
#         output_dict["features_js"] = self.js
#
#         output_dict["features_menu"] = self._create_features_menu()
#
#         info_grid_script, info_grid_div = self.info_grid.create_grid_elements(histogram_data, self.chosen_feature)
#         output_dict["bokeh_script_info_grid"] = info_grid_script
#         output_dict["info_grid"] = info_grid_div
#
#         scatter_plot_grid_script, scatter_plot_grid_div = self.scatter_plot_grid.create_grid_elements(
#             scatter_data, categorical_columns, self.features, self.chosen_feature)
#         output_dict["scatter_plot_grid"] = scatter_plot_grid_div
#         output_dict["bokeh_script_scatter_plot_grid"] = scatter_plot_grid_script
#
#         return self.template.render(**output_dict)
#
#     def _create_features_menu(self):
#         html = "<div class='features-menu-title'><div>Features</div><div class='close-button'>x</div></div>"
#         template = "<div class='single-feature'>{:03}. {}</div>"
#         i = 0
#         for feat in sorted(self.features.features()):
#             html += template.format(i, feat)
#             i += 1
#         return html
