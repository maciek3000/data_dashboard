import os
from bs4 import BeautifulSoup
from bokeh.embed import components


class BaseView:
    """Parent Class for Views (Subpages) in the created HTML.

        BaseView defines several variables that are common to all HTML templates:
            - base_css name
            - created_on Div (includes date and time of creation of the whole analysis)
            - addresses (filepaths) to all subpages.

        standard_params() method should be called to get the dictionary of those HTML variables:
            jinja template id: content
    """
    base_css_id = "base_css"
    creation_date_id = "created_on"
    file_suffix = "_file"

    def __init__(self):
        pass

    def standard_params(self, base_css, creation_date, hyperlinks):
        output = {
            self.base_css_id: base_css,
            self.creation_date_id: creation_date
        }

        for view, path in hyperlinks.items():
            output[(view + self.file_suffix)] = path

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

    def render(self, base_css, creation_date, hyperlinks, numerical_df, categorical_df, unused_features, head_df, pairplot, features):

        output = {}

        # Standard params
        standard = super().standard_params(base_css, creation_date, hyperlinks)
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

        return self.template.render(**output)

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

        path = os.path.join(self.output_directory, self.assets, (self.pairplot_filename + ".png"))
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

    def render(self, base_css, creation_date, hyperlinks, histogram, scatterplot, feature_list, first_feature):

        output = {}

        # Standard variables
        standard = super().standard_params(base_css, creation_date, hyperlinks)
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
