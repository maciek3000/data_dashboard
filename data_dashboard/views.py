import pandas as pd
from bs4 import BeautifulSoup
from bokeh.embed import components
from bokeh.core.validation.warnings import MISSING_RENDERERS
from bokeh.core.validation import silence
from .functions import append_description, series_to_dict, replace_duplicate_str, assess_models_names

placeholder_features_limit_crossed = "<div>Plot was turned off because of too many features present in the data." \
                                     "Reduce the number of features or call 'create_dashboard' with " \
                                     "'force_pairplot=True' argument.</div>"
"""str: hardcoded HTML element to be used when a given Plot is turned off with a bool flag."""


class BaseView:
    """Parent Class for HTML Views (Subpages).

    Jinja Variables defined in the BaseView are:
        - base_css name for css file common to all Views
        - created_on Div (includes date and time of creation of the Dashboard)
        - hyperlinks to all subpages

    standard_params method should be called to get the dictionary of those HTML variables in a dictionary structure of
        jinja template id: content
    """
    _base_css = "base_css"
    _creation_date = "created_on"
    _file_suffix = "_file"

    def __init__(self):
        """Create BaseView object."""
        pass

    def standard_params(self, base_css, creation_date, hyperlinks):
        """Create dictionary of jinja id: HTML content pairs.

        Every view in hyperlinks keys is appended with file suffix as defined in the HTML template.

        Args:
            base_css (str): address of base css file
            creation_date (date): creation date of HTML
            hyperlinks (dict): 'view name': hyperlink pairs

        Returns:
            dict: 'jinja id': HTML content pairs
        """
        output = {
            self._base_css: base_css,
            self._creation_date: creation_date
        }

        for view, path in hyperlinks.items():
            output[(view + self._file_suffix)] = path

        return output


class Overview(BaseView):
    """Overview Subpage (View) of the Dashboard. Inherits from BaseView.

    As the name suggests, Overview gives a general information about the data - what's the distribution of features,
    what are their summary statistics, etc. Jinja templates defines following elements:
        - 'describe' table of Numerical features data
        - 'describe' table of Categorical features data
        - list of unused columns
        - head (first 5 rows) of data
        - seaborn pairplot visualization

    Note:
        Every feature name is also 'hoverable', meaning that upon hovering, additional box with the description (and
        optional mapping) will appear.

    Attributes:
        template (jinja2.Template): loaded HTML template
        css (str): file path to Overview specific CSS file that will be included in HTML
        feature_name_with_description_class (str): HTML (CSS) class shared between different objects indicating HTML
            element with hidden (hoverable) description
        Rest of the Attributes are inherited from BaseView.
    """
    # Internal Limit for listing categories in descriptions
    _max_categories_limit = 10

    # jinja template ids
    _numerical_table = "described_numeric"
    _categorical_table = "described_categorical"
    _table_head = "head"
    _unused_columns = "unused_cols"
    _pairplot_id = "pairplot"
    _overview_css = "overview_css"

    # Strings for HTML
    _table_None = "Provided X, y data do not include this Table."
    _mapping_title = "Category - Original"
    _mapping_format = "{mapped} - {original}"
    _too_many_categories = "(...) Showing only first {} categories"

    def __init__(self, template, css_path, feature_description_class):
        """Create Overview object. Overrides __init__ from BaseView.

        Calls BaseView __init__ method.

        Args:
            template (jinja2.Template): loaded HTML template
            css_path (str): file path to Overview specific CSS file that will be included in HTML
            feature_description_class (str): HTML (CSS) class shared between different objects indicating HTML element
                with hidden (hoverable) description
        """
        super().__init__()
        self.template = template
        self.css = css_path
        self.feature_name_with_description_class = feature_description_class

    def render(self,
               base_css,  # base template params
               creation_date,
               hyperlinks,
               numerical_df,  # main elements of the View
               categorical_df,
               unused_features,
               head_df,
               do_pairplot_flag,
               pairplot_path,
               mapping,  # auxilliary dictionaries
               descriptions
               ):
        """Create HTML from loaded template and with provided arguments.

        Dict of 'jinja id': content pairs is created, fed into render method of provided template and returned. Standard
        params are obtained from BaseView. Link to the Pairplot might be included or not, depending on provided
        do_pairplot_flag.

        Note:
            Every row header in tables is appended with a <span> HTML element holding a description of the feature
            (hidden with CSS).

        Args:
            base_css (str): address of base css file
            creation_date (date): creation date of HTML
            hyperlinks (dict): 'view name': hyperlink pairs
            numerical_df (pandas.DataFrame, None): 'describe' DataFrame of Numerical features or None
            categorical_df (pandas.DataFrame, None): 'describe' DataFrame of Categorical features or None
            unused_features (list): list of names of unused features in the analysis
            head_df (pandas.DataFrame, None): first 5 rows of data (transposed) or None
            do_pairplot_flag (bool): flag indicating if pairplot was generated or not
            pairplot_path (str, None): file path to created pairplot visualization or None
            mapping (dict): 'feature name': category mapping pairs
            descriptions (dict): 'feature name': descriptions pairs

        Returns:
            str: rendered HTML template
        """
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

        # pairplot depending on the flag
        if do_pairplot_flag:
            output[self._pairplot_id] = self._pairplot(pairplot_path)
        else:
            output[self._pairplot_id] = placeholder_features_limit_crossed

        return self.template.render(**output)

    def _tables(self, numerical_df, categorical_df, head_df, mapping, descriptions):
        """Create dict of 'jinja ids': rendered tables HTML.

        Tables are stylized similarly and descriptions are appended whenever possible. If a given table is None,
        placeholder text is put in it's place.

        Args:
            numerical_df (pandas.DataFrame, None): 'describe' DataFrame of Numerical features or None
            categorical_df (pandas.DataFrame, None): 'describe' DataFrame of Categorical features or None
            head_df (pandas.DataFrame, None): first 5 rows of data (transposed) or None
            mapping (dict): 'feature name': category mapping pairs
            descriptions (dict): 'feature name': descriptions pairs

        Returns:
            dict: 'jinja id': HTML table content
        """
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
        """Convert pandas dataframe to HTML string.

        Numbers are rounded to the 2nd decimal place.

        Args:
            dataframe (pandas.DataFrame): DataFrame

        Returns:
            str: table HTML string
        """
        return dataframe.to_html(float_format="{:.2f}".format)

    def _stylize_html_table(self, html_table, mapping, descriptions):
        """Stylize HTML table element with additional content.

        Every header in HTML row gets appended with appropriate descriptions and mappings where applicable. Final HTML
        description element has CSS class from feature_name_with_description_class attribute added as well.

        Args:
            html_table (str): HTML table
            mapping (dict): 'feature name': category mapping pairs
            descriptions (dict): 'feature name': descriptions pairs

        Returns:
            str: HTML table with stylized (added) elements
        """
        # all operations done with table object change it internally
        table = BeautifulSoup(html_table, "html.parser")
        headers = table.select("table tbody tr th")

        for header in headers:
            description = descriptions[header.string]
            header_mapping = mapping[header.string]

            # create new HTML element with description and append mapping to it
            new_description = append_description(description, table)
            self._append_mapping(new_description, header_mapping, table)

            # wrap it in <p> element and add class
            header.string.wrap(table.new_tag("p"))
            header.p.append(new_description)
            header.p["class"] = self.feature_name_with_description_class

        return str(table)

    def _append_mapping(self, html, mapping, parsed_html):
        """Append mappings between values in table and their 'logical counterparts' to parsed_html at the end of
        html element.

        html is an element to which mappings are appended, whereas parsed_html is Soup object with all HTML code in
        question. Every value mapping has its own line with <br> tag at the end. If the number of mapping elements
        exceeds _max_categories_limit class attribute then only those first elements are included in the HTML and the
        rest gets cut off with a placeholder message.

        Nothing is returned from the function as all operations are done on parsed_html object.

        Note:
            if mapping is None, then no changes are introduced.

        Args:
            html (Tag): new Tag created with bs4, linked to parsed_html
            mapping (dict): value in data: external value pairs from a give feature data
            parsed_html (bs4.BeautifulSoup): Soup of the entire HTML table

        """
        # appending mappings to descriptions as long as they exist (they are not none)
        if mapping:
            html.append(parsed_html.new_tag("br"))
            html.append(parsed_html.new_tag("br"))
            html.append(parsed_html.new_string(self._mapping_title))
            i = 1
            for mapped, original in mapping.items():
                if i > self._max_categories_limit:  # 0 indexing
                    html.append(parsed_html.new_tag("br"))
                    html.append(parsed_html.new_string(self._too_many_categories.format(i - 1)))
                    break
                html.append(parsed_html.new_tag("br"))
                html.append(parsed_html.new_string(self._mapping_format.format(mapped=mapped, original=original)))
                i += 1

    def _unused_features_html(self, unused_features):
        """Create list of unused features.

        Elements of unused_features list are wrapped in <ul> and <li> Tags.

        Args:
            unused_features (list): unused_features (list): list of names of unused features in the analysis

        Returns:
            str: HTML with unused features
        """
        html = "<ul>"
        for feature in unused_features:
            html += "<li>" + feature + "</li>"
        html += "</ul>"
        return html

    def _pairplot(self, pairplot_path):
        """Create <img> HTML tag with a file path to created pairplot visualization.

        Args:
            pairplot_path (str): file path to pairplot visualization image

        Returns:
            str: HTML img tag
        """
        template = "<a href={path}><img src={path} title='Click to open larger version'></img></a>"
        html = template.format(path=pairplot_path)
        return html


class FeatureView(BaseView):
    """Features Subpage (View) of the Dashboard. Inherits from BaseView.

    Features View provides detailed information on a particular feature. Jinja template defines:
        - Menu layout for the user to choose which feature they want to investigate at a given time
        - Summary Statistics and Histogram pane on a chosen feature
        - Transformations applied on a chosen Feature
        - Features Correlations Plot
        - Scatter Plot grid

    Feature Correlations Plot is the only 'static' element on the page, as it doesn't change upon choosing another
    feature. The rest of the View gets updated every time different feature is chosen from the Menu. JavaScript needed
    to achieve that level of interactivity is also included in the View.

    Note:
        Scatter Plot grid needs the most resources to be created and updated. It might be turned off to speed up
        the creation of the Dashboard.

    Attributes:
        template (jinja2.Template): loaded HTML template
        css (str): file path to FeaturesView specific CSS file that will be included in HTML
        js (str): file path to FeaturesView specific JS file that will be included in HTML
        target_name (str): name of the target feature
        pre_transformed_columns (list): list of feature names that are already pre-transformed
        Rest of the Attributes are inherited from BaseView.
    """
    # jinja template ids
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

    _transformed_feature = "transformed_feature"
    _transformed_feature_normal_transformations_plots_script = "bokeh_script_normal_transformations_plots"

    # HTML Templates
    _transformed_feature_template = '<div class="{feature_class}" id="{feature_name}">{content}</div>'
    _single_transformed_feature_template = """
        <div class='{transformed_feature_grid_class}'>
            {transformers_html}
            <div class='{transformations_table_class}'>
                {df_html}
            </div>
        </div>"""

    _transformed_feature_original_prefix = "Original_"
    _transformed_feature_transformed_df_title = "Applied Transformations (First 5 Rows) - Test Data"
    _transformed_feature_transformers_title = "Transformers (fitted on Train Data)"
    _transformed_feature_normal_transformations_title = "Normal Transformations applied on a Feature"
    _transformed_feature_normal_transformations_plots_div = "<div class='{plots_class}'><div class='{subtitle_div}'>" \
                                                            "{title}</div><div>{plot_row}</div></div>"
    _transformed_feature_pretransformed_feature_template = "Feature pre-transformed - no information provided."
    _transformed_feature_single_transformer_template = "<div class='{transformed_feature_single_transformer}'>" \
                                                       "{transformer}</div>"
    _transformed_feature_transformers_listing_template = """
        <div class='{transformer_list_class}'>
            <div class='{transformed_feature_subtitle_class}'>{transformers_title}</div>
            <div>{transformers}</div>
            </div>
        """
    _transformed_feature_transformed_html_table_template = "<div class='{transformed_feature_subtitle_class}'>" \
                                                           "{transformed_df_title}</div>"

    _feature_menu_header = "<div class='features-menu-title'><div>Features</div><div class='close-button'>x</div></div>"
    _feature_menu_single_feature = "<div class='{}'><span>{:03}. {}</span></div>"

    # CSS
    _menu_single_feature_class = "single-feature"
    _menu_target_feature_class = "target-feature"
    _first_feature_transformed = "chosen-feature-transformed"
    _transformed_feature_div = "transformed-feature"
    _transformed_feature_grid = "transformed-grid"
    _transformed_feature_transformations_table = "transformations-table"
    _transformed_feature_subtitle_div = "subtitle"
    _transformed_feature_transformer_list = "transformer-list"
    _transformed_feature_single_transformer = "single-transformer"
    _transformed_feature_plots_grid = "transformed-feature-plots"

    def __init__(self, template, css_path, js_path, target_name, pre_transformed_columns):
        """Create FeaturesView object. Overrides __init__ from BaseView.

        Calls BaseView __init__ method.

        Args:
            template (jinja2.Template): loaded HTML template
            css_path (str): file path to FeaturesView specific CSS file that will be included in HTML
            js_path (str): file path to FeaturesView specific JS file that will be included in HTML
            target_name (str): name of the target feature
            pre_transformed_columns (list): list of feature names that are already pre-transformed
        """
        super().__init__()
        self.template = template
        self.css = css_path
        self.js = js_path
        self.target_name = target_name
        self.pre_transformed_columns = pre_transformed_columns

    def render(
            self,
            base_css,  # base template params
            creation_date,
            hyperlinks,
            feature_list,  # main elements of the View
            summary_grid,
            X_transformations,  # transformation part
            y_transformations,
            test_features_df,
            test_transformed_features_df,
            normal_transformations_plots,
            correlations_plot,  # correlations
            do_scatterplot_flag,
            scatterplot,
            numerical_features,  # auxilliary
            initial_feature
    ):
        """Create HTML from loaded template and with provided arguments.

        Dict of 'jinja id': content pairs is created, fed into render method of provided template and returned. Standard
        params are obtained from BaseView. Features Menu bar is created and located on the left of the View.

        Normal Transformations Plot will be included if a given feature is in numerical_features list.

        Scatter Plot grid might be included or not, depending on provided do_scatterplot_flag.

        Args:
            base_css (str): address of base css file
            creation_date (date): creation date of HTML
            hyperlinks (dict): 'view name': hyperlink pairs
            feature_list (list): list of features names
            summary_grid (bokeh.Row): generated InfoGrid layout Row Plot
            X_transformations (dict): 'feature name': (transformers, transformations) tuple pairs
            y_transformations (list): list of transformers used to transform y
            test_features_df (pandas.DataFrame): 'raw' pandas DataFrame with original test data
            test_transformed_features_df (pandas.DataFrame): pandas DataFrame of transformed test data
            normal_transformations_plots (bokeh.Row): generated NormalTransformationsPlots bokeh Row Plot
            correlations_plot (bokeh.Plot): generated CorrelationPlot bokeh Plot
            do_scatterplot_flag (bool): flag indicating if Scatter Plot Grid was generated or not
            scatterplot (bokeh.Grid, None): bokeh Grid of Scatter Plots or None
            numerical_features (list): list of numerical features names
            initial_feature (str): feature name to be used as an initial choice

        Returns:
            str: rendered HTML template
        """
        output = {}

        # Standard variables
        standard = super().standard_params(base_css, creation_date, hyperlinks)
        output.update(standard)

        # JS/CSS
        output[self._features_css] = self.css
        output[self._features_js] = self.js

        # First Feature
        output[self._first_feature] = initial_feature

        # Features Menu
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
            output[self._scatterplot] = placeholder_features_limit_crossed

        # Transformed Features

        # adding target transformation to all transformations
        transformations = X_transformations
        transformations[self.target_name] = (y_transformations, self.target_name)

        # only first 5 rows are going to be shown
        df = test_features_df.head()
        transformed_df = test_transformed_features_df.head()

        # adding scripts from bokeh plots at the end of the HTML so they dont break other JS
        normal_plots_scripts = []
        normal_plots_divs = {}
        for feature, plots in normal_transformations_plots.items():
            script, div = components(plots)
            normal_plots_scripts.append(script)
            normal_plots_divs[feature] = div

        transformed_divs = self._transformed_features_divs(
            df=df,
            transformed_df=transformed_df,
            transformations=transformations,
            numerical_features=numerical_features,
            normal_plots=normal_plots_divs,
            initial_feature=initial_feature
        )
        output[self._transformed_feature] = transformed_divs

        # adding transformation scripts
        output[self._transformed_feature_normal_transformations_plots_script] = "".join(normal_plots_scripts)

        return self.template.render(**output)

    def _create_features_menu(self, features):
        """Create Features Menu HTML.

        Features are listed one by one and wrapped in Divs to make them clickable and interactive. Every feature Div
        has _menu_single_feature_class class attribute appended as a CSS class for querying later on. Target variable
        gets additional CSS class appended. All styling and interactivity comes from CSS and JS.

        Args:
            features (list): list of features names

        Returns:
            str: HTML of Features Menu
        """
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

    def _transformed_features_divs(self,
                                   df,
                                   transformed_df,
                                   transformations,
                                   numerical_features,
                                   normal_plots,
                                   initial_feature
                                   ):
        """Create Transformations Divs for every feature present in the test data.

        If feature is in transformed_columns attribute list, no plots or information are included, only the
        placeholder text for pre-transformed feature. Otherwise, all necessary transformations and transformed
        DataFrames are included in the HTML output.

        Args:
            df (pandas.DataFrame): original DataFrame
            transformed_df (pandas.DataFrame): transformed original DataFrame
            transformations (dict): 'feature name': (transformers, transformations) tuple pairs (including target)
            numerical_features (list): list of numerical features names
            normal_plots (dict): 'feature name': Div for bokeh Plot pairs
            initial_feature (str): feature name to be used as an initial choice

        Returns:
            str: created HTML for features transformations
        """
        output = ""
        for col in df.columns:
            feature_class = self._transformed_feature_div
            # Adding CSS class for initial feature
            if col == initial_feature:
                feature_class += " " + self._first_feature_transformed

            if col not in self.pre_transformed_columns:
                output += self._transformed_column_div(
                    col, feature_class, df, transformed_df, transformations, numerical_features, normal_plots
                )
            else:
                output += self._pre_transformed_column_div(col, feature_class)

        return output

    def _transformed_column_div(self,
                                col,
                                col_class,
                                df,
                                transformed_df,
                                transformations,
                                numerical_features,
                                normal_plots
                                ):
        """Create single feature transformation Div.

        Transformations, transformers and appropriate columns from both df and transformed_df are extracted based on
        a provided col and fed into an HTML template. If col is in numerical features, bokeh Plot Divs are also
        included.

        Args:
            col (str): feature (column) to create transformation Div for
            col_class (str): CSS class to include in the HTML template
            df (pandas.DataFrame): original DataFrame
            transformed_df (pandas.DataFrame): transformed original DataFrame
            transformations (dict): 'feature name': (transformers, transformations) tuple pairs (including target)
            numerical_features (list): list of numerical features names
            normal_plots (dict): 'feature name': Div for bokeh Plot pairs

        Returns:
            str: single feature transformations HTML
        """
        transformers = transformations[col][0]
        new_cols = transformations[col][1]
        content = self._single_transformed_feature(df[col], transformed_df[new_cols], transformers)
        if col in numerical_features:
            plot_row = normal_plots[col]
            content += self._transformed_feature_normal_transformations_plots_div.format(
                title=self._transformed_feature_normal_transformations_title,
                subtitle_div=self._transformed_feature_subtitle_div,
                plots_class=self._transformed_feature_plots_grid,
                plot_row=plot_row
            )

        col_id = "_" + col  # adding underscore to mitigate scenarios when JS doesnt work, e.g. #1
        html = self._transformed_feature_template.format(
            feature_class=col_class,
            title=col,
            content=content,
            feature_name=col_id
        )
        return html

    def _pre_transformed_column_div(self, col, col_class):
        """Create transformation Div for a feature that was pre-transformed.

        Pre-transformed feature does not have any data to include, as transformations happened externally. However,
        structure of Transformation Div should still be preserved to not break any CSS/JS interactions.

        Args:
            col (str): feature (column) to create pre-transformed Div for
            col_class (str): CSS class to include in the HTML template

        Returns:
            str: pre-transformed feature transformation HTML
        """
        content = self._single_transformed_feature_template.format(
            transformed_feature_grid_class=self._transformed_feature_grid,
            transformers_html=self._transformed_feature_pretransformed_feature_template,
            transformations_table_class=self._transformed_feature_transformations_table,
            df_html=""  # empty table
        )
        col_id = "_" + col  # adding underscore to mitigate scenarios when JS doesnt work, e.g. #1
        html = self._transformed_feature_template.format(
            feature_class=col_class,
            title=col,
            content=content,
            feature_name=col_id
        )
        return html

    def _single_transformed_feature(self, series, transformed_output, transformers):
        """Create Transformations Div that is the same for both numerical and categorical features.

        Transformation Div consist of listing of Transformers used and a comparison of original DataFrame vs transformed
        DataFrame (in a same HTML table).

        Args:
            series (pandas.Series): original series of a given feature
            transformed_output (pandas.Series, pandas.DataFrame): transformed data of a given feature
            transformers (list): list of Transformers

        Returns:
            str: feature transformations HTML
        """
        transformers_html = self._transformers_html(transformers)
        df_html = self._transformed_dataframe_html(series, transformed_output)

        template = self._single_transformed_feature_template
        output = template.format(
            transformed_feature_grid_class=self._transformed_feature_grid,
            transformers_html=transformers_html,
            transformations_table_class=self._transformed_feature_transformations_table,
            df_html=df_html
        )
        return output

    def _transformers_html(self, transformers):
        """Create HTML Div with transformers listed.

        Args:
            transformers (list): list of Transformers used

        Returns:
            str: feature Transformers HTML
        """
        single_transformer_template = self._transformed_feature_single_transformer_template

        _ = []
        for transformer in transformers:
            _.append(
                single_transformer_template.format(
                    transformed_feature_single_transformer=self._transformed_feature_single_transformer,
                    transformer=str(transformer)
                )
            )

        template = self._transformed_feature_transformers_listing_template

        output = template.format(
            transformer_list_class=self._transformed_feature_transformer_list,
            transformers_title=self._transformed_feature_transformers_title,
            transformed_feature_subtitle_class=self._transformed_feature_subtitle_div,
            transformers="".join(_)  # joining Divs of Transformers
        )
        return output

    def _transformed_dataframe_html(self, series, transformed):
        """Create HTML table from concatted series and transformed DataFrame/Series.

        Args:
            series (pandas.Series): data Series to be used as a first column of the new DataFrame
            transformed (pandas.Series, pandas.DataFrame): output of transformed series

        Returns:
            str: feature HTML table
        """
        series.name = self._transformed_feature_original_prefix + str(series.name)
        df = pd.concat([series, transformed], axis=1)

        output = self._transformed_feature_transformed_html_table_template.format(
            transformed_feature_subtitle_class=self._transformed_feature_subtitle_div,
            transformed_df_title=self._transformed_feature_transformed_df_title
        )
        output += df.to_html(index=False)
        return output


class ModelsView(BaseView):
    """Base ModelsView. Inherits from BaseView.

    ModelsView is a Base class for different Models Views in different Machine Learning problems.

    Standard params are created from calling BaseView standard_params method. ModelsView also creates Models Result
    table that is shared across all ModelsView child classes.

    Note:
        render method should be overridden in Child class.

    Attributes:
        template (jinja2.Template): loaded HTML template
        css (str): file path to FeaturesView specific CSS file that will be included in HTML
        params_name (str): name of the params column in the Models result table
        model_with_description_class (str): HTML (CSS) class shared between different objects indicating HTML element
                with hidden (hoverable) description
        Rest of the Attributes are inherited from BaseView.
    """
    # jinja template ids
    _models_css = "models_css"
    _models_table = "models_left_upper"

    _models_plot_title = "models_right_title"
    _models_plot = "models_right_plot"
    _models_plot_script = "bokeh_script_models_right"

    _models_left_bottom_title = "models_left_bottom_title"
    _models_left_bottom = "models_left_bottom"
    _models_left_bottom_script = "bokeh_script_models_left_bottom"

    _predictions_table = "predictions_table"
    _predictions_table_script = "bokeh_script_predictions_table"

    # CSS
    _first_model_class = "first-model"
    _other_model_class = "other-model"

    def __init__(self, template, css_path, params_name, model_with_description_class):
        """Create ModelsView object. Overrides __init__ from BaseView.

        Calls BaseView __init__ method.

        Args:
            template (jinja2.Template): loaded HTML template
            css_path (str): file path to FeaturesView specific CSS file that will be included in HTML
            params_name (str): name of the params column in the Models result table
            model_with_description_class (str): HTML (CSS) class shared between different objects indicating HTML
                element with hidden (hoverable) description
        """
        super().__init__()
        self.template = template
        self.css = css_path
        self.params_name = params_name
        self.model_with_description_class = model_with_description_class

    def render(self,
               base_css,
               creation_date,
               hyperlinks,
               model_results,
               models_right,
               models_left_bottom,
               predictions_table
               ):
        """To be implemented by Child class.

        Arguments are defined to adhere to the render method structure.

        Args:
            base_css (str): address of base css file
            creation_date (date): creation date of HTML
            hyperlinks (dict): 'view name': hyperlink pairs
            model_results (pandas.DataFrame): results of Model search
            models_right (Any): Plots or results specific to a View
            models_left_bottom (Any): Plots or results specific to a View
            predictions_table (bokeh.DataTable): DataTable with actual y, predictions from different Models
                and 'raw' original data

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def _base_output(self, base_css, creation_date, hyperlinks, model_results, predictions_table):
        """Create dict of 'jinja template ids': HTML elements that are used across all ModelsView classes.

        standard_params method from BaseView is called to create BaseView HTML elements. Models search results HTML
        Table is also created and bokeh DataTable of predictions from different Models is included as well.

        Note:
            this method should be called by ModelsView child classes to get those elements set up

        Args:
            base_css (str): address of base css file
            creation_date (date): creation date of HTML
            hyperlinks (dict): 'view name': hyperlink pairs
            model_results (pandas.DataFrame): results of Model search
            predictions_table (bokeh.DataTable): DataTable with actual y, predictions from different Models
                and 'raw' original data

        Returns:
            dict: 'jinja template id': HTML element pairs
        """
        output = {}

        # Standard variables
        standard = super().standard_params(base_css, creation_date, hyperlinks)
        output.update(standard)

        # ModelsView CSS
        output[self._models_css] = self.css

        # Model results table
        output[self._models_table] = self._models_result_table(model_results)

        # Predictions bokeh DataTable
        predictions_script, predictions_div = components(predictions_table)
        output[self._predictions_table_script] = predictions_script
        output[self._predictions_table] = predictions_div

        return output

    def _models_result_table(self, results_dataframe):
        """Create HTML Table from results DataFrame.

        Score names are included as headers, whereas Model names are used as row headers. Model Names are checked and
        potential duplicates are changed. Params column is removed from the table, but instead included as hidden
        HTML element that becomes visible when Model name is hovered over. Additionally, different CSS classes are
        added to different rows.

        Args:
              results_dataframe (pandas.DataFrame): DataFrame with Models search results

        Returns:
              str: results HTML table
        """
        new_df = results_dataframe
        new_df.index = replace_duplicate_str(results_dataframe.index.tolist())
        new_params = series_to_dict(results_dataframe[self.params_name])

        # removing params from the dataframe
        df = results_dataframe.drop([self.params_name], axis=1)
        df.index.name = None  # no index name for styling purposes
        html_table = df.to_html(float_format="{:.5f}".format)

        table = BeautifulSoup(html_table, "html.parser")
        headers = table.select("table tbody tr th")

        # appending params to Model names
        for header in headers:
            single_model_params = new_params[header.string]
            params_html = append_description(single_model_params, table)

            header.string.wrap(table.new_tag("p"))
            header.p.append(params_html)
            header.p["class"] = self.model_with_description_class

        # adding different CSS classes to distinguish different Models
        rows = table.select("table tbody tr")
        rows[0]["class"] = self._first_model_class
        for row in rows[1:-1]:
            row["class"] = self._other_model_class

        output = str(table)
        return output


class ModelsViewClassification(ModelsView):
    """ModelsView for Classification problems. Inherits from ModelsView.

    ModelsViewClassification has:
        - ROC/Precision Recall/DET curves for all Models on the right of the View
        - HTML Confusion Matrices for all Models on the left bottom of the View

    Standard Params and Predictions Table are taken from ModelsView Parent class.

    Attributes:
        Inherited from ModelsView.
    """
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
        """Create ModelsViewClassification object. Overrides __init__ from ModelsView.

        Calls ModelsView __init__ method.

        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(*args, **kwargs)

    def render(self,
               base_css,
               creation_date,
               hyperlinks,
               model_results,
               models_right,
               models_left_bottom,
               predictions_table
               ):
        """Create HTML from loaded template and with provided arguments. Overrides render from ModelsView.

        Dict of 'jinja id': content pairs is created, fed into render method of provided template and returned. Models
        search result HTML table and Predictions bokeh DataTable are created with ModelsView _base_output method.

        View specific content is:
            - models_right: ROC/Precision Recall/DET curves Plots for every Model
            - models_left_bottom: HTML Confusion Matrices

        Args:
            base_css (str): address of base css file
            creation_date (date): creation date of HTML
            hyperlinks (dict): 'view name': hyperlink pairs
            model_results (pandas.DataFrame): results of Model search
            models_right (bokeh.Plot): bokeh Tabs Plot with different Performance Curves for Models
            models_left_bottom (list): list of tuples (Model, confusion matrix data)
            predictions_table (bokeh.DataTable): DataTable with actual y, predictions from different Models
                and 'raw' original data

        Returns:
            str: rendered HTML template
        """
        # Standard Params
        output = super()._base_output(base_css, creation_date, hyperlinks, model_results, predictions_table)

        # View specific Content
        models_plot = models_right
        confusion_matrices = assess_models_names(models_left_bottom)

        # Performance Assessments Curves on the right
        models_plot_script, models_plot_div = components(models_plot)
        output[self._models_plot_title] = self._models_plot_title_text
        output[self._models_plot_script] = models_plot_script
        output[self._models_plot] = models_plot_div

        # HTML Confusion Matrices on the bottom left
        output[self._models_left_bottom_title] = self._models_confusion_matrix_title_text
        output[self._models_left_bottom] = self._confusion_matrices(confusion_matrices)

        return self.template.render(**output)

    def _confusion_matrices(self, models_confusion_matrices):
        """Create HTML from Models confusion matrices results.

        Confusion Matrices for all Models are converted to HTML tables and wrapped in their own Divs.

        Args:
            models_confusion_matrices (list): list of tuples (Model, confusion matrix data)

        Returns:
            str: Confusion Matrices HTML
        """
        output = "<div class='{}'>".format(self._confusion_matrices_class)
        i = 0

        for model, matrix in models_confusion_matrices:
            # differentiating Model colors based on the order
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
        """Create HTML table from confusion array.

        Args:
            confusion_array (numpy.ndarray): [2, 2] array

        Returns:
            str: Confusion Matrix HTML table
        """
        tn, fp, fn, tp = confusion_array.ravel()
        table = self._single_confusion_matrix_html_template.format(tn=tn, fp=fp, fn=fn, tp=tp).replace("\n", "")
        output = "<div class='{}'>{}</div>".format(self._confusion_matrices_single_matrix_table, table)
        return output


class ModelsViewRegression(ModelsView):
    """ModelsView for Regression problems. Inherits from ModelsView.

    ModelsViewRegression has:
        - Prediction Errors Plots for all Models on the right of the View
        - Residual Plots for all Models on the left bottom of the View

    Standard Params and Predictions Table are taken from ModelsView Parent class.

    Attributes:
        Inherited from ModelsView.
    """
    _prediction_errors_title = "Prediction Error Plots"
    _residuals_title = "Residual Plots"

    def __init__(self, *args, **kwargs):
        """Create ModelsViewRegression object. Overrides __init__ from ModelsView.

        Calls ModelsView __init__ method.

        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(*args, **kwargs)

    def render(self,
               base_css,
               creation_date,
               hyperlinks,
               model_results,
               models_right,
               models_left_bottom,
               predictions_table
               ):
        """Create HTML from loaded template and with provided arguments. Overrides render from ModelsView.

        Dict of 'jinja id': content pairs is created, fed into render method of provided template and returned. Models
        search result HTML table and Predictions bokeh DataTable are created with ModelsView _base_output method.

        View specific content is:
            - models_right: Prediction Error Plots for every Model
            - models_left_bottom: Residual Plots for every Model

        Args:
            base_css (str): address of base css file
            creation_date (date): creation date of HTML
            hyperlinks (dict): 'view name': hyperlink pairs
            model_results (pandas.DataFrame): results of Model search
            models_right (bokeh.Plot): bokeh Tabs Plot with Prediction Errors for every Model
            models_left_bottom (list): bokeh Tabs Plot with Residuals for every Model
            predictions_table (bokeh.DataTable): DataTable with actual y, predictions from different Models
                and 'raw' original data

        Returns:
            str: rendered HTML template
        """
        # Standard Params
        output = self._base_output(base_css, creation_date, hyperlinks, model_results, predictions_table)

        # View Specific Content
        prediction_errors_plot = models_right
        residual_plot = models_left_bottom

        # Prediction Error Plots
        models_right_plot_script, models_right_plot_div = components(prediction_errors_plot)
        output[self._models_plot_title] = self._prediction_errors_title
        output[self._models_plot_script] = models_right_plot_script
        output[self._models_plot] = models_right_plot_div

        # Residual Plots
        models_left_bottom_plot_script, models_left_bottom_plot_div = components(residual_plot)
        output[self._models_left_bottom_title] = self._residuals_title
        output[self._models_left_bottom_script] = models_left_bottom_plot_script
        output[self._models_left_bottom] = models_left_bottom_plot_div

        return self.template.render(**output)


class ModelsViewMulticlass(ModelsView):
    """ModelsView for Multiclass problems. Inherits from ModelsView.

    ModelsViewMulticlass has:
        - NO content on the right of the View
        - Confusion Matrices Plots on the bottom left of the View

    Standard Params and Predictions Table are taken from ModelsView Parent class.

    Attributes:
        Inherited from ModelsView.
    """
    _confusion_matrices_title = "Confusion Matrices"

    def __init__(self, *args, **kwargs):
        """Create ModelsViewMulticlass object. Overrides __init__ from ModelsView.

        Calls ModelsView __init__ method.

        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(*args, **kwargs)

    def render(self,
               base_css,
               creation_date,
               hyperlinks,
               model_results,
               models_right,
               models_left_bottom,
               predictions_table
               ):
        """Create HTML from loaded template and with provided arguments. Overrides render from ModelsView.

        Dict of 'jinja id': content pairs is created, fed into render method of provided template and returned. Models
        search result HTML table and Predictions bokeh DataTable are created with ModelsView _base_output method.

        View specific content is:
            - models_right: NO content
            - models_left_bottom: Confusion Matrices bokeh Plots

        Args:
            base_css (str): address of base css file
            creation_date (date): creation date of HTML
            hyperlinks (dict): 'view name': hyperlink pairs
            model_results (pandas.DataFrame): results of Model search
            models_right (None): no content
            models_left_bottom (bokeh.Row): bokeh Row Layout with Confusion Matrices for every Model
            predictions_table (bokeh.DataTable): DataTable with actual y, predictions from different Models
                and 'raw' original data

        Returns:
            str: rendered HTML template
        """
        # Standard Params
        output = self._base_output(base_css, creation_date, hyperlinks, model_results, predictions_table)

        # View Specific Content
        confusion_matrices = models_left_bottom

        # Confusion Matrices Plots
        models_left_bottom_plot_script, models_left_bottom_plot_div = components(confusion_matrices)
        output[self._models_left_bottom_title] = self._confusion_matrices_title
        output[self._models_left_bottom_script] = models_left_bottom_plot_script
        output[self._models_left_bottom] = models_left_bottom_plot_div

        return self.template.render(**output)
