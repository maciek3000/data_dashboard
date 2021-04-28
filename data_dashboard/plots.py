import functools
import scipy.stats
import pandas as pd
import seaborn as sns
from bs4 import BeautifulSoup
from bokeh.plotting import figure
from bokeh.layouts import column, row, Spacer
from bokeh.models import ColumnDataSource, FuncTickFormatter, Span, Slope
from bokeh.models.widgets import Select, Div, HTMLTemplateFormatter
from bokeh.models import CustomJS, ColorBar, BasicTicker, LinearColorMapper, Panel, Tabs, LabelSet, NumeralTickFormatter
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.palettes import Category10  # , Reds4
from bokeh.models.widgets.tables import DataTable, TableColumn
from .functions import append_description, assess_models_names


def stylize(plot_func):
    """Return function stylizing Plot figure.

    Used as a decorator for functions returning Plot so that all figures are stylized similarly.

    Returns:
        function: function stylizing Plot figure
    """

    @functools.wraps(plot_func)
    def wrapper(self, *args, **kwargs):
        p = plot_func(self, *args, **kwargs)

        text_color = "#8C8C8C"

        p.axis.minor_tick_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_text_font = "Lato"
        p.axis.major_label_text_color = text_color
        p.axis.axis_line_color = text_color

        p.axis.axis_label_text_font = "Lato"
        p.axis.axis_label_text_color = text_color
        p.axis.axis_label_text_font_style = "normal"

        p.xaxis.axis_label_text_font_size = "16px"
        p.xaxis.major_label_text_font_size = "14px"
        p.yaxis.axis_label_text_font_size = "16px"
        p.yaxis.major_label_text_font_size = "14px"

        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None

        p.title.text_font = "Lato"
        p.title.text_color = text_color
        p.title.text_font_size = "20px"

        return p

    return wrapper


def default_figure(plot_specific_kwargs=None):
    """Create and return bokeh figure with predefined settings that should be consistent across different plots.

    Additional arguments can be provided in plot_specific_kwargs argument as a kwargs dictionary of param: value pairs.

    Note:
        Keep in mind that supplying stylizing attributes to Figure() doesn't always work, as apparently some methods
        (e.g. grids) require axes to be in place to change anything in them. That's the reason some of the stylizing
        takes place here and some other in stylize decorator.

    Args:
        plot_specific_kwargs (dict, optional): dict of parameter: value pairs for the figure, defaults to None

    Returns:
        bokeh.Figure: bokeh Plot Figure
    """
    default_kwargs = {
        "tools": [],
        "toolbar_location": None,
        "outline_line_color": None
    }

    if plot_specific_kwargs:
        default_kwargs.update(plot_specific_kwargs)

    p = figure(**default_kwargs)
    return p


class PairPlot:
    """Seaborn pairplot of features.

    Create Seaborn pairplot of features (scatter plots for combinations of different features, histogram for the
    same feature in one Plot).

    Attributes:
        plot_design (PlotDesign): PlotDesign object with predefined style elements
    """

    def __init__(self, plot_design):
        """Create PairPlot object.

        Set custom style to seaborn module.

        Args:
            plot_design (PlotDesign): PlotDesign object with predefined style elements
        """
        self.plot_design = plot_design

        text_color = self.plot_design.text_color
        text_font = self.plot_design.text_font
        sns.set_style("white",
                      {
                          "axes.edgecolor": text_color,
                          "axes.labelcolor": text_color,
                          "text.color": text_color,
                          "font.sans-serif": [text_font],
                          "xtick.color": text_color,
                          "ytick.color": text_color,
                      }
                      )

    def pairplot(self, dataframe):
        """Create seaborn pairplot with data provided in the dataframe.

        Args:
            dataframe (pandas.DataFrame): DataFrame to create pairplot visualization with

        Returns:
            seaborn.PairGrid: pairplot visualization
        """
        colors = {"color": self.plot_design.pairplot_color}
        p = sns.pairplot(dataframe, plot_kws=colors, diag_kws=colors)
        return p


class CorrelationPlot:
    """HeatMap of correlations between features.

    Plot is made of two separate HeatMaps - one where features are normalized and one where features are provided
    'raw', without any transformations.

    Note:
        Only Absolute Values of correlations are shown - even though there is a statistical difference between
        correlation coefficient of 0.8 and -0.8, the Plot will show them as the same to demonstrate only the strength
        of the correlation, not the direction.

     Attributes:
        plot_design (PlotDesign): PlotDesign object with predefined style elements
        target_name (str): target feature name
    """
    # CSS
    _correlation = "correlation-plot"

    # parameters for Plot/Source
    _correlation_x = "x"
    _correlation_y = "correlation_y"
    _correlation_values_normalized = "correlation_values_normalized"
    _correlation_values_normalized_abs = "correlation_values_normalized_abs"
    _correlation_values_normalized_title = "Normalized Data Correlation"
    _correlation_values_raw = "correlation_values_raw"
    _correlation_values_raw_abs = "correlation_values_raw_abs"
    _correlation_values_raw_title = "Raw Data Correlation"

    def __init__(self, plot_design, target_name):
        """Create Correlation Plot object.

        Args:
            plot_design (PlotDesign): PlotDesign object with predefined style elements
            target_name (str): target feature name
        """
        self.plot_design = plot_design
        self.target_name = target_name

    def correlation_plot(self, correlation_data_normalized, correlation_data_raw):
        """Create Correlation Plot with provided correlation data.

        Args:
            correlation_data_normalized (pandas.DataFrame): DataFrame of correlations between normalized columns
            correlation_data_raw (pandas.DataFrame): DataFrame of correlations between 'raw' columns

        Returns:
            bokeh.Tabs: Plot with two tabs
        """
        # correlation source is left in case callback is needed in the future
        correlation_source, correlation_plot = self._create_correlation(
            correlation_data_normalized, correlation_data_raw
        )

        return correlation_plot

    def _create_correlation(self, data_normalized, data_raw):
        """Create Tabs Plot with two HeatMaps based on provided data.

        Plots included in Tabs are identical when it comes to style attributes, but each of them have different title
        and data underneath.

        Args:
            data_normalized (pandas.DataFrame): DataFrame of correlations between normalized columns
            data_raw (pandas.DataFrame): DataFrame of correlations between 'raw' columns

        Returns:
            tuple: (bokeh.ColumnDataSource, Tabs Plot with two Panels)
        """
        source, cols_in_order = self._create_correlation_source(data_normalized, data_raw)
        mapper = self._create_correlation_color_mapper()
        plots = []
        for name, value in [
            (self._correlation_values_normalized_title, self._correlation_values_normalized_abs),
            (self._correlation_values_raw_title, self._correlation_values_raw_abs)
        ]:
            plot = self._create_correlation_plot(source, cols_in_order, mapper, value)
            plots.append(Panel(child=plot, title=name))

        main_plot = Tabs(tabs=plots)
        return source, main_plot

    def _create_correlation_source(self, data_normalized, data_raw):
        """Create ColumnDataSource needed for Plots.

        Both normalized and 'raw' data are included in the source so that both plots can have access to all data. Cols
        are returned for setting up ranges on the Plot later on.

        Note:
            target name is injected in the beginning of columns list so that it located at the start of the axes
            in the Plot

        Args:
            data_normalized (pandas.DataFrame): DataFrame of correlations between normalized columns
            data_raw (pandas.DataFrame): DataFrame of correlations between 'raw' columns

        Returns:
            tuple: (ColumnDataSource, column list)
        """

        source = ColumnDataSource()
        cols = sorted(data_normalized.columns.to_list())
        cols.remove(self.target_name)
        cols.insert(0, self.target_name)  # cols are needed for x_range and y_range in the plot

        # stacking to get 1d array
        stacked_normalized = data_normalized.stack()
        stacked_raw = data_raw.stack()

        features_x = stacked_normalized.index.droplevel(0).to_list()  # one of the indexes
        features_y = stacked_normalized.index.droplevel(1).to_list()  # second of the indexes

        values_normalized = stacked_normalized.to_list()
        values_normalized_abs = stacked_normalized.apply(lambda x: abs(x)).to_list()
        values_raw = stacked_raw.to_list()
        values_raw_abs = stacked_raw.apply(lambda x: abs(x)).to_list()

        source.data = {
            self._correlation_x: features_x,
            self._correlation_y: features_y,
            self._correlation_values_normalized: values_normalized,
            self._correlation_values_normalized_abs: values_normalized_abs,
            self._correlation_values_raw: values_raw,
            self._correlation_values_raw_abs: values_raw_abs
        }

        return source, cols

    @stylize
    def _create_correlation_plot(self, source, cols_for_range, color_mapper, value_to_color):
        """Create Correlation HeatMap Plot.

        value_to_color determines which values from ColumnDataSource source are used as a fill value.

        Args:
            source (bokeh.ColumnDataSource): ColumnDataSource with correlation data
            cols_for_range (list): list of columns (features) to be put into axes ranges
            color_mapper (bokeh.LinearColorMapper): LinearColorMapper for HeatMap
            value_to_color (str): name of the feature based on which values will be colored

        Returns
            bokeh.Figure: created Plot Figure
        """
        # tooltip
        tooltip_text = [
            (self._correlation_values_normalized_title, "@" + self._correlation_values_normalized),
            (self._correlation_values_raw_title, "@" + self._correlation_values_raw)
        ]

        # figure
        kwargs = {
            "css_classes": [self._correlation],
            "x_range": cols_for_range,
            "y_range": cols_for_range[::-1],  # first value to be at the top of the axis
            "tooltips": tooltip_text,
        }
        p = default_figure(kwargs)

        # heatmap
        p.rect(
            x=self._correlation_x,
            y=self._correlation_y,
            source=source,
            fill_color={"field": value_to_color, "transform": color_mapper},
            width=1,
            height=1,
            line_color=None,
        )

        # plot specific styling
        p.xaxis.major_label_orientation = -1  # in radians
        p.add_layout(ColorBar(color_mapper=color_mapper, width=40), "right")  # legend Bar

        return p

    def _create_correlation_color_mapper(self):
        """Create LinearColorMapper used for Correlation HeatMap plots.

        Returns:
            bokeh.LinearColorMapper: color mapper used to map values to specific colors
        """
        tints = self.plot_design.contrary_color_tints

        no_correlation = [tints[9]]
        small_correlation = [tints[7]] * 2
        medium_correlation = [tints[6]] * 2
        high_correlation = [tints[4]] * 4
        very_high_correlation = [tints[2]] * 1
        palette = no_correlation + small_correlation + medium_correlation + high_correlation + very_high_correlation

        linear_correlation = tints[0]  # perfect correlations gets it's own color
        cmap = LinearColorMapper(
            palette=palette,
            low=0,
            high=0.9999,
            high_color=linear_correlation
        )
        return cmap


class NormalTransformationsPlots:
    """Row of Histograms representing Normal Distribution Transformations.

    Attributes:
        plot_design (PlotDesign): PlotDesign object with predefined style elements
    """
    # hardcoded Plot titles
    _box_cox_title = "Box-Cox"
    _yeo_johnson_title = "Yeo-Johnson"
    _quantile_transformer_title = "QuantileTransformer"

    def __init__(self, plot_design):
        """Create NormalTransformationsPlots object.

        Args:
            plot_design (PlotDesign): PlotDesign object with predefined style elements
        """
        self.plot_design = plot_design

    def plots(self, histogram_data):
        """Create pairs of 'feature name': histogram plot rows from histogram_data.

        Transformations of every feature are represented in their own Histogram Plots, which are all placed together
        in one row (for a feature).

        Args:
            histogram_data (dict): dictionary of 'feature name': tuple (Transformer, histogram data) pairs

        Returns:
            dict: dictionary of 'feature name': created Plot pairs
        """
        output = {}
        for feature, transformer_data in histogram_data.items():
            plot_row = self._plot_row(transformer_data)
            output[feature] = plot_row

        return output

    def _plot_row(self, transformer_data):
        """Create row of Histogram plots depending on how many Transformations are in transformer_data.

        Plots get a Spacer squeezed between them so that they aren't too cluttered.

        Args:
            transformer_data (list): list of tuples of (Transformer, histogram data of transformation)

        Returns:
            bokeh.Row: bokeh Row layout of histograms of transformed data
        """
        plots = []
        for transformer, histogram_data in transformer_data:
            tr_name = str(transformer)
            if "box-cox" in tr_name:
                plot_name = self._box_cox_title
            elif "PowerTransformer" in tr_name:
                plot_name = self._yeo_johnson_title  # PlotTransformer() as yeo-johnson is default
            elif "QuantileTransformer" in tr_name:
                plot_name = self._quantile_transformer_title
            else:
                plot_name = "Undefined"
            plot = self._single_histogram(plot_name, histogram_data)
            plots.append(plot)
            plots.append(Spacer(width=50))

        output = row(*plots)
        return output

    @stylize
    def _single_histogram(self, plot_name, histogram_data):
        """Return Histogram Plot bokeh Figure.

        Plot is created with plot_name as a title and unpacked histogram_data as values.

        Args:
            plot_name (str): title of the Plot
            histogram_data (tuple): histogram values, left edges, right edges

        Returns:
            bokeh.Plot: Histogram Plot Figure

        """
        hist, left_edges, right_edges = histogram_data

        # figure
        kwargs = {
            "plot_height": 250,
            "height_policy": "fit",
            "plot_width": 250,
            "title": plot_name,
        }
        p = default_figure(kwargs)

        # histogram
        fcolor = self.plot_design.base_color_tints[-3]
        p.quad(top=hist, bottom=0, left=left_edges, right=right_edges, fill_color=fcolor, line_color=fcolor)

        # plot specific styling
        p.y_range.start = 0
        p.xaxis.ticker = BasicTicker(desired_num_ticks=5)
        p.xaxis.formatter = NumeralTickFormatter(format="0.[0]")

        return p


class MainGrid:
    """Base class for Grid Plots.

    Include dropdown with feature names for interactivity. This dropdown is usually hidden with help of CSS styling,
    but it can still be accessed with JS scripts.

    Attributes:
        features (list): list of features names
        plot_design (PlotDesign): PlotDesign object with predefined style elements
        feature_description_class (str): HTML (CSS) class shared between different objects indicating HTML element
            with hidden (hoverable) description
    """
    _features_dropdown = "features-dropdown"

    def __init__(self, features, plot_design, feature_description_class):
        """Create MainGrid object.

        Args:
            features (list): list of features names
            plot_design (PlotDesign): PlotDesign object with predefined style elements
            feature_description_class (str): HTML (CSS) class shared between different objects indicating HTML element
                with hidden (hoverable) description
        """
        # Font won't be updated in plots until any change is made (e.g. choosing different Feature).
        # This is a bug in bokeh: https://github.com/bokeh/bokeh/issues/9448
        # Issue is relatively minor, I won't be doing any workaround for now.

        self.features = features
        self.plot_design = plot_design
        self.feature_description_class = feature_description_class

    def _create_features_dropdown(self, name=_features_dropdown):
        """Create bokeh Dropdown Widget with feature names as available selections.

        Name can be customized in case it is needed for Bokeh element searching.

        Args:
            name (str, optional): name of the dropdown, defaults to _features_dropdown class attribute.

        Returns:
            bokeh.Dropdown: Dropdown with feature names as available options
        """
        fts = sorted(self.features)
        d = Select(options=fts, css_classes=[self._features_dropdown], name=name)
        return d

    def _create_features_dropdown_callbacks(self, **kwargs):
        """Create JS callback associated with the dropdown.

        Child classes should override this method.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Raises:
            NotImplementedError: Classes should implement this method.
        """
        raise NotImplementedError


class InfoGrid(MainGrid):
    """InfoGrid used as a basic statistic element in FeaturesView. Inherits from MainGrid.

    Grid consists of summary Div defining some of the common summary statistics (e.g. mean, median, etc.) and a
    Histogram plot of a given feature. Based on changes to the Dropdown, values in those elements adjust dynamically
    to a chosen feature.

    Note:
        Attributes are inherited from MainGrid.
    """

    # HTML elements
    _info_div_html = """
    <div id="info-div-description"><p class="{feature_description_class}">Description<span id="info_div_description">{description}</span></p></div>
    <div>Type: <span id="info_div_type">{type}</span></div>
    <div>Mean: <span id="info_div_mean">{mean:.4f}</span></div>
    <div>Median: <span id="info_div_median">{median:.4f}</span></div>
    <div>Min: <span id="info_div_min">{min:.4f}</span></div>
    <div>Max: <span id="info_div_max">{max:.4f}</span></div>
    <div>Standard deviation: <span id="info_div_std">{std:.4f}</span></div>
    <div># of Missing: <span id="info_div_missing">{missing:.4f}</span></div>
    """

    # CSS elements
    _infogrid_dropdown = "info_grid_dropdown"
    _feature_name = "feature_name"
    _infogrid_left_pane = "info-div-left-pane"
    _info_div = "info-div"
    _info_div_content = "info-div-content"
    _infogrid_row = "infogrid-row"
    _infogrid_all = "infogrid"
    _histogram = "histogram-plot"

    # JS callbacks
    _info_div_callback = """
                // new values
                var new_feature = cb_obj.value;  // new feature
                
                // updating 
                
                function update_span(id, group) {
                    document.querySelector(id).innerText = summary_statistics[new_feature][group];
                };
                
                var ids = ["#info_div_type", "#info_div_description", "#info_div_mean", "#info_div_median", 
                        "#info_div_min", "#info_div_max", "#info_div_std", "#info_div_missing"];
                var types = ["type", "description", "mean", "50%", "min", "max", "std", "missing"];
                
                for (i=0; i < ids.length; i++) {
                    update_span(ids[i], types[i]);
                };
            """

    _histogram_callback = """
                // new dropdown value
                var new_val = cb_obj.value;  
                
                // new histogram data 
                var new_hist = hist_data[new_val];
                var hist = new_hist[0];
                var left_edges = new_hist[1];
                var right_edges = new_hist[2];
                
                // histogram source updated
                hist_source.data["{hist_source_data}"] = hist;
                hist_source.data["{hist_source_left_edges}"] = left_edges;
                hist_source.data["{hist_source_right_edges}"] = right_edges;
                
                // updating ColumnDataSources
                hist_source.change.emit();
            """

    # plot elements
    _histogram_title = "Feature Distribution"
    _hist_source_data = "hist"
    _hist_source_left_edges = "left_edges"
    _hist_source_right_edges = "right_edges"

    def __init__(self, features, plot_design, feature_description_class):
        """Create InfoGrid object.

        Call MainGrid __init__ with necessary arguments.

        Args:
            features (list): list of features names
            plot_design (PlotDesign): PlotDesign object with predefined style elements
            feature_description_class (str): HTML (CSS) class shared between different objects indicating HTML element
                with hidden (hoverable) description
        """
        super().__init__(features, plot_design, feature_description_class)

    def summary_grid(self, summary_statistics, histogram_data, initial_feature):
        """Create Summary Grid with Summary Statistics and distribution histogram.

        Grid has a design of:
            - hidden Dropdown at the top
            - Summary Statistics Div on the left
            - Histogram Plot on the right

        On change of value in dropdown, underlying data also changes to represent statistics of a chosen feature.

        Args:
            summary_statistics (dict): 'feature name': summary dict pairs
            histogram_data (dict): 'feature name': histogram data tuple pairs
            initial_feature (str): feature name used to extract initial data

        Returns:
            bokeh.Column: Column of Plot elements
        """
        # dropdown
        dropdown = self._create_features_dropdown(self._infogrid_dropdown)

        # histogram
        histogram_source, histogram_plot = self._create_histogram(histogram_data, initial_feature)

        # summary div
        info_div = self._create_info_div(summary_statistics, initial_feature)

        # JS callbacks
        callbacks = self._create_features_dropdown_callbacks(
            summary_statistics=summary_statistics,
            histogram_data=histogram_data,
            histogram_source=histogram_source,
        )
        for callback in callbacks:
            dropdown.js_on_change("value", callback)

        # output
        output = column(
            dropdown,  # this dropdown will be invisible (display: none)
            row(
                info_div, Spacer(width=150), histogram_plot, css_classes=[self._infogrid_left_pane]
            ),
            css_classes=[self._infogrid_all]
        )
        return output

    def _create_features_dropdown_callbacks(self, summary_statistics, histogram_data, histogram_source):
        """Create callbacks that will be triggered upon value change in Grid Dropdown. Implementation of method
        defined in MainGrid.

        Args:
            summary_statistics (dict): 'feature name': summary dict pairs
            histogram_data (dict): 'feature name': histogram data tuple pairs
            histogram_source (bokeh.ColumnDataSource): ColumnDataSource used to provide data to the Plot

        Returns:
            list: created callbacks
        """
        callbacks = []

        for call in [
            self._create_histogram_callback(histogram_data, histogram_source),
            self._create_info_div_callback(summary_statistics),
        ]:
            callbacks.append(call)

        return callbacks

    def _create_histogram_callback(self, histogram_data, histogram_source):
        """Create callback responsible for changing data in Histogram Plot when the feature in the Dropdown changes.

        JS Code updates Histogram ColumnDataSource with data from histogram_data depending on the feature that was
        chosen.

        Args:
            histogram_data (dict): 'feature name': histogram data tuple pairs
            histogram_source (bokeh.ColumnDataSource): ColumnDataSource used to provide data to the Plot

        Return:
            bokeh.CustomJS: dropdown callback
        """
        kwargs = {
            "hist_source": histogram_source,
            "hist_data": histogram_data
        }

        code = self._histogram_callback.format(
            hist_source_data=self._hist_source_data,
            hist_source_left_edges=self._hist_source_left_edges,
            hist_source_right_edges=self._hist_source_right_edges
        )

        callback = CustomJS(args=kwargs, code=code)
        return callback

    def _create_info_div_callback(self, summary_statistics):
        """Create callback responsible for changing data in Summary Div when the feature in the Dropdown changes.

        JS Code updates Summary Div elements with data from summary_statistics depending on the feature that was
        chosen. Elements are identified via their IDs hardcoded in the JS code and during the creation of Summary Div.

        Args:
            summary_statistics (dict): 'feature name': summary dict pairs

        Return:
            bokeh.CustomJS: dropdown callback
        """
        kwargs = {
            "summary_statistics": summary_statistics
        }
        callback = CustomJS(
            args=kwargs,
            code=self._info_div_callback
        )
        return callback

    def _create_info_div(self, summary_statistics, feature):
        """Create Div containing summary statistics for a chosen feature.

        Provided feature is used as an initial choice upon creation of the Div.

        Args:
            summary_statistics (dict): 'feature name': summary dict pairs
            feature (str): feature name used to extract initial data

        Returns:
            bokeh.Div: bokeh Div Widget
        """
        feature_dict = summary_statistics[feature]

        # statistics using describe method of pandas.DataFrame
        text = self._info_div_html.format(
            feature_description_class=self.feature_description_class,
            info_div_content=self._info_div_content,
            type=feature_dict["type"],
            description=feature_dict["description"],
            mean=feature_dict["mean"],
            median=feature_dict["50%"],
            min=feature_dict["min"],
            max=feature_dict["max"],
            std=feature_dict["std"],
            missing=feature_dict["missing"]
        )
        d = Div(name=self._info_div_content, css_classes=[self._info_div_content], text=text)

        return d

    def _create_histogram(self, histogram_data, feature):
        """Create Histogram Plot and underlying Histogram ColumnDataSource.

        Args:
            histogram_data (dict): 'feature name': histogram data tuple pairs
            feature (str): feature name used to extract initial data

        Returns:
            tuple: (Histogram ColumnDataSource, Histogram Plot Figure)
        """
        hist_source = self._create_histogram_source(histogram_data, feature)
        hist_plot = self._create_histogram_plot(hist_source)
        return hist_source, hist_plot

    def _create_histogram_source(self, histogram_data, feature):
        """Create ColumnDataSource used in Histogram Plot.

        Provided feature is used as an initial choice upon creation of the Div.

        Args:
            histogram_data (dict): 'feature name': histogram data tuple pairs
            feature (str): feature name used to extract initial data

        Returns:
            bokeh.ColumnDataSource: Histogram ColumnDataSource
        """
        source = ColumnDataSource()
        first_values = histogram_data[feature]

        source.data = {
            self._hist_source_data: first_values[0],
            self._hist_source_left_edges: first_values[1],
            self._hist_source_right_edges: first_values[2]
        }
        return source

    @stylize
    def _create_histogram_plot(self, source):
        """Create Histogram Plot Figure.

        Args:
            source (bokeh.ColumnDataSource): ColumnDataSource used to provide data to the Plot

        Returns:
            bokeh.Figure: bokeh Histogram Plot Figure
        """
        # figure
        kwargs = {
            "plot_height": 300,
            "height_policy": "fit",
            "plot_width": 300,
            "title": self._histogram_title,
            "css_classes": [self._histogram]
        }
        p = default_figure(kwargs)

        # histogram
        fcolor = self.plot_design.fill_color
        p.quad(top=self._hist_source_data,
               bottom=0,
               left=self._hist_source_left_edges,
               right=self._hist_source_right_edges,
               source=source,
               fill_color=fcolor,
               line_color=fcolor
               )

        # plot specific styling
        p.y_range.start = 0
        p.yaxis.visible = False
        p.xaxis.ticker = BasicTicker(desired_num_ticks=5)
        p.xaxis.formatter = NumeralTickFormatter(format="0.[0]")

        return p


class ScatterPlotGrid(MainGrid):
    """ScatterPlot Grid visualizing Scatters of every feature against another. Inherits from MainGrid.

    Features are plotted against each other in a similar fashion to Seaborn's pairplot. The additional layers of
    complexity is added by introducing coloring - every feature is also used as a hue. Because of that, the number
    of scatter plots to draw increases in comparison to sns pairplot.

    Note:
        Row with a feature that is chosen (via Dropdown) is greyed out to minimize confusion (I've tried to remove
        it from the Plot all together but it proved to be quite troublesome as the whole layout would break -
        therefore it was decided to simply grey it out).

    Attributes:
        categorical_columns (list): list of categorical columns
        feature_descriptions (dict): 'feature name': description pairs
        feature_mapping (dict): 'feature name': mapping pairs
        categorical_suffix (str, optional): categorical suffix to identify categorical columns for coloring, defaults
            to "_categorical" during __init__
        Rest of the attributes are inherited from MainGrid.
    """
    # Bokeh Name
    _scatterplot_grid_dropdown = "scatter_plot_grid_dropdown"

    # HTML elements
    _row_description_html = "{hue}"
    _legend_no_hue_html = "No color - too many categories!"
    _legend_template_html = """
        <div class='legend-row'><div style='background-color: {color}' class='legend-marker'>
        </div><div class='legend-description'>{category}</div></div>
    """

    # JS Callbacks
    _scatterplot_callback_js = """
                // new dropdown value
                var new_val = cb_obj.value;  
                
                // new x 
                var new_x = new_val;
                
                document.querySelector("." + "{chosen_feature_scatter}").innerText = new_val;
                
                // scatter source updated
                for (i=0; i<scatter_sources.length; i++) {{
                    scatter_sources[i].data["x"] = scatter_sources[i].data[new_x];
                    scatter_sources[i].change.emit();
                }};
                
                // removing previous greying out
                var all_scatter_rows = document.getElementsByClassName("{scatter_plot_row}");
                for (j=0; j<all_scatter_rows.length; j++) {{
                    all_scatter_rows[j].classList.remove("{active_feature_hue}");
                }};
                
                // greying out
                var scatter_row = document.getElementsByClassName("{scatter_plot_row}" + "-" + new_val);
                scatter_row[0].classList.add("{active_feature_hue}");
            """

    # hardcoded limit
    _max_categories_internal_limit = 10

    # Color Scheme
    _categorical_palette = Category10

    # changing palette for 1 and 2 elements to be of similar color to the palette for 3 elements
    _categorical_palette[2] = Category10[3][:2]
    _categorical_palette[1] = Category10[3][:1]

    # Scatter source
    _scatter_x_axis = "x"
    _scatter_y_axis = "y"

    # CSS element
    _legend = "legend"
    _legend_categorical = "legend-categorical"
    _chosen_feature_scatter_title = "chosen-feature-scatter"
    _active_feature_hue = "active-feature-hue"
    _scatterplot_row = "scatter-plot-row"
    _hue_title = "hue-title"
    _row_description = "row-description"

    def __init__(self,
                 features,
                 plot_design,
                 feature_description_class,
                 categorical_features,
                 feature_descriptions,
                 feature_mapping,
                 categorical_suffix="_categorical"
                 ):
        """Create ScatterPlotGrid object.

        Call MainGrid __init__ with necessary arguments.

        Args:
            features (list): list of features names
            plot_design (PlotDesign): PlotDesign object with predefined style elements
            feature_description_class (str): HTML (CSS) class shared between different HTML elements
            categorical_features (list): list of categorical columns
            feature_descriptions (dict): 'feature name': description pairs
            feature_mapping (dict): 'feature name': mapping pairs
            categorical_suffix (str, optional): categorical suffix to identify categorical columns for coloring,
                defaults to "_categorical"
        """
        self.categorical_columns = categorical_features
        self.feature_descriptions = feature_descriptions
        self.feature_mapping = feature_mapping
        self.categorical_suffix = categorical_suffix
        super().__init__(features, plot_design, feature_description_class)

    def scattergrid(self, scatter_data, initial_feature):
        """Create ScatterGrid Visualization.

        ScatterGrid consists of several rows of ScatterPlots, each row colored (hued) by the values of different
        feature. X axis of all Scatter plots represents the chosen feature in the Dropdown - change in the Dropdown
        triggers change in all Scatter Plots. The exact structure is:
            - Dropdown (hidden)
            - Title with the Chosen Feature name
            - Rows of Scatter Plots

        Additionally, first column of every row includes either ColorBar indicating coloring by Numerical feature or
        legend for Categorical feature (if the number of unique values does not exceed the limit).

        Args:
            scatter_data (dict): 'feature name': data values pairs
            initial_feature (str): feature name used to extract initial data

        Returns:
            bokeh.Grid: bokeh Grid Layout of Plots
        """
        # features
        features = self.features  # taken from MainGrid

        # Scatter
        scatter_row_sources, scatter_rows = self._create_scatter_rows(scatter_data, features, initial_feature)

        # Dropdown
        dropdown = self._create_features_dropdown(self._scatterplot_grid_dropdown)
        callbacks = self._create_features_dropdown_callbacks(scatter_row_sources)
        for callback in callbacks:
            dropdown.js_on_change("value", callback)

        # output Grid
        grid = column(
            dropdown,
            Div(text=initial_feature, css_classes=[self._chosen_feature_scatter_title]),
            *scatter_rows,
        )
        return grid

    def _create_features_dropdown_callbacks(self, scatter_source):
        """Create callbacks that will be triggered upon value change in Grid Dropdown. Implementation of method
        defined in MainGrid.

        Args:
            scatter_source (list): ColumnDataSource sources for scatter plots

        Returns:
            list: created callbacks
        """
        callbacks = []
        for call in [
            self._create_scatter_plot_callback(scatter_source),
        ]:
            callbacks.append(call)

        return callbacks

    def _create_scatter_plot_callback(self, sources):
        """Create callback responsible for changing data in Scatter Plots when the feature in the Dropdown changes.

        JS Code updates X axis in all ColumnDataSource sources so that new chosen feature is on the X axis.
        Additionally greys out row with the coloring by the new chosen feature and removes previous greying out.

        Args:
            sources (list): list of ColumnDataSource sources for Scatter Plots

        Return:
            bokeh.CustomJS: dropdown callback
        """
        kwargs = {
            "scatter_sources": sources
        }

        code = self._scatterplot_callback_js.format(
            chosen_feature_scatter=self._chosen_feature_scatter_title,
            scatter_plot_row=self._scatterplot_row,
            active_feature_hue=self._active_feature_hue
        )

        callback = CustomJS(args=kwargs, code=code)
        return callback

    def _create_scatter_rows(self, scatter_data, features, initial_feature):
        """Create all rows of Scatter Plots and all ColumnDataSource sources.

        Additionally add greying out to the row corresponding to the initial feature.

        Args:
            scatter_data (dict): 'feature name': data values pairs
            features (list): list of feature names
            initial_feature (str): feature name used to extract initial data

        Returns:
            tuple: (list of ColumnDataSource sources, list of all created Row Plots)
        """
        all_sources = []
        all_rows = []
        for feature in features:
            sources, single_row = self._create_single_scatter_row(scatter_data, features, initial_feature, feature)

            if feature == initial_feature:
                single_row.css_classes.append(self._active_feature_hue)

            all_sources.extend(sources)
            all_rows.append(single_row)

        return all_sources, all_rows

    def _create_single_scatter_row(self, scatter_data, features, initial_feature, hue):
        """Create single row of Scatter Plots and corresponding ColumnDataSource sources.

        First element of the Scatter Row is provided as a Color Legend depending on the feature that is used as a hue -
        either ColorBar for Numerical Range or Unique Values colors in Categorical features (if their number does not
        exceed the limit. Additionally the title of the first element is a hoverable HTML element showing description
        for a given feature.

        Args:
            scatter_data (dict): 'feature name': data values pairs
            features (list): list of feature names
            initial_feature (str): feature name used to extract initial data
            hue (str): feature name that will be used as coloring in created row

        Returns:
            tuple: (ColumnDataSource sources for Plots in that row, bokeh.Row of Scatter Plots)
        """
        sources = []
        plots = []

        color_map = self._create_color_map(hue, scatter_data)
        for feature in features:
            src = self._create_scatter_source(scatter_data, initial_feature, feature)
            plot = self._create_scatter_plot(src, initial_feature, feature, color_map)
            sources.append(src)
            plots.append(plot)

        color_legend = self._create_row_description(hue, color_map)
        r = row(
            color_legend,
            *plots,
            css_classes=[self._scatterplot_row, self._scatterplot_row + "-" + hue],
            margin=(0, 48, 0, 0)
        )

        return sources, r

    def _create_scatter_source(self, scatter_data, x, y):
        """Create single ColumnDataSource to be used in one of the Scatter Plots.

        Full scatter_data is included in ColumnDataSource to enable dynamic changes with JS in HTML output.

        Args:
            scatter_data (dict): 'feature name': data values pairs
            x (str): feature name chosen to be on X axis
            y (str): feature name chosen to be on Y axis

        Returns:
            bokeh.ColumnDataSource: ColumnDataSource with ScatterPlot data

        """
        source = ColumnDataSource(scatter_data)
        # additional 2 columns for x and y in plots
        source.data.update(
            {
                self._scatter_x_axis: source.data[x],
                self._scatter_y_axis: source.data[y]
            }
        )
        return source

    @stylize
    def _create_scatter_plot(self, source, x, y, cmap):
        """Create single Scatter Plot with provided data and coloring.

        Note:
            x argument (X feature name) is not used, but is left in case it is needed in the future.

        Args:
            source (bokeh.ColumnDataSource): ColumnDataSource for a single Scatter Plot
            x (str): feature name chosen to be on X axis
            y (str): feature name chosen to be on Y axis
            cmap (bokeh.ColorMapper): color mapper to be used to color points in Scatter Plot

        Returns:
            bokeh.Figure: bokeh Scatter Plot Figure
        """
        # figure
        p = default_figure()

        # Scatter Plot
        kwargs = {
            "x": self._scatter_x_axis,
            "y": self._scatter_y_axis,
            "source": source,
            "size": 10,
            "fill_color": self.plot_design.fill_color,
        }

        if cmap:  # overriding plain fill color if cmap is not None
            kwargs.update(
                {
                    "fill_color": cmap,
                }
            )
        p.scatter(**kwargs)

        # plot specific styling
        p.plot_width = 200
        p.plot_height = 200

        p.yaxis.axis_label = y

        p.xaxis.ticker = BasicTicker(desired_num_ticks=4)
        p.yaxis.ticker = BasicTicker(desired_num_ticks=4)
        p.xaxis.formatter = NumeralTickFormatter(format="0.[0]")
        p.yaxis.formatter = NumeralTickFormatter(format="0.[0]")

        # turned off, as it apparently clogs up plots and JS interactions
        # p.xaxis.major_label_orientation = -0.75  # in radians

        return p

    def _create_color_map(self, hue, data):
        """Create Color Mapper depending on the type of the feature.

        If feature to color by is Categorical then unique values in the data are treated as separate colors - unless
        their number exceeds limit of max categories, in which case None is returned. If the feature is Numerical, then
        Linear Color Mapper is created.

        Note:
            Categorical data for coloring is taken from columns ending in categorical_suffix attribute, as values for
            coloring must be str, but values for plotting on X or Y axes must be numeric.

        Args:
            hue (str): feature name that will be used as coloring in created row
            data (dict): 'feature name': data values pairs

        Returns:
            bokeh.ColorMapper, None: LinearColorMapper for Numerical feature as a hue, CategoricalColorMapper for
                Categorical feature as a hue if not exceeding the limit of unique values, None if limit is exceeded
        """
        if hue in self.categorical_columns:
            # adding suffix to column name to get unique str values
            factors = sorted(set(data[hue + self.categorical_suffix]))
            if len(factors) <= self._max_categories_internal_limit:
                cmap = factor_cmap(
                    hue + self.categorical_suffix,
                    palette=self._categorical_palette[len(factors)],
                    factors=factors
                )
            else:
                # If there is too many categories, None is returned
                cmap = None
        else:
            values = data[hue]
            cmap = linear_cmap(
                hue,
                palette=self.plot_design.contrary_color_linear_palette,
                low=min(values),
                high=max(values)
            )

        return cmap

    def _create_row_description(self, hue, cmap):
        """Create Row description - first element of a given Scatter Plots row.

        First element follows a fixed structure of:
            - Name of a Feature used as a hue in a given row (Title)
            - Color Legend

        Title element is hoverable in HTML and on hover will show description of a Hue Feature. Color Legend will show
        coloring used in that Scatter Plot row or placeholder Div text if ColorMapper was not used.

        Args:
            hue (str): feature name that will be used as coloring in created row
            cmap (bokeh.ColorMapper, None): ColorMapper used to color values in Scatter Plots or None

        Returns:
            bokeh.Column: Column Layout with Feature Name title and Color Legend
        """
        # HTML needs to be prepared so that description is hidden/hoverable
        desc = self.feature_descriptions[hue]

        parsed_html = BeautifulSoup(self._row_description_html.format(hue=hue), "html.parser")
        parsed_html.string.wrap(parsed_html.new_tag("p"))
        parsed_html.p.append(append_description(desc, parsed_html))
        parsed_html.p["class"] = self.feature_description_class

        feature_with_description = str(parsed_html)

        # Color Legend
        legend = self._create_legend(hue, cmap)

        # Feature Title Div
        kwargs = {
            "text": feature_with_description,
            "css_classes": [self._hue_title]
        }
        d = Div(**kwargs)

        # output Column
        c = column(
            d,
            legend,
            width=200,
            height=195,
            width_policy="fixed",
            css_classes=[self._row_description]
        )
        return c

    def _create_legend(self, hue, cmap):
        """Create Legend HTML element depending on coloring used in a given Scatter Plots row.

        If feature for coloring is Categorical, then custom legend is created with color - category pairs of HTML
        elements. If feature is Numerical then bokeh ColorBar is appended to the Div (by doing a trick of creating
        an empty bokeh Figure and appending ColorBar to it).

        If provided cmap is None, then placeholder Div text is included instead.

        Args:
            hue (str): feature name that will be used as coloring in created row
            cmap (bokeh.ColorMapper, None): ColorMapper used to color values in Scatter Plots or None

        Returns:
            bokeh.Div or bokeh.Figure: Div in case of Categorical feature or cmap being None, Figure in case of
                numerical hue
        """
        if cmap:
            if hue in self.categorical_columns:
                mapping = self.feature_mapping[hue]
                categories = cmap["transform"].factors
                colors = cmap["transform"].palette
                text = ""
                template = self._legend_template_html

                for category, color in zip(categories, colors):
                    mapped_category = mapping[float(category)]  # float as keys in mapping dicts are numerical
                    text += template.format(
                        color=color,
                        category=mapped_category
                    )
                legend = Div(text=text, css_classes=[self._legend, self._legend_categorical])

            else:

                colorbar = ColorBar(color_mapper=cmap["transform"],
                                    ticker=BasicTicker(desired_num_ticks=4),
                                    formatter=NumeralTickFormatter(format="0.[0000]"),
                                    label_standoff=7,
                                    border_line_color=None,
                                    bar_line_color=self.plot_design.text_color,
                                    major_label_text_font_size="14px",
                                    major_label_text_color=self.plot_design.text_color,
                                    major_tick_line_color=self.plot_design.text_color,
                                    major_tick_in=0,
                                    location=(-100, 0),  # by default ColorBar is placed to the side of the Figure
                                    width=30
                                    )
                legend = default_figure(
                    {
                        "height": 120,
                        "width": 120,
                        "css_classes": [self._legend]
                    }
                )
                legend.add_layout(colorbar, "right")

        else:
            legend = Div(
                text=self._legend_no_hue_html,
                css_classes=[self._legend]
            )

        return legend


class ModelsPlotClassification:
    """Models Performance Plots in Classification Problems.

    ModelsPlotClassification creates Tabs Plot with 3 different Model performance Assessments in each Panel:
        - ROC Curve
        - Precision Recall Curve
        - Detection Error Tradeoff Curve

    All Plots define some sort of a baseline to compare the results. All Plots share the same styling attributes
    as well.

    Attributes:
        plot_design (PlotDesign): PlotDesign object with predefined style elements
    """
    _roc_plot_name = "ROC Curve"
    _precision_recall_plot_name = "Precision Recall Plot"
    _det_plot_name = "Detection Error Tradeoff"

    def __init__(self, plot_design):
        """Create ModelsPlotClassification object.

        Args:
            plot_design (PlotDesign): PlotDesign object with predefined style elements
        """
        self.plot_design = plot_design

    def models_comparison_plot(self, roc_curves, precision_recall_curves, det_curves, target_proportion):
        """Create Models Comparison Plots to compare their performances in different aspects.

        3 Panels included in the final Tabs Plot are:
            - ROC: https://en.wikipedia.org/wiki/Receiver_operating_characteristic
            - Precision Recall: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
            - Detection Error Tradeoff: https://scikit-learn.org/stable/auto_examples/model_selection/plot_det.html?highlight=detection%20error%20tradeoff

        Args:
            roc_curves (list): tuples of (Model, roc results)
            precision_recall_curves (list): tuples of (Model, precision-recall results)
            det_curves (list): tuples of (Model, det results)
            target_proportion (float): proportion of positive label in the data used to calculate curves results

        Returns:
            bokeh.Tabs: Tabs Figure with 3 Panels of ROC, Precision-Recall and DET curves
        """
        new_tps = [assess_models_names(tp) for tp in [roc_curves, precision_recall_curves, det_curves]]
        roc_curves, precision_recall_curves, det_curves = new_tps

        # ROC
        roc_plot = Panel(
            child=self._roc_plot(roc_curves),
            title=self._roc_plot_name
        )
        # Precision-Recall
        precision_recall_plot = Panel(
            child=self._precision_recall_plot(precision_recall_curves, target_proportion),
            title=self._precision_recall_plot_name
        )
        # DET
        det_plot = Panel(
            child=self._det_plot(det_curves),
            title=self._det_plot_name
        )

        # Final Plot
        main_plot = Tabs(tabs=[roc_plot, precision_recall_plot, det_plot])

        return main_plot

    @stylize
    def _roc_plot(self, roc_curves):
        """Create Plot with ROC Curves calculated for different Models.

        Added Legend is interactive and can be used to turn off (mute) given Models results (plotted Line).

        Args:
            roc_curves (list): tuples of (Model, roc curves)

        Returns:
            bokeh.Figure: bokeh Plot Figure
        """
        # figure
        p = default_figure(
            {
                "x_range": (-0.01, 1.1),
                "y_range": (-0.01, 1.1),
                "tools": "pan,wheel_zoom,box_zoom,reset",
                "toolbar_location": "right"
            }
        )

        # main lines added to the plot
        self._default_models_lines(p, roc_curves)

        # baseline comparison
        p.line(
            [0, 1],  # line x=y
            [0, 1],
            line_dash="dashed",
            line_width=1,
            color=self.plot_design.models_dummy_color,
            legend_label="Random Baseline",
            muted_alpha=0.5  # clicked line in the Legend will be muted
        )

        # plot specific styling
        p.legend.location = "bottom_right"
        p.xaxis.axis_label = "False Positive Rate"
        p.yaxis.axis_label = "True Positive Rate"

        return p

    @stylize
    def _precision_recall_plot(self, precision_recall_curves, target_proportion):
        """Create Plot with Precision Recall Curves calculated for different Models.

        Added Legend is interactive and can be used to turn off (mute) given Models results (plotted Line).

        Note:
            Target Proportion is used to create a parallel line to X axis (y = target_proportion).

        Args:
            precision_recall_curves (list): tuples of (Model, precision-recall curves)
            target_proportion (float): proportion of positive label in the data used to calculate curves results

        Returns:
            bokeh.Figure: bokeh Plot Figure
        """
        # figure
        p = default_figure(
            {
                "x_range": (-0.01, 1.1),
                "y_range": (-0.01, 1.1),
                "tools": "pan,wheel_zoom,box_zoom,reset",
                "toolbar_location": "right"
            }
        )
        # changing the order of default precision-recall calculations
        # new order is: recall (X-axis), precision (Y-axis), thresholds
        curves = [(model, (values[1], values[0], values[2])) for model, values in precision_recall_curves]

        # main lines added to the plot
        self._default_models_lines(p, curves)

        # baseline comparison
        p.line(
            [0, 1],
            [target_proportion, target_proportion],  # y = target_proportion
            line_dash="dashed",
            line_width=1,
            color=self.plot_design.models_dummy_color,
            legend_label="Random Baseline",
            muted_alpha=0.5  # clicked line in the Legend will be muted
        )

        # plot specific styling
        p.legend.location = "bottom_left"
        p.xaxis.axis_label = "Recall"
        p.yaxis.axis_label = "Precision"

        return p

    @stylize
    def _det_plot(self, det_curves):
        """Create Plot with DET Curves calculated for different Models.

        Added Legend is interactive and can be used to turn off (mute) given Models results (plotted Line).

        Note:
            DET curves represent straight lines in normal deviate scale - Axes do not follow regular linear scale
            to accommodate for that.

        Args:
            det_curves (list): tuples of (Model, det curves)

        Returns:
            bokeh.Figure: bokeh Plot Figure
        """
        # figure
        p = default_figure({
            "x_range": (-3, 3),
            "y_range": (-3, 3),
            "tools": "pan,wheel_zoom,box_zoom,reset",
            "toolbar_location": "right"
        })

        # transforming calculated results to normal deviate scale
        # https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/metrics/_plot/det_curve.py#L100
        new_curves = []
        for model, curve in det_curves:
            f = scipy.stats.norm.ppf
            new_tuple = (f(curve[0]), f(curve[1]))
            new_curves.append((model, new_tuple))

        # main lines added to the plot
        self._default_models_lines(p, new_curves)

        # Custom ticks on Axes
        ticks = [0.001, 0.01, 0.05, 0.20, 0.4999, 0.80, 0.95, 0.99, 0.999]
        tick_location = scipy.stats.norm.ppf(ticks)
        mapper = {norm_tick: tick for norm_tick, tick in zip(tick_location, ticks)}
        # 0.4999 was included instead of 0.5 as after normal transformation, 0.5 becomes 0.0.
        # FuncTickFormatter would then try to access dictionary of ticks with a key of 0, which JS evaluates
        # to undefined and raises an error.

        p.xaxis.ticker = tick_location
        p.yaxis.ticker = tick_location

        formatter = FuncTickFormatter(args={"mapper": mapper}, code="""
            return (mapper[tick] * 100).toString() + "%";
        """)
        p.xaxis.formatter = formatter
        p.yaxis.formatter = formatter

        # plot specific styling
        p.legend.location = "top_right"
        p.xaxis.axis_label = "False Positive Rate"
        p.yaxis.axis_label = "False Negative Rate"

        return p

    def _default_models_lines(self, plot, model_values_tuple):
        """Add similarly stylized Lines to the plot on values provided in model_values_tuple.

        model_values_tuple is pre-sorted so that the first entry is the best scoring Model. In order to plot that
        Model at the top, it needs to be drawn as last - therefore the order is reversed. The best Model (new last) is
        plotted with a different color to make it more outstanding in the final visualization.

        Note:
            Legend is defined here as 'mutable' - records can be interactively clicked and muted.

        Args:
            plot (bokeh.Figure): bokeh Plot Figure
            model_values_tuple (list): list of tuples (Model, curve results (X, Y))
        """
        new_tuples = list(reversed(model_values_tuple))
        lw = 5

        # models are plotted in reverse order (without the first one)
        for model, values in new_tuples[:-1]:
            plot.step(
                values[0],
                values[1],
                line_width=lw - 2,
                legend_label=model,
                line_color=self.plot_design.models_color_tuple[1],
                muted_alpha=0.2  # clicked line in the Legend will be muted
            )

        # the best model is plotted as last to be on top of other lines
        first_model, first_values = new_tuples[-1]
        plot.step(
            first_values[0],
            first_values[1],
            line_width=lw,
            legend_label=first_model,
            line_color=self.plot_design.models_color_tuple[0],
            muted_alpha=0.2  # clicked line in the Legend will be muted
        )

        plot.legend.click_policy = "mute"
        plot.toolbar.autohide = True


class ModelsPlotRegression:
    """Models Performance Plots in Regression Problems.

        ModelsPlotRegression can create 2 different Plots:
            - Prediction Error Tabs, where every assessed Model has it's own Panel
            - Residual Plot Tabs, where every assessed Model has it's own Panel

        Attributes:
            plot_design (PlotDesign): PlotDesign object with predefined style elements
        """
    _formatter_code = """return String(tick);"""

    def __init__(self, plot_design):
        """Create ModelsPlotRegression object.

        Args:
            plot_design (PlotDesign): PlotDesign object with predefined style elements
        """
        self.plot_design = plot_design

    def prediction_error_plot(self, prediction_errors):
        """Create Prediction Error Tabs for all Models in prediction_errors results.

        As prediction_errors list is sorted in descending order, the first Model is the one that achieved the best
        results. Therefore, it gets another color than the rest to be more outstanding in the visualization.

        Args:
            prediction_errors (list): list of tuples (Model, (actual results, predicted results))

        Returns:
            bokeh.Tabs: bokeh Tabs Figure
        """
        prediction_errors = assess_models_names(prediction_errors)
        _ = []
        i = 0
        for model, scatter_points in prediction_errors:
            if i == 0:
                color = self.plot_design.models_color_tuple[0]
                i += 1
            else:
                color = self.plot_design.models_color_tuple[1]
            plot = self._single_prediction_error_plot(scatter_points, color)
            _.append(Panel(child=plot, title=model))

        main_plot = Tabs(tabs=_)
        return main_plot

    def residual_plot(self, residual_tuples):
        """Create Residual Plots Tabs for all Models in residual_tuples results.

        As residual tuples list is sorted in descending order, the first Model is the one that achieved the best
        results. Therefore, it gets another color than the rest to be more outstanding in the visualization.

        Args:
            residual_tuples (list): list of tuples (Model, (predictions, difference between predicted and actual))

        Returns:
            bokeh.Tabs: bokeh Tabs Figure
        """
        residual_tuples = assess_models_names(residual_tuples)
        _ = []
        i = 0
        for model, scatter_points in residual_tuples:
            if i == 0:
                color = self.plot_design.models_color_tuple[0]
                i += 1
            else:
                color = self.plot_design.models_color_tuple[1]
            plot = self._single_residual_plot(scatter_points, color)
            _.append(Panel(child=plot, title=model))

        main_plot = Tabs(tabs=_)
        return main_plot

    @stylize
    def _single_prediction_error_plot(self, scatter_data, color):
        """Create a single Prediction Errors Plot.

        Actual y is plotted on X axis, whereas predicted y is plotted on Y axis. If predictions were 100% correct,
        points would make a straight line (x=y) - as it's usually not the case, the bigger the difference from that
        line the bigger the error.

        Args:
            scatter_data (tuple): tuple of (actual y, predicted y)
            color (str): color of the scatter points

        Returns:
            bokeh.Figure: bokeh Scatter Plot Figure
        """
        # figure
        p = default_figure(
            {
                "tools": "pan,wheel_zoom,box_zoom,reset",
                "toolbar_location": "right"
            }
        )

        # scatter
        p.scatter(scatter_data[0], scatter_data[1], color=color, size=16, fill_alpha=0.8)

        # baseline of x=y
        slope = Slope(
            gradient=1,
            y_intercept=0,
            line_width=1,
            line_color=self.plot_design.models_dummy_color,
            line_dash="dashed"
        )
        p.add_layout(slope)

        # plot specific styling
        p.xaxis.axis_label = "Actual"
        p.yaxis.axis_label = "Predicted"

        formatter = FuncTickFormatter(code=self._formatter_code)  # negative numbers are having a wacky formatting
        # formatters must be created independently, cannot be reused between plots
        p.xaxis.formatter = formatter
        p.yaxis.formatter = formatter

        p.toolbar.autohide = True

        return p

    @stylize
    def _single_residual_plot(self, scatter_data, color):
        """Create a single Residual Plot.

        Predictions are plotted on X axis, whereas difference between predicted and actual y is plotted on Y axis.
        Baseline goes through y = 0 - Residuals should be located seemingly at random, on both sides of the baseline.
        If there are any noticeable patterns then it means that the Model is learning incorrectly.

        Args:
            scatter_data (tuple): tuple of (predicted y, difference between predicted y and actual y)
            color (str): color of the scatter points

        Returns:
            bokeh.Figure: bokeh Scatter Plot Figure
        """
        # figure
        p = default_figure(
            {
                "tools": "pan,wheel_zoom,box_zoom,reset",
                "toolbar_location": "right",
                "width": 900,
                "height": 300
            }
        )

        # Residuals
        p.scatter(scatter_data[0], scatter_data[1], color=color, size=10, fill_alpha=0.8)

        # Baseline
        baseline = Span(location=0, dimension="width", line_color=self.plot_design.models_dummy_color, line_width=2)
        p.add_layout(baseline)

        # plot specific styling
        p.xaxis.axis_label = "Predicted"
        p.yaxis.axis_label = "Residual"

        formatter = FuncTickFormatter(code=self._formatter_code)  # negative numbers are having a wacky formatting
        # formatters must be created independently, cannot be reused
        p.xaxis.formatter = formatter
        p.yaxis.formatter = formatter

        p.toolbar.autohide = True

        return p


class ModelsPlotMulticlass:
    """Models Confusion Matrices in Multiclass Classification Problems.

    ModelsPlotMulticlass creates Confusion Matrices with labels present in target variable. Confusion Matrices
    are represented as HeatMaps, with higher values being colored more intensively.

    As confusion matrices come as arrays with the same ordering as labels (i-th row corresponds to i-th class), mapping
    between indices and values is created. If the original mapping (from features_descriptions_dict) is present then
    it is used. Otherwise, simple conversion of classes to their str counterparts is done.

    Attributes:
        plot_design (PlotDesign): PlotDesign object with predefined style elements
        labels (list): list of string labels present in y variable.
        label_mapping (dict): i-th index number: str mapping pairs between confusion matrix and corresponding values
    """
    _x = "x"
    _y = "y"
    _values = "values"

    def __init__(self, plot_design, label_classes, original_label_mapping):
        """Create ModelsPlotMulticlass object.

        labels and label_mapping are assessed and created to allow changing indices of confusion matrices to
        corresponding values.

        Args:
            plot_design (PlotDesign): PlotDesign object with predefined style elements
            label_classes (list, numpy.ndarray): sequence of label classes present in target variable
            original_label_mapping (dict, None): label: corresponding label value pairs of target variable, can be None
        """
        self.plot_design = plot_design
        self.labels, self.label_mapping = self._create_labels_and_mapping(label_classes, original_label_mapping)

        print(self.labels)
        print(self.label_mapping)

    def confusion_matrices_plot(self, confusion_matrices):
        """Create bokeh Row of Confusion Matrices (HeatMaps).

        Confusion Matrix for every Model is plotted in the row alongside others in the same order as they are passed.
        As the first Model is also the best one in terms of performance, it gets another color for better visibility
        in the visualization.

        Plots have also additional space added between them with Spacer element.

        Args:
            confusion_matrices (list): tuples of (Model, confusion_matrix)

        Returns:
            bokeh.Row: bokeh Row of Plots
        """
        confusion_matrices = assess_models_names(confusion_matrices)
        _ = []
        i = 0
        for model, array in confusion_matrices:
            if i == 0:
                palette = self.plot_design.contrary_half_color_tints
                i += 1
            else:
                palette = self.plot_design.base_half_color_tints
            plot = self._single_confusion_matrix_plot(array, palette, model)
            _.append(plot)
            _.append(Spacer(width=100))

        main_plot = row(*_)
        return main_plot

    @stylize
    def _single_confusion_matrix_plot(self, confusion_array, palette, model_name):
        """Create single Confusion Matrix Plot (HeatMap).

        Values are included in the ColumnDataSource, from where Plot and Coloring use them. Rectangles in the Plot
        are colored based on the values of predictions - the higher the value, the more intense the color. Numbers
        are additionally plotted at the center of the rectangles.

        Args:
            confusion_array (numpy.ndarray): confusion matrix numpy.ndarray
            palette (list): list of colors to use as a Mapper
            model_name (str): name of the Model

        Returns:
            bokeh.Figure: bokeh Plot Figure
        """
        # source and cmap
        source = self._create_column_data_source(confusion_array)
        cmap = LinearColorMapper(palette=palette[::-1], low=0, high=max(confusion_array.ravel()))

        # figure
        p = default_figure(
            {
                "title": model_name,
                "height": 300,
                "width": 300,
                "x_range": self.labels,
                "y_range": self.labels[::-1]
            }
        )

        # Rectangles (HeatMap)
        p.rect(
            x=self._x,
            y=self._y,
            source=source,
            fill_color={"field": self._values, "transform": cmap},
            width=1,
            height=1,
            line_color=None,
        )

        # Numbers on Rectangles
        labels = LabelSet(
            x=self._x,
            y=self._y,
            text=self._values,
            source=source,
            render_mode="canvas",
            x_offset=-7,  # trying to center the numbers manually
            y_offset=-7,
            text_color="black",
            text_font_size="11px",
        )
        p.add_layout(labels)

        # plot specific styling
        p.yaxis.axis_label = "Actual"
        p.xaxis.axis_label = "Predicted"
        p.xaxis.major_label_orientation = -1.57  # in radians

        return p

    def _create_column_data_source(self, confusion_array):
        """Create ColumnDataSource from confusion matrix.

        Confusion Matrix is converted to pandas.DataFrame, from which different levels of index are taken for easy
        mapping between Axes ranges and values. Every index is mapped with label_mapping attribute to it's corresponding
        string value.

        Args:
            confusion_array (numpy.ndarray): confusion matrix numpy.ndarray

        Returns:
            bokeh.ColumnDataSource: ColumnDataSource
        """
        cds = ColumnDataSource()
        df = pd.DataFrame(confusion_array).stack()
        old_x = df.index.droplevel(0).to_list()  # one of the indexes  astype(str).
        x = [self.label_mapping[ind] for ind in old_x]
        old_y = df.index.droplevel(1).to_list()  # second of the indexes  astype(str).
        y = [self.label_mapping[ind] for ind in old_y]
        values = df.to_list()

        cds.data = {
            self._x: x,
            self._y: y,
            self._values: values
        }

        return cds

    def _create_labels_and_mapping(self, labels, mapping):
        """Create string counterparts of labels and appropriate label mapping to be used in Figure axes.

        Keys of mapping are simple enumerations of labels list (corresponding to indices in confusion matrix). Values
        of mapping are either labels converted to string (if mapping is None) or their corresponding values (converted
        to strings as well) from mapping.

        Labels are values from that newly created mapping in order.

        Args:
            labels (numpy.ndarray): array of target classes (labels)
            mapping (dict, None): original mapping of labels (external)

        Returns:
            tuple: (list of string labels, dict of mapping between indices and string labels)
        """
        numbered_classes = list(enumerate(list(labels), start=0))
        if mapping:
            new_mapping = {number: str(mapping[label]) for number, label in numbered_classes}
        else:
            new_mapping = {number: str(label) for number, label in numbered_classes}
        new_labels = [new_mapping[numbered[0]] for numbered in numbered_classes]

        return new_labels, new_mapping


class ModelsDataTable:
    """ModelsDataTable plotting 'nicely' looking and interactive Table.

    Predictions Table is constructed with different elements, concatted together:
        - actual y results
        - predictions from all Models
        - 'raw' (original) DataFrame

    Attributes:
        plot_design (PlotDesign): PlotDesign object with predefined style elements
    """
    _model_column_template = '<div style="color:{color};font-weight:bold;font-size:1.15em"><%= value %></div>'

    def __init__(self, plot_design):
        """Create ModelsDataTable object.

        Args:
            plot_design (PlotDesign): PlotDesign object with predefined style elements
        """
        self.plot_design = plot_design

    def data_table(self, X, y, models_predictions):
        """Create bokeh DataTable with X, y and predictions data.

        bokeh DataTable is an interactive Table that can be sorted, columns can be dragged around and it has a nice
        modern look. In the Table the first column of the Table is y - actual results. Predictions from Models follow
        and at the end, the 'raw' X data is added.

        Note:
            y and predictions columns are colored differently for better visibility.

        Args:
            X (pandas.DataFrame): X features space used to predict y
            y (pandas.Series, numpy.ndarray): actual target variable
            models_predictions (list): tuples of (Model, predictions)

        Returns:
            bokeh.DataTable: DataTable with predictions from different Models
        """
        models_predictions = assess_models_names(models_predictions)
        base_color = self.plot_design.base_color_tints[0]

        # formatter for y and prediction columns to color and style them separately
        cols = [TableColumn(
            field=y.name,
            title=y.name,
            formatter=HTMLTemplateFormatter(template=self._model_column_template.format(color=base_color))
        )]

        # predictions
        _ = []
        i = 0
        for model, predictions in models_predictions:
            if i == 0:
                color = self.plot_design.models_color_tuple[0]
                i += 1
            else:
                color = self.plot_design.models_color_tuple[1]

            predictions = pd.Series(predictions, name=model).round(6)
            _.append(predictions)
            cols.append(
                TableColumn(
                    field=model,
                    title=model,
                    formatter=HTMLTemplateFormatter(template=self._model_column_template.format(color=color)))
            )

        for col in X.columns:
            cols.append(TableColumn(field=col, title=col))
        scores = pd.DataFrame(_).T  # by default, wide table is created instead of a long one

        # final DataFrame and DataTable
        df = pd.concat([y, scores, X], axis=1)
        source = ColumnDataSource(df)
        dt = DataTable(source=source, columns=cols, editable=False, sizing_mode="stretch_width")

        return dt
