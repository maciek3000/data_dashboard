from .views import append_description
from bokeh.plotting import figure
from bokeh.layouts import column, row, Spacer
from bokeh.models import ColumnDataSource, FuncTickFormatter
from bokeh.models.widgets import Select, Div, HTMLTemplateFormatter
from bokeh.models import CustomJS, ColorBar, BasicTicker, PrintfTickFormatter, LinearColorMapper, Panel, Tabs, \
    HoverTool, LabelSet
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.palettes import Reds4, Category10
from bokeh.models.widgets.tables import DataTable, TableColumn
import functools
import seaborn as sns
import scipy.stats
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

from bokeh.core.validation.check import silence

from .model_finder import obj_name
from .views import assess_models_names

contrary_color_palette = ["#FFF7F3", "#FFB695", "#EB6F54", "#9C2B19"]


def stylize(force=False):
    """Decorator for functions that return Plot figures so that all of them would be stylized similarly"""

    def decorator_stylize_attributes(plot_func):
        @functools.wraps(plot_func)
        def wrapper(self, *args, **kwargs):
            p = plot_func(self, *args, **kwargs)

            if force:
                return p

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

    return decorator_stylize_attributes


def default_figure(plot_specific_kwargs=None):
    # supplying stylizing attributes to Figure() doesn't always work, as apparently some methods (e.g. grids)
    # require axes from a created figure to change something

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

        When it comes to visualizing features, pairplot consists of many scatter plots, plotting every feature
        against one another and one histogram for every feature, presenting its distribution.

        This is plotted with seaborn, as static visualization was deemed sufficient for this and also because
        recreating it manually in bokeh would mean reinventing the wheel for no reason.
    """

    def __init__(self, plot_design):
        self.plot_design = plot_design

        text_color = self.plot_design.text_color
        sns.set_style("white", {
            "axes.edgecolor": text_color,
            "axes.labelcolor": text_color,
            "text.color": text_color,
            "font.sans-serif": ["Lato"],
            "xtick.color": text_color,
            "ytick.color": text_color,
        })

    def pairplot(self, dataframe):
        colors = {"color": self.plot_design.pairplot_color}
        p = sns.pairplot(dataframe, plot_kws=colors, diag_kws=colors)
        return p


class MainGrid:
    """Parent class for Grid Plots.

        Can create dropdown with feature names for Grid plots, to which JS callback can be associated.
        This dropdown serves as a main tool for interactivity in bokeh Grid plots. It's usually hidden from
        an end-user, but it can still be accessed (and clicked) by JS scripts in the background.
    """

    _features_dropdown = "features_dropdown"

    def __init__(self, features, plot_design, feature_description_class):
        self.features = features
        self.plot_design = plot_design
        self.feature_description_class = feature_description_class

    def _create_features_dropdown(self, name=None):
        if name is None:
            name = self._features_dropdown
        fts = sorted(self.features)
        d = Select(options=fts, css_classes=[self._features_dropdown], name=name)
        return d

    def _create_features_dropdown_callbacks(self, **kwargs):
        raise NotImplementedError


class InfoGrid(MainGrid):
    """Creates a bokeh Grid plot with visualizations/summary statistics/etc. for a single feature.

        As of now it consists of a Histogram and a Div which includes some basic info on the specific feature.

        Dropdown from a parent MainGrid class is connected via JS to dynamically change histogram and div when
        another feature is chosen.
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

    _correlation_tooltip_text = """<div>
    <div>Normalized data correlation: @{normalized}</div>
    <div>Raw data correlation: @{raw}</div>
    </div>
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
    _correlation = "correlation-plot"

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
                
                /* var edges = new_hist[1];
                var left_edges = edges.slice(0, edges.length - 1);
                var right_edges = edges.slice(1, edges.length); */
                
                // histogram source updated
                hist_source.data["hist"] = hist;
                hist_source.data["left_edges"] = left_edges;
                hist_source.data["right_edges"] = right_edges;
                
                // updating ColumnDataSources
                hist_source.change.emit();
            """

    # plot elements
    _histogram_title = "Feature Distribution"
    _correlation_x = "x"
    _correlation_y = "correlation_y"
    _correlation_values_normalized = "correlation_values_normalized"
    _correlation_values_normalized_abs = "correlation_values_normalized_abs"
    _correlation_values_normalized_title = "Normalized Data Correlation"
    _correlation_values_raw = "correlation_values_raw"
    _correlation_values_raw_abs = "correlation_values_raw_abs"
    _correlation_values_raw_title = "Raw Data Correlation"

    def __init__(self, features, plot_design, feature_description_class, target_name):
        super().__init__(features, plot_design, feature_description_class)
        self.target_name = target_name

    def summary_grid(self, summary_statistics, histogram_data, initial_feature):
        dropdown = self._create_features_dropdown(self._infogrid_dropdown)

        histogram_source, histogram_plot = self._create_histogram(histogram_data, initial_feature)
        info_div = self._create_info_div(summary_statistics, initial_feature)

        callbacks = self._create_features_dropdown_callbacks(
            summary_statistics=summary_statistics,
            histogram_data=histogram_data,
            histogram_source=histogram_source,
        )
        for callback in callbacks:
            dropdown.js_on_change("value", callback)

        output = column(
            dropdown,  # this dropdown will be invisible (display: none)
            column(
                info_div, histogram_plot, css_classes=[self._infogrid_left_pane]
            ),
            # height_policy="max",
            # height=500,
            css_classes=[self._infogrid_all]
        )

        return output

    def correlation_plot(self, correlation_data_normalized, correlation_data_raw):
        # correlation source is left in case it is decided later on that the callback is needed
        correlation_source, correlation_plot = self._create_correlation(correlation_data_normalized,
                                                                        correlation_data_raw)

        return correlation_plot

    # def infogrid(self, summary_statistics, histogram_data, correlation_data_normalized, correlation_data_raw,
    #              initial_feature):
    #
    #     dropdown = self._create_features_dropdown(self._infogrid_dropdown)
    #
    #     histogram_source, histogram_plot = self._create_histogram(histogram_data, initial_feature)
    #     info_div = self._create_info_div(summary_statistics, initial_feature)
    #
    #     # correlation source is left in case it is decided later on that the callback is needed
    #     correlation_source, correlation_plot = self._create_correlation(correlation_data_normalized,
    #                                                                     correlation_data_raw)
    #
    #     callbacks = self._create_features_dropdown_callbacks(
    #         summary_statistics=summary_statistics,
    #         histogram_data=histogram_data,
    #         histogram_source=histogram_source,
    #     )
    #     for callback in callbacks:
    #         dropdown.js_on_change("value", callback)
    #
    #     infogrid_row = row(
    #         column(
    #             info_div, histogram_plot, css_classes=[self._infogrid_left_pane]
    #         ),
    #         correlation_plot,
    #         css_classes=[self._infogrid_row],
    #         height_policy="max",
    #         height=500
    #     )
    #
    #     output = column(
    #         dropdown,  # this dropdown will be invisible (display: none)
    #         infogrid_row,
    #         css_classes=[self._infogrid_all]
    #     )
    #     return output

    def _create_features_dropdown_callbacks(self, summary_statistics, histogram_data, histogram_source):
        callbacks = []

        for call in [
            self._create_histogram_callback(histogram_data, histogram_source),
            self._create_info_div_callback(summary_statistics),
        ]:
            callbacks.append(call)

        return callbacks

    def _create_histogram_callback(self, histogram_data, histogram_source):
        kwargs = {
            "hist_source": histogram_source,
            "hist_data": histogram_data
        }

        callback = CustomJS(args=kwargs, code=self._histogram_callback)
        return callback

    def _create_info_div_callback(self, summary_statistics):
        kwargs = {
            "summary_statistics": summary_statistics
        }
        callback = CustomJS(
            args=kwargs,
            code=self._info_div_callback
        )
        return callback

    def _create_info_div(self, summary_statistics, feature):

        feature_dict = summary_statistics[feature]

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

        # histogram source contains calculated histogram edges for all features in the data
        # they are preemptively calculated so that JS can then call ColumnDataSource object and change
        # which edges should be shown on the plot based on a chosen feature

        hist_source = self._create_histogram_source(histogram_data, feature)
        hist_plot = self._create_histogram_plot(hist_source)
        return hist_source, hist_plot

    def _create_histogram_source(self, histogram_data, feature):
        source = ColumnDataSource()
        first_values = histogram_data[feature]

        # TODO: move names to properties
        source.data = {
            "hist": first_values[0],
            "left_edges": first_values[1],
            "right_edges": first_values[2]
        }
        return source

    @stylize()
    def _create_histogram_plot(self, source):

        kwargs = {
            "plot_height": 300,
            "height_policy": "fit",
            "plot_width": 300,
            "title": self._histogram_title,
            "css_classes": [self._histogram]
        }

        fcolor = self.plot_design.fill_color
        p = default_figure(kwargs)
        p.quad(top="hist", bottom=0, left="left_edges", right="right_edges", source=source,
               fill_color=fcolor, line_color=fcolor)

        p.y_range.start = 0
        p.yaxis.visible = False
        return p

    def _create_correlation(self, data_normalized, data_raw):
        # Creating identical plots but with different coloring depending on the values

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
        # absolute value is needed for coloring - -1.0 and 1.0 is a strong correlation regardless of direction
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

    @stylize()
    def _create_correlation_plot(self, source, cols_for_range, color_mapper, value_to_color):

        tooltip_text = [
            (self._correlation_values_normalized_title, "@" + self._correlation_values_normalized),
            (self._correlation_values_raw_title, "@" + self._correlation_values_raw)
        ]

        kwargs = {
            "css_classes": [self._correlation],
            "x_range": cols_for_range,
            "y_range": cols_for_range[::-1],  # first value to be at the top of the axis
            "tooltips": tooltip_text,
        }

        p = default_figure(kwargs)

        p.rect(
            x=self._correlation_x,
            y=self._correlation_y,
            source=source,
            fill_color={"field": value_to_color, "transform": color_mapper},
            width=1,
            height=1,
            line_color=None,
        )

        p.xaxis.major_label_orientation = -1  # in radians
        p.add_layout(ColorBar(color_mapper=color_mapper, width=40), "right")

        return p

    def _create_correlation_color_mapper(self):

        # palette
        tints = self.plot_design.contrary_color_tints
        linear_correlation = tints[0]
        no_correlation = [tints[9]]
        small_correlation = [tints[7]] * 2
        medium_correlation = [tints[6]] * 2
        high_correlation = [tints[4]] * 4
        very_high_correlation = [tints[2]] * 1
        palette = no_correlation + small_correlation + medium_correlation + high_correlation + very_high_correlation

        cmap = LinearColorMapper(palette=palette, low=0, high=0.9999,
                                 high_color=linear_correlation)  # contrary_color_palette

        return cmap


class ScatterPlotGrid(MainGrid):
    """Grid of Scatter Plots created with bokeh.

        Similar to seaborn Pairplot, every feature is plotted against another, creating a grid of scatter plots.
        However, there is an additional layer applied - every feature is also used as a hue (coloring) for every
        scatter plot. The idea behind it is to visualize any specific subgroups within correlations between features.

        ScatterPlotGrid is interactive - it has a hidden feature dropdown, which drives changes in the rows when
        the value in it is changed. Chosen feature is always plotted as X. Additionally, row with a feature that
        is chosen is greyed out so that it can be identified and neglected when looking for any insights (initially
        I wanted to remove it from the plot, but it posed a lot of difficulties - there was a blank space left
        and other plots didn't want to move again; so I dropped that idea and went with greying it out).

        Dropdown from a parent MainGrid class is connected via JS to dynamically change X values in scatter plots.
    """

    # HTML elements
    _row_description_html = "{hue}"
    _legend_no_hue_html = "No color - too many categories!"
    _legend_template_html = """
        <div class='legend-row'><div style='background-color: {color}' class='legend-marker'>
        </div><div class='legend-description'>{category}</div></div>
    """

    # CSS elements
    _scatterplot_grid_dropdown = "scatter_plot_grid_dropdown"
    _active_feature_hue = "active_feature_hue"
    _scatterplot_row = "scatter_plot_row"
    _hue_title = "hue-title"
    _row_description = "row-description"

    # TODO: make js callback dynamic
    # JS Callbacks
    _scatterplot_callback_js = """
                // new dropdown value
                var new_val = cb_obj.value;  
                
                // new x 
                var new_x = new_val;
                
                // TODO: class here
                document.querySelector("." + "chosen-feature-scatter").innerText = new_val;
                
                // scatter source updated
                for (i=0; i<scatter_sources.length; i++) {
                    scatter_sources[i].data["x"] = scatter_sources[i].data[new_x];
                    scatter_sources[i].change.emit();
                };
                
                // TODO: classes here
                // removing previous greying out
                var all_scatter_rows = document.getElementsByClassName("scatter_plot_row");
                for (j=0; j<all_scatter_rows.length; j++) {
                    all_scatter_rows[j].classList.remove("active_feature_hue");
                };
                
                // TODO: class here
                // greying out
                var scatter_row = document.getElementsByClassName("scatter_plot_row_" + new_val);
                scatter_row[0].classList.add("active_feature_hue");

            """

    # hardcoded limit
    _max_categories_internal_limit = 10

    # Color Scheme
    _categorical_palette = Category10
    _linear_palette = contrary_color_palette  # Reds4[::-1]

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

    def __init__(self, features, plot_design, categorical_features, feature_descriptions, feature_mapping,
                 feature_description_class, categorical_suffix="_categorical"):
        self.categorical_columns = categorical_features
        self.feature_descriptions = feature_descriptions
        self.feature_mapping = feature_mapping
        self.categorical_suffix = categorical_suffix
        super().__init__(features, plot_design, feature_description_class)

        # self._linear_palette = self.plot_design.contrary_color_tints[::-2][:4]

    def scattergrid(self, scatter_data, initial_feature):

        # Font won't be updated in plots until any change is made (e.g. choosing different Feature).
        # This is a bug in bokeh: https://github.com/bokeh/bokeh/issues/9448
        # Issue is relatively minor, I won't be doing any workaround for now.

        features = self.features
        scatter_row_sources, scatter_rows = self._create_scatter_rows(scatter_data, features, initial_feature)

        dropdown = self._create_features_dropdown(self._scatterplot_grid_dropdown)
        callbacks = self._create_features_dropdown_callbacks(scatter_row_sources)
        for callback in callbacks:
            dropdown.js_on_change("value", callback)

        grid = column(
            dropdown,
            Div(text=initial_feature, css_classes=[self._chosen_feature_scatter_title]),
            *scatter_rows,
        )

        return grid

    def _create_features_dropdown_callbacks(self, scatter_source):
        callbacks = []

        for call in [
            self._create_scatter_plot_callback(scatter_source),
        ]:
            callbacks.append(call)

        return callbacks

    def _create_scatter_plot_callback(self, sources):
        kwargs = {
            "scatter_sources": sources
        }

        callback = CustomJS(args=kwargs, code=self._scatterplot_callback_js)
        return callback

    def _create_scatter_rows(self, scatter_data, features, initial_feature):
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
            css_classes=[self._scatterplot_row, self._scatterplot_row + "_" + hue],
            margin=(0, 48, 0, 0)
        )

        return sources, r

    def _create_scatter_source(self, scatter_data, x, y):
        source = ColumnDataSource(scatter_data)
        source.data.update(
            {self._scatter_x_axis: source.data[x],
             self._scatter_y_axis: source.data[y]
             }
        )
        return source

    @stylize()
    def _create_scatter_plot(self, source, x, y, cmap):
        # x is left in case it is needed for any styling of the plot later in the development

        kwargs = {
            "x": self._scatter_x_axis,
            "y": self._scatter_y_axis,
            "source": source,
            "size": 10,
            "fill_color": self.plot_design.fill_color
        }

        if cmap:  # overriding plain fill color
            kwargs.update(
                {
                    "fill_color": cmap,
                }
            )

        # TODO: No X axis label as no axis are being updated in the js callback - decide if needed somehow?
        p = default_figure()
        p.plot_width = 200
        p.plot_height = 200
        p.scatter(**kwargs)
        p.yaxis.axis_label = y

        return p

    def _create_color_map(self, hue, data):
        if hue in self.categorical_columns:
            # adding suffix to column name as described in analyzer._scatter_data
            factors = sorted(set(data[hue + self.categorical_suffix]))
            if len(factors) <= self._max_categories_internal_limit:
                cmap = factor_cmap(hue + self.categorical_suffix, palette=self._categorical_palette[len(factors)],
                                   factors=factors)
            else:
                # If there is too many categories, None is returned
                cmap = None
        else:
            values = data[hue]
            cmap = linear_cmap(hue, palette=self._linear_palette, low=min(values), high=max(values))

        return cmap

    def _create_row_description(self, hue, cmap):

        # HTML needs to be prepared so that description is hidden/hoverable
        desc = self.feature_descriptions[hue]

        # TODO: this is a repeated snippet from Overview as well - make into one function?
        parsed_html = BeautifulSoup(self._row_description_html.format(hue=hue), "html.parser")
        parsed_html.string.wrap(parsed_html.new_tag("p"))
        parsed_html.p.append(append_description(desc, parsed_html))
        parsed_html.p["class"] = self.feature_description_class

        feature_with_description = str(parsed_html)

        kwargs = {
            "text": feature_with_description,
            "css_classes": [self._hue_title]
        }

        legend = self._create_legend(hue, cmap)

        d = Div(**kwargs)

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
        if cmap:
            if hue in self.categorical_columns:
                mapping = self.feature_mapping[hue]
                categories = cmap["transform"].factors
                colors = cmap["transform"].palette
                text = ""
                template = self._legend_template_html
                for category, color in zip(categories, colors):
                    mapped_category = mapping[float(category)]
                    text += template.format(color=color, category=mapped_category)

                legend = Div(text=text, css_classes=[self._legend, self._legend_categorical])

            else:
                colorbar = ColorBar(color_mapper=cmap["transform"], ticker=BasicTicker(desired_num_ticks=4),
                                    formatter=PrintfTickFormatter(), label_standoff=7, border_line_color=None,
                                    bar_line_color=self.plot_design.text_color,
                                    major_label_text_font_size="14px", location=(-100, 0),
                                    major_label_text_color=self.plot_design.text_color, width=30,
                                    major_tick_line_color=self.plot_design.text_color, major_tick_in=0)
                legend = default_figure({"height": 120, "width": 120, "css_classes": [self._legend]})
                # TODO: supress warning
                legend.add_layout(colorbar, "right")
        else:
            legend = Div(text=self._legend_no_hue_html, css_classes=[self._legend])

        return legend


class ModelsPlotClassification:
    _roc_plot_name = "ROC Curve"
    _precision_recall_plot_name = "Precision Recall Plot"
    _det_plot_name = "Detection Error Tradeoff"

    def __init__(self, plot_design):
        self.plot_design = plot_design

    def models_comparison_plot(self, roc_curves, precision_recall_curves, det_curves, target_proportion):

        new_tps = [assess_models_names(tp) for tp in [roc_curves, precision_recall_curves, det_curves]]
        roc_curves, precision_recall_curves, det_curves = new_tps

        roc_plot = Panel(child=self._roc_plot(roc_curves), title=self._roc_plot_name)
        precision_recall_plot = Panel(child=self._precision_recall_plot(precision_recall_curves, target_proportion),
                                      title=self._precision_recall_plot_name)
        det_plot = Panel(child=self._det_plot(det_curves), title=self._det_plot_name)
        main_plot = Tabs(tabs=[roc_plot, precision_recall_plot, det_plot])

        return main_plot

    @stylize()
    def _roc_plot(self, roc_curves):
        p = default_figure(
            {
                "x_range": (-0.01, 1.1),
                "y_range": (-0.01, 1.1),
                "tools": "pan,wheel_zoom,box_zoom,reset",
                "toolbar_location": "right"
            }
        )
        self._default_models_lines(p, roc_curves)

        p.legend.location = "bottom_right"
        p.line([0, 1], [0, 1], line_dash="dashed", line_width=1,
               color=self.plot_design.models_dummy_color, legend_label="Random Baseline", muted_alpha=0.5)

        p.xaxis.axis_label = "False Positive Rate"
        p.yaxis.axis_label = "True Positive Rate"

        return p

    @stylize()
    def _precision_recall_plot(self, precision_recall_curves, target_proportion):
        p = default_figure(
            {
                "x_range": (-0.01, 1.1),
                "y_range": (-0.01, 1.1),
                "tools": "pan,wheel_zoom,box_zoom,reset",
                "toolbar_location": "right"
            }
        )

        curves = [(model, (values[1], values[0], values[2])) for model, values in precision_recall_curves]
        self._default_models_lines(p, curves)

        p.legend.location = "bottom_left"
        p.line([0, 1], [target_proportion, target_proportion], line_dash="dashed", line_width=1,
               color=self.plot_design.models_dummy_color, legend_label="Random Baseline", muted_alpha=0.5)

        p.xaxis.axis_label = "Recall"
        p.yaxis.axis_label = "Precision"

        return p

    @stylize()
    def _det_plot(self, det_curves):

        # https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/metrics/_plot/det_curve.py#L100
        p = default_figure({
            "x_range": (-3, 3),
            "y_range": (-3, 3),
            "tools": "pan,wheel_zoom,box_zoom,reset",
            "toolbar_location": "right"
        })

        new_curves = []
        for model, curve in det_curves:
            f = scipy.stats.norm.ppf
            new_tuple = (f(curve[0]), f(curve[1]))
            new_curves.append((model, new_tuple))

        self._default_models_lines(p, new_curves)
        p.legend.location = "top_right"

        # 0.4999 was included instead of 0.5 as after normal transformation, 0.5 becomes 0.0. FuncTickFormatter would
        # then try to access dictionary of ticks with a key of 0, which JS evaluated to undefined and error was raised.
        ticks = [0.001, 0.01, 0.05, 0.20, 0.4999, 0.80, 0.95, 0.99, 0.999]
        tick_location = scipy.stats.norm.ppf(ticks)
        mapper = {norm_tick: tick for norm_tick, tick in zip(tick_location, ticks)}

        p.xaxis.ticker = tick_location
        p.yaxis.ticker = tick_location

        formatter = FuncTickFormatter(args={"mapper": mapper}, code="""
            return (mapper[tick] * 100).toString() + "%";
        """)

        p.xaxis.formatter = formatter
        p.yaxis.formatter = formatter

        p.xaxis.axis_label = "False Positive Rate"
        p.yaxis.axis_label = "False Negative Rate"

        return p

    def _default_models_lines(self, plot, model_values_tuple):

        new_tuples = list(reversed(model_values_tuple))
        lw = 5

        # models are plotted in reverse order (without the first one)
        for model, values in new_tuples[:-1]:
            plot.step(values[0], values[1], line_width=lw - 2, legend_label=model,
                      line_color=self.plot_design.models_color_tuple[1], muted_alpha=0.2)

        # the best model is plotted as last to be on top of other lines
        first_model, first_values = new_tuples[-1]
        plot.step(first_values[0], first_values[1], line_width=lw, legend_label=first_model,
                  line_color=self.plot_design.models_color_tuple[0], muted_alpha=0.2)

        plot.legend.click_policy = "mute"
        plot.toolbar.autohide = True


class ModelsPlotRegression:
    # negative numbers are having a wacky formatting
    _formatter_code = """return String(tick);"""

    def __init__(self, plot_design):
        self.plot_design = plot_design

    def prediction_error_plot(self, prediction_errors):

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

    @stylize()
    def _single_prediction_error_plot(self, scatter_data, color):
        p = default_figure(
            {
                "tools": "pan,wheel_zoom,box_zoom,reset",
                "toolbar_location": "right"
            }
        )

        p.scatter(scatter_data[0], scatter_data[1], color=color, size=16, fill_alpha=0.8)

        # TODO: will it really work or is it plotted always as a straight line?
        y_line = []
        for q in np.linspace(0, 1, 11):
            y_line.append(scatter_data[0].quantile(q=q))

        p.line(y_line, y_line, line_width=1, color=self.plot_design.models_dummy_color, line_dash="dashed")

        p.xaxis.axis_label = "Actual"
        p.yaxis.axis_label = "Predicted"

        formatter = FuncTickFormatter(code=self._formatter_code)
        # formatters must be created independently, cannot be reused between plots
        p.xaxis.formatter = formatter
        p.yaxis.formatter = formatter

        p.toolbar.autohide = True

        return p

    def residual_plot(self, residual_tuples):
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

    @stylize()
    def _single_residual_plot(self, scatter_data, color):
        p = default_figure(
            {
                "tools": "pan,wheel_zoom,box_zoom,reset",
                "toolbar_location": "right",
                "width": 900,
                "height": 300
            }
        )

        p.scatter(scatter_data[0], scatter_data[1], color=color, size=10, fill_alpha=0.8)

        p.line([min(scatter_data[0]), max(scatter_data[0])], [0, 0], line_width=2,
               color=self.plot_design.models_dummy_color)

        p.xaxis.axis_label = "Predicted"
        p.yaxis.axis_label = "Residual"

        formatter = FuncTickFormatter(code=self._formatter_code)
        # formatters must be created independently, cannot be reused
        p.xaxis.formatter = formatter
        p.yaxis.formatter = formatter

        p.toolbar.autohide = True

        return p


class ModelsPlotMulticlass:
    _x = "x"
    _y = "y"
    _values = "values"

    def __init__(self, plot_design, label_classes):
        self.plot_design = plot_design
        self.labels = list(map(str, label_classes))

    def confusion_matrices_plot(self, confusion_matrices):

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

    @stylize()
    def _single_confusion_matrix_plot(self, confusion_array, palette, model_name):

        source = self._create_column_data_source(confusion_array)
        cmap = LinearColorMapper(palette=palette[::-1], low=0, high=max(confusion_array.ravel()))

        p = default_figure(
            {
                "title": model_name,
                "height": 300,
                "width": 300,
                "x_range": self.labels,
                "y_range": self.labels[::-1]
            }
        )

        p.rect(
            x=self._x,
            y=self._y,
            source=source,
            fill_color={"field": self._values, "transform": cmap},
            width=1,
            height=1,
            line_color=None,
        )

        labels = LabelSet(
            x=self._x,
            y=self._y,
            text=self._values,
            source=source,
            render_mode="canvas",
            x_offset=-7,
            y_offset=-7,
            text_color="black",
            text_font_size="11px",
        )
        p.add_layout(labels)

        p.xaxis.major_label_orientation = -0.5  # in radians

        return p

    def _create_column_data_source(self, confusion_array):

        cds = ColumnDataSource()
        df = pd.DataFrame(confusion_array).stack()
        x = df.index.droplevel(0).astype(str).to_list()  # one of the indexes
        y = df.index.droplevel(1).astype(str).to_list()  # second of the indexes
        values = df.to_list()

        cds.data = {
            self._x: x,
            self._y: y,
            self._values: values
        }

        return cds


class ModelsDataTable:
    _model_column_template = '<div style="color:{color};font-weight:bold;font-size:1.15em"><%= value %></div>'

    def __init__(self, plot_design):
        self.plot_design = plot_design

    def data_table(self, X, y, models_predictions):
        models_predictions = assess_models_names(models_predictions)
        base_color = self.plot_design.base_color_tints[0]

        cols = [TableColumn(
            field=y.name,
            title=y.name,
            formatter=HTMLTemplateFormatter(template=self._model_column_template.format(color=base_color))
        )]

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
        df = pd.concat([y, scores, X], axis=1)

        source = ColumnDataSource(df)
        dt = DataTable(source=source, columns=cols, editable=False, sizing_mode="stretch_width")

        return dt

    # def _single_data_table(self, X, y, single_predictions):
    #     df = pd.concat([X, y, single_predictions], axis=1)
    #
    #     source = ColumnDataSource(df)
    #
    #     cols = []
    #     for col in df.columns:
    #         cols.append(TableColumn(field=col, title=col))
    #
    #     dt = DataTable(source=source, columns=cols, editable=False, sizing_mode="stretch_width")
    #
    #     return dt
