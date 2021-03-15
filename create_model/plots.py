from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Select, Div
from bokeh.models import CustomJS, ColorBar, BasicTicker, PrintfTickFormatter
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.palettes import Reds4, Category10
import functools
import seaborn as sns


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

            p.xgrid.grid_line_color = None
            p.ygrid.grid_line_color = None

            p.title.text_font = "Lato"
            p.title.text_color = text_color
            p.title.text_font_size = "16px"

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

    def __init__(self, features):
        self.features = features

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
    _info_div_html = "Feature to be discussed: <p id={feature_name_id}>{feature}</p>"

    # CSS elements
    _infogrid_dropdown = "info_grid_dropdown"
    _feature_name_id = "feature_name"
    _info_div = "info_div"

    # JS callbacks
    _histogram_callback_js = """
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

    _info_div_callback = """
                // new values
                var new_feature = cb_obj.value;  // new feature
                
                // updating 
                document.querySelector("#" + "{feature_name_id}").innerText = new_feature;
            """

    # plot elements
    _fill_color = "#8CA8CD"
    _histogram_title = "Feature Distribution"

    def __init__(self, features):
        super().__init__(features)

    def infogrid(self, histogram_data, initial_feature):

        histogram_source, histogram_plot = self._create_histogram(histogram_data, initial_feature)
        info_div = self._create_info_div(initial_feature)
        dropdown = self._create_features_dropdown(self._infogrid_dropdown)

        callbacks = self._create_features_dropdown_callbacks(
            histogram_data=histogram_data,
            histogram_source=histogram_source,
        )
        for callback in callbacks:
            dropdown.js_on_change("value", callback)

        output = column(
            dropdown,  # this dropdown will be invisible (display: none)
            row(
                histogram_plot, info_div,
            ),
        )
        return output

    def _create_features_dropdown_callbacks(self, histogram_data, histogram_source):
        callbacks = []

        for call in [
            self._create_histogram_callback(histogram_data, histogram_source),
            self._create_info_div_callback()
        ]:
            callbacks.append(call)

        return callbacks

    def _create_histogram_callback(self, histogram_data, histogram_source):
        kwargs = {
            "hist_source": histogram_source,
            "hist_data": histogram_data
        }

        callback = CustomJS(args=kwargs, code=self._histogram_callback_js)
        return callback

    def _create_info_div_callback(self):
        # this code will need to be updated with every detail added to the Info (Summary) Div
        callback = CustomJS(code=self._info_div_callback.format(feature_name_id=self._feature_name_id))
        return callback

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

        source.data = {
            "hist": first_values[0],
            "left_edges": first_values[1],
            "right_edges": first_values[2]
        }
        return source

    @stylize()
    def _create_histogram_plot(self, source):

        kwargs = {
            "plot_height": 460,
            "plot_width": 460,
            "title": self._histogram_title
        }

        p = default_figure(kwargs)
        p.quad(top="hist", bottom=0, left="left_edges", right="right_edges", source=source,
               fill_color=self._fill_color, line_color=self._fill_color)

        p.y_range.start = 0
        p.yaxis.visible = False
        return p

    def _create_info_div(self, feature):
        text = self._info_div_html.format(
            feature_name_id=self._feature_name_id,
            feature=feature
        )
        d = Div(name=self._info_div, css_classes=[self._info_div], text=text)

        return d


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
        <div class='legend-row'><span style='background-color: {color}' class='legend-marker'>
        </span>{category}</div>
    """

    # CSS elements
    _scatterplot_grid_dropdown = "scatter_plot_grid_dropdown"
    _active_feature_hue = "active_feature_hue"
    _scatterplot_row = "scatter_plot_row"
    _hue_title = "hue-title"
    _row_description = "row-description"

    # JS Callbacks
    _scatterplot_callback_js = """
                // new dropdown value
                var new_val = cb_obj.value;  
                
                // new x 
                var new_x = new_val;
                
                // scatter source updated
                for (i=0; i<scatter_sources.length; i++) {
                    scatter_sources[i].data["x"] = scatter_sources[i].data[new_x];
                    scatter_sources[i].change.emit();
                };
                
                // removing previous greying out
                var all_scatter_rows = document.getElementsByClassName("scatter_plot_row");
                for (j=0; j<all_scatter_rows.length; j++) {
                    all_scatter_rows[j].classList.remove("active_feature_hue");
                };
                
                // greying out
                var scatter_row = document.getElementsByClassName("scatter_plot_row_" + new_val);
                scatter_row[0].classList.add("active_feature_hue");

            """

    # Color Scheme
    categorical_palette = Category10
    linear_palette = Reds4[::-1]

    # changing palette for 1 and 2 elements to be of similar color to the palette for 3 elements
    categorical_palette[2] = Category10[3][:2]
    categorical_palette[1] = Category10[3][:1]

    _fill_color = "#8CA8CD"

    def __init__(self, features, max_categories, categorical_suffix="_categorical"):
        self.features = features
        self.max_categories = max_categories
        self.categorical_suffix = categorical_suffix
        super().__init__(features)

    def scattergrid(self, scatter_data, categorical_columns, initial_feature, feature_mapping):

        # Font won't be updated in plots until any change is made (e.g. choosing different Feature).
        # This is a bug in bokeh: https://github.com/bokeh/bokeh/issues/9448
        # Issue is relatively minor, I won't be doing any workaround for now.

        features = sorted(self.features)
        scatter_row_sources, scatter_rows = self._create_scatter_rows(
            scatter_data, features, initial_feature, categorical_columns, feature_mapping
        )

        dropdown = self._create_features_dropdown(self._scatterplot_grid_dropdown)
        callbacks = self._create_features_dropdown_callbacks(scatter_row_sources)
        for callback in callbacks:
            dropdown.js_on_change("value", callback)

        grid = column(
            dropdown,
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

    def _create_scatter_rows(self, scatter_data, features, initial_feature, categorical_columns, feature_mapping):
        all_sources = []
        all_rows = []

        for feature in features:
            sources, single_row = self._create_single_scatter_row(scatter_data, features, initial_feature,
                                                                  feature, categorical_columns, feature_mapping)

            if feature == initial_feature:
                single_row.css_classes.append(self._active_feature_hue)

            all_sources.extend(sources)
            all_rows.append(single_row)

        return all_sources, all_rows

    def _create_single_scatter_row(self, scatter_data, features, initial_feature, hue, categorical_columns, feature_mapping):
        sources = []
        plots = []

        color_map = self._create_color_map(hue, scatter_data, categorical_columns)
        for feature in features:
            src = self._create_scatter_source(scatter_data, initial_feature, feature)
            plot = self._create_scatter_plot(src, initial_feature, feature, color_map)

            sources.append(src)
            plots.append(plot)

        color_legend = self._create_row_description(hue, color_map, categorical_columns, feature_mapping)

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
            {"x": source.data[x],
             "y": source.data[y]
             }
        )
        return source

    @stylize()
    def _create_scatter_plot(self, source, x, y, cmap):
        # x is left in case it is needed for any styling of the plot later in the development

        kwargs = {
            "x": "x",
            "y": "y",
            "source": source,
            "size": 10,
            "fill_color": self._fill_color
        }

        if cmap:
            kwargs.update(
                {
                    "fill_color": cmap,
                }
            )

        p = default_figure()
        p.plot_width = 200
        p.plot_height = 200
        p.scatter(**kwargs)
        p.yaxis.axis_label = y

        return p

    def _create_color_map(self, hue, data, categorical_columns):
        if hue in categorical_columns:
            # adding suffix to column name as described in analyzer._scatter_data
            factors = sorted(set(data[hue + self.categorical_suffix]))
            if len(factors) <= self.max_categories:
                cmap = factor_cmap(hue + self.categorical_suffix, palette=self.categorical_palette[len(factors)],
                                   factors=factors)
            else:
                # If there is too many categories, None is returned
                cmap = None
        else:
            values = data[hue]
            cmap = linear_cmap(hue, palette=self.linear_palette, low=min(values), high=max(values))

        return cmap

    def _create_row_description(self, hue, cmap, categorical_columns, feature_mapping):

        kwargs = {
            "text": self._row_description_html.format(hue=hue),
            "css_classes": [self._hue_title]
        }

        legend = self._create_legend(hue, cmap, categorical_columns, feature_mapping)

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

    def _create_legend(self, hue, cmap, categorical_columns, feature_mapping):
        legend = Div(text=self._legend_no_hue_html)
        if cmap:
            if hue in categorical_columns:
                mapping = feature_mapping[hue]
                categories = cmap["transform"].factors
                colors = cmap["transform"].palette
                text = ""
                template = self._legend_template_html
                for category, color in zip(categories, colors):
                    mapped_category = mapping[float(category)]
                    text += template.format(color=color, category=mapped_category)

                legend = Div(text=text, css_classes=["legend"])

            else:
                colorbar = ColorBar(color_mapper=cmap["transform"], ticker=BasicTicker(desired_num_ticks=4),
                                    formatter=PrintfTickFormatter(), label_standoff=10, border_line_color=None,
                                    location=(0, 0), major_label_text_font_size="12px")
                legend = default_figure({"height": 100, "width": 100})
                legend.add_layout(colorbar)

        return legend
