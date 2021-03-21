from .views import append_description
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Select, Div
from bokeh.models import CustomJS, ColorBar, BasicTicker, PrintfTickFormatter
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.palettes import Reds4, Category10
import functools
import seaborn as sns
from bs4 import BeautifulSoup


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
    _info_div_html = """<div class="{info_div_content}">
    <div id="info-div-description"><p class="{feature_description_class}">Description<span id="info_div_description">{description}</span></p></div>
    <div>Type: <span id="info_div_type">{type}</span></div>
    <div>Mean: <span id="info_div_mean">{mean:.4f}</span></div>
    <div>Median: <span id="info_div_median">{median:.4f}</span></div>
    <div>Min: <span id="info_div_min">{min:.4f}</span></div>
    <div>Max: <span id="info_div_max">{max:.4f}</span></div>
    <div>Standard deviation: <span id="info_div_std">{std:.4f}</span></div>
    <div># of Missing: <span id="info_div_missing">{missing:.4f}</span></div>
    </div>"""

    # CSS elements
    _infogrid_dropdown = "info_grid_dropdown"
    _feature_name = "feature_name"
    _info_div = "info-div"
    _info_div_content = "info-div-content"
    _infogrid_row = "infogrid-row"
    _infogrid_all = "infogrid"

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

    # plot elements
    _histogram_title = "Feature Distribution"

    def __init__(self, features, plot_design, feature_description_class):
        super().__init__(features, plot_design, feature_description_class)

    def infogrid(self, histogram_data, summary_statistics, initial_feature):

        histogram_source, histogram_plot = self._create_histogram(histogram_data, initial_feature)
        info_div = self._create_info_div(summary_statistics, initial_feature)
        dropdown = self._create_features_dropdown(self._infogrid_dropdown)

        callbacks = self._create_features_dropdown_callbacks(
            histogram_data=histogram_data,
            histogram_source=histogram_source,
            summary_statistics=summary_statistics
        )
        for callback in callbacks:
            dropdown.js_on_change("value", callback)

        output = column(
            dropdown,  # this dropdown will be invisible (display: none)
            row(
                info_div, histogram_plot, css_classes=[self._infogrid_row]
            ), css_classes=[self._infogrid_all]
        )
        return output

    def _create_features_dropdown_callbacks(self, histogram_data, histogram_source, summary_statistics):
        callbacks = []

        for call in [
            self._create_histogram_callback(histogram_data, histogram_source),
            self._create_info_div_callback(summary_statistics)
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

    def _create_info_div_callback(self, summary_statistics):
        # this code will need to be updated with every detail added to the Info (Summary) Div
        kwargs = {
            "summary_statistics": summary_statistics
        }
        callback = CustomJS(
            args=kwargs,
            code=self._info_div_callback
        )
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

        fcolor = self.plot_design.fill_color
        p = default_figure(kwargs)
        p.quad(top="hist", bottom=0, left="left_edges", right="right_edges", source=source,
               fill_color=fcolor, line_color=fcolor)

        p.y_range.start = 0
        p.yaxis.visible = False
        return p

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
    _linear_palette = ["#FFF7F3", "#FFB695", "#EB6F54", "#9C2B19"]  # Reds4[::-1]

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

    def __init__(self, features, plot_design, categorical_features, feature_descriptions, feature_mapping, feature_description_class, categorical_suffix="_categorical"):
        self.categorical_columns = categorical_features
        self.feature_descriptions = feature_descriptions
        self.feature_mapping = feature_mapping
        self.categorical_suffix = categorical_suffix
        super().__init__(features, plot_design, feature_description_class)

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
                                    formatter=PrintfTickFormatter(), label_standoff=10, border_line_color=None,
                                    location=(0, 0), major_label_text_font_size="12px",
                                    major_label_text_color=self.plot_design.text_color)
                legend = default_figure({"height": 100, "width": 100, "css_classes": [self._legend]})
                legend.add_layout(colorbar)
        else:
            legend = Div(text=self._legend_no_hue_html, css_classes=[self._legend])

        return legend
