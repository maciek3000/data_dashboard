from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource
from bokeh.embed import components
from bokeh.models.widgets import Select, Div
from bokeh.models import CustomJS, ColorBar, BasicTicker, PrintfTickFormatter, Legend
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.palettes import Reds4, Category10

import functools


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

            for axis in [p.xaxis, p.yaxis]:
                pass

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


class MainGrid:

    def __init__(self, features):
        self.features = features

    def _create_features_dropdown(self, name="features_dropdown"):
        fts = sorted(self.features.keys())
        d = Select(options=fts, css_classes=["features_dropdown"], name=name)
        return d

    def _create_features_dropdown_callbacks(self, **kwargs):
        raise NotImplementedError


class InfoGrid(MainGrid):

    def __init__(self, features):
        super().__init__(features)

    def create_grid_elements(self, histogram_data, initial_feature):
        return components(self._create_info_grid(histogram_data, initial_feature))

    def _create_info_grid(self, histogram_data, initial_feature):

        histogram_source, histogram_plot = self._create_histogram(histogram_data, initial_feature)
        info_mapping, info_div = self._create_info_div(initial_feature)
        dropdown = self._create_features_dropdown("info_grid_dropdown")

        dropdown_kwargs = {
            # histogram
            "histogram_data": histogram_data,
            "histogram_source": histogram_source,

            # info div
            "info_mapping": info_mapping,
        }

        callbacks = self._create_features_dropdown_callbacks(**dropdown_kwargs)
        for callback in callbacks:
            dropdown.js_on_change("value", callback)

        output = column(
            dropdown,  # this dropdown will be invisible (display: none)
            row(
                histogram_plot, info_div,
            ),
        )
        return output

    # def _create_features_dropdown(self):
    #     fts = sorted(self.features.keys())
    #     d = Select(options=fts, css_classes=["features_dropdown"], name="features_dropdown")
    #     return d

    def _create_features_dropdown_callbacks(self, histogram_data, histogram_source, info_mapping):
        callbacks = []

        for call in [
            self._create_histogram_callback(histogram_data, histogram_source),
            self._create_info_div_callback(info_mapping)
        ]:
            callbacks.append(call)

        return callbacks

    def _create_histogram_callback(self, histogram_data, histogram_source):
        kwargs = {
            "hist_source": histogram_source,
            "hist_data": histogram_data
        }

        callback = CustomJS(args=kwargs, code="""
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
                
            """)
        return callback

    def _create_info_div_callback(self, info_mapping):
        mapp = {
            "info_mapping": info_mapping
        }

        callback = CustomJS(args=mapp, code="""
                // new values
                var new_feature = cb_obj.value;  // new feature
                
                // updating 
                document.querySelector("#" + info_mapping["feature_name"]).innerText = new_feature;
            """)

        return callback

    def _create_histogram(self, histogram_data, feature):
        hist_source = self._create_histogram_source(histogram_data, feature)
        hist_plot = self._create_histogram_plot(hist_source)
        return hist_source, hist_plot

    def _create_histogram_source(self, histogram_data, feature):
        source = ColumnDataSource()
        first_values = histogram_data[feature]

        source.data = {
            "hist": first_values[0],
            "left_edges": first_values[1],  #first_values[1][:-1],
            "right_edges": first_values[2]  #first_values[1][1:],
        }
        return source

    @stylize()
    def _create_histogram_plot(self, source):

        kwargs = {
            "plot_height": 460,
            "plot_width": 460,
            "title": "Feature Distribution"
        }

        p = default_figure(kwargs)
        p.quad(top="hist", bottom=0, left="left_edges", right="right_edges", source=source,
               fill_color="#8CA8CD", line_color="#8CA8CD")

        p.y_range.start = 0
        p.yaxis.visible = False
        return p

    def _create_info_div(self, feature):
        id_name = "feature_name"
        mapping = {
            "feature_name": id_name
        }
        text = "Feature to be discussed: <p id={id_name}>{feature}</p>".format(id_name=id_name,
                                                                               feature=feature)
        d = Div(name="info_div", css_classes=["info_div"], text=text)
        return mapping, d


class ScatterPlotGrid(MainGrid):

    def __init__(self, features):
        super().__init__(features)

        self.categorical_palette = Category10
        self.linear_palette = Reds4[::-1]

        self.categorical_palette[2] = Category10[3][:2]
        self.categorical_palette[1] = Category10[3][:1]

    def create_grid_elements(self, scatter_data, categorical_columns, features, initial_feature):
        return components(self._create_scatter(scatter_data, categorical_columns, features, initial_feature))

    def _create_scatter(self, scatter_data, categorical_columns, features, initial_feature):

        features = sorted(features.keys())
        scatter_row_sources, scatter_rows = self._create_scatter_rows(scatter_data, features, initial_feature, categorical_columns)

        dropdown = self._create_features_dropdown("scatter_plot_grid_dropdown")
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

        callback = CustomJS(args=kwargs, code="""
                // new dropdown value
                var new_val = cb_obj.value;  
                
                // new x 
                var new_x = new_val;
                
                // scatter source updated
                for (i=0; i<scatter_sources.length; i++) {
                    scatter_sources[i].data["x"] = scatter_sources[i].data[new_x];
                    scatter_sources[i].change.emit();
                };
                
                var all_scatter_rows = document.getElementsByClassName("scatter_plot_row");
                for (j=0; j<all_scatter_rows.length; j++) {
                    all_scatter_rows[j].classList.remove("active_feature_hue");
                };
                
                var scatter_row = document.getElementsByClassName("scatter_plot_row_" + new_val);
                scatter_row[0].classList.add("active_feature_hue");

            """)
        return callback

    def _create_scatter_rows(self, data, features, initial_feature, categorical_columns):
        all_sources = []
        all_rows = []

        for feature in features:
            sources, single_row = self._create_single_scatter_row(data, features, initial_feature, feature, categorical_columns)

            if feature == initial_feature:
                single_row.css_classes.append("active_feature_hue")

            all_sources.extend(sources)
            all_rows.append(single_row)

        return all_sources, all_rows

    def _create_single_scatter_row(self, data, features, x, hue, categorical_columns):
        sources = []
        plots = []

        color_map = self._create_color_map(hue, data, categorical_columns)
        for feature in features:
            src = self._create_scatter_source(data, x, feature)
            plot = self._create_scatter_plot(src, x, feature, color_map)

            sources.append(src)
            plots.append(plot)


        # p = default_figure({"width": 100, "height": 100})
        # if color_map and (hue in categorical_columns):
        #     p.renderers = plots[0].renderers
        #     items = []
        #     for item in plots[0].legend[0].items:
        #         items.append((item.label["value"], item.renderers))
        #     colorbar = Legend(items=items)
        #
        # if colorbar:
        #     p.add_layout(colorbar)

        color_legend = self._create_row_description(hue, color_map, categorical_columns)

        r = row(
            color_legend,
            *plots,
            css_classes=["scatter_plot_row", "scatter_plot_row_" + hue]
        )

        return sources, r


    def _create_scatter_source(self, scatter_data, x, y):
        source = ColumnDataSource(scatter_data)
        # x = self.chosen_feature
        # cols_wo_x = sorted(scatter_data.keys() - {x,})
        # y = cols_wo_x[0]
        # # feature to be chosen at first when the page loads for the first time
        # source.data = {
        #     "x": scatter_data[x],
        #     "y": scatter_data[y],
        # }
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
            "fill_color": "#8CA8CD"
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
            factors = sorted(set(data[hue + "_categorical"]))
            if len(factors) <= 10:
                cmap = factor_cmap(hue + "_categorical", palette=self.categorical_palette[len(factors)], factors=factors)
            else:
                cmap = None
        else:
            values = data[hue]
            cmap = linear_cmap(hue, palette=self.linear_palette, low=min(values), high=max(values))

        return cmap

    def _create_row_description(self, hue, cmap, categorical_columns):

        kwargs = {
            "text": "Color: {hue}".format(hue=hue),
        }

        legend = self._create_legend(hue, cmap, categorical_columns)

        d = Div(**kwargs)

        c = column(
            d,
            legend,
            width=200,
            height=200,
            width_policy="fixed"
        )

        return c

    def _create_legend(self, hue, cmap, categorical_columns):
        
        legend = Div(text="No hue - too many categories!")
        if cmap:
            if hue in categorical_columns:
                categories = cmap["transform"].factors
                colors = cmap["transform"].palette
                text = "<ul style='list-style-type: circle'>"
                template = "<li style='color: {color}'>{category}</li>"
                for category, color in zip(categories, colors):
                    text += template.format(color=color, category=category)
                text += "</ul>"

                legend = Div(text=text)

            else:
                colorbar = ColorBar(color_mapper=cmap["transform"], ticker=BasicTicker(desired_num_ticks=4),
                                formatter=PrintfTickFormatter(), label_standoff=10, border_line_color=None,
                                location=(0, 0), major_label_text_font_size="12px")
                legend = default_figure({"height": 100, "width": 100})
                legend.add_layout(colorbar)

        return legend