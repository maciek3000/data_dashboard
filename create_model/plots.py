from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource
from bokeh.embed import components
from bokeh.models.widgets import Select, Div
from bokeh.models import CustomJS
from bokeh.transform import factor_cmap
from bokeh.palettes import cividis

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


class InfoGrid:

    def __init__(self, features):
        self.features = features

    def create_grid_elements(self, histogram_data, initial_feature):
        return components(self._create_info_grid(histogram_data, initial_feature))

    def _create_info_grid(self, histogram_data, initial_feature):

        histogram_source, histogram_plot = self._create_histogram(histogram_data, initial_feature)
        info_mapping, info_div = self._create_info_div(initial_feature)
        dropdown = self._create_features_dropdown()

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

    def _create_features_dropdown(self):
        fts = sorted(self.features.keys())
        d = Select(options=fts, css_classes=["features_dropdown"], name="features_dropdown")
        return d

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


class ScatterPlotGrid:

    def __init__(self, features):
        self.features = features

    def create_grid_elements(self, scatter_data, categorical_columns, initial_feature):
        return components(self._create_scatter(scatter_data, categorical_columns, initial_feature))

    def _create_scatter(self, scatter_data, categorical_columns, initial_feature):
        scatter_source = self._create_scatter_source(scatter_data)
        scatter_plot = self._create_scatter_plot(scatter_source, categorical_columns, initial_feature)

        grid = column(
            #row(one_dropdown, two_dropdown),
            row(scatter_plot)
        )

        return grid

    def _create_scatter_source(self, scatter_data):
        scatter_data = scatter_data.dropna().to_dict(orient="list")
        scatter_data["Embarked"] = list(map(str, scatter_data["Embarked"]))
        source = ColumnDataSource(scatter_data)
        # x = self.chosen_feature
        # cols_wo_x = sorted(scatter_data.keys() - {x,})
        # y = cols_wo_x[0]
        # # feature to be chosen at first when the page loads for the first time
        # source.data = {
        #     "x": scatter_data[x],
        #     "y": scatter_data[y],
        # }
        return source

    @stylize()
    def _create_scatter_plot(self, source, categorical_columns, feature):
        hue_cols = {key: arg for key, arg in source.data.items() if (key in categorical_columns)}

        x = feature
        cols_wo_x = sorted(source.data.keys() - {x, })
        y = cols_wo_x[0]

        x = "Fare"
        y = "Parch"

        print(source.data.keys())

        # factor_cmap expects categorical data to be Str, not Int/Float

        # hue = sorted(hue_cols.keys())[0]
        hue = "Embarked"
        hue_cmap = factor_cmap(hue, palette=cividis(3), factors=sorted(set(source.data[hue])))

        p = default_figure()
        p.scatter(x=x, y=y, fill_color=hue_cmap, source=source)
        # fill_color=hue_cmap,
        return p
