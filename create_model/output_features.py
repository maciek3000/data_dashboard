from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource
from bokeh.embed import components
from bokeh.models.widgets import Select, Div
from bokeh.models import CustomJS
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral5
from bs4 import BeautifulSoup

import functools


class FeatureView:

    def __init__(self, template, css_path, js_path, features, naive_mapping):
        self.template = template
        self.css = css_path
        self.js = js_path
        self.features = features
        self.naive_mapping = naive_mapping

        self.chosen_feature = None

    def _stylize_attributes(force=False):
        """Decorator for functions that return Plot figures that all of them would be stylized similarly"""

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

    def _default_figure(self, plot_specific_kwargs=None):
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

    def render(self, base_dict, histogram_data, scatter_data, categorical_columns):

        self.chosen_feature = sorted(self.features.keys())[0]
        output_dict = {}
        output_dict.update(base_dict)

        output_dict["chosen_feature"] = self.chosen_feature
        output_dict["features_css"] = self.css
        output_dict["features_js"] = self.js

        output_dict["features_menu"] = self._create_features_menu()

        grid = self._create_gridplot(histogram_data, scatter_data, categorical_columns)
        script, div = components(grid)
        output_dict["bokeh_script"] = script
        output_dict["test_plot"] = div

        return self.template.render(**output_dict)

    def _create_features_menu(self):
        html = "<div class='features-menu-title'><div>Features</div><div class='close-button'>x</div></div>"
        template = "<div class='single-feature'>{:03}. {}</div>"
        i = 0
        for feat in self.features:
            html += template.format(i, feat)
            i += 1
        return html

    def _create_gridplot(self, histogram_data, scatter_data, categorical_columns):

        histogram_source, histogram_plot = self._create_histogram(histogram_data)
        info_mapping, info_div = self._create_info_div()
        scatter_source, scatter_plot = self._create_scatter(scatter_data, categorical_columns)
        dropdown = self._create_features_dropdown()

        dropdown_kwargs = {
            # histogram
            "histogram_data": histogram_data,
            "histogram_source": histogram_source,
            # info div
            "info_mapping": info_mapping,
            # scatter
            "scatter_data": scatter_data,
            "scatter_source": scatter_source
        }

        callbacks = self._create_features_dropdown_callbacks(**dropdown_kwargs)
        for callback in callbacks:
            dropdown.js_on_change("value", callback)

        output = column(
            dropdown,  # this dropdown will be invisible (display: none)
            row(
                histogram_plot, info_div, scatter_plot
            )
        )
        return output

    def _create_features_dropdown(self):
        fts = sorted(self.features.keys())
        d = Select(options=fts, css_classes=["features_dropdown"], name="features_dropdown")
        return d

    def _create_features_dropdown_callbacks(self, histogram_data, histogram_source, info_mapping,
                                            scatter_data, scatter_source):
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
            var edges = new_hist[1];
            var left_edges = edges.slice(0, edges.length - 1);
            var right_edges = edges.slice(1, edges.length);
            
            // histogram source updated
            hist_source.data["hist"] = hist;
            hist_source.data["left_edges"] = left_edges;
            hist_source.data["right_edges"] = right_edges;
            
            // updating ColumnDataSources
            hist_source.change.emit();
            
        """)
        return callback

    def _create_info_div_callback(self, info_mapping):
        # feature : feature_name

        callback = CustomJS(args=info_mapping, code="""
            // new values
            var new_feature = cb_obj.value;  // new feature
            
            // updating 
            document.querySelector("#" + info_mapping["feature_name"]).innerText = new_feature;
        """)

        return callback

    def _create_histogram(self, histogram_data):
        hist_source = self._create_histogram_source(histogram_data)
        hist_plot = self._create_histogram_plot(hist_source)
        return hist_source, hist_plot

    def _create_histogram_source(self, histogram_data):
        source = ColumnDataSource()
        first_values = histogram_data[self.chosen_feature]

        source.data = {
            "hist": first_values[0],
            "left_edges": first_values[1][:-1],
            "right_edges": first_values[1][1:],
        }
        return source

    @_stylize_attributes()
    def _create_histogram_plot(self, source):

        kwargs = {
            "plot_height": 460,
            "plot_width": 460,
            "title": "Feature Distribution"
        }

        p = self._default_figure(kwargs)
        p.quad(top="hist", bottom=0, left="left_edges", right="right_edges", source=source,
               fill_color="#8CA8CD")

        p.y_range.start = 0
        p.yaxis.visible = False
        return p

    def _create_info_div(self):
        id_name = "feature_name"
        mapping = {
            "feature_name": id_name
        }
        text = "Feature to be discussed: <p id={id_name}>{feature}</p>".format(id_name=id_name,
                                                                               feature=self.chosen_feature)
        d = Div(name="info_div", css_classes=["info_div"], text=text)
        return mapping, d

    def _create_scatter(self, scatter_data, categorical_columns):
        scatter_source = self._create_scatter_source(scatter_data, categorical_columns)
        scatter_plot = self._create_scatter_plot(scatter_source)

        grid = column(
            #row(one_dropdown, two_dropdown),
            row(scatter_plot)
        )

        return scatter_source, grid

    def _create_scatter_source(self, scatter_data, categorical_columns):
        source = ColumnDataSource()
        x = self.chosen_feature
        cols_wo_x = sorted(scatter_data.keys() - {x,})
        y = cols_wo_x[0]
        hue_cols = {key: arg for key, arg in scatter_data.items() if (key in categorical_columns) and (key not in [x, y])}
        hue = sorted(hue_cols.keys())[0]
        # feature to be chosen at first when the page loads for the first time
        source.data = {
            "x": scatter_data[x],
            "y": scatter_data[y],
            "color": factor_cmap(hue, palette=Spectral5, factors=list(map(str, sorted(set(scatter_data[hue])))))
        }

    @_stylize_attributes()
    def _create_scatter_plot(self, source):
        # This won't work - JS would need to make a lot of calculations on data (e.g. remove NaN values) and create
        # new colormaps for every chosen category for hue
        p = self._default_figure()
        p.scatter(x="x", y="y", fill_color="hue", source=source)
        return p