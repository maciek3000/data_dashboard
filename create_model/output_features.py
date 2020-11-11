from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource
from bokeh.embed import components
from bokeh.models.widgets import Select
from bokeh.models import CustomJS
from bs4 import BeautifulSoup

import functools

class FeatureView:


    def __init__(self, template, css_path, js_path, features, naive_mapping):
        self.template = template
        self.css = css_path
        self.js = js_path
        self.features = features
        self.naive_mapping = naive_mapping

    def _stylize_attributes(force=False):
        def decorator_stylize_attributes(plot_func):
            @functools.wraps(plot_func)
            def wrapper(self, *args, **kwargs):
                p = plot_func(self, *args, **kwargs)

                if force:
                    return p

                for axis in [p.xaxis, p.yaxis]:
                    axis.minor_tick_line_color = None
                    axis.major_tick_line_color = None
                    axis.major_label_text_color = "#8C8C8C"
                    axis.axis_label_text_font = "Lato"
                    axis.axis_line_color = "#8C8C8C"

                p.xgrid.grid_line_color = None
                p.ygrid.grid_line_color = None

                return p
            return wrapper
        return decorator_stylize_attributes

    def render(self, base_dict, histogram_data):

        output_dict = {}
        output_dict.update(base_dict)

        output_dict["features_menu"] = self._create_features_menu()
        output_dict["chosen_feature"] = sorted(histogram_data)[0]
        grid = self._create_gridplot(histogram_data)

        script, div = components(grid)

        output_dict["bokeh_script"] = script
        output_dict["test_plot"] = div
        output_dict["features_css"] = self.css
        output_dict["features_js"] = self.js
        return self.template.render(**output_dict)

    def _create_features_menu(self):
        html = "<div class='features-menu-title'><div>Features</div><div class='close-button'>x</div></div>"
        template = "<div class='single-feature'>{:03}. {}</div>"
        i = 0
        for feat in self.features:
            html += template.format(i, feat)
            i += 1
        return html


    def _create_gridplot(self, histogram_data):

        histogram_source = ColumnDataSource()

        unique_features = sorted(histogram_data.keys())
        new = histogram_data[unique_features[0]]

        histogram_source.data = {
            "hist": new[0],
            "left_edges": new[1][:-1],
            "right_edges": new[1][1:]
        }

        histogram_plot = self._create_histogram_plot(histogram_source)
        dropdown = self._create_dropdown(unique_features)

        callback = CustomJS(args=dict(source=histogram_source, all_data=histogram_data), code="""
            var new_val = cb_obj.value;
            var new_hist = all_data[new_val];
            var hist = new_hist[0];
            var edges = new_hist[1];
            var left_edges = edges.slice(0, edges.length - 1);
            var right_edges = edges.slice(1, edges.length);
            
            var data = source.data;
            data["hist"] = hist;
            data["left_edges"] = left_edges;
            data["right_edges"] = right_edges;
            
            console.log(left_edges);
            console.log(right_edges);
            
            source.change.emit();
            
        """)

        dropdown.js_on_change("value", callback)

        output = row(
            histogram_plot, dropdown
        )

        return output

    @_stylize_attributes()
    def _create_histogram_plot(self, source):

        kwargs = {
            "plot_height": 460,
            "plot_width": 460
        }

        p = self._default_figure(kwargs)

        p.quad(top="hist", bottom=0, left="left_edges", right="right_edges", source=source, fill_color="#8CA8CD")
        return p

    def _create_dropdown(self, vals):
        d = Select(options=vals, css_classes=["features_dropdown"], name="features_dropdown")
        return d

    def _default_figure(self, plot_specific_kwargs=None):
        # supplying stylizing attributes to Figure() doesn't always work, as apparently some methods (e.g. grids)
        # require axes from a created figure to change something

        default_kwargs = {
            "tools": [],
        }

        if plot_specific_kwargs:
            default_kwargs.update(plot_specific_kwargs)

        p = figure(**default_kwargs)

        return p
