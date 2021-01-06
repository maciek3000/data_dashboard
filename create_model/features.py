from .plots import InfoGrid, ScatterPlotGrid


class FeatureView:

    def __init__(self, template, css_path, js_path, features, naive_mapping):
        # TODO: get path to templates folder instead of a instanced template

        self.template = template
        self.css = css_path
        self.js = js_path
        self.features = features
        self.naive_mapping = naive_mapping

        self.chosen_feature = None

        self.info_grid = InfoGrid(features)
        self.scatter_plot_grid = ScatterPlotGrid(features)

    def render(self, base_dict, histogram_data, scatter_data, categorical_columns):
        self.chosen_feature = sorted(self.features.keys())[0]
        output_dict = {}
        output_dict.update(base_dict)

        output_dict["chosen_feature"] = self.chosen_feature
        output_dict["features_css"] = self.css
        output_dict["features_js"] = self.js

        output_dict["features_menu"] = self._create_features_menu()

        info_grid_script, info_grid_div = self.info_grid.create_grid_elements(histogram_data, self.chosen_feature)
        output_dict["bokeh_script_info_grid"] = info_grid_script
        output_dict["info_grid"] = info_grid_div

        scatter_plot_grid_script, scatter_plot_grid_div = self.scatter_plot_grid.create_grid_elements(
            scatter_data, categorical_columns, self.chosen_feature)
        output_dict["scatter_plot_grid"] = scatter_plot_grid_div
        output_dict["bokeh_script_scatter_plot_grid"] = scatter_plot_grid_script

        return self.template.render(**output_dict)

    def _create_features_menu(self):
        html = "<div class='features-menu-title'><div>Features</div><div class='close-button'>x</div></div>"
        template = "<div class='single-feature'>{:03}. {}</div>"
        i = 0
        for feat in self.features:
            html += template.format(i, feat)
            i += 1
        return html
