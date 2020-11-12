import os, datetime, copy
from jinja2 import Environment, FileSystemLoader
from .output_overview import Overview
from .output_features import FeatureView


class Output:

    time_format = "%d-%b-%Y %H:%M:%S"
    footer_note = "Created on {time}"

    def __init__(self, root_path, features, naive_mapping, data_name, package_name):

        # TODO:
        # this solution is sufficient right now but nowhere near satisfying
        # if the Coordinator is imported as a package, this whole facade might crumble with directories
        # being created in seemingly random places.
        self.root_path = root_path
        self.features = features
        self.naive_mapping = naive_mapping

        self.output_directory = os.path.join(self.root_path, "output")
        self.templates_path = os.path.join(self.root_path, package_name, "templates")
        self.static_path = os.path.join(self.root_path, package_name, "static")
        self.env = Environment(loader=FileSystemLoader(self.templates_path))

        # TODO: rethink if the feature mapping should be done in overview / every other view or done here once
        self.overview = Overview(
            template=self.env.get_template("overview.html"),
            css_path=os.path.join(self.static_path, "overview.css"),
            features=self.features,
            naive_mapping=self.naive_mapping
        )

        self.feature_view = FeatureView(
            template=self.env.get_template("features.html"),
            css_path=os.path.join(self.static_path, "features.css"),
            js_path=os.path.join(self.static_path, "features.js"),
            features=self.features,
            naive_mapping=self.naive_mapping
        )

    def create_html_output(self, data_objects):
        # extracting different objects from data_objects that are needed for different templates

        # Overview elements
        tables = data_objects["explainer_tables"]
        lists = data_objects["explainer_lists"]
        figures = data_objects["explainer_figures"]

        histograms = data_objects["explainer_histograms"]
        scatter = data_objects["explainer_scatter"]

        # figure directory is needed for views to save figures if they need to
        figure_directory = os.path.join(self.output_directory, "assets")

        # base params include dynamic items for the base of templates
        base_params = self._get_standard_variables()

        overview = "overview"
        feature_view = "features"

        views = [overview, feature_view]
        view_paths = self._paths_to_views(views)

        base_params.update(view_paths)

        rendered_templates = {
            overview: self.overview.render(copy.copy(base_params), tables, lists, figures, figure_directory),
            feature_view: self.feature_view.render(copy.copy(base_params), histograms, scatter),
        }

        self._write_html(rendered_templates, view_paths)

    def _get_standard_variables(self):
        current_time = datetime.datetime.now().strftime(self.time_format)
        html_dict = {
            "created_on": self.footer_note.format(time=current_time),
            "base_css": os.path.join(self.static_path, "style.css")
        }
        return html_dict

    def _write_html(self, html_templates, html_file_paths):
        for name in html_templates:
            with open(html_file_paths[name + "_file"], "w") as f:
                f.write(html_templates[name])

    def _paths_to_views(self, views):
        _ = {}
        for view in views:
            _[(view + "_file")] = os.path.join(self.output_directory, (view + ".html"))
        return _
