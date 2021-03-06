import os, datetime
from jinja2 import Environment, FileSystemLoader
from .views import Overview
from .views import FeatureView


class Output:
    time_format = "%d-%b-%Y %H:%M:%S"
    footer_note = "Created on {time}"

    view_overview_filename = "overview"
    view_features_filename = "features"

    def __init__(self, root_path, analyzer, data_name, package_name):

        self.analyzer = analyzer
        # self.naive_mapping = self.analyzer.features.mapping()  # TODO

        # TODO:
        # this solution is sufficient right now but nowhere near satisfying
        # if the Coordinator is imported as a package, this whole facade might crumble with directories
        # being created in seemingly random places.
        self.root_path = root_path
        self.output_directory = os.path.join(self.root_path, "output")
        self.templates_path = os.path.join(self.root_path, package_name, "templates")
        self.static_path = os.path.join(self.root_path, package_name, "static")
        self.env = Environment(loader=FileSystemLoader(self.templates_path))

        self.view_overview = Overview(
                    template=self.env.get_template("overview.html"),
                    css_path=os.path.join(self.static_path, "overview.css"),
                    output_directory=self.output_directory
                )

        self.view_features = FeatureView(
                    template=self.env.get_template("features.html"),
                    css_path=os.path.join(self.static_path, "features.css"),
                    js_path=os.path.join(self.static_path, "features.js"),
                )

    def create_html(self):

        base_css = os.path.join(self.static_path, "style.css")

        current_time = datetime.datetime.now().strftime(self.time_format)
        created_on = self.footer_note.format(time=current_time)

        hyperlinks = self._views_hyperlinks()

        feature_list = self.analyzer.feature_list()
        first_feature = sorted(feature_list)[0]


        overview_rendered = self.view_overview.render(
            base_css=base_css,
            creation_date=created_on,
            hyperlinks=hyperlinks,
            numerical_df=self.analyzer.numerical_describe_df(),
            categorical_df=self.analyzer.categorical_describe_df(),
            unused_features=self.analyzer.unused_features(),
            head_df=self.analyzer.df_head(),
            pairplot=self.analyzer.features_pairplot_static(),
            features=self.analyzer.features  # features object needed for descriptions/mappings
        )

        features_rendered = self.view_features.render(
            base_css=base_css,
            creation_date=created_on,
            hyperlinks=hyperlinks,
            histogram=self.analyzer.histogram(first_feature),
            scatterplot=self.analyzer.scatterplot(first_feature),
            feature_list=feature_list,
            first_feature=first_feature
        )

        rendered_templates = {
            self.view_overview_filename: overview_rendered,
            self.view_features_filename: features_rendered
        }

        for template_filename, template in rendered_templates.items():
            self._write_html(template_filename, template)

    def _views_hyperlinks(self):
        views_links = {}
        for filename in [self.view_overview_filename, self.view_features_filename]:
            views_links[filename] = self._path_to_file(filename)
        return views_links

    def _write_html(self, template_filename, template):
        template_filepath = self._path_to_file(template_filename)
        with open(template_filepath, "w") as f:
            f.write(template)

    def _path_to_file(self, filename):
        return os.path.join(self.output_directory, (filename + ".html"))


# class Output:
#
#     time_format = "%d-%b-%Y %H:%M:%S"
#     footer_note = "Created on {time}"
#
#     def __init__(self, root_path, features, data_name, package_name):
#
#         # TODO:
#         # this solution is sufficient right now but nowhere near satisfying
#         # if the Coordinator is imported as a package, this whole facade might crumble with directories
#         # being created in seemingly random places.
#         self.root_path = root_path
#         self.features = features
#         self.naive_mapping = features.mapping()  # TODO
#
#         self.output_directory = os.path.join(self.root_path, "output")
#         self.templates_path = os.path.join(self.root_path, package_name, "templates")
#         self.static_path = os.path.join(self.root_path, package_name, "static")
#         self.env = Environment(loader=FileSystemLoader(self.templates_path))
#
#         # TODO: rethink if the feature mapping should be done in overview / every other view or done here once
#         self.overview = Overview(
#             template=self.env.get_template("overview.html"),
#             css_path=os.path.join(self.static_path, "overview.css"),
#             features=self.features,
#             naive_mapping=self.naive_mapping
#         )
#
#         self.feature_view = FeatureView(
#             template=self.env.get_template("features.html"),
#             css_path=os.path.join(self.static_path, "features.css"),
#             js_path=os.path.join(self.static_path, "features.js"),
#             features=self.features,
#             naive_mapping=self.naive_mapping
#         )
#
#     def create_html_output(self, data_objects):
#         # extracting different objects from data_objects that are needed for different templates
#
#         # Overview elements
#         # TODO: change hardcoded keys
#         tables = data_objects["explainer_tables"]
#         lists = data_objects["explainer_lists"]
#         figures = data_objects["explainer_figures"]
#
#         histograms = data_objects["explainer_histograms"]
#         scatter = data_objects["explainer_scatter"]
#         categorical_columns = data_objects["explainer_categorical"]
#
#         # figure directory is needed for views to save figures if they need to
#         figure_directory = os.path.join(self.output_directory, "assets")
#
#         # base params include dynamic items for the base of templates
#         base_params = self._get_standard_variables()
#
#         overview = "overview"
#         feature_view = "features"
#
#         views = [overview, feature_view]
#         view_paths = self._paths_to_views(views)
#
#         base_params.update(view_paths)
#
#         rendered_templates = {
#             overview: self.overview.render(copy.copy(base_params), tables, lists, figures, figure_directory),
#             feature_view: self.feature_view.render(copy.copy(base_params), histograms, scatter, categorical_columns),
#         }
#
#         self._write_html(rendered_templates, view_paths)
#
#     def _get_standard_variables(self):
#         current_time = datetime.datetime.now().strftime(self.time_format)
#         html_dict = {
#             "created_on": self.footer_note.format(time=current_time),
#             "base_css": os.path.join(self.static_path, "style.css")
#         }
#         return html_dict
#
#     def _write_html(self, html_templates, html_file_paths):
#         for name in html_templates:
#             with open(html_file_paths[name + "_file"], "w") as f:
#                 f.write(html_templates[name])
#
#     def _paths_to_views(self, views):
#         _ = {}
#         for view in views:
#             _[(view + "_file")] = os.path.join(self.output_directory, (view + ".html"))
#         return _
