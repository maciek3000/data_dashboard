import os, datetime
from jinja2 import Environment, FileSystemLoader
from .views import Overview
from .views import FeatureView


class Output:
    """Class for producing HTML output.

        Creates several Views (Subpages) and joins them together to form an interactive WebPage/Dashboard.
        Every view is static - there is no server communication between html template and provided data. Every
        possible interaction is created with CSS/HTML or JS. This was a conscious design choice - albeit much slower
        than emulating server interaction, files can be easily shared between parties of interest.
        Keep in mind that basically all the data and the calculations are embedded into the files - if you'd wish
        to post them on the server you have to be aware if the data itself can be posted for public scrutiny.

        Output class uses jinja2 templates to provide them to Views, which are later populated by them with
        adequate plots/calculations/text.
        Output additionally defines filepaths to created HTML files and a bunch of standard variables used across
        all templates.
    """

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
