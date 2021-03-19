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

    # base template
    _base_css = "style.css"
    _time_format = "%d-%b-%Y %H:%M:%S"
    _footer_note = "Created on {time}"

    # views
    _view_overview = "overview"
    _view_overview_html = "overview.html"
    _view_overview_css = "overview.css"
    _view_features = "features"
    _view_features_html = "features.html"
    _view_features_css = "features.css"
    _view_features_js = "features.js"

    # directories
    _output_directory_name = "output"
    _static_directory_name = "static"
    _templates_directory_name = "templates"

    # CSS elements
    _feature_name_with_description_class = "feature-name-w-desc"

    def __init__(self, root_path, analyzer, package_name):

        self.analyzer = analyzer

        # TODO:
        # this solution is sufficient right now but nowhere near satisfying
        # if the Coordinator is imported as a package, this whole facade might crumble with directories
        # being created in seemingly random places.
        self.root_path = root_path
        self.output_directory = os.path.join(self.root_path, self._output_directory_name)
        self.templates_path = os.path.join(self.root_path, package_name, self._templates_directory_name)
        self.static_path = os.path.join(self.root_path, package_name, self._static_directory_name)
        self.env = Environment(loader=FileSystemLoader(self.templates_path))

        self.view_overview = Overview(
                    template=self.env.get_template(self._view_overview_html),
                    css_path=os.path.join(self.static_path, self._view_overview_css),
                    output_directory=self.output_directory,
                    max_categories=self.analyzer.max_categories,
                    feature_description_class=self._feature_name_with_description_class
                )

        self.view_features = FeatureView(
                    template=self.env.get_template(self._view_features_html),
                    css_path=os.path.join(self.static_path, self._view_features_css),
                    js_path=os.path.join(self.static_path, self._view_features_js),
                )

    def create_html(self):

        base_css = os.path.join(self.static_path, self._base_css)

        current_time = datetime.datetime.now().strftime(self._time_format)
        created_on = self._footer_note.format(time=current_time)

        hyperlinks = {
            self._view_overview: self._path_to_file(self._view_overview_html),
            self._view_features: self._path_to_file(self._view_features_html)
        }

        feature_list = self.analyzer.feature_list()
        first_feature = feature_list[0]

        overview_rendered = self.view_overview.render(
            base_css=base_css,
            creation_date=created_on,
            hyperlinks=hyperlinks,
            numerical_df=self.analyzer.numerical_describe_df(),
            categorical_df=self.analyzer.categorical_describe_df(),
            unused_features=self.analyzer.unused_features(),
            head_df=self.analyzer.df_head(),
            pairplot=self.analyzer.features_pairplot_static(),
            mapping=self.analyzer.features_mapping(),
            descriptions=self.analyzer.features_descriptions()
        )

        features_rendered = self.view_features.render(
            base_css=base_css,
            creation_date=created_on,
            hyperlinks=hyperlinks,
            histogram=self.analyzer.histogram(first_feature, self._feature_name_with_description_class),
            scatterplot=self.analyzer.scatterplot(first_feature, self._feature_name_with_description_class),
            feature_list=feature_list,
            first_feature=first_feature
        )

        rendered_templates = {
            self._view_overview_html: overview_rendered,
            self._view_features_html: features_rendered
        }

        for template_filepath, template in rendered_templates.items():
            self._write_html(template_filepath, template)

    def _write_html(self, template_filename, template):
        template_filepath = self._path_to_file(template_filename)
        with open(template_filepath, "w") as f:
            f.write(template)

    def _path_to_file(self, filename):
        return os.path.join(self.output_directory, filename)
