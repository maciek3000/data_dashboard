import os, datetime
from jinja2 import Environment, FileSystemLoader
from .views import Overview, FeatureView, ModelsViewClassification, ModelsViewRegression, ModelsViewMulticlass
from .plots import PairPlot, InfoGrid, ScatterPlotGrid, CorrelationPlot
from .plots import ModelsPlotClassification, ModelsPlotRegression, ModelsPlotMulticlass, ModelsDataTable
from .plot_design import PlotDesign

import pandas as pd


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
    _view_models = "models"
    _view_models_html = "models.html"
    _view_models_css = "models.css"
    _view_models_js = "models.js"

    # directories
    _static_directory_name = "static"
    _templates_directory_name = "templates"

    # CSS elements
    _element_with_description_class = "elem-w-desc"

    # view specific properties
    _view_models_model_limit = 3

    def __init__(self, root_path, output_directory, package_name, features, analyzer, transformer, model_finder, X_transformed, y_transformed, X_test, y_test):

        self.features = features
        self.analyzer = analyzer
        self.transformer = transformer
        self.model_finder = model_finder

        self.X_transformed = X_transformed
        self.y_transformed = y_transformed
        self.X_test = X_test
        self.y_test = y_test

        # TODO:
        # this solution is sufficient right now but nowhere near satisfying
        # if the Coordinator is imported as a package, this whole facade might crumble with directories
        # being created in seemingly random places.
        self.root_path = root_path
        self.templates_path = os.path.join(self.root_path, package_name, self._templates_directory_name)
        self.static_path = os.path.join(self.root_path, package_name, self._static_directory_name)
        self.env = Environment(loader=FileSystemLoader(self.templates_path))

        self.output_directory = output_directory

        self.view_overview = Overview(
                    template=self.env.get_template(self._view_overview_html),
                    css_path=os.path.join(self.static_path, self._view_overview_css),
                    output_directory=self.output_directory,
                    max_categories=self.analyzer.max_categories,
                    feature_description_class=self._element_with_description_class
                )

        self.view_features = FeatureView(
                    template=self.env.get_template(self._view_features_html),
                    css_path=os.path.join(self.static_path, self._view_features_css),
                    js_path=os.path.join(self.static_path, self._view_features_js),
                    target_name=self.features.target
                )

        self.view_models = self._models_view_creator(self.model_finder.problem)

        self.plot_design = PlotDesign()
        self.pairplot = PairPlot(self.plot_design)

        self.infogrid = InfoGrid(
            features=self.features.features(),
            plot_design=self.plot_design,
            feature_description_class=self._element_with_description_class,
        )
        self.correlation_plot = CorrelationPlot(
            plot_design=self.plot_design,
            target_name=self.features.target
        )

        self.scattergrid = ScatterPlotGrid(
            features=self.features.features(),
            plot_design=self.plot_design,
            categorical_features=self.features.categorical_features(),
            feature_descriptions=self.features.descriptions(),
            feature_mapping=self.features.mapping(),
            feature_description_class=self._element_with_description_class
        )

    def create_html(self):

        base_css = os.path.join(self.static_path, self._base_css)

        current_time = datetime.datetime.now().strftime(self._time_format)
        created_on = self._footer_note.format(time=current_time)

        hyperlinks = {
            self._view_overview: self._path_to_file(self._view_overview_html),
            self._view_features: self._path_to_file(self._view_features_html),
            self._view_models: self._path_to_file(self._view_models_html)
        }

        feature_list = self.analyzer.feature_list()
        first_feature = feature_list[0]

        generated_pairplot = self.pairplot.pairplot(self.analyzer.features_pairplot_df())
        generated_infogrid_summary = self.infogrid.summary_grid(
            summary_statistics=self.analyzer.summary_statistics(),
            histogram_data=self.analyzer.histogram_data(),
            initial_feature=first_feature
        )

        generated_correlation_plot = self.correlation_plot.correlation_plot(
            correlation_data_normalized=self.analyzer.correlation_data_normalized(),
            correlation_data_raw=self.analyzer.correlation_data_raw()
        )

        generated_scattergrid = self.scattergrid.scattergrid(self.analyzer.scatter_data(), first_feature)

        # TODO: transformed is sometimes csrmatrix, sometimes array
        transformed_df = pd.DataFrame(data=self.X_transformed.toarray(), columns=self.transformer.transformed_columns())
        transformed_df = pd.concat([transformed_df, pd.Series(self.y_transformed, name=self.features.target)], axis=1)

        overview_rendered = self.view_overview.render(
            base_css=base_css,
            creation_date=created_on,
            hyperlinks=hyperlinks,
            numerical_df=self.analyzer.numerical_describe_df(),
            categorical_df=self.analyzer.categorical_describe_df(),
            unused_features=self.analyzer.unused_features(),
            head_df=self.analyzer.df_head(),
            pairplot=generated_pairplot,
            mapping=self.analyzer.features_mapping(),
            descriptions=self.analyzer.features_descriptions()
        )

        features_rendered = self.view_features.render(
            base_css=base_css,
            creation_date=created_on,
            hyperlinks=hyperlinks,
            summary_grid=generated_infogrid_summary,
            correlations_plot=generated_correlation_plot,
            scatterplot=generated_scattergrid,
            feature_list=feature_list,
            features_df=self.features.raw_data().head(),
            transformed_features_df=transformed_df.head(),
            X_transformations=self.transformer.transformations(),
            y_transformations=self.transformer.y_transformers(),
            first_feature=first_feature
        )

        models_right, models_left_bottom = self._models_plot_output(self.model_finder.problem)

        predicted_y = self.model_finder.predictions_X_test(self._view_models_model_limit)
        table = ModelsDataTable(self.plot_design).data_table(self.X_test, self.y_test, predicted_y)

        models_rendered = self.view_models.render(
            base_css=base_css,
            creation_date=created_on,
            hyperlinks=hyperlinks,
            model_results=self.model_finder.search_results(self._view_models_model_limit),
            models_right=models_right,
            models_left_bottom=models_left_bottom,
            incorrect_predictions_table=table
        )

        rendered_templates = {
            self._view_overview_html: overview_rendered,
            self._view_features_html: features_rendered,
            self._view_models_html: models_rendered
        }

        for template_filepath, template in rendered_templates.items():
            self._write_html(template_filepath, template)

    def _write_html(self, template_filename, template):
        template_filepath = self._path_to_file(template_filename)
        with open(template_filepath, "w") as f:
            f.write(template)

    def _path_to_file(self, filename):
        return os.path.join(self.output_directory, filename)

    def _models_view_creator(self, problem_type):

        kwargs = {
            "template": self.env.get_template(self._view_models_html),
            "css_path": os.path.join(self.static_path, self._view_models_css),
            "js_path": os.path.join(self.static_path, self._view_models_js),
            "params_name": self.model_finder.dataframe_params_name(),
            "model_with_description_class": self._element_with_description_class,
        }

        if problem_type == self.model_finder._classification:
            mv = ModelsViewClassification

        elif problem_type == self.model_finder._regression:
            mv = ModelsViewRegression

        elif problem_type == self.model_finder._multiclass:
            mv = ModelsViewMulticlass

        else:
            raise ValueError("Incorrect problem type provided: {problem_type}".format(problem_type=problem_type))

        return mv(**kwargs)

    def _models_plot_output(self, problem_type):

        pd = self.plot_design

        if problem_type == self.model_finder._classification:
            mp = ModelsPlotClassification(pd)
            models_right = mp.models_comparison_plot(
                roc_curves=self.model_finder.roc_curves(self._view_models_model_limit),
                precision_recall_curves=self.model_finder.precision_recall_curves(self._view_models_model_limit),
                det_curves=self.model_finder.det_curves(self._view_models_model_limit),
                target_proportion=self.model_finder.test_target_proportion()
            )
            models_left_bottom = self.model_finder.confusion_matrices(self._view_models_model_limit)

        elif problem_type == self.model_finder._regression:
            mp = ModelsPlotRegression(pd)
            models_right = mp.prediction_error_plot(self.model_finder.prediction_errors(self._view_models_model_limit))
            models_left_bottom = mp.residual_plot(self.model_finder.residuals(self._view_models_model_limit))

        elif problem_type == self.model_finder._multiclass:
            mp = ModelsPlotMulticlass(pd, self.transformer.y_classes())
            models_right = None
            models_left_bottom = mp.confusion_matrices_plot(self.model_finder.confusion_matrices(self._view_models_model_limit))

        else:
            raise ValueError("Incorrect problem type provided: {problem_type}".format(problem_type=problem_type))

        return models_right, models_left_bottom
