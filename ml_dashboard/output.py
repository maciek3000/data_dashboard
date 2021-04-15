import os, datetime
from jinja2 import Environment, FileSystemLoader
from .views import Overview, FeatureView, ModelsViewClassification, ModelsViewRegression, ModelsViewMulticlass
from .plots import PairPlot, InfoGrid, ScatterPlotGrid, CorrelationPlot, NormalTransformationsPlots
from .plots import ModelsPlotClassification, ModelsPlotRegression, ModelsPlotMulticlass, ModelsDataTable
from .plot_design import PlotDesign

import pandas as pd
import numpy as np
import shutil
import pathlib


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

    # output structure
    _created_assets_directory = "assets"
    _created_static_directory = "static"

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
    _pairplot_name = "pairplot.png"
    _static_files_names = [
        _base_css,
        _view_overview_css,
        _view_features_css,
        _view_features_js,
        _view_models_css,
        _view_models_js
    ]

    def __init__(self,
                 root_path, output_directory, package_name,
                 features, analyzer, transformer, model_finder,
                 X_train, X_test, y_train, y_test,
                 transformed_X_train, transformed_X_test, transformed_y_train, transformed_y_test
                 ):

        # objects needed to create the output
        self.features = features
        self.analyzer = analyzer
        self.transformer = transformer
        self.model_finder = model_finder

        # data used to create the output
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.transformed_X_train = transformed_X_train
        self.transformed_X_test = transformed_X_test
        self.transformed_y_train = transformed_y_train
        self.transformed_y_test = transformed_y_test

        # directory where the dashboard will be created
        self.output_directory = output_directory

        # TODO:
        # this solution is sufficient right now but nowhere near satisfying
        # if the Coordinator is imported as a package, this whole facade might crumble with directories
        # being created in seemingly random places.
        self.root_path = root_path
        self._templates_path = os.path.join(self.root_path, package_name, self._templates_directory_name)
        self._static_template_path = os.path.join(self.root_path, package_name, self._static_directory_name)
        self.env = Environment(loader=FileSystemLoader(self._templates_path))

        # Views
        self.view_overview = Overview(
                    template=self.env.get_template(self._view_overview_html),
                    css_path=os.path.join(self.static_path(), self._view_overview_css),
                    max_categories=self.analyzer.max_categories,
                    feature_description_class=self._element_with_description_class
                )

        self.view_features = FeatureView(
                    template=self.env.get_template(self._view_features_html),
                    css_path=os.path.join(self.static_path(), self._view_features_css),
                    js_path=os.path.join(self.static_path(), self._view_features_js),
                    target_name=self.features.target
                )

        self.view_models = self._models_view_creator(
            problem_type=self.model_finder.problem
        )

        # Plots
        self.plot_design = PlotDesign()

        self.pairplot = PairPlot(
            plot_design=self.plot_design
        )

        self.infogrid = InfoGrid(
            features=self.features.features(),  # list of features
            plot_design=self.plot_design,
            feature_description_class=self._element_with_description_class,  # class of hoverable description
        )

        self.correlation_plot = CorrelationPlot(
            plot_design=self.plot_design,
            target_name=self.features.target
        )

        self.scattergrid = ScatterPlotGrid(
            features=self.features.features(),  # list of features
            plot_design=self.plot_design,
            categorical_features=self.features.categorical_features(),  # categorical features including target
            feature_descriptions=self.features.descriptions(),  # descriptions of all features including target
            feature_mapping=self.features.mapping(),  # mapping of variables in features
            feature_description_class=self._element_with_description_class  # class of hoverable description
        )

        self.normal_transformations_plot = NormalTransformationsPlots(
            plot_design=self.plot_design
        )

        self.models_data_table = ModelsDataTable(
            plot_design=self.plot_design
        )

    def create_html(self):

        # base variables needed by every view
        base_css = os.path.join(self.static_path(), self._base_css)  # path to output_directory
        current_time = datetime.datetime.now().strftime(self._time_format)
        created_on = self._footer_note.format(time=current_time)
        hyperlinks = {
            self._view_overview: self._path_to_file(self._view_overview_html),
            self._view_features: self._path_to_file(self._view_features_html),
            self._view_models: self._path_to_file(self._view_models_html)
        }

        # feature that will be chosen in the beginning
        feature_list = self.analyzer.feature_list()
        first_feature = feature_list[0]

        # seaborn pairplot
        generated_pairplot = self.pairplot.pairplot(
            dataframe=self.analyzer.features_pairplot_df()
        )
        pairplot_path = os.path.join(self.assets_path(), (self._pairplot_name))

        # InfoGrid
        generated_infogrid_summary = self.infogrid.summary_grid(
            summary_statistics=self.analyzer.summary_statistics(),
            histogram_data=self.analyzer.histogram_data(),
            initial_feature=first_feature
        )

        # CorrelationsPlot
        generated_correlation_plot = self.correlation_plot.correlation_plot(
            correlation_data_normalized=self.analyzer.correlation_data_normalized(),
            correlation_data_raw=self.analyzer.correlation_data_raw()
        )

        # ScatterGrid
        generated_scattergrid = self.scattergrid.scattergrid(
            scatter_data=self.analyzer.scatter_data(),
            initial_feature=first_feature
        )

        # data needed for FeaturesView
        # train/test splits of X and y, numerical features only
        train_data = pd.concat([self.X_train, self.y_train], axis=1)[self.features.numerical_features()]
        test_data = pd.concat([self.X_test, self.y_test], axis=1)[self.features.numerical_features()]

        # histogram data for "normal" transformations of numerical features
        normal_transformations_hist_data = self.transformer.normal_transformations_histograms(train_data, test_data)
        generated_normal_transformations_plots = self.normal_transformations_plot.plots(normal_transformations_hist_data)

        # test split of original data
        original_test_df = pd.concat([self.X_test, self.y_test], axis=1)

        # test split of transformed data, with user-friendly column names
        transformed_df = pd.DataFrame(data=self.transformed_X_test, columns=self.transformer.transformed_columns())
        transformed_df = pd.concat(
            [transformed_df, pd.Series(self.transformed_y_test, name=self.features.target)],
            axis=1
        )

        # data needed for ModelsView
        models_right, models_left_bottom = self._models_plot_output(self.model_finder.problem)

        predicted_y = self.model_finder.predictions_X_test(self._view_models_model_limit)
        table = self.models_data_table.data_table(self.X_test, self.y_test, predicted_y)

        # Overview
        overview_rendered = self.view_overview.render(
            base_css=base_css,
            creation_date=created_on,
            hyperlinks=hyperlinks,
            numerical_df=self.analyzer.numerical_describe_df(),
            categorical_df=self.analyzer.categorical_describe_df(),
            unused_features=self.analyzer.unused_features(),
            head_df=self.analyzer.df_head(),
            pairplot_path=pairplot_path,
            mapping=self.analyzer.features_mapping(),
            descriptions=self.analyzer.features_descriptions()
        )

        # FeaturesView
        features_rendered = self.view_features.render(
            base_css=base_css,
            creation_date=created_on,
            hyperlinks=hyperlinks,
            summary_grid=generated_infogrid_summary,
            correlations_plot=generated_correlation_plot,
            scatterplot=generated_scattergrid,
            feature_list=feature_list,
            numerical_features=self.features.numerical_features(),
            test_features_df=original_test_df,
            test_transformed_features_df=transformed_df,
            X_transformations=self.transformer.transformations(),
            y_transformations=self.transformer.y_transformers(),
            normal_transformations_plots=generated_normal_transformations_plots,
            initial_feature=first_feature
        )

        # ModelsView
        models_rendered = self.view_models.render(
            base_css=base_css,
            creation_date=created_on,
            hyperlinks=hyperlinks,
            model_results=self.model_finder.search_results(self._view_models_model_limit),
            models_right=models_right,
            models_left_bottom=models_left_bottom,
            incorrect_predictions_table=table
        )

        # Writing files to the HDD

        self._create_subdirectories()

        # HTML files
        rendered_templates = {
            self._view_overview_html: overview_rendered,
            self._view_features_html: features_rendered,
            self._view_models_html: models_rendered
        }
        for template_filepath, template in rendered_templates.items():
            self._write_html(template_filepath, template)

        # CSS and JS files
        self._copy_static()

        # Plots
        generated_pairplot.savefig(pairplot_path)

    def static_path(self):
        return os.path.join(self.output_directory, self._created_static_directory)

    def assets_path(self):
        return os.path.join(self.output_directory, self._created_assets_directory)

    def _write_html(self, template_filename, template):
        template_filepath = self._path_to_file(template_filename)
        with open(template_filepath, "w") as f:
            f.write(template)

    def _create_subdirectories(self):
        # creating directories for static and assets files
        for directory_path in [self.static_path(), self.assets_path()]:
            pathlib.Path(directory_path).mkdir(exist_ok=True)

    def _copy_static(self):
        for f in self._static_files_names:
            shutil.copy(os.path.join(self._static_template_path, f), os.path.join(self.static_path(), f))

    def _path_to_file(self, filename):
        return os.path.join(self.output_directory, filename)

    def _models_view_creator(self, problem_type):

        kwargs = {
            "template": self.env.get_template(self._view_models_html),
            "css_path": os.path.join(self.static_path(), self._view_models_css),
            "js_path": os.path.join(self.static_path(), self._view_models_js),
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

        plot_design = self.plot_design

        if problem_type == self.model_finder._classification:
            mp = ModelsPlotClassification(plot_design)
            models_right = mp.models_comparison_plot(
                roc_curves=self.model_finder.roc_curves(self._view_models_model_limit),
                precision_recall_curves=self.model_finder.precision_recall_curves(self._view_models_model_limit),
                det_curves=self.model_finder.det_curves(self._view_models_model_limit),
                target_proportion=self.model_finder.test_target_proportion()
            )
            models_left_bottom = self.model_finder.confusion_matrices(self._view_models_model_limit)

        elif problem_type == self.model_finder._regression:
            mp = ModelsPlotRegression(plot_design)
            models_right = mp.prediction_error_plot(self.model_finder.prediction_errors(self._view_models_model_limit))
            models_left_bottom = mp.residual_plot(self.model_finder.residuals(self._view_models_model_limit))

        elif problem_type == self.model_finder._multiclass:
            mp = ModelsPlotMulticlass(plot_design, self.transformer.y_classes())
            models_right = None
            models_left_bottom = mp.confusion_matrices_plot(self.model_finder.confusion_matrices(self._view_models_model_limit))

        else:
            raise ValueError("Incorrect problem type provided: {problem_type}".format(problem_type=problem_type))

        return models_right, models_left_bottom
