import os
import datetime
import pathlib
import pkgutil
import pandas as pd
from jinja2 import Environment, PackageLoader
from .views import Overview, FeatureView, ModelsViewClassification, ModelsViewRegression, ModelsViewMulticlass
from .plots import PairPlot, InfoGrid, ScatterPlotGrid, CorrelationPlot, NormalTransformationsPlots
from .plots import ModelsPlotClassification, ModelsPlotRegression, ModelsPlotMulticlass, ModelsDataTable
from .plot_design import PlotDesign
from .functions import make_pandas_data


class Output:
    """Class for producing HTML output.

    Output loads jinja2 templates from templates folder and provides them to Views object alongside all required
    plots, calculated data and dynamic references to static files to create HTML pages. Pages are linked together
    and can be navigated as regular HTML pages.

    Created HTML and static files are saved into the output directory. As links to pages/CSS/JS are relative, the whole
    output directory can be copied and moved around.

    Note:
        Plots for ModelsView are created during create_html method call in comparison to every other Plot object and
        are not included as instance attributes.

    Attributes:
        output_directory (str): directory where HTML output will be placed
        package_name (str): name of the data_dashboard package
        pre_transformed_columns (list): list of feature names that are already pre-transformed

        features (Features): Features object
        analyzer (Analyzer): Analyzer object
        transformer (Transformer): Transformer object
        model_finder (ModelFinder): ModelFinder object

        X_train (pandas.DataFrame): train split of X features
        X_test (pandas.DataFrame): test split of X features
        y_train (pandas.Series): train split of y target variable
        y_test (pandas.Series): test split of y target variable
        transformed_X_train (numpy.ndarray, scipy.csr_matrix): transformed X train split
        transformed_X_test (numpy.ndarray, scipy.csr_matrix): transformed X test split
        transformed_y_train (numpy.ndarray): transformed y train split
        transformed_y_test (numpy.ndarray): transformed y test split

        random_state (int, None): integer for reproducibility on fitting and transformations, defaults to None if not
            provided during __init__

        env (jinja2.Environment): jinja2 Environment to load HTML templates

        hyperlinks (dict): 'view name': view HTML path pairs (relative path)
        view_overview (Overview): Overview HTML Subpage View
        view_features (FeatureView): FeatureView HTML Subpage View
        view_models (ModelsView):  ModelsView HTML Subpage View (depending on problem type)
        plot_design (PlotDesign): PlotDesign object with predefined style elements

        pairplot (PairPlot): PairPlot object for Overview page visualizations
        infogrid (InfoGrid): InfoGrid object for FeatureView page visualizations
        correlation_plot (CorrelationPlot): CorrelationPlot object for FeatureView page visualizations
        scattergrid (ScatterPlotGrid): ScatterPlotGrid object for FeatureView page visualizations
        normal_transformations_plot (NormalTransformationsPlots): NormalTransformationsPlots object for FeatureView
            page visualizations
        models_data_table (ModelsDataTable): ModelsDataTable object for ModelsView page visualizations
    """

    # base template
    _base_css = "style.css"
    _time_format = "%d-%b-%Y %H:%M:%S"
    _logs_time_format = "%d%m%Y%H%M%S"
    _footer_note = "Created on {time}"

    # output structure
    _created_assets_directory = "assets"
    _created_static_directory = "static"
    _created_logs_directory = "logs"

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

    # directories
    _assets_directory_name = "assets"
    _static_directory_name = "static"
    _templates_directory_name = "templates"

    # logs
    _search_results_csv = "search.csv"
    _quicksearch_results_csv = "quicksearch.csv"
    _gridsearch_results_csv = "gridsearch.csv"

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
    ]

    def __init__(self,
                 output_directory,
                 package_name,
                 pre_transformed_columns,
                 features,
                 analyzer,
                 transformer,
                 model_finder,
                 X_train,
                 X_test,
                 y_train,
                 y_test,
                 transformed_X_train,
                 transformed_X_test,
                 transformed_y_train,
                 transformed_y_test,
                 random_state=None
                 ):
        """Create Output object.

        Set jinja2 Environment to load HTML templates. Create View objects and Plot objects that are needed for
        creating HTML output.

        Args:
            output_directory (str): directory where HTML output will be placed
            package_name (str): name of the data_dashboard package
            pre_transformed_columns (list): list of feature names that are already pre-transformed

            features (Features): Features object
            analyzer (Analyzer): Analyzer object
            transformer (Transformer): Transformer object
            model_finder (ModelFinder): ModelFinder object

            X_train (pandas.DataFrame): train split of X features
            X_test (pandas.DataFrame): test split of X features
            y_train (pandas.Series): train split of y target variable
            y_test (pandas.Series): test split of y target variable
            transformed_X_train (numpy.ndarray, scipy.csr_matrix): transformed X train split
            transformed_X_test (numpy.ndarray, scipy.csr_matrix): transformed X test split
            transformed_y_train (numpy.ndarray): transformed y train split
            transformed_y_test (numpy.ndarray): transformed y test split

            random_state (int, optional): integer for reproducibility on fitting and transformations, defaults to None
        """
        self.output_directory = output_directory
        self.package_name = package_name
        self.pre_transformed_columns = pre_transformed_columns

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

        # random state
        self.random_state = random_state

        # jinja2 Environment
        self.env = Environment(loader=PackageLoader(package_name, self._templates_directory_name))

        # Views
        self.hyperlinks = None
        self.view_overview = Overview(
                    template=self.env.get_template(self._view_overview_html),
                    css_path=(self._static_directory_name + "/" + self._view_overview_css),
                    feature_description_class=self._element_with_description_class
                )

        self.view_features = FeatureView(
                    template=self.env.get_template(self._view_features_html),
                    css_path=(self._static_directory_name + "/" + self._view_features_css),
                    js_path=(self._static_directory_name + "/" + self._view_features_js),
                    target_name=self.features.target,
                    pre_transformed_columns=self.pre_transformed_columns
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
            feature_description_class=self._element_with_description_class,  # class of hoverable description
            categorical_features=self.features.categorical_features(),  # categorical features including target
            feature_descriptions=self.features.descriptions(),  # descriptions of all features including target
            feature_mapping=self.features.mapping()  # mapping of variables in features
        )

        self.normal_transformations_plot = NormalTransformationsPlots(
            plot_design=self.plot_design
        )

        self.models_data_table = ModelsDataTable(
            plot_design=self.plot_design
        )

    def create_html(self, do_pairplots, do_logs):
        """Create HTML output.

        HTML output is put into output_directory attribute directory. HTML and static files are copied to the output
        directory in a predefined structure and relative paths are used as hyperlinks to join them across HTML pages.
        Necessary data for View objects and Plot objects is taken from features, analyzer, model_finder and transformer
        objects passed during initialization.

        Depending on do_pairplots flag, pairplot and scatterplot Plots might not be calculated and included in the HTML
        output.

        Depending on do_logs flag, .csv files of search results from model_finder might not be saved onto the output
        directory.

        Args:
            do_pairplots (bool): flag indicating if pairplot and scatterplot Plot objects should generate the
                appropriate Plots
            do_logs (bool): flag indicating if log files of different model_finder search results should be written
                down to the output directory
        """
        ###################################################
        # =============== Base Parameters =============== #
        ###################################################

        # base variables needed by every view
        base_css = (self._static_directory_name + "/" + self._base_css)  # relative path to static directory
        time_started = datetime.datetime.now()

        current_time = time_started.strftime(self._time_format)
        created_on = self._footer_note.format(time=current_time)

        self.hyperlinks = {
            self._view_overview: self._view_overview_html,
            self._view_features: self._view_features_html,
            self._view_models: self._view_models_html
        }

        # feature that will be chosen in the beginning
        feature_list = self.analyzer.feature_list()  # feature list is already sorted
        first_feature = feature_list[0]

        ############################################################
        # =============== Plots and Necessary Data =============== #
        ############################################################

        # seaborn pairplot
        if do_pairplots:
            generated_pairplot = self.pairplot.pairplot(
                dataframe=self.analyzer.features_pairplot_df()
            )
            pairplot_path = os.path.join(self.assets_path(), self._pairplot_name)
        else:
            generated_pairplot = None
            pairplot_path = None

        # InfoGrid
        generated_infogrid_summary = self.infogrid.summary_grid(
            summary_statistics=self.analyzer.summary_statistics(),
            histogram_data=self.analyzer.histogram_data(),
            initial_feature=first_feature
        )

        # CorrelationsPlot
        generated_correlation_plot = self.correlation_plot.correlation_plot(
            correlation_data_normalized=self.analyzer.correlation_data_normalized(random_state=self.random_state),
            correlation_data_raw=self.analyzer.correlation_data_raw()
        )

        # ScatterGrid
        if do_pairplots:
            generated_scattergrid = self.scattergrid.scattergrid(
                scatter_data=self.analyzer.scatter_data(),
                initial_feature=first_feature
            )
        else:
            generated_scattergrid = None

        # data needed for FeaturesView
        # train/test splits of X and y, numerical features only
        train_data = pd.concat([self.X_train, self.y_train], axis=1)[self.features.numerical_features()]
        test_data = pd.concat([self.X_test, self.y_test], axis=1)[self.features.numerical_features()]

        # histogram data for "normal" transformations of numerical features
        normal_transformations_hist_data = self.transformer.normal_transformations_histograms(train_data, test_data)
        generated_normal_transformations_plots = self.normal_transformations_plot.plots(
            normal_transformations_hist_data
        )

        # test split of original data
        original_test_df = pd.concat([self.X_test, self.y_test], axis=1)

        # test split of transformed data, with user-friendly column names
        tr_X = make_pandas_data(self.transformed_X_test, pd.DataFrame)
        tr_y = make_pandas_data(self.transformed_y_test, pd.Series)

        # accommodating for any pre-transformed columns with self.transformed_columns
        tr_X.columns = self.transformer.transformed_columns() + sorted(self.pre_transformed_columns)
        tr_y.name = self.features.target
        transformed_df = pd.concat([tr_X, tr_y], axis=1)

        # data needed for ModelsView
        models_right, models_left_bottom = self._models_plot_output(self.model_finder.problem)

        predicted_y = self.model_finder.predictions_X_test(self._view_models_model_limit)
        predictions_table = self.models_data_table.data_table(self.X_test, self.y_test, predicted_y)

        ###################################################
        # =============== Rendering Views =============== #
        ###################################################

        # Overview
        overview_rendered = self.view_overview.render(
            base_css=base_css,
            creation_date=created_on,
            hyperlinks=self.hyperlinks,
            numerical_df=self.analyzer.numerical_describe_df(),
            categorical_df=self.analyzer.categorical_describe_df(),
            unused_features=self.analyzer.unused_features(),
            head_df=self.analyzer.df_head(),
            do_pairplot_flag=do_pairplots,
            pairplot_path=(self._assets_directory_name + "/" + self._pairplot_name),
            mapping=self.analyzer.features_mapping(),
            descriptions=self.analyzer.features_descriptions()
        )

        # FeaturesView
        features_rendered = self.view_features.render(
            base_css=base_css,
            creation_date=created_on,
            hyperlinks=self.hyperlinks,
            feature_list=feature_list,
            summary_grid=generated_infogrid_summary,
            X_transformations=self.transformer.transformations(),
            y_transformations=self.transformer.y_transformations(),
            test_features_df=original_test_df,
            test_transformed_features_df=transformed_df,
            normal_transformations_plots=generated_normal_transformations_plots,
            correlations_plot=generated_correlation_plot,
            do_scatterplot_flag=do_pairplots,
            scatterplot=generated_scattergrid,
            numerical_features=self.features.numerical_features(),
            initial_feature=first_feature
        )

        # ModelsView
        models_rendered = self.view_models.render(
            base_css=base_css,
            creation_date=created_on,
            hyperlinks=self.hyperlinks,
            model_results=self.model_finder.search_results(self._view_models_model_limit),
            models_right=models_right,
            models_left_bottom=models_left_bottom,
            predictions_table=predictions_table
        )

        #################################################
        # =============== Writing Files =============== #
        #################################################

        self._create_output_directory()
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
        if do_pairplots:
            generated_pairplot.savefig(pairplot_path)

        # Logging
        if do_logs:
            self._write_logs(time_started)

    def static_path(self):
        """Return absolute path to the static folder in the output directory.

        Returns:
            str: static directory path in output directory
        """
        return os.path.join(self.output_directory, self._created_static_directory)

    def assets_path(self):
        """Return absolute path to the assets folder in the output directory.

        Returns:
            str: assets directory path in output directory
        """
        return os.path.join(self.output_directory, self._created_assets_directory)

    def logs_path(self):
        """Return absolute path to the logs folder in the output directory.

        Returns:
            str: logs directory path in output directory
        """
        return os.path.join(self.output_directory, self._created_logs_directory)

    def overview_file(self):
        """Return absolute path to the Overview HTML file in the output directory.

        Returns:
            str: Overview HTML file path
        """
        return os.path.join(self.output_directory, self._view_overview_html)

    def features_file(self):
        """Return absolute path to the FeatureView HTML file in the output directory.

        Returns:
            str: FeatureView HTML file path
        """
        return os.path.join(self.output_directory, self._view_features_html)

    def models_file(self):
        """Return absolute path to the ModelsView HTML file in the output directory.

        Returns:
            str: ModelsView HTML file path
        """
        return os.path.join(self.output_directory, self._view_models_html)

    def _create_output_directory(self):
        """Create output directory in case it doesn't exist.

        Include creation of parents directories if they don't exist as well.
        """
        pathlib.Path(self.output_directory).mkdir(exist_ok=True, parents=True)

    def _create_subdirectories(self):
        """Create directories for static and assets in case they don't exist.

        Logs directory is not included as it might not be needed based on flags provided to 'create_html' method.
        """
        # creating directories for static and assets files
        for directory_path in [self.static_path(), self.assets_path()]:
            pathlib.Path(directory_path).mkdir(exist_ok=True)

    def _write_html(self, template_filename, template):
        """Write template HTML content into HTML file in output directory based on template filename.

        template_filename is used to dynamically create HTML file path in output directory.

        Args:
            template_filename (str): HTML file name
            template (str): rendered HTML content
        """
        template_filepath = self._path_to_file(template_filename)
        with open(template_filepath, "w") as f:
            f.write(template)

    def _write_logs(self, time_started):
        """Write down .csv files of model_finder search results into logs directory in output directory.

        3 different search results are available in model_finder object:
            - search results
            - quicksearch results
            - gridsearch results

        Log .csv files are created in the new subdirectory for all of them as long as the result is not None.

        Args:
            time_started (datetime.datetime): start time of HTML Output creation
        """
        directory = self._create_logs_directory(time_started)
        mf = self.model_finder
        dfs = [mf.search_results(model_limit=None), mf.quicksearch_results(), mf.gridsearch_results()]
        names = [self._search_results_csv, self._quicksearch_results_csv, self._gridsearch_results_csv]

        for filename, df in zip(names, dfs):
            if df is not None:
                df.to_csv(os.path.join(directory, filename))

    def _create_logs_directory(self, time_started):
        """Create subdirectory in logs in output directory.

        New subdirectory is created in Logs with the formatted start time of the HTML output creation.

        Args:
            time_started (datetime.datetime): start time of HTML Output creation
        """
        directory = time_started.strftime(self._logs_time_format)
        logs_directory = os.path.join(self.logs_path(), directory)
        pathlib.Path(logs_directory).mkdir(parents=True, exist_ok=True)
        return logs_directory

    def _copy_static(self):
        """Copy static files to static folder in the output directory."""
        for f in self._static_files_names:  # predefined static files to copy
            f_copy = pkgutil.get_data(self.package_name, os.path.join(self._static_directory_name, f)).decode("utf-8")
            with open(os.path.join(self.static_path(), f), "w", newline="") as fw:
                fw.write(f_copy)

    def _path_to_file(self, filename):
        """Create output directory file path to provided filename.

        Returns:
            str: output directory file path
        """
        return os.path.join(self.output_directory, filename)

    def _models_view_creator(self, problem_type):
        """Create appropriate ModelsView based on a problem type.

        Following ModelsViews can be created:
            - classification: ModelsViewClassification
            - regression: ModelsViewRegression
            - multiclass: ModelsViewMulticlass

        Args:
            problem_type (str): problem type

        Returns:
            ModelsView: ModelsView specific to a given problem

        Raises:
            ValueError: when provided problem_type is incorrect
        """
        kwargs = {
            "template": self.env.get_template(self._view_models_html),
            "css_path": (self._static_directory_name + "/" + self._view_models_css),
            "params_name": self.model_finder.dataframe_params_name(),
            "model_with_description_class": self._element_with_description_class,
        }

        # Creating ModelsView based on the type of the ML problem
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
        """Create appropriate Plots and/or Data based on a problem type.

        Plot objects are instantiated and appropriately called based on a provided problem_type. Following ModelPlots
        can be created:
            - classification: ModelsPlotClassification
            - regression: ModelsPlotRegression
            - multiclass: ModelsPlotMulticlass

        Output Plot/data is not restricted to those Plot objects and can be obtained via different methods specific
        to a problem type.

        Args:
            problem_type (str): problem type

        Returns:
            tuple: (models_right Plot/Data, models_left_bottom Plot/Data)

        Raises:
            ValueError: when provided problem_type is incorrect
        """
        # Creating Plots based on type of ML problem
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
            mp = ModelsPlotMulticlass(
                plot_design,
                self.transformer.y_classes(),
                self.features[self.features.target].original_mapping  # original mapping for confusion matrices
            )
            models_right = None
            models_left_bottom = mp.confusion_matrices_plot(
                self.model_finder.confusion_matrices(self._view_models_model_limit)
            )

        else:
            raise ValueError("Incorrect problem type provided: {problem_type}".format(problem_type=problem_type))

        return models_right, models_left_bottom
