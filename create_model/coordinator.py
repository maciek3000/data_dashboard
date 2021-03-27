from .analyzer import Analyzer
from .features import Features
from .output import Output
from .transformer import Transformer
from .model_finder import ModelFinder
from .descriptor import FeatureDescriptor
import os
import pandas as pd

class Coordinator:
    """Main object of the package.

        The role of this class is to analyze the provided data (in terms of summary statistics, visualizing any
        "easy" correlations, showing any potential new features, etc.), to transform the data for Machine Learning
        models and then train and find the best model it could find, additionally providing all the metadata on it.
        The end-goal is to create a series of static HTML files, which can be accessed like a dashboard to easily
        navigate through them in search for any potential insights in the machine learning pipeline.

        Coordinator encompasses all the classes that are used in creating the analysis and finding a model of
        given data. Coordinator exposes several functions that might be called separately (if there is a need
        for just a singular task, e.g. creating a dashboard for analyzing the data), but it also exposes a
        .full_analysis() function which tries to build the whole analysis from the ground up.
    """

    _name = "create_model"
    _output_created_text = "Created output at {directory}"
    _model_found_text = "Model: {name}\nScore: {score}\nParams: {params}"

    def __init__(self, X, y, output_directory, scoring=None, feature_descriptions_dict=None, root_path=None, random_state=None):

        # copy original dataframes to avoid changing the originals
        self.X = X.copy()
        self.y = y.copy()

        self.transformed_X = None
        self.transformed_y = None

        self.scoring = scoring

        if root_path is None:
            self.root_path = os.getcwd()
        else:
            self.root_path = root_path

        self.features_descriptions = FeatureDescriptor(feature_descriptions_dict)
        self.features = Features(self.X, self.y, self.features_descriptions)
        self.analyzer = Analyzer(self.features)

        self.transformer = Transformer(self.features)
        self.transformer.fit(self.X)
        self.transformer.fit_y(self.y)
        self.model_finder = ModelFinder(
            X=self.transformer.transform(self.X),
            y=self.transformer.transform_y(self.y),
            target_type=self.features[self.features.target].type.lower(),
            random_state=random_state
        )

        self.output = Output(self.root_path, output_directory, analyzer=self.analyzer, package_name=self._name)

    def find_and_fit(self, models=None, scoring=None, mode="quick", random_state=None):
        clf = self.model_finder.find_and_fit(models, scoring, mode)
        return clf

    def set_and_fit(self, model):
        self.model_finder.set_and_fit(model)

    def predict(self, X):
        transformed = self.transformer.transform(X)
        output = self.model_finder.predict(transformed)
        return output

    def create_dashboard(self):
        self.output.create_html()
        print(self._output_created_text.format(directory=self.output.output_directory))

    # exposed method in case only transformation is needed
    def transform(self, X):
        return self.transformer.transform(X)

