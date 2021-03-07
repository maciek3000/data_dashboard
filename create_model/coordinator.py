from .analyzer import Analyzer
from .features import Features
from .output import Output
from .transformer import Transformer
from .model_finder import ModelFinder
from .descriptor import FeatureDescriptor
import pandas as pd

import os


class Coordinator:

    name = "create_model"

    def __init__(self, X, y, scoring=None, feature_json=None, root_path=None):

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

        self.features_descriptions = FeatureDescriptor(feature_json)
        self.features = Features(self.X, self.y, self.features_descriptions)
        self.analyzer = Analyzer(self.features)

        # self.mapping = self.features.mapping()

        self.transformer = Transformer(self.X, self.y,
                                       self.features.numerical_features(drop_target=True),
                                       self.features.categorical_features(drop_target=True))

        self.output = Output(self.root_path, analyzer=self.analyzer, data_name="test", package_name=self.name)

    def find_model(self):
        if not self.transformed_X:
            self.transformer = self.transformer.fit()
            self.transformed_X = self.transformer.transform()

        model_finder = ModelFinder(self.transformed_X, self.y, scoring=self.scoring, random_state=2862)

        model, score, params = model_finder.find_best_model()
        print("Model: {}\nScore: {}\nParams: {}".format(model.__name__, score, params))

        return model(**params).fit(self.transformed_X, self.y)

    def transform(self, X):
        return self.transformer.transform(X)

    def create_html(self):
        # # TODO: change hardcoded keys
        # data_explained = self.analyzer.analyze()
        # explainer_data_keys = ["figures", "tables", "lists", "histograms", "scatter", "categorical"]
        # explainer_data = {"explainer_" + key: data_explained[key] for key in explainer_data_keys}
        #
        #
        # # transformer_data_keys = ["transformations"]
        # # transformer_data = {"transformer_" + key: self.transformer.data_objects[key] for key in transformer_data_keys}
        # #
        # # output = {}
        # # for _ in [explainer_data, transformer_data]:
        # #     output.update(_)
        #
        # # TODO: change when more objects will provide their data_objects
        # output = explainer_data
        #
        # self.output.create_html_output(output)

        self.output.create_html()
        print("Created output at {directory}".format(directory=self.output.output_directory))
