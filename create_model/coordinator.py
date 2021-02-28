from .explainer import DataExplainer, DataFeatures
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

        if root_path is None:
            self.root_path = os.getcwd()
        else:
            self.root_path = root_path

        self.features_descriptions = FeatureDescriptor(feature_json)
        self.features = DataFeatures(pd.concat([self.X, self.y], axis=1), self.y.name, self.features_descriptions)

        self.explainer = DataExplainer(self.features)
        # TODO: rethink if data_explained can be moved to .data_objects property of DataExplainer
        self.data_explained = self.explainer.analyze()
        #self.explainer_mapping = self.explainer.mapping
        self.mapping = self.features.mapping()



        # TODO: consider lazy instancing
        self.output = Output(self.root_path, descriptions=self.features_descriptions, naive_mapping=self.mapping, data_name="test", package_name=self.name)
        self.transformer = Transformer(self.X, self.y, self.data_explained["numerical"], self.data_explained["categorical"])
        self.scoring = scoring

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
        # TODO: change hardcoded keys
        explainer_data_keys = ["figures", "tables", "lists", "histograms", "scatter", "categorical"]
        explainer_data = {"explainer_" + key: self.data_explained[key] for key in explainer_data_keys}


        # transformer_data_keys = ["transformations"]
        # transformer_data = {"transformer_" + key: self.transformer.data_objects[key] for key in transformer_data_keys}
        #
        # output = {}
        # for _ in [explainer_data, transformer_data]:
        #     output.update(_)

        # TODO: change when more objects will provide their data_objects
        output = explainer_data

        self.output.create_html_output(output)
        print("Created output at {directory}".format(directory=self.output.output_directory))
