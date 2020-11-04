from .data_explainer import DataExplainer
from .output import Output
from .data_transformer import Transformer
from .model_finder import ModelFinder
from .feature_descriptor import FeatureDescriptor

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

        self.explainer = DataExplainer(self.X, self.y)
        self.data_explained = self.explainer.analyze()
        self.explainer_mapping = self.explainer.mapping

        self.features = FeatureDescriptor(feature_json)

        # TODO: consider lazy instancing
        self.output = Output(self.root_path, features=self.features, naive_mapping=self.explainer_mapping, data_name="test", package_name=self.name)
        self.transformer = Transformer(self.X, self.y, self.data_explained["columns"]["columns_without_target"])
        self.scoring = scoring

    def eda(self):
        output_keys = ["figures", "tables", "lists"]
        output = {key: self.data_explained[key] for key in output_keys}
        self.output.create_html_output(output)
        print("Created output at {directory}".format(directory=self.output.output_directory))

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
