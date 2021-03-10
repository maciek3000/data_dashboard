from .analyzer import Analyzer
from .features import Features
from .output import Output
from .transformer import Transformer
from .model_finder import ModelFinder
from .descriptor import FeatureDescriptor
import os


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

    name = "create_model"

    def __init__(self, X, y, scoring=None, feature_descriptions_dict=None, root_path=None):

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
        self.output.create_html()
        print("Created output at {directory}".format(directory=self.output.output_directory))

    def full_analysis(self):
        raise NotImplementedError
