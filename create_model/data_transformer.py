from sklearn.pipeline import make_pipeline

class Transformer:
    """Wrapper for pipeline for transformations of input and output data."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def transform(self):
        pass
