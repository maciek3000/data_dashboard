

class DataExplainer:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def explain(self):
        self.analyze()
        self.create_html()

    def analyze(self):

        for col in self.X.columns:
            pass

    def create_html(self):
        pass

