import seaborn as sns
import os
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

class DataExplainer:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def explain(self):
        self.analyze()
        self.create_html()

    def analyze(self):
        plot = sns.pairplot(self.X)
        return plot.fig

    def create_html(self):
        print(os.path.join(os.getcwd(), "templates"))
        env = Environment(loader=FileSystemLoader(os.path.join(os.getcwd(), "create_model", "templates")))
        template = env.get_template("test.html")
        rendered = template.render(title="Test Title", content=self.X.to_html())
        output_directory = os.path.join(Path(os.getcwd()), "output")
        with open(os.path.join(output_directory, "output.html"), "w") as f:
            f.write(rendered)


