import os
from jinja2 import Environment, FileSystemLoader


class Output:

    def __init__(self, root_path):
        self.root_path = root_path
        self.output_directory = os.path.join(self.root_path, "output")
        self.templates_path = os.path.join(self.root_path, "create_model", "templates")
        self.env = Environment(loader=FileSystemLoader(self.templates_path))

    def create_html_output(self, output):
        params = self._analyze_data_output(output)
        template = self.env.get_template("test.html")

        params["Title"] = "Test Title"

        rendered = template.render(**params)
        with open(os.path.join(self.output_directory, "output.html"), "w") as f:
            f.write(rendered)

    def _analyze_data_output(self, output_dict):

        fig_dict = output_dict["figures"]
        html_dict = output_dict["html"]
        params = html_dict

        for fig in fig_dict:
            name = fig
            plot = fig_dict[fig]
            path = os.path.join(self.output_directory, "assets", (name + ".png"))
            params[name] = "<img src={path}></img>".format(path=path)
            plot.savefig(path)

        return params
