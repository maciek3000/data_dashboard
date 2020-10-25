import os, datetime
from jinja2 import Environment, FileSystemLoader


class Output:

    def __init__(self, root_path, data_name, package_name):

        # TODO:
        # this solution is sufficient right now but nowhere near satisfying
        # if the Coordinator is imported as a package, this whole facade might crumble with directories
        # being created in seemingly random places.
        self.root_path = root_path
        self.output_directory = os.path.join(self.root_path, "output")
        self.templates_path = os.path.join(self.root_path, package_name, "templates")
        self.static_path = os.path.join(self.root_path, package_name, "static")
        self.env = Environment(loader=FileSystemLoader(self.templates_path))

    def create_html_output(self, output):
        params = self._analyze_data_output(output)
        template = self.env.get_template("overview.html")

        params["base_css"] = os.path.join(self.static_path, "style.css")
        time = datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        params["created_on"] = "Created on {time}".format(time=time)

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
