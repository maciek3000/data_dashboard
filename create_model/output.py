import os, datetime, copy
from jinja2 import Environment, FileSystemLoader


class Output:

    time_format = "%d-%b-%Y %H:%M:%S"
    footer_note = "Created on {time}"

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

    def create_html_output(self, data_objects):
        templates = {
            "overview": self._overview_template,
        }
        rendered_templates = []

        # params dict encompasses shared arguments for all templates
        # it's up to the template to decide which they want to use and which not
        params = self._get_standard_variables(data_objects)
        for template, method in templates.items():
            rendered_templates.append((template, method(copy.copy(params))))

        self._write_html(rendered_templates)

    def _get_standard_variables(self, data_objects):
        current_time = datetime.datetime.now().strftime(self.time_format)
        html_dict = self._convert_data_to_web_elements(data_objects)
        html_dict["created_on"] = self.footer_note.format(time=current_time)
        html_dict["base_css"] = os.path.join(self.static_path, "style.css")
        return html_dict

    def _overview_template(self, base_dict):
        base_dict["overview_css"] = os.path.join(self.static_path, "overview.css")
        template = self.env.get_template("overview.html")
        return template.render(**base_dict)

    def _write_html(self, html_templates):
        for name, html in html_templates:
            with open(os.path.join(self.output_directory, (name + ".html")), "w") as f:
                f.write(html)

    def _convert_data_to_web_elements(self, output_dict):
        # TODO: change to separate functions
        tables_dict = output_dict["tables"]
        params = {key: arg.to_html(float_format="{:.2f}".format) for key, arg in tables_dict.items()}

        list_dict = output_dict["lists"]
        # TODO: redesign
        for key, l in list_dict.items():
            _ = ""
            for x in l:
                _ += "<li>" + x + "</li>"
            list_dict[key] = _
        params.update(list_dict)

        fig_dict = output_dict["figures"]
        for name, plot in fig_dict.items():
            path = os.path.join(self.output_directory, "assets", (name + ".png"))
            params[name] = "<img src={path}></img>".format(path=path)
            plot.savefig(path)

        return params
