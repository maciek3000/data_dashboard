import os, datetime, copy
from jinja2 import Environment, FileSystemLoader
from bs4 import BeautifulSoup


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

        tables = self.__create_tables_html(output_dict["tables"])
        lists = self.__create_lists_html((output_dict["lists"]))
        figures = self.__create_figures(output_dict["figures"])
        html = [tables, lists, figures]

        params = {}
        for d in html:
            params.update(d)  # might consider checking for duplicate keys

        return params

    def __create_tables_html(self, tables):
        params = {}
        for key, arg in tables.items():
            html_table = arg.to_html(float_format="{:.2f}".format)
            html_table = self.__append_description(html_table)
            params[key] = html_table

        return params

    def __create_lists_html(self, lists):
        d = {}

        # Was thinking of redesigning it with bs4, but its a very simple structure so it would be an overkill
        for key, l in lists.items():
            _ = "<ul>"
            for x in l:
                _ += "<li>" + x + "</li>"
            _ += "</ul>"
            d[key] = _
        return d

    def __create_figures(self, figures):
        d = {}
        for name, plot in figures.items():
            path = os.path.join(self.output_directory, "assets", (name + ".png"))
            d[name] = "<a href={path}><img src={path} title='Click to open larger version'></img></a>".format(path=path)
            plot.savefig(path)
        return d

    def __append_description(self, html_table):

        placeholder_names = ["Age", "Fare", "PassengerId", "Cabin", "Embarked"]
        placeholder_desc = " - Test Description"

        table = BeautifulSoup(html_table, "html.parser")
        headers = table.select("table tbody tr th")
        for header in headers:
            if header.string in placeholder_names:
                header.string.wrap(table.new_tag("p"))
                new_tag = table.new_tag("span", class_="hover-box")
                new_tag.string = placeholder_desc
                header.p.append(new_tag)
        return str(table)
