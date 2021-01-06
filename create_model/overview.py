from bs4 import BeautifulSoup
import os


class Overview:

    def __init__(self, template, css_path, features, naive_mapping):
        self.template = template
        self.css = css_path
        self.features = features
        self.naive_mapping = naive_mapping

    def render(self, base_dict, tables, lists, figures, figure_directory):
        tables = self.__create_tables_html(tables)
        lists = self.__create_lists_html(lists)
        figures = self.__create_figures(figures, figure_directory)

        output_dict = {}
        for d in [tables, lists, figures, base_dict]:
            output_dict.update(d)

        # adding overview specific css
        output_dict["overview_css"] = self.css
        return self.template.render(**output_dict)

    def __create_tables_html(self, tables):
        params = {}
        for key, arg in tables.items():
            html_table = arg.to_html(float_format="{:.2f}".format)
            html_table = self.__append_description(html_table)
            params[key] = html_table

        return params

    def __append_description(self, html_table):
        if self.features.initialized:
            table = BeautifulSoup(html_table, "html.parser")
            headers = table.select("table tbody tr th")
            for header in headers:
                try:
                    description = self.features[header.string]
                except KeyError:
                    continue

                # adding <span> that will hold description of a feature
                # every \n is replaced with <br> tag
                header.string.wrap(table.new_tag("p"))
                new_tag = table.new_tag("span")
                lines = description.split("\n")
                new_tag.string = lines[0]
                if len(lines) > 1:
                    for line in lines[1:]:
                        new_tag.append(table.new_tag("br"))
                        new_tag.append(table.new_string("{}".format(line)))

                # appending mappings to descriptions as long as they exist (they are not none)
                mappings = self._consolidate_mappings(header.string)
                if mappings:
                    new_tag.append(table.new_tag("br"))
                    new_tag.append(table.new_tag("br"))
                    new_tag.append(table.new_string("Category - Original (Transformed)"))
                    i = 0
                    for key, val in mappings.items():
                        new_tag.append(table.new_tag("br"))
                        new_tag.append(table.new_string("{} - {}".format(key, val)))
                        if i >= 10:
                            new_tag.append(table.new_tag("br"))
                            new_tag.append(table.new_string("(...) Showing only first 10 categories"))
                            break
                        i += 1

                header.p.append(new_tag)
            return str(table)
        else:
            return html_table

    def _consolidate_mappings(self, feature):
        # description mapping comes from json so all keys are strings
        description_mapping = self.features.feature_mapping(feature)
        try:
            naive_mapping = self.naive_mapping[feature]
        except KeyError:
            naive_mapping = None

        if (description_mapping is None) and (naive_mapping is None):
            return ""

        # we're expecting naive mapping to be always present in case of categorical values
        converted_naive_mapping = {str(key): str(val) for key, val in naive_mapping.items()}
        if description_mapping is None:
            # changing to strings to accomodate for json keys
            new_pairs = converted_naive_mapping
        else:
            _ = {}
            for key in converted_naive_mapping:
                _[description_mapping[key]] = "{} ({})".format(key, converted_naive_mapping[key])
            new_pairs = _

        # app = ""
        # if len(new_pairs) > 10:
        #     new_pairs = dict(list(new_pairs.items())[:10])
        #     app = "Showing only first 10 categories"
        #
        # mapping_string += "<br>".join((" - ".join([str(key), str(val)]) for key, val in new_pairs.items()))
        # mapping_string += app

        return new_pairs

    def __create_lists_html(self, lists):
        d = {}

        # Was thinking of redesigning it with bs4, but its a very simple structure so it would be an overkill
        # TODO: redesign with bs4 and append descriptions
        for key, l in lists.items():
            _ = "<ul>"
            for x in l:
                _ += "<li>" + x + "</li>"
            _ += "</ul>"
            d[key] = _
        return d

    def __create_figures(self, figures, output_directory):
        d = {}
        for name, plot in figures.items():
            path = os.path.join(output_directory, (name + ".png"))
            d[name] = "<a href={path}><img src={path} title='Click to open larger version'></img></a>".format(path=path)
            plot.savefig(path)
        return d
