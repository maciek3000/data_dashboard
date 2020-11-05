

class FeatureView:

    def __init__(self, template, css_path):
        self.template = template
        self.css = css_path

    def render(self, base_dict):

        output_dict = {}
        output_dict.update(base_dict)
        output_dict["feature_css"] = self.css
        return self.template.render(**output_dict)
