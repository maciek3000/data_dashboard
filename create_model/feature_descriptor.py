import json


class FeatureDescriptor:
    """Provides Descriptions and optional mapping of Features.

        Class reads json_file (must be already opened) and creates internal dictionary of it.
        obj[feature] syntax returns a description of a provided feature name.
        .feature_mapping(feature) returns mapping of categories to data of a given feature.
    """

    _description = "description"
    _mapping = "mapping"

    def __init__(self, json_file):
        self.json = json.load(json_file)

    def __getitem__(self, arg):
        if arg not in self.json:
            raise KeyError

        return self.json[arg][self._description]

    def feature_mapping(self, arg):
        if arg not in self.json:
            raise KeyError

        try:
            return self.json[arg][self._mapping]
        except KeyError:
            return None
