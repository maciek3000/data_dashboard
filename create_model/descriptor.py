import json


class FeatureDescriptor(dict):
    """Provides Descriptions and optional mapping of Features.

        Class reads json_file (must be already opened) and creates internal dictionary of it.
        obj[feature] syntax returns a description of a provided feature name.
        .feature_mapping(feature) returns mapping of categories to data of a given feature.
    """

    _description = "description"
    _mapping = "mapping"

    def __init__(self, json_file):
        self.initialized = False
        if json_file:
            self.json = json.load(json_file)
            self.initialized = True
            super().__init__(self.json)

    def feature_mapping(self, arg):
        if arg not in self.json:
            raise KeyError

        try:
            return self.json[arg][self._mapping]
        except KeyError:
            return None

    def __getitem__(self, arg):
        if arg not in self.json:
            raise KeyError

        return self.json[arg][self._description]

    def __iter__(self):
        self.__iter_items = list(sorted(self.json.keys()))
        self.__counter = 0
        return self

    def __next__(self):
        try:
            self.__counter += 1
            return self.__iter_items[self.__counter - 1]
        except IndexError:
            raise StopIteration

    # def keys(self):
    #     return self.json.keys()
