import json


class FeatureDescriptor(dict):
    """Provides metadata on features (known before any analysis takes place).

        Class reads json_file (must be already opened) and creates internal dictionary of it.
        obj[feature] syntax returns a description of a provided feature name.

        Defines different methods to extracting different information from JSON.
        As for now they are:
            description()
            mapping()
            category()

        JSON might provide different metadata, but every "key" in JSON should also be defined in the class
        (in case any changes are introduced to JSON). Keys should be included as ._property properties and
        corresponding methods for extracting them should also be defined.

        Including .initialized bool property as sometimes there might not be any corresponding json_file available.
    """

    _description = "description"
    _mapping = "mapping"
    _category = "category"

    def __init__(self, json_file):
        self.initialized = False
        if json_file:
            self.json = json.load(json_file)
            self.initialized = True
            super().__init__(self.json)

    def mapping(self, arg):
        if arg not in self.json:
            raise KeyError

        try:
            return self.json[arg][self._mapping]
        except KeyError:
            return None

    def description(self, arg):
        if arg not in self.json:
            raise KeyError

        try:
            return self.json[arg][self._description]
        except KeyError:
            return None

    def category(self, arg):
        if arg not in self.json:
            raise KeyError

        try:
            return self.json[arg][self._category]
        except KeyError:
            return None

    def __getitem__(self, arg):
        if arg not in self.json:
            raise KeyError

        return self.json[arg]

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
