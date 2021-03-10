
class FeatureDescriptor:
    """Provides metadata on features (known before any analysis takes place).

        features_descriptions_dict must be a dict with all the metadata for the features, where key is the name
        of the feature. Item of the appropriate feature should be another dictionary, which will contain such
        information as description, mapping of variables within the feature or the category:
            key (feature): dict()
        Keys for the internal dictionary are defined in class properties - this isn't the most intuitive way of
        doing that, but should be "rigid" enough to force some sort of a structure in the dictionary. It might
        be changed in the future, but there are no plans for it at the moment.

        obj[feature] syntax returns a description of a provided feature name.

        Defines different methods to extracting different information from features dictionary.
        As for now they are:
            description()
            mapping()
            category()

        Including .initialized bool property as sometimes there might not be any corresponding descriptions available.
    """

    _description = "description"
    _mapping = "mapping"
    _category = "category"

    def __init__(self, features_descriptions_dict):
        self.initialized = False
        if features_descriptions_dict:
            self.features_descriptions = features_descriptions_dict
            self.initialized = True

    def mapping(self, arg):
        if arg not in self.features_descriptions:
            raise KeyError

        try:
            return self.features_descriptions[arg][self._mapping]
        except KeyError:
            return None

    def description(self, arg):
        if arg not in self.features_descriptions:
            raise KeyError

        try:
            return self.features_descriptions[arg][self._description]
        except KeyError:
            return None

    def category(self, arg):
        if arg not in self.features_descriptions:
            raise KeyError

        try:
            return self.features_descriptions[arg][self._category]
        except KeyError:
            return None

    def __getitem__(self, arg):
        if arg not in self.features_descriptions:
            raise KeyError

        return self.features_descriptions[arg]

    def __iter__(self):
        self.__iter_items = list(sorted(self.features_descriptions.keys()))
        self.__counter = 0
        return self

    def __next__(self):
        try:
            self.__counter += 1
            return self.__iter_items[self.__counter - 1]
        except IndexError:
            raise StopIteration

    def keys(self):
        return self.features_descriptions.keys()
