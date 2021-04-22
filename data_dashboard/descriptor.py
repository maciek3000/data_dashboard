class FeatureDescriptor:
    """Provides external information on Features.

    Attributes:
        initialized (bool): bool flag indicating if the FeatureDescriptor was initialized with
            feature_descriptions_dict or not, defaults to False
        features_descriptions (dict): dictionary of Feature descriptions, categories and mappings that must follow
            a set structure

    Example:
        Example of features_descriptions attribute and how it should be structured:
            feature_name: {
                _description: 'description of the Feature',
                _category: 'category of the Feature',
                _mapping: {
                    '1st unique value in data': 'mapping1',
                    '2nd unique value in data': 'mapping2',
                    (...)
                    }
                }
    """
    _description = "description"
    _mapping = "mapping"
    _category = "category"

    def __init__(self, features_descriptions_dict):
        """Create FeatureDescriptor object. If features_descriptions_dict is not None, set initialized attribute to
        True.

        Args:
            features_descriptions_dict (dict): dictionary that should follow structure described in `class` docstring
        """
        self.initialized = False
        if features_descriptions_dict:
            self.features_descriptions = features_descriptions_dict
            self.initialized = True

    def mapping(self, arg):
        """Return _mapping attribute value from internal dictionary of arg in features_descriptions attribute. If
        _mapping key does not exist in internal dictionary, return None.

        Args:
            arg (str): Feature name, key in features_descriptions attribute dictionary

        Returns:
            {dict, None}: mapping dictionary of arg Feature or None if it doesn't exist

        Raises:
            KeyError: if arg is not in features_descriptions dictionary
        """
        if arg not in self.features_descriptions:
            raise KeyError

        try:
            return self.features_descriptions[arg][self._mapping]
        except KeyError:
            return None

    def description(self, arg):
        """Return _description attribute value from internal dictionary of arg in features_descriptions attribute. If
        _description key does not exist in internal dictionary, return None.

        Args:
            arg (str): Feature name, key in features_descriptions attribute dictionary

        Returns:
            {str, None}: description string of arg Feature or None if it doesn't exist

        Raises:
            KeyError: if arg is not in features_descriptions dictionary
        """
        if arg not in self.features_descriptions:
            raise KeyError

        try:
            return self.features_descriptions[arg][self._description]
        except KeyError:
            return None

    def category(self, arg):
        """Return _category attribute value from internal dictionary of arg in features_descriptions attribute. If
        _category key does not exist in internal dictionary, return None.

        Args:
            arg (str): Feature name, key in features_descriptions attribute dictionary

        Returns:
            {str, None}: category string of arg Feature or None if it doesn't exist

        Raises:
            KeyError: if arg is not in features_descriptions dictionary
        """
        if arg not in self.features_descriptions:
            raise KeyError

        try:
            return self.features_descriptions[arg][self._category]
        except KeyError:
            return None

    def keys(self):
        """Return keys from features_descriptions dictionary.

        Returns:
            dict_keys: keys from features_descriptions
        """
        return self.features_descriptions.keys()

    def __getitem__(self, arg):
        """Return arg item from features_descriptions attribute dictionary.

        Returns:
            dict: internal dictionary of arg from features_descriptions

        Raises:
             KeyError: if arg is not present in features_descriptions keys
        """
        if arg not in self.features_descriptions:
            raise KeyError

        return self.features_descriptions[arg]

    def __iter__(self):
        """Implement iterator protocol with features_descriptions.keys() as iterable."""
        self.__iter_items = list(sorted(self.features_descriptions.keys()))
        self.__counter = 0
        return self

    def __next__(self):
        """Implement iterator protocol with features_descriptions.keys() as iterable.

        Raises:
            StopIteration: on IndexError
        """
        try:
            self.__counter += 1
            return self.__iter_items[self.__counter - 1]
        except IndexError:
            raise StopIteration
