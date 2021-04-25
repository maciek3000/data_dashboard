class PlotDesign:
    """Simple Container of hardcoded style elements to be used across all modules.

    Attributes:
        text_color (str): text color in hex
        text_font (str): text font
        pairplot_color (str): pairplot color in hex
        fill_color (str): fill color of plot elements in hex
        base_color_tints (list): list of tints (hex) for a base color (main color is first, lighter tints follow)
        contrary_color_tints (list): list of tints (hex) for a contrary  color (main color is first,
            lighter tints follow)
        models_color_tuple (tuple): 2 element tuple of colors for Models View - contrary_color, tint of base color
        models_dummy_color (str): color for the dummy Model in hex
        contrary_half_color_tints (list): list of custom contrary color tints used in Models View Confusion Matrices
        base_half_color_tints (list): list of custom base color tints used in Models View Confusion Matrices
        contrary_color_linear_palette (list): list of contrary color tints used in Scatter Plot Linear Legend
    """
    def __init__(self):
        """Create PlotDesign object."""
        self.text_color = "#8C8C8C"
        self.text_font = "Lato"
        self.pairplot_color = "#19529c"
        self.fill_color = "#8CA8CD"

        self.base_color_tints = [
            "#19529c",
            "#3063a6",
            "#4775b0",
            "#5e86ba",
            "#7597c4",
            "#8ca9ce",
            "#a3bad7",
            "#bacbe1",
            "#d1dceb",
            "#e8eef5"
        ]

        self.contrary_color_tints = [
            "#9c2b19",
            "#a64030",
            "#b05547",
            "#ba6b5e",
            "#c48075",
            "#ce958c",
            "#d7aaa3",
            "#e1bfba",
            "#ebd5d1",
            "#f5eae8"
        ]

        self.models_color_tuple = (self.contrary_color_tints[0], self.base_color_tints[4])
        self.models_dummy_color = "#ABABAB"

        self.contrary_half_color_tints = [
            "#c48075",
            "#d09991",
            "#dcb3ac",
            "#e7ccc8",
            "#f3e6e3"
        ]

        self.base_half_color_tints = [
            "#7597c4",
            "#91acd0",
            "#acc1dc",
            "#c8d5e7",
            "#e3eaf3"
        ]

        self.contrary_color_linear_palette = ["#FFF7F3", "#FFB695", "#EB6F54", "#9C2B19"]
