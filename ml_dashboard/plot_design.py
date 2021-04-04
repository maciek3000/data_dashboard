class PlotDesign:

    def __init__(self):
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
        self.models_dummy_color = "#DCDCDC"