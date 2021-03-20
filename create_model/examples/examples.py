from sklearn.datasets import load_iris, load_boston, load_diabetes, load_digits, load_wine, load_breast_cancer
import pandas as pd


def iris():
    _ = load_iris(as_frame=True)

    X = _["data"]
    y = _["target"]

    X = X.rename(
        {
            "sepal width (cm)": "sepal_width",
            "sepal length (cm)": "sepal_length",
            "petal width (cm)": "petal_width",
            "petal length (cm)": "petal_length"
        },
        axis=1
    )

    descriptions = {
        "target": {
            "mapping": {
                0: "Iris-Setosa",
                1: "Iris-Versicolour",
                2: "Iris-Virginica"
            }
        }
    }

    return X, y, descriptions


def boston():
    _ = load_boston()

    X = pd.DataFrame(_["data"], columns=_["feature_names"])
    y = pd.Series(_["target"], name="MEDV")

    d = "description"

    descriptions = {
        "CRIM": {
            d: "per capita crime rate by town",
        },
        "ZN": {
            d: "proportion of residential land zoned for lots over 25,000 sq.ft."
        },
        "INDUS": {
            d: "proportion of non-retail business acres per town"
        },
        "CHAS": {
            d: "Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)"
        },
        "NOX": {
            d: "nitric oxides concentration (parts per 10 million)"
        },
        "RM": {
            d: "average number of rooms per dwelling"
        },
        "AGE": {
            d: "proportion of owner-occupied units built prior to 1940"
        },
        "DIS": {
            d: "weighted distances to five Boston employment centres"
        },
        "RAD": {
            d: "index of accessibility to radial highways"
        },
        "TAX": {
            d: "full-value property-tax rate per $10,000"
        },
        "PTRATIO": {
            d: "pupil-teacher ratio by town"
        },
        "B": {
            d: "1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town"
        },
        "LSTAT": {
            d: "% lower status of the population"
        },
        "MEDV": {
            d: "Median value of owner-occupied homes in $1000's"
        }
    }

    return X, y, descriptions


def diabetes():
    _ = load_diabetes(as_frame=True)

    X = _["data"]
    y = _["target"]

    d = "description"

    descriptions = {
        "age": {
            d: "age in years"
        },
        "bmi": {
            d: "body mass index"
        },
        "bp": {
            d: "average blood pressure"
        },
        "s1": {
            d: "tc, T-Cells (a type of white blood cells)"
        },
        "s2": {
            d: "ldl, low-density lipoproteins"
        },
        "s3": {
            d: "hdl, high-density lipoproteins"
        },
        "s4": {
            d: "tch, thyroid stimulating hormone"
        },
        "s5": {
            d: "ltg, lamotrigine"
        },
        "s6": {
            d: "glu, blood sugar level"
        }
    }

    return X, y, descriptions


def digits(n_class=10):
    _ = load_digits(n_class=n_class, as_frame=True)
    X = _["data"]
    y = _["target"]
    descriptions = None

    return X, y, descriptions


def wine():
    _ = load_wine(as_frame=True)

    X = _["data"]
    y = _["target"]
    descriptions = None

    return X, y, descriptions


def breast_cancer():
    _ = load_breast_cancer(as_frame=True)

    X = _["data"]
    y = _["target"]
    descriptions = None

    return X, y, descriptions