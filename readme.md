# data-dashboard

## Short description
Creates a simple static HTML dashboard with provided X, y data to help 
users see what's going on in their data, help make decisions regarding 
features and finds the best "baseline" model to predict y.

### Instructions

You can install the package via pip:

``pip install data-dashboard``

To make it work, you need to have the data loaded in a format of:
- X: data on which predictions will happen
- y: target feature
- descriptions *(Optional)*: dict-like collection, that will be described
later, can be left as None

If you don't have any data handy, you can use predefined datasets in
`examples.py`*`:

```python
from data_dashboard.examples.examples import iris

# All examples that you can use are:
# iris - multiclass
# boston - regression
# diabetes - regression
# digits - multiclass
# wine - multiclass
# breast_cancer - classification

X, y, descriptions = iris()
```

With data loaded into memory, you are able to proceed with `Dashboard`:

```python
# importing Dashboard
from data_dashboard.dashboard import Dashboard
import os

# define an output directory where the HTML files will be created
output_path = os.path.join(os.getcwd(), "output")

# create an instance of a Dashboard
dsh = Dashboard(output_path, X, y, descriptions)

# create HTML Dashboard with default arguments
dsh.create_dashboard()
```

HTML Dashboard will be created in the defined output directory. Dashboard
can be created for **classification**, **regression** and **multiclass** 
problems.

#### Dashboard is not able right now to deal with multi-label problems.

`Dashboard` object can be customized depending on the data that you have:

```python
Dashboard(
    
    output_directory,  # directory where HTML dashboard will be placed
    
    X,  # data without target, preferably pandas DataFrame
    y,  # target data, preferably pandas Series,
    feature_descriptions_dict=None  # optional dict with descriptions of features

    random_state=None,  # integer representing random state for repeatable results
                 
    classification_pos_label=None,  # one of the labels in target that will
    # be forced as a positive label
    force_classification_pos_label_multiclass=None,  # forcing label in target
    # in a multiclass problem, making it a binary classification problem
    
    already_transformed_columns=None  # list of columns that are already transformed

)
```

`create_dashboard()` first searches for the 'best' model across predefined
set of default models and creates HTML dashboard only after finding it.
Function can take `scoring` argument (which should be a metric function
from sklearn) which will be used to evaluate models. If `scoring` is `None`, 
then the default metric (for a particular problem) is used.

Depending on provided arguments, search can happen in 4 different ways:
- if `models` are provided as a sequence of instantiated Models, then
  each Model is fitted on the train part of the data and score is 
  calculated on test set of the data.
- if `models` are provided as a `dictionary` of Model: param_grid pairs, then:
    - if `mode` == `quick` then each Model is instantiated with default 
    parameters (similar to `LazyPredict` package), score is evaluated and 
      only few of the best scoring models are then GridSearched (with 
      HalvingGridSearch)
      
    - if `mode` == `detailed` then all Models are GridSearched with provided
    grid_params.
      
- if `models` is `None`, then default models (for a particular problem) are used,
again depending on provided `mode` (either instantiated with default params in 
  `quick` and then only some of them are GridSearched or all of them being 
  GridSearched in a `detailed` mode.)
  
At the end, the best model (depending on the `scoring`) is chosen.


```python

dsh.create_dashboard(
    
    models=None,  # can be sequence of instantiated Models, dict of 
    # Model: param_grid pairs or None
    
    scoring=None,  # should be a sklearn metric function
    mode='quick',  # either 'quick' or 'detailed'
    logging=True,  # turning logging (search results) on/off
    
    disable_pairplots=False,  # turning pairplots on/off as this is 
    # a potential bottleneck of the application
    force_pairplot=False  # forcing pairplots when Dashboard decided
    # to turn them off (when there are too many features in X).
)
```

### Known Issues/Drawbacks

- Multi-label classification is not included
- Pairplots are turned off when the number of features crosses a threshold 
of 15, to prevent any MemoryErrors and save time on visualizations that 
  degrade in the usefulness when the # of features increases
  
- Features HTML page might be laggy depending on the # of features
- CSS might be wonky on some resolutions
