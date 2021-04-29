# data-dashboard

## Short description
Creates a simple static HTML dashboard with provided ``X``, ``y`` data to help 
users see what's going on in their data, assists in making decisions regarding 
features and finds the best "baseline" Machine Learning Model.

![data_dashboard](./docs/source/screenshots/dashboard.gif)

## Longer Description

### Installation and Usage

``data_dashboard`` library allows you to build HTML Dashboard visualizing not only the data and relationships between
features but also automatically search for the best 'baseline' sklearn compatible Model.


You can install the package via pip:

```pip install data-dashboard```


To create a Dashboard you need the data: ``X``, ``y`` and the ``output_directory`` where the HTML files will be placed.
You can use toy datasets from ``examples`` (e.g. ``iris`` dataset) included in the library as well:

```python
from data_dashboard import Dashboard
from data_dashboard.examples import iris
output_directory = "your_path/dashboard_output"
X, y, descriptions = iris()

# descriptions is additional argument described further in docs
dsh = Dashboard(X, y, output_directory, descriptions)
dsh.create_dashboard()
```
Created HTML Dashboard will contain 3 subpages for you to look:

  * Overview with summary statistics of the data
  * Features where you can dig deeper into each feature in the data
  * Models showing search results and models performances

### Description

``data_dashboard`` aims to help you in those initial moments when you get your hands on new data and you ask yourself
a question "Now what?". 

Instead of going through Jupyter Notebook tables and visualizations, ``data_dashboard``
gathers all that information in one place in a user-friendly interface. What is more, automated 'baseline' sklearn Model
is created - you can then adjust not only parameters and algorithms, but also see how your changes affect the
performance.

There is also an educational twist to ``data_dashboard``, as explanations on correlations and transformations might 
alleviate pains that beginner Data Scientists might encounter (e.g. what do Transformers really do with my features?). 
Furthermore, provided visualizations might also assist in manual feature engineering.

Last but not least, ``data_dashboard`` tries to put more emphasis on the design of both HTML and Visualization 
elements. If you are not the biggest fan of default matplotlib plots, then you can give ``data_dashboard`` a try - 
perhaps the styling will suit your taste!

### Documentation

Documentation can be found here: <https://data-dashboard.readthedocs.io/>

### Author

MIT License Copyright (c) 2021 Maciej Dowgird

For contact use: [dowgird.maciej@gmail.com](mailto:dowgird.maciej@gmail.com)
