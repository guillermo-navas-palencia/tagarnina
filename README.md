<p align="center">
  <img src="https://raw.githubusercontent.com/guillermo-navas-palencia/tagarnina/main/branding/tagarnina.png" alt="tagarnina logo" width="300">
</p>

[![PyPI version](https://img.shields.io/pypi/v/tagarnina.svg)](https://pypi.org/project/tagarnina/)
[![CI](https://github.com/guillermo-navas-palencia/tagarnina/actions/workflows/ci.yml/badge.svg)](https://github.com/guillermo-navas-palencia/tagarnina/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/guillermo-navas-palencia/tagarnina/blob/main/LICENSE)

**tagarnina** is a Python library for feature selection with user-defined constraints. It implements constrained recursive feature elimination (CRFE), where a constraint solver decides which features can be removed at each step, and cross-validated CRFE (CRFECV).

**Algorithms**

- `CRFE`: recursive feature elimination with constraints. Without constraints it behaves like scikit-learn's `RFE`. Strategies: `"probing"` (default) and `"greedy"`.
- `CRFECV`: `CRFE` with cross-validation to choose the number of features.

**Constraints** (in `tagarnina.optimization.constraints`)

- `MaxCorrelationConstraint`: limits the pairwise correlation between selected features.
- `GroupConstraint`, `AtMostKConstraint`, `AtLeastKConstraint`: bound the number of selected features within a group.
- `MinMaxConstraint`: bounds the total number of selected features.
- `OnlyEnforceIfConstraint`: if `feature1` is selected, `feature2` must be selected.
- `OnlyExcludeIfConstraint`: if `feature1` is selected, `feature2` cannot be selected.
- `FixedConstraint`: always selects a feature.
- `AllOrNoneConstraint`: selects all or none of a set of features.
- `KnapsackConstraint`: limits the weighted sum of the selected features.

## Table of Contents

- [Installation](#installation)
- [Dependencies](#dependencies)
- [Getting Started](#getting-started)
- [Example](#example)
- [Citation](#citation)
- [License](#license)

## Installation

`tagarnina` requires Python 3.12 or newer. Install the latest release from PyPI:

```bash
pip install tagarnina
```

To install from a local clone of the repository instead:

```bash
git clone https://github.com/guillermo-navas-palencia/tagarnina.git
cd tagarnina
pip install .
```

To install the development tools (tests, linting, type checking):

```bash
pip install ".[dev]"
```

To build the documentation and run the examples (adds Sphinx, matplotlib, and SHAP):

```bash
pip install ".[docs]"
```

## Dependencies

Installing the package installs its required dependencies:

- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [OR-Tools](https://developers.google.com/optimization) (CP-SAT solver)
- [scikit-learn](https://scikit-learn.org/)

Optional extras:

- `dev`: pytest, pytest-cov, ruff, mypy
- `docs`: sphinx, pydata-sphinx-theme, numpydoc, sphinx-gallery, matplotlib, shap

## Getting Started

- **Documentation and API reference:** https://gnpalencia.org/tagarnina/
- **Examples:** https://gnpalencia.org/tagarnina/auto_examples/index.html

## Example

The example uses the [Ames Housing](https://www.openml.org/d/43926) dataset from OpenML, fetched with `fetch_openml`. It contains 2,930 house sales with 80 features. The target is the continuous sale price, `Sale_Price`, and there are no missing values. The task selects 20 features under two constraints:

- `AllOrNoneConstraint`: one-hot encoded columns of each categorical feature are selected together or not at all.
- `MaxCorrelationConstraint`: no two selected numerical features have an absolute Pearson correlation above 0.7.

```python
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

from tagarnina.crfe import CRFE
from tagarnina.optimization.constraints import (
    AllOrNoneConstraint,
    Constraints,
    MaxCorrelationConstraint,
)

X, y = fetch_openml(data_id=43926, as_frame=True, return_X_y=True, parser="liac-arff")

categorical_features = X.select_dtypes("category").columns.tolist()
numerical_features = X.select_dtypes("number").columns.tolist()

print(
    f"{len(categorical_features)} categorical and {len(numerical_features)} numerical features"
)
```

```text
46 categorical and 34 numerical features
```

One-hot encode the categorical features:

```python
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                sparse_output=False,
                feature_name_combiner=lambda feature, category: f"{feature}={category}"
         ),
         make_column_selector(dtype_include="category"))
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)
X_transformed = pd.DataFrame(
    data=preprocessor.fit_transform(X),
    columns=preprocessor.get_feature_names_out(),
)

print(X_transformed.shape)
```

```text
(2930, 352)
```

Create the all-or-none and maximum correlation constraints:

```python
all_or_none_constraints = [
    AllOrNoneConstraint(
        features=[c for c in X_transformed.columns if c.startswith(f"{feature}=")]
    )
    for feature in categorical_features
]

max_corr_constraint = MaxCorrelationConstraint(
    features=numerical_features,
    max_corr=0.7,
    absolute=True,
    method="pearson",
)

constraints = Constraints(
    constraints=[
        max_corr_constraint,
        *all_or_none_constraints
    ]
)
```

Use random forest regressor as an estimator. Run CRFE with elimination step=0.1 (10% of the features) and R^2 regression score function.

```python
estimator = RandomForestRegressor(max_depth=10, min_samples_leaf=20, random_state=0)

crfe = CRFE(
    estimator=estimator,
    constraints=constraints,
    step=0.1,
    n_features_to_select=20,
    scoring="r2",
    verbose=2,
)
crfe.fit(X_transformed, y)
```

With `verbose=2`, the output begins with a banner, the settings, the constraints, and the solver and model details. It then shows a table with one row per iteration and ends with a summary. The last lines are the selected feature names.

Output of the example above (timings vary between machines):

```text
----------------------------------------------------------------------------------------------------------
                                             tagarnina v0.1.0                                             
                                Constrained Recursive Feature Elimination                                 
                          (c) Guillermo Navas-Palencia, 2026, Apache License 2.0                          
----------------------------------------------------------------------------------------------------------
Estimator    : RandomForestRegressor
Data         : 2930 samples, 352 features
Settings     : step=0.1, n_features_to_select=20, strategy=probing, scoring=r2
Constraints  : 47 (MaxCorrelation 1, AllOrNone 46)
Solver       : CP-SAT 9.15.6755, workers=1, time_limit=10s, seed=0
Model        : 704 variables, 628 constraints, built in 0.875s

----------------------------------------------------------------------------------------------------------
 iter   n_features   removed     probes    fit(s)   solve(s)        score   retained       gap  elapsed(s)
----------------------------------------------------------------------------------------------------------
    0          352        78        0|0     4.533      0.041   8.9217e-01     0.9918    0.0082       5.580
    1          274        55        0|0     3.742      0.027   8.9030e-01     1.0000    0.0000       9.434
    2          219        41        0|0     3.324      0.027   8.8920e-01     1.0000    0.0000      12.880
    3          178        57        0|0     3.371      0.021   8.8920e-01     0.9794    0.0206      16.360
    4          121        49        0|0     2.631      0.028   8.8471e-01     0.9739    0.0261      19.126
    5           72        41        0|0     2.651      0.020   8.7967e-01     1.0000    0.0000      21.893
    6           31        11        0|0     1.716      0.014   8.7942e-01     1.0000    0.0000      23.704
    7           20         -          -     1.399          -   8.7879e-01          -         -      25.149
----------------------------------------------------------------------------------------------------------

Termination  : reached the target number of features
Selected     : 20 of 352 features
Probes       : 0 feasible, 0 infeasible, 0 blocked
Time         : wall 25.151s (fit 23.366, solve 0.178, score 0.446), cpu 24.078s
==========================================================================================================
```

Show the selected features:
```python
selected_features = crfe.get_feature_names_out()
print(selected_features)
```

```text
['Central_Air=N' 'Central_Air=Y' 'Lot_Area' 'Year_Built' 'Year_Remod_Add'
 'Mas_Vnr_Area' 'BsmtFin_SF_1' 'Bsmt_Unf_SF' 'Total_Bsmt_SF'
 'Second_Flr_SF' 'Gr_Liv_Area' 'Bsmt_Full_Bath' 'Full_Bath'
 'Bedroom_AbvGr' 'Fireplaces' 'Garage_Cars' 'Wood_Deck_SF' 'Open_Porch_SF'
 'Longitude' 'Latitude']
```

Count the selected categorical and numerical features, including both `Central_Air` columns.

```python
selected_categorical = []
for feature in categorical_features:
    columns = [c for c in X_transformed.columns if c.startswith(f"{feature}=")]
    n_selected = sum(c in selected_features for c in columns)
    if n_selected:
        selected_categorical.append(feature)
        print(f"{feature}: {n_selected} of {len(columns)} columns selected")

selected_numerical = [c for c in numerical_features if c in selected_features]

print("categorical selected:", len(selected_categorical))
print("numerical selected:", len(selected_numerical))
```

```text
Central_Air: 2 of 2 columns selected
categorical selected: 1
numerical selected: 18
```

## Citation

If you use `tagarnina` in academic work, please cite it:

```bibtex
@software{navas_palencia_tagarnina,
  author  = {Navas-Palencia, Guillermo},
  title   = {tagarnina: Python library for feature selection with user-defined constraints},
  year    = {2026},
  url     = {https://github.com/guillermo-navas-palencia/tagarnina},
}
```

## License

Distributed under the Apache License 2.0. See [LICENSE](https://github.com/guillermo-navas-palencia/tagarnina/blob/main/LICENSE) for details.