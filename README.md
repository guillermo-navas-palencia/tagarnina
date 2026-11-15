<p align="center">
  <img src="branding/tagarnina.png" alt="tagarnina logo" width="300">
</p>

[![CI](https://github.com/guillermo-navas-palencia/tagarnina/actions/workflows/ci.yml/badge.svg)](https://github.com/guillermo-navas-palencia/tagarnina/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

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

`tagarnina` requires Python 3.12 or newer. From a local clone of the repository:

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
- **Examples:** the [examples/](examples/) folder contains sphinx-gallery scripts covering
  basic and advanced constraints, greedy versus probing strategies, and SHAP-based
  importance. See [examples/README.rst](examples/README.rst) for an overview.

## Example

The example uses the [Ames Housing](https://www.openml.org/d/43926) dataset from OpenML, fetched with `fetch_openml`. It contains 2,930 house sales with 80 features, 46 of them categorical (for example `Neighborhood` and `Central_Air`). The target is the continuous sale price, `Sale_Price`, and there are no missing values. The task selects 20 features under two constraints:

- `AllOrNoneConstraint`: the one-hot encoded columns of each categorical feature are selected together or not at all.
- `MaxCorrelationConstraint`: no two selected numerical features have an absolute Pearson correlation above 0.7.

```python
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor

from tagarnina.crfe import CRFE
from tagarnina.optimization.constraints import (
    AllOrNoneConstraint,
    Constraints,
    MaxCorrelationConstraint,
)

X, y = fetch_openml(data_id=43926, as_frame=True, return_X_y=True, parser="liac-arff")

categorical_features = X.select_dtypes(exclude="number").columns.tolist()
numerical_features = X.select_dtypes(include="number").columns.tolist()
X = pd.get_dummies(X, columns=categorical_features, prefix_sep="=", dtype=int)
groups = {
    feature: [c for c in X.columns if c.startswith(f"{feature}=")]
    for feature in categorical_features
}
numerical_features = [
    c for c in X.columns if not any(c in cols for cols in groups.values())
]

constraints = Constraints(
    constraints=[
        *[AllOrNoneConstraint(features=cols) for cols in groups.values()],
        MaxCorrelationConstraint(
            features=numerical_features,
            max_corr=0.7,
            absolute=True,
            method="pearson",
        ),
    ]
)

crfe = CRFE(
    estimator=RandomForestRegressor(max_depth=10, min_samples_leaf=20, random_state=0),
    constraints=constraints,
    step=0.1,
    n_features_to_select=20,
    scoring="r2",
    verbose=2,
)
crfe.fit(X, y)

print(crfe.get_feature_names_out())
```

With `verbose=2`, the output begins with a banner, the settings, the constraints, and the
solver and model details. It then shows a table with one row per iteration (`n_features`,
`removed`, `score`, `retained`, `gap`, and so on) and ends with a summary. The last lines are
the selected feature names.

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
Constraints  : 47 (AllOrNone 46, MaxCorrelation 1)
Solver       : CP-SAT 9.15.6755, workers=1, time_limit=10s, seed=0
Model        : 704 variables, 628 constraints, built in 2.828s

----------------------------------------------------------------------------------------------------------
 iter   n_features   removed     probes    fit(s)   solve(s)        score   retained       gap  elapsed(s)
----------------------------------------------------------------------------------------------------------
    0          352        49        0|0    17.325      0.075   8.9219e-01     0.9918    0.0082      21.161
    1          303        45        0|0    11.196      0.066   8.9044e-01     1.0000    0.0000      32.597
    2          258        54        0|0    11.488      0.053   8.9030e-01     1.0000    0.0000      44.293
    3          204        47        0|0     9.433      0.049   8.8919e-01     0.9942    0.0058      53.945
    4          157        54        0|0     6.052      0.037   8.8804e-01     0.9854    0.0146      60.159
    5          103        51        0|0     9.962      0.260   8.8465e-01     0.9737    0.0263      70.762
    6           52        32        0|0     5.053      0.011   8.7951e-01     1.0000    0.0000      75.896
    7           20         -          -     1.403          -   8.7881e-01          -         -      77.345
----------------------------------------------------------------------------------------------------------

Termination  : reached the target number of features
Selected     : 20 of 352 features
Probes       : 0 feasible, 0 infeasible, 0 blocked
Time         : wall 77.346s (fit 71.911, solve 0.551, score 0.960), cpu 46.312s
==========================================================================================================
['Lot_Area' 'Year_Built' 'Year_Remod_Add' 'Mas_Vnr_Area' 'BsmtFin_SF_1'
 'Bsmt_Unf_SF' 'Total_Bsmt_SF' 'Second_Flr_SF' 'Gr_Liv_Area'
 'Bsmt_Full_Bath' 'Full_Bath' 'Bedroom_AbvGr' 'Fireplaces' 'Garage_Cars'
 'Wood_Deck_SF' 'Open_Porch_SF' 'Longitude' 'Latitude' 'Central_Air=N'
 'Central_Air=Y']
```

The next snippet counts the selected categorical and continuous features. It reuses `groups`,
`numerical_features`, and `selected` from the example above.

```python
selected_categorical = [f for f, cols in groups.items() if any(c in selected for c in cols)]
selected_continuous = [c for c in numerical_features if c in selected]

print("categorical selected:", len(selected_categorical))
print("continuous selected:", len(selected_continuous))
```

```text
categorical selected: 1
continuous selected: 18
```

## Citation

If you use `tagarnina` in academic work, please cite it:

```bibtex
@software{navas_palencia_tagarnina,
  author  = {Navas-Palencia, Guillermo},
  title   = {tagarnina: Constrained Feature Selection},
  year    = {2026},
  url     = {https://github.com/guillermo-navas-palencia/tagarnina},
}
```

## License

Distributed under the Apache License 2.0. See [LICENSE](LICENSE) for details.