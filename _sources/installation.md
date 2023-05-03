---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Installation

The package can be installed directly with `pip` or from GitHub.

## Pip

GStatSim can be installed using the `pip` command:

```{code-cell}
:tags: [hide-output]
pip install gstatsim
```

## GitHub

The most recent version can be installed from GitHub via:

```{code-cell}
:tags: [hide-output]
git clone https://github.com/GatorGlaciology/GStatSim
```


## Package dependencies
* Numpy
* Pandas
* Scipy
* tqdm
* Sklearn

## Requirements for visualization and variogram analysis
* Matplotlib
* SciKit-GStat

These can all be installed using `pip`. We have included a plot_utils.py file with plotting routines for the tutorials.



