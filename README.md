## Rhythmic Investigation of Timeseries with Multiday Oscillations (RITMO)

---
This package provides a python toolbox for assessing the slow-drifting correlation and causation between two osciallting timeseries with multiday patterns. It includes three well-estabilised approaches:

* Empirical Dynamic Modelling (also known as EDM)
* Phase locking value
* Mutual information

---

## Installation

Command line using the Python pip module: `python -m pip install ritmo`

---

## Usage

Example usage at the python prompt:

```python
>>> from ritmo import Ritmo
>>> import numpy as np
>>> x = np.arange(0, 100*24*3.6e6, 3.6e6) # UNIX timestamps in milliseconds
>>> y1 = np.random.random(x.size) # first random timeseries
>>> y2 = np.random.random(x.size) # second random timeseries
>>> ritmo = Ritmo(y1 = y1, y2 = y2, x1 = x, y1_name = 'Random_TS_1', y2_name = 'Random_TS_2')
>>> ritmo.run() # Runs all modules at once
```

To run individual modules:

```python
>>> ritmo.run_edm() # Empirical dynamical modelling module
>>> ritmo.run_plv() # Phase locking value module
>>> ritmo.run_mutual_information() # Mutual information module
```

---

## Releasing

Releases are published automatically when a tag is pushed to GitHub.

```python
export RELEASE=x.x.x
git commit --allow-empty -m "Release $RELEASE"
git tag -a $RELEASE -m "Version $RELEASE"
git push upstream --tags
```
