## Research Investigation of Timeseries with Multiday Oscillations (RITMO)

---
This package provides a python toolbox for assessing the slow-drifting correlation and causation between two oscialltory timeseries' with multiday patterns. It includes three well-estabilised approaches:

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
>>> ritmo = Ritmo(y1 = y1, y2 = y2, x1 = x)
>>> ritmo.run()
```

---

### References

Stirling et al. 2022. A methodology to assess cyclical correlates: case study of the heart and the epileptic brain.
