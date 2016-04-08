# clusterify
A python library to reorganize numpy ndarray according to Euclidean distances

## Requirements

* Numpy 1.10.1

## Usage

```python
import matplotlib.pyplot as plt
import numpy as np
import clusterify as c
data = np.random.rand(100,100)
f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
ax1.pcolor(data, cmap=plt.cm.Blues)
ax2.pcolor(c.sort_array(data), cmap=plt.cm.Blues)
plt.show()
```