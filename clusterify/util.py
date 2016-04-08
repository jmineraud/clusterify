import math
import numpy as np

def distance(c1, c2, indexes=None, weights=None, nan_value=None):
    """ Calculate the Euclidean distance between two clusters
    :param c1 The first cluster
    :param c2 The second cluster
    :param indexes the indexes to compute the Euclidean distance
    :param weights the to weight the some of the values, default is None where all indexes have equal weights 
    :param nan_value the value that will replace nan in the clusters' serie. Default is None which does not replace nan
    """
    s1 = c1.serie if indexes is None else [c1.serie[i] for i in indexes]
    s2 = c2.serie if indexes is None else [c2.serie[i] for i in indexes]
    assert len(s1) == len(s2) and len(s1) > 0
    if nan_value is not None:
        # Replace nan values in the series
        s1 = [ nan_value if np.isnan(x) else x for x in s1]
        s2 = [ nan_value if np.isnan(x) else x for x in s2]
    # Handle weights
    if weights is None:
        weights = [1 for _ in range(len(s1))]
    else:
        assert len(weights) == len(s1)
    return np.sqrt((np.multiply(weights, np.power(np.subtract(s1, s2), 2))).sum())
    
def get_indexes (minIndex, nClusters):
    """ Get the index of the clusters based on the index of the minimum distance between clusters
    """
    index1 = 0
    while minIndex >= (nClusters - 1):
        minIndex -= nClusters - 1
        nClusters -= 1
        index1 += 1
    index2 = minIndex % (nClusters - 1)
    return (index1, index2 + index1 + 1)
