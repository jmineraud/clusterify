import numpy as np
from clusterify import util

class ClusterNode ():
    """A class to manage cluster nodes for the clusterisation of the numpy ndarray.
    TODO I might want to save a fitness attribute for the cluster to know when there is
    significant difference.
    """
    def __init__(self, serie=None, left=None, right=None, fuse=None):
        """The constructor for the Cluster node. A cluster node can be either a leaf with the original 
        data series or a cluster that has two nodes left and right
        :param serie The series of number that will be used to perform the clusterization
        :param left the left cluster if the cluster node is not a leaf
        :param right the right cluster if the cluster node is not a leaf
        :param fuse is a function that fuse two data series (x,y -> z). Default is a weighted sum of sub clusters based on size
        """
        # Check if the cluster node is valid or not (ie, either a leaf or a parent of two sub-clusters)
        assert (left != None and right != None and isinstance(left, ClusterNode) and isinstance(right, ClusterNode)) or \
            (left == None and right == None)
        # If cluster is a leaf then series should not be empty
        assert (left is None) == (not (serie is None or (len(serie) == 0)))
        # If cluster is not a leaf, series from clusters on left and right side should have the same size
        assert left == None or len(left.serie) == len(right.serie)
        self.left = left
        self.right = right
        self.serie = serie
        
        # In the event we are creating a parent cluster, we need to calculate the parent cluster series using fuse
        if self.serie is None or len(self.serie) == 0:
            if fuse is None:
                sizeLeft = self.left.size() # Numbers of leaf cluster node in left cluster
                sizeRight = self.right.size() # Numbers of leaf cluster node in right cluster
                # We fuse data by default by having a weighted sum of sub clusters based on size 
                fuse = lambda l,r : (l * sizeLeft + r * sizeRight) / self.size()
                # Init the series
                self.serie = []
                for i in range(len(self.left.serie)):
                    self.serie.append(fuse(self.left.serie[i], self.right.serie[i]))
            else:
                self.serie = fuse(self.left, self.right)
                assert self.serie is not None and len(self.serie) == len(self.left.serie)
    
    def is_leaf(self):
        """Return true if cluster node is a leaf
        """
        return self.left is None and self.right is None

    def size(self):
        """Numbers of leaf cluster node in the current cluster
        """
        if self.is_leaf():
            # If node is leaf, return 1
            return 1
        else:
            # Otherwise, add the size of left and right sub clusters
            return int(self.left.size () + self.right.size ())

    def most_left(self):
        """Get the most left leaf
        """
        if self.is_leaf():
            return self
        else:
            return self.left.most_left()
    
    def most_right(self):
        """Get the most right leaf
        """
        if self.is_leaf():
            return self
        else:
            return self.right.most_right()

    def sorted_leaves(self):
        """Return a numpy ndarray of the leaves based on left or right positions
        """
        if self.is_leaf():
            # For leaves, I just return the serie. The serie is embedded in an array
            return np.array([self.serie])
        else:
            # For parent cluster, we concatenate the left and right cluster on axis 0
            return np.concatenate((self.left.sorted_leaves(), self.right.sorted_leaves()), axis=0)

def init_clusters(array, useRow=True):
    """ Initialiaze a numpy ndarray as cluster
    :param array The numpy ndarray
    :param useRow tell the algorithm the create clusters from rows instead of columns
    """
    assert array is not None
    if useRow:
        return list(map(lambda x: ClusterNode(x), array))
    else:
        return list(map(lambda x: ClusterNode(x), array.T))

def clusterify(array, indexes=None, weights=None, useRow=True):
    """Return a ClusterNode corresponding to the ndarray in inputs
    :param array the ndarray
    :param indexes the indexes used to compute the Euclidean distance. Default is none which means all values are used
    :param weights the weights used for each of the indexes used. Default is none which means equal weights
    :param useRow defines if we do the classification by row or by column. Default is by row (ie. True)
    :return a ClusterNode
    """
    # Just a quick check on the shape of the array
    sx, sy = array.shape
    assert sx > 0 and sy > 0
    # Currently, we use the minimum of the array for the nan values, we may need a more clever trick in the future
    nan_value = np.min(array)

    # Initialize the list of clusters (these clusters are the leaves)
    clusters = init_clusters(array, useRow=useRow)
    assert clusters is not None and len(clusters) > 0

    # Compute the distances
    distances = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            distances.append(util.distance(clusters[i], clusters[j], indexes=indexes, weights=weights, nan_value=nan_value))
    # we only need to calculate N * (N - 1) / 2 distances
    assert len(distances) == (len(clusters) * (len(clusters) - 1)) / 2, \
        "len(distance) = %r, expected = %r" % (len(distances), (len(clusters) * (len(clusters) - 1)) / 2)

    # Clusterization process ongoing while the number of clusters in clusters is 1
    while len(clusters) > 1:
        nClusters = len(clusters)
        # Pick the index of the minimum distance
        indexMinDist = np.argmin(distances)
        # Get the indexes of the clusters which where used to calculate the distance
        clustersIndex = util.get_indexes(indexMinDist, nClusters)
        
        # We can retrieve our clusters from the indexes
        assert clustersIndex[0] != clustersIndex[1]
        c1 = clusters[clustersIndex[0]]
        c2 = clusters[clustersIndex[1]]
        
        # Remove the two clusters from the list of clusters
        clusters.remove(c1)
        clusters.remove(c2)
        # Delete the corresponding distances
        distances = [x for i, x in enumerate(distances) if not np.any(np.in1d(util.get_indexes(i, nClusters), clustersIndex))]
        if len(clusters) > 0:
            # The number of distances should be N*N-1/2 with N = len(clusters)
            assert len(distances) == (len(clusters) * (len(clusters) - 1)) / 2, \
                "len(distance) = %r, expected = %r" % (len(distances), (len(clusters) * (len(clusters) - 1)) / 2)


        # Now we can add our new cluster to the left of the array
        # TODO check if there would be a better way to select right from left than using size (maybe variance)
        distNormal = util.distance(c1.most_right(), c2.most_left(), indexes=indexes, weights=weights, nan_value=nan_value)
        distReverse = util.distance(c2.most_right(), c1.most_left(), indexes=indexes, weights=weights, nan_value=nan_value)
        cluster = ClusterNode(left=c1, right=c2) if distNormal < distReverse else ClusterNode(left=c2, right=c1)
        clusters.insert(0, cluster)
        # We compute the distance between the new clusters and all the other clusters
        clusterDistances = [util.distance(cluster, c, indexes=indexes, weights=weights, nan_value=nan_value) for c in clusters[1:]]
        # Finally we concatenate the computed distances with other distances
        distances = clusterDistances + distances
        # The number of distances should be N*N-1/2 with N = len(clusters)
        assert len(distances) == (len(clusters) * (len(clusters) - 1)) / 2, \
            "len(distance) = %r, expected = %r" % (len(distances), (len(clusters) * (len(clusters) - 1)) / 2)


    # The clustering process is completed
    assert len(clusters) == 1
    return clusters[0]

def sort_array(array, indexes=None, weights=None, useRow=True):
    """Return a sorted array based on the euclidean distance and cluster process
    :param array the ndarray
    :param indexes the indexes used to compute the Euclidean distance. Default is none which means all values are used
    :param weights the weights used for each of the indexes used. Default is none which means equal weights
    :param useRow defines if we do the classification by row or by column. Default is by row (ie. True)
    :return a ndarray with the same shape as array
    """
    # Cluster all the series in the array according to parameters
    cluster = clusterify(array, indexes=indexes, weights=weights, useRow=useRow)
    # Get the sorted array by getting the leaves in proper order
    sorted_array = cluster.sorted_leaves()
    # Use transpose if columns are used instead of rows
    res = sorted_array if useRow else sorted_array.T
    # Verify the shape
    assert array.shape == res.shape
    # Return result
    return res

