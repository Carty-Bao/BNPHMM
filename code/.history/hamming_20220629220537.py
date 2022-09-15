import munkres
import numpy as np

def as1D(x):
    """ Convert input into to 1D numpy array.

    Returns
    -------
    x : 1D array

    Examples
    -------
    >>> as1D(5)
    array([5])
    >>> as1D([1,2,3])
    array([1, 2, 3])
    >>> as1D([[3,4,5,6]])
    array([3, 4, 5, 6])
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray_chkfinite(x)
    if x.ndim < 1:
        x = np.asarray_chkfinite([x])
    elif x.ndim > 1:
        x = np.squeeze(x)
    return x

def buildCostMatrix(zHat, zTrue):
    ''' Construct cost matrix for alignment of estimated and true sequences

    Args
    --------
    zHat : 1D array
        each entry is an integer label in {0, 1, ... Kest-1}
    zTrue : 1D array
        each entry is an integer label in {0, 1, ... Ktrue-1}
        with optional negative state labels

    Returns
    --------
    CostMatrix : 2D array, size Ktrue x Kest
        CostMatrix[j,k] = count of events across all timesteps,
        where j is assigned, but k is not.
    '''
    zHat = as1D(zHat)
    zTrue = as1D(zTrue)
    Ktrue = int(np.max(zTrue)) + 1
    Kest = int(np.max(zHat)) + 1
    K = np.maximum(Ktrue, Kest)
    CostMatrix = np.zeros((K, K))
    for ktrue in xrange(K):
        for kest in xrange(K):
            CostMatrix[ktrue, kest] = np.sum(np.logical_and(zTrue == ktrue,
                                                            zHat != kest))
    return CostMatrix


zHat = as1D(zHat)
zTrue = as1D(zTrue)
Kest = zHat.max() + 1
Ktrue = zTrue.max() + 1
CostMatrix = buildCostMatrix(zHat, zTrue)
    MunkresAlg = munkres.Munkres()
    tmpAlignedRowColPairs = MunkresAlg.compute(CostMatrix)
    AlignedRowColPairs = list()
    OrigToAlignedMap = dict()
    AlignedToOrigMap = dict()
    for (ktrue, kest) in tmpAlignedRowColPairs:
        if kest < Kest:
            AlignedRowColPairs.append((ktrue, kest))
            OrigToAlignedMap[kest] = ktrue
            AlignedToOrigMap[ktrue] = kest