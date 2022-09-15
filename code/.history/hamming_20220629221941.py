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
    for ktrue in range(K):
        for kest in range(K):
            CostMatrix[ktrue, kest] = np.sum(np.logical_and(zTrue == ktrue,
                                                            zHat != kest))
    return CostMatrix

def alignEstimatedStateSeqToTruth(zHat, zTrue, useInfo=None, returnInfo=False):

    ''' Relabel the states in zHat to minimize the hamming-distance to zTrue

    Args
    --------
    zHat : 1D array
        each entry is an integer label in {0, 1, ... Kest-1}
    zTrue : 1D array
        each entry is an integer label in {0, 1, ... Ktrue-1}

    Returns
    --------
    zHatAligned : 1D array
        relabeled version of zHat that aligns to zTrue
    AInfo : dict
        information about the alignment
    '''
    try:
        import munkres
    except ImportError:
        raise ImportError('Required third-party module munkres not found.\n' +
                          'To fix, add $BNPYROOT/third-party/ to your path.')
    zHat = as1D(zHat)
    zTrue = as1D(zTrue)
    Kest = zHat.max() + 1
    Ktrue = zTrue.max() + 1

    if useInfo is None:
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
    else:
        # Unpack existing alignment info
        AlignedRowColPairs = useInfo['AlignedRowColPairs']
        CostMatrix = useInfo['CostMatrix']
        AlignedToOrigMap = useInfo['AlignedToOrigMap']
        OrigToAlignedMap = useInfo['OrigToAlignedMap']
        Ktrue = useInfo['Ktrue']
        Kest = useInfo['Kest']

        assert np.allclose(Ktrue, zTrue.max() + 1)
        Khat = zHat.max() + 1

        # Account for extra states present in zHat
        # that have never been aligned before.
        # They should align to the next available UID in set
        # [Ktrue, Ktrue+1, Ktrue+2, ...]
        # so they don't get confused for a true label
        ktrueextra = np.max([r for r, c in AlignedRowColPairs])
        ktrueextra = int(np.maximum(ktrueextra + 1, Ktrue))
        for khat in np.arange(Kest, Khat + 1):
            if khat in OrigToAlignedMap:
                continue
            OrigToAlignedMap[khat] = ktrueextra
            AlignedToOrigMap[ktrueextra] = khat
            AlignedRowColPairs.append((ktrueextra, khat))
            ktrueextra += 1

    zHatA = -1 * np.ones_like(zHat)
    for kest in np.unique(zHat):
        mask = zHat == kest
        zHatA[mask] = OrigToAlignedMap[kest]
    assert np.all(zHatA >= 0)

    if returnInfo:
        return zHatA, dict(CostMatrix=CostMatrix,
                           AlignedRowColPairs=AlignedRowColPairs,
                           OrigToAlignedMap=OrigToAlignedMap,
                           AlignedToOrigMap=AlignedToOrigMap,
                           Ktrue=Ktrue,
                           Kest=Kest)
    else:
        return zHatA

Z_n = np.array([1,0,0,1,1])
Ztrue = np.array([1,0,1,1,1])

curZA = alignEstimatedStateSeqToTruth(Z_n, Ztrue)