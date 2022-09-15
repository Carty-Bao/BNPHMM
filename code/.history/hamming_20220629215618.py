import munkres
import bumpy as np

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

zHat = as1D(zHat)
zTrue = as1D(zTrue)
Kest = zHat.max() + 1
Ktrue = zTrue.max() + 1