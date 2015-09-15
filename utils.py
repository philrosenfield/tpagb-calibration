import numpy as np

def extrema(func, arr1, arr2):
    return func([func(arr1), func(arr2)])


def minmax(arr1, arr2):
    return extrema(np.min, arr1, arr2), extrema(np.max, arr1, arr2)
