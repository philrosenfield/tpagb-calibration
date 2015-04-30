import numpy as np

def check_astcor(filters):
    """add _cor to filter names if it isn't already there"""
    if type(filters) is str:
        filters = [filters]

    for i, f in enumerate(filters):
        if not f.endswith('cor'):
            filters[i] = f + '_cor'
    return filters

def extrema(func, arr1, arr2):
    return func([func(arr1), func(arr2)])


def minmax(arr1, arr2):
    return extrema(np.min, arr1, arr2), extrema(np.max, arr1, arr2)
