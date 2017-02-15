import os
import re
import numpy as np
import sys
import argparse


def add_data(old_data, names, new_data):
    '''
    use with Starpop, Track, or any object with data attribute that is a
    np.recarray

    add columns to self.data, update self.key_dict
    see numpy.lib.recfunctions.append_fields.__doc__

    Parameters
    ----------
    old_data : recarray
        original data to add columns to

    new_data : array or sequence of arrays
        new columns to add to old_data

    names : string, sequence
        String or sequence of strings corresponding to the names
        of the new_data.

    Returns
    -------
    array with old_data and new_data
    '''
    import numpy.lib.recfunctions as nlr
    data = nlr.append_fields(np.asarray(old_data), names, new_data).data
    data = data.view(np.recarray)
    return data


def float2sci(num):
    """mpl has a better way of doing this?"""
    _, exnt = '{0:.0e}'.format(num).split('e')
    exnt = int(exnt)
    if exnt == 0:
        # 10 ** 0 = 1
        retv = ''
    else:
        retv = r'$10^{{{0:d}}}$'.format(exnt)
    return retv


def between(arr, mdim, mbrt, inds=None):
    '''indices of arr or arr[inds] between mdim and mbrt'''
    if mdim < mbrt:
        mtmp = mbrt
        mbrt = mdim
        mdim = mtmp
    i, = np.nonzero((arr < mdim) & (arr > mbrt))
    if inds is not None:
        i = np.intersect1d(i, inds)
    return i


def parse_pipeline(filename):
    '''find target and filters from the filename'''
    name = os.path.split(filename)[1].upper()

    # filters are assumed to be like F814W
    starts = [m.start() for m in re.finditer('_F', name)]
    starts.extend([m.start() for m in re.finditer('-F', name)])
    starts = np.array(starts)
    starts += 1

    filters = [name[s: s+5] for s in starts]
    for i, f in enumerate(filters):
        try:
            # sometimes you get FIELD...
            int(f[1])
        except:
            filters.pop(i)
    # the target name is assumed to be before the filters in the filename
    try:
        pref = name[:starts[0]-1]
        for t in pref.split('_'):
            if t == 'IR':
                continue
            try:
                # this could be the proposal ID
                int(t)
            except:
                # a mix of str and int should be the target
                target = t
    except:
        target = filename.split('.')[0]

    return target, filters


def convertz(z=None, oh=None, mh=None, feh=None, oh_sun=8.76, z_sun=0.01524,
             y0=.2485, dy_dz=1.80):
    '''
    input:
    metallicity as z
    [O/H] as oh
    [M/H] as mh
    [Fe/H] as feh

    initial args can be oh_sun, z_sun, y0, and dy_dz

    returns oh, z, y, x, feh, mh where y = He and X = H mass fractions
    '''

    if oh is not None:
        feh = oh - oh_sun
        z = z_sun * 10 ** (feh)

    if mh is not None:
        z = (1 - y0) / ((10**(-1. * mh) / 0.0207) + (1. + dy_dz))

    if z is not None:
        feh = np.log10(z / z_sun)

    if feh is not None:
        z = z_sun * 10**feh

    oh = feh + oh_sun
    y = y0 + dy_dz * z
    x = 1. - z - y
    if mh is None:
        mh = np.log10((z / x) / 0.0207)

    if __name__ == "__main__":
        print('''
                 [O/H] = %2f
                 z = %.4f
                 y = %.4f
                 x = %.4f
                 [Fe/H] = %.4f
                 [M/H] = %.4f''' % (oh, z, y, x, feh, mh))
    return oh, z, y, x, feh, mh


def extrema(func, arr1, arr2):
    return func([func(arr1), func(arr2)])


def minmax(arr1, arr2):
    return extrema(np.min, arr1, arr2), extrema(np.max, arr1, arr2)


def points_inside_poly(points, all_verts):
    """ Proxy to the correct way with mpl """
    from matplotlib.path import Path
    return Path(all_verts).contains_points(points)


def sort_dict(d, reverse=False):
    from collections import OrderedDict
    return OrderedDict(sorted(d.items(), key=lambda k: k[0], reverse=reverse))


def count_uncert_ratio(numerator, denominator):
    ''' combine poisson error to calculate ratio uncertainty'''
    n = float(numerator)
    d = float(denominator)
    try:
        cur = (n / d) * (1./np.sqrt(n) + 1./np.sqrt(d))
    except ZeroDivisionError:
        cur = np.nan
    return cur


def shift(fnames, off=0):
    """
    example:
    out_ugc4305-2_f555w_f814w_caf09_v1.2s_m36_s12d_ns_nas_017.dat
    off = 25
    new fomrat:
    out_ugc4305-2_f555w_f814w_caf09_v1.2s_m36_s12d_ns_nas_042.dat
    """
    line = ''
    for f in fnames:
            pref, idxext = '_'.join(f.split('_')[:-1]), f.split('_')[-1]
            idx, ext = idxext.split('.')
            nidx = int(idx) + off
            nf = '_'.join([pref, '{:03d}.{}'.format(nidx, ext)])
            line += 'mv {} {}\n'.format(f, nf)
    return line



def find_peaks(arr):
    '''
    find maxs and mins of an array
    from
    http://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
    Parameters
    ----------
    arr : array
        input array to find maxs and mins

    Returns
    -------
    turning_points : dict
        keys:
        maxima_number: int, how many maxima in arr
        minima_number: int, how many minima in arr
        maxima_locations: list, indicies of maxima
        minima_locations: list, indicies of minima
    '''
    gradients = np.diff(arr)
    #print gradients

    maxima_num = 0
    minima_num = 0
    max_locations = []
    min_locations = []
    count = 0
    for i in gradients[:-1]:
        count += 1

        if ((cmp(i, 0) > 0) & (cmp(gradients[count], 0) < 0) &
           (i != gradients[count])):
            maxima_num += 1
            max_locations.append(count)

        if ((cmp(i, 0) < 0) & (cmp(gradients[count], 0) > 0) &
           (i != gradients[count])):
            minima_num += 1
            min_locations.append(count)

    turning_points = {'maxima_number': maxima_num,
                      'minima_number': minima_num,
                      'maxima_locations': max_locations,
                      'minima_locations': min_locations}

    return turning_points



def extrap1d(x, y, xout_arr):
    '''
    linear extapolation from interp1d class with a way around bounds_error.
    Adapted from:
    http://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate-give-an-extrapolated-result-beyond-the-input-range

    Parameters
    ----------
    x, y : arrays
        values to interpolate

    xout_arr : array
        x array to extrapolate to

    Returns
    -------
    f, yo : interpolator class and extrapolated y array
    '''
    from scipy.interpolate import interp1d
    # Interpolator class
    f = interp1d(x, y)
    xo = xout_arr
    # Boolean indexing approach
    # Generate an empty output array for "y" values
    yo = np.empty_like(xo)

    # Values lower than the minimum "x" are extrapolated at the same time
    low = xo < f.x[0]
    yo[low] = f.y[0] + (xo[low] - f.x[0]) * (f.y[1] - f.y[0]) / (f.x[1] - f.x[0])

    # Values higher than the maximum "x" are extrapolated at same time
    high = xo > f.x[-1]
    yo[high] = f.y[-1] + (xo[high] - f.x[-1]) * (f.y[-1] - f.y[-2]) / (f.x[-1] - f.x[-2])

    # Values inside the interpolation range are interpolated directly
    inside = np.logical_and(xo >= f.x[0], xo <= f.x[-1])
    yo[inside] = f(xo[inside])
    return f, yo



def main(argv):
    description = "Write a scripts to shift filename numbers"

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-o', '--outfile', type=str, default='shift.sh',
                        help='output file')

    parser.add_argument('-s', '--offset', type=int, default=0,
                        help='numeric offset')

    parser.add_argument('files', type=str, nargs='*', help='input files')

    args = parser.parse_args(argv)

    lines = shift(args.files, off=args.offset)

    with open(args.outfile, 'w') as out:
        out.write(lines)

if __name__ == "__main__":
    main(sys.argv[1:])
