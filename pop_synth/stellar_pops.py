import logging
import os
import numpy as np
from ResolvedStellarPops.galaxies.starpop import stars_in_region
from ResolvedStellarPops.galaxies.asts import ASTs
from ResolvedStellarPops.utils import points_inside_poly

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rgb_agb_regions(mag, offset=None, trgb_exclude=None, trgb=None, col_min=None,
                    col_max=None, mag1=None, mag_bright=None,
                    mag_faint=None):
    """the indices of mag in rgb and agb regions"""
    # define RGB regions
    if mag_bright is not None:
        low = mag_faint
        mid = mag_bright
    else:
        assert offset is not None, \
            'rgb_agb_regions: need either offset or mag limits'
        low = trgb + offset
        mid = trgb + trgb_exclude

    # Recovered stars in simulated RGB region.
    srgb = stars_in_region(mag, low, mid, col_min=col_min, col_max=col_max,
                           mag1=mag1)

    # define AGB regions
    mid = trgb - trgb_exclude
    high = 10

    # Recovered stars in simulated AGB region.
    sagb = stars_in_region(mag, mid, high)

    return srgb, sagb


def normalize_simulation(mag, nrgb, srgb, sagb):
    """normalization factor, all indices of a random sample of the distribution
       random sample of the rgb and agb"""
    norm = nrgb / float(len(srgb))

    logger.info('Normalization: %f' % norm)

    # random sample the data distribution
    rands = np.random.random(len(mag))
    ind, = np.nonzero(rands < norm)

    # scaled rgb: norm + in rgb
    rgb = list(set(ind) & set(srgb))

    # scaled agb
    agb = list(set(ind) & set(sagb))
    return norm, ind, rgb, agb


def limiting_mag(fakefile, comp_frac):
    """
    find the completeness fraction in each filter of the fake file
    for details see ResolvedStellarPops.galaxies.asts.ASTs.__doc__.

    Parameters
    ----------
    fakefile : str
        match fake file (mag1in, mag2in, mag1diff, mag2diff)
    comp_frac : float
        completeness fraction e.g, 0.9 means 90% completeness

    Returns
    -------
    comp1, comp2 : float, float
        the completeness fraction in each filter
    """
    assert os.path.isfile(fakefile), \
        'limiting mag: fakefile %s not found' % fakefile
    ast = ASTs(fakefile)
    ast.completeness(combined_filters=True, interpolate=True)
    comp1, comp2 = ast.get_completeness_fraction(comp_frac)
    return comp1, comp2


def completeness_corrections(fakefile, mag_bins, mag2=True):
    '''
    get the completeness fraction for a given list of magnitudes.
    for details see ResolvedStellarPops.galaxies.asts.ASTs.__doc__.

    Parameters
    ----------
    fakefile : str
        match fake file (mag1in, mag2in, mag1diff, mag2diff)
    mag_bins : array
        array of magnitudes to find completeness interpolation
    mag2 : bool
        True use fcomp2, False use fcomp1

    Returns
    -------
    ast_c : array len(mag_bins)
        completeness corrections to mag_bins
    '''
    assert os.path.isfile(fakefile), \
        'completeness corrections: fakefile %s not found' % fakefile
    ast = ASTs(fakefile)
    ast.completeness(combined_filters=True, interpolate=True)

    if mag2:
        ast_c = ast.fcomp2(mag_bins)
    else:
        ast_c = ast.fcomp1(mag_bins)

    return ast_c

def exclude_gate_inds(mag1, mag2, match_param=None, exclude_gates=None,
                      ms_color_cut=False):
    """
    return an array of all points outside the exclude gates region if
    find_ms_color_cut is set, will include all points bluer than the median
    exclude gates color.
    """
    if exclude_gates is None:
        exclude_gates = get_exclude_gates(match_param)
        if False in np.isfinite(exclude_gates):
            return exclude_gates
        
    color = mag1 - mag2
    mag = mag1
    points = np.column_stack((color, mag))
    inds, = np.nonzero(points_inside_poly(points, exclude_gates))
    
    decontam = np.arange(len(mag))
    decontam = np.delete(decontam, inds)
    
    if ms_color_cut:
        color_cut = np.median(color[inds])
        blue = np.nonzero(color[decontam] < color_cut)
        decontam = np.delete(decontam, blue)
    
    return decontam


def get_exclude_gates(match_param):
    exg = np.ndarray(shape=(10,)) * np.nan
    ex_line = open(match_param, 'r').readlines()[7]
    nexgs = np.int(ex_line[0])
    if nexgs == 0:
        logger.warning('no exclude gates')
    elif nexgs > 1:
        logger.error('get_exclude_gates only works on 1 gate... code!')
    else:
        exg = np.array(ex_line.split()[1:-1], dtype=float)
        exg = np.append(exg, exg[:2])
    return exg.reshape(5, 2)