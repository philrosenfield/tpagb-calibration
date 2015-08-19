import logging
import os
import numpy as np

from ResolvedStellarPops.galaxies.starpop import stars_in_region
from ResolvedStellarPops.galaxies.asts import ASTs
from ResolvedStellarPops.utils import points_inside_poly


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rgb_agb_regions(mag, offset=None, trgb_exclude=None, trgb=None, col_min=None,
                    col_max=None, mag1=None, mag_bright=None, color=None,
                    mag_faint=None):
    """
    Return indices of mag in rgb and agb regions

    Parameters
    ----------
    mag : array
        (probably should be filter2) list of magnitudes

    offset : float
        magnitude offset below the trgb to include as rgb stars

    trgb_exclude : float
        +/- magnitude around the trgb to exclude from rgb and agb star
        determination

    trgb : float
        magnitude (same filter as mag) of the rgb tip

    col_min : float
        blue color cut (need mag1)

    col_max : float
        red color cut (need mag1)

    mag1 : array
        filter1 magnitude list

    mag_bright : float
        brightest mag to include as rgb (agb is set to mag=10)

    mag_faint : float
        faintest mag to include as rgb

    Returns
    -------
    srgb, sagb : array, array
        indices of stars in rgb region and agb region

    Note
    ----
    if both (offset, trgb, trbg_exclude) and (mag_faint, mag_bright) are
    passed, will default to the former for rgb retions

    """
    # define RGB regions
    if offset is not None and np.isfinite(trgb):
        low = trgb + offset
        mid = trgb + trgb_exclude
    else:
        assert mag_bright is not None, \
            'rgb_agb_regions: need either offset or mag limits'
        low = mag_faint
        mid = mag_bright

    # Recovered stars in simulated RGB region.
    srgb = stars_in_region(mag, low, mid, col_min=col_min, col_max=col_max,
                           mag1=mag1, color=color)
    if len(srgb) == 0:
        import pdb; pdb.set_trace()
    # define AGB regions
    if offset is not None:
        mid = trgb - trgb_exclude
    else:
        mid = mag_bright
    high = 10

    # Recovered stars in simulated AGB region.
    sagb = stars_in_region(mag, mid, high, col_min=col_min, col_max=10,
                           mag1=mag1, color=color)

    return srgb, sagb


def normalize_simulation(mag, nrgb, srgb, sagb, norm=None):
    """
    scale simulation to have the same number of nrgb (data) as srgb (from mag)

    Parameters
    ----------
    mag : array
        magnitude list (model)
    nrgb : float
        number of rgb stars in the data
    srgb : array
        indices of mag that are rgb
    sagb :
        indices of mag that are agb

    Returns
    -------
    norm : float
        normalization factor
    ind : array
        random sample of mag scaled to nrgb
    rgb : array
        random sample of srgb scaled to nrgb
    agb : array
        random sample of sagb scaled to nrgb

    """
    if norm is None:
        norm = nrgb / float(len(srgb))

    logger.info('Normalization: %f' % norm)
    if norm >= 0.5:
        logger.warning('Not many simulated stars, need larger region or larger simulation')

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
                      ms_color_cut=True):
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
    inds = np.array([])

    for exg in exclude_gates:
        ind, = np.nonzero(points_inside_poly(points, exg))
        inds = np.append(inds, ind)
    inds = list(np.unique(inds.flatten()))

    decontam = np.arange(len(mag))
    decontam = np.delete(decontam, inds)

    if ms_color_cut:
        color_cut = np.median(color[inds])
        blue = np.nonzero(color[decontam] < color_cut)
        decontam = np.delete(decontam, blue)

    return decontam


def get_exclude_gates(match_param):

    ex_line = open(match_param, 'r').readlines()[7]
    nexgs = np.int(ex_line[0])
    if nexgs == 0:
        logger.warning('no exclude gates')
        exgs = np.inf
    else:
        ex_data = np.array(ex_line.split()[1:-1], dtype=float)
        npts = len(ex_data)
        splits = np.arange(0, npts + 1, 8)
        exgs = [ex_data[splits[i]: splits[i+1]] for i in range(len(splits)-1)]
        # complete the polygon
        for i in range(len(exgs)):
            exgs[i] = np.append(exgs[i], exgs[i][:2])
            exgs[i] = exgs[i].reshape(5, 2)

    return exgs
