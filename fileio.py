import logging
import os
import sys
import glob
import numpy as np

from astropy.io import ascii
from astropy.table import Table
from .TPAGBparams import snap_src, phat_src

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_file(f, mad=True):
    '''
    input
    f (string): if f is not a file will print "no file"
    optional
    mad (bool)[True]: if mad is True, will exit program.
    '''
    test = os.path.isfile(f)
    if test is False:
        logging.warning('{} not found'.format(f))
        if mad:
            sys.exit()
    return test


def replace_ext(filename, ext):
    '''
    input
    filename string with .ext
    new_ext replace ext with new ext
    eg:
    $ replace_ext('data.02.SSS.v4.dat', '.log')
    data.02.SSS.v4.log
    '''
    return split_file_extention(filename)[0] + ext


def split_file_extention(filename):
    '''
    split the filename from its extension
    '''
    return '.'.join(filename.split('.')[:-1]), filename.split('.')[-1]


def get_files(src, search_string):
    '''
    returns a list of files, similar to ls src/search_string
    '''
    if not src.endswith('/'):
        src += '/'
    try:
        files = glob.glob1(src, search_string)
    except IndexError:
        logging.error('Can''t find %s in %s' % (search_string, src))
        sys.exit(2)
    files = [os.path.join(src, f)
             for f in files if ensure_file(os.path.join(src, f), mad=False)]
    return files


def load_table(filename, target, optfilter1=None, opt=True):

    if opt:
        filt1 = optfilter1
        assert optfilter1 is not None
    else:
        filt1 = nirfilter1

    tbl = ascii.read(filename)

    ifilts = list(np.nonzero((tbl['filter1'] == filt1))[0])
    itargs = [i for i in range(len(tbl['target'])) if target.upper() in tbl['target'][i]]
    indx = list(set(ifilts) & set(itargs))

    if len(indx) == 0:
        logger.error('{}, {} not found in table'.format(target, filt1))
        sys.exit(2)

    return tbl[indx]


def load_phat(target):
    from .pop_synth.starpop import StarPop
    galname, = get_files(phat_data_loc, '*{}*match'.format(target))
    optgal = StarPop()
    optgal.data = np.genfromtxt(galname, unpack=True, names=['F475W', 'F814W'])
    return optgal, optgal

def find_phatfake(target):
    fake, = get_files(phat_match_run_loc,
                                '*{}*.matchfake'.format(target))
    return fake, fake

def load_obs(target, optfilter1=''):
    """return in NIR and OPT galaxy as StarPop objects -- not 4 filter"""
    from .pop_synth.starpop import StarPop
    from astropy.io import fits
    if 'm31' in target or 'B' in target:
        optgal, nirgal = load_phat(target)
    else:
        nirgalname, = get_files(data_loc, '*{}*fits'.format(target.upper()))

        optgalname, = get_files(data_loc,
                                ('*{}*{}*fits'.format(target, optfilter1).lower()))

        nirgal = StarPop()
        nirgal.data = fits.getdata(nirgalname)

        optgal = StarPop()
        optgal.data = fits.getdata(optgalname)
    return optgal, nirgal

def find_fakes(target, optfilter1=''):
    """return matchfake filenames"""
    if 'm31' in target or 'B' in target:
        optfake, nirfake = find_phatfake(target)
    else:
        search_str = '*{}*.matchfake'.format(target.upper())
        fakes = get_files(data_loc, search_str)

        nirfake, = [f for f in fakes if 'IR' in f]
        optfake = [f for f in fakes if not 'IR' in f]

        if len(optfake) > 1:
            if optfilter1 != '':
                optfake, = [o for o in optfake if optfilter1 in o]
            else:
                optfake = optfake[0]
        elif len(optfake) == 1:
            optfake, = optfake
    return optfake, nirfake

def find_match_param(target, optfilter1=''):
    """return matchparam filename"""
    search_str = '*{}*.param'.format(optfilter1)
    loc = os.path.join(match_run_loc, target)

    if not os.path.isdir(loc):
        logger.error('{} directory not found!'.format(loc))
        sys.exit(2)

    try:
        mparam, = get_files(loc, search_str)
    except ValueError:
        if optfilter1 == '':
            print('Need to pass optfilter1')
            raise ValueError
        else:
            raise ValueError

    return mparam


def load_lf_file(lf_file):
    """
    Read a strange formatted file... basically, the rows are transformed.

    Each data row of the file corresponds to a header key.

    For example, data row 1 corresponds to header key 1, data row 2 corresponds
    to header key 2 ... up to the number of header keys, and then repeats.
    So if there are 5 header keys, data row 6 corresponds to header key 1, etc.

    It was done this way because some header keys correspond to a float,
    some correspond to an array of variable length.
    """
    header = open(lf_file).readline().replace('#', '').split()
    ncols = len(header)
    lfd = {}
    with open(lf_file, 'r') as lf:
        lines = [l.strip() for l in lf.readlines() if not l.startswith('#')]

    dtypes = [float, float, float, float, int, int, int, int, int, float, str]
    assert len(dtypes) == ncols, 'lf file format is not recognized'

    for i, key in enumerate(header):
        lfd[key] = [np.array(l.split(), dtype=dtypes[i])
                    for l in lines[i::ncols] if len(l.split())>0]
    return lfd

def remove_all(string, badchars):
    for b in badchars:
        string = string.replace(b, '')
    return string

def readfile(filename, col_key_line=0, comment_char='#', string_column=None,
             string_length=16, only_keys=None, delimiter=' '):
    '''
    reads a file as a np array, uses the comment char and col_key_line
    to get the name of the columns.
    '''
    if col_key_line == 0:
        with open(filename, 'r') as f:
            line = f.readline()
        col_line = line.replace(comment_char, '')
        col_keys = remove_all(col_line, '/[]-').split()
    else:
        with open(filename, 'r') as f:
            lines = f.readlines()
        col_keys = lines[col_key_line].replace(comment_char, '').strip().translate(None, '/[]').split()
    usecols = range(len(col_keys))

    if only_keys is not None:
        only_keys = [o for o in only_keys if o in col_keys]
        usecols = list(np.sort([col_keys.index(i) for i in only_keys]))
        col_keys = list(np.array(col_keys)[usecols])

    dtype = [(c, '<f8') for c in col_keys]
    if string_column is not None:
        if type(string_column) is list:
            for s in string_column:
                dtype[s] = (col_keys[s], '|U%i' % string_length)
        else:
            dtype[string_column] = (col_keys[string_column], '|U%i' % string_length)
    data = np.genfromtxt(filename, dtype=dtype, invalid_raise=False,
                         usecols=usecols, skip_header=col_key_line + 1)
    return data


def load_observation(filename, colname1=None, colname2=None, match_param=None,
                     exclude_gates=None, filterset=0):
    """
    Convienince routine for loading mags from a fits file or photometry file

    Note on using exclude gates:
    Be sure the filters sent to exclude_gate_inds match the filters in the
    match_param file.

    Parameters
    ----------
    filename : string
        path to observation

    colname1, colname2 : string, string
        name of mag1, mag2 column (if fits file)

    match_param : string
        path to match_param file to extract exclude_gates

    exculde_gates : N,2 np.array exclude_gates (color, mag1), N probably is 5 to
        match MATCH, but doesn't need to be if this is re-purposed.

    Returns
    -------
    mag1, mag2 : np.arrays
        could be sliced to exclude the indices within the exclude gates.
    """
    from .pop_synth.stellar_pops import exclude_gate_inds
    if filename.endswith('fits'):
        data = Table.read(filename, format='fits')
        mag1 = data[colname1]
        mag2 = data[colname2]
        if 'MAG2' in colname1 and match_param is not None:
            # this is likely a 4-filter matched catalog MAG1_ACS, MAG3_IR...
            cam = colname1.split('_')[1]
            if not cam in ['ACS', 'WFPC2']:
                logger.warning('Using {} for exclude gates! Probably should be optical!'.format(cam))
            m1 = data['MAG1_{}'.format(cam)]
            m2 = data['MAG2_{}'.format(cam)]
            inds = exclude_gate_inds(m1, m2, match_param=match_param)
            mag1 = m1[inds]
            mag2 = m2[inds]
            logger.info('using exclude gates')
    else:
        try:
            mag1, mag2 = np.loadtxt(filename, unpack=True)
        except:
            mag1, mag2, mag3, mag4 = np.loadtxt(filename, unpack=True)
            if filterset != 0:
                mag1 = mag3
                mag2 = mag4
        if match_param is not None:
            inds = exclude_gate_inds(mag1, mag2, match_param=match_param)
            mag1 = mag1[inds]
            mag2 = mag2[inds]
            logger.info('using exclude gates')

    inds, = np.nonzero(mag1-mag2 < -5)
    if len(inds) > 10:
        logger.warning('Lots ({}) of super blue shit found in {}! Cutting them out.'.format(filename, len(inds)))
        ginds, = np.nonzero(mag1-mag2 > -5)
        mag1 = mag1[ginds]
        mag2 = mag2[ginds]
    return mag1, mag2

def load_from_lf_file(lf_file, filter1='F814W_cor', filter2='F160W_cor',
                      flatten=True):

    # load models
    lf_dict = load_lf_file(lf_file)

    # idx_norm are the normalized indices of the simulation set to match RGB
    idx_norm = lf_dict['idx_norm']
    nmodels = 1

    # take all the scaled cmds in the file together and make an average hess

    smag1 = np.array([lf_dict[filter1][i][idx_norm[i]]
                            for i in range(len(idx_norm))])
    smag2 = np.array([lf_dict[filter2][i][idx_norm[i]]
                            for i in range(len(idx_norm))])
    if flatten:
        smag1 = np.concatenate(smag1)
        smag2 = np.concatenate(smag2)
        nmodels = len(idx_norm)
    return smag1, smag2, nmodels


def load_photometry(lf_file, observation, filter1='F814W_cor',
                     filter2='F160W_cor', col1='MAG2_ACS',
                     col2='MAG4_IR', yfilter='I', flatten=True,
                     optfake=None, comp_frac=None, nirfake=None):
    """
    Load photometry from a lf_file (see load_lf_file) and observation
    (see load_observation) will also cut lf_file to be within observation mags
    and cut both to be within comp_frac.
    If flatten is true, will concatenate all mags from the lf_file.
    If it's false, mags coming back will be an array of arrays.
    """
    from .pop_synth.stellar_pops import limiting_mag
    def limit_mags(mag1, mag2, smag1, smag2, comp2=90.):
        # for opt_nir_matched data, take the obs limits from the data
        inds, = np.nonzero((smag1 <= mag1.max()) & (smag2 <= mag2.max()) &
                           (smag1 >= mag1.min()) & (smag2 >= mag2.min()))

        sinds, = np.nonzero((smag2 <= comp2))# & (smag1 <= comp1))
        inds = list(set(sinds) & set(inds))
        smag1 = smag1[inds]
        smag2 = smag2[inds]

        oinds, = np.nonzero((mag2 <= comp2))# & (mag1 <= comp1))
        mag1 = mag1[oinds]
        mag2 = mag2[oinds]

        return mag1, mag2, smag1, smag2

    smag1, smag2, nmodels = load_from_lf_file(lf_file, flatten=flatten,
                                              filter1=filter1, filter2=filter2)

    if comp_frac is not None:
        if optfake is None:
            target = os.path.split(lf_file)[1].split('_')[0]
            optfake, nirfake = find_fakes(target)
        #_, comp1 = limiting_mag(optfake, comp_frac)
        _, comp2 = limiting_mag(nirfake, comp_frac)

    # load observation
    try:
        mag1, mag2 = load_observation(observation, col1, col2)
        lf2 = False
    except:
        # maybe send another LF file?
        mag1, mag2, nmodels = load_from_lf_file(observation, flatten=flatten,
                                                filter1=filter1, filter2=filter2)
        lf2 = True

    if flatten:
        mag1s, mag2s, smag1, smag2 = limit_mags(mag1, mag2, smag1, smag2, comp2)
    else:
        mag1s = []
        mag2s = []
        smag1s = []
        smag2s = []

        for i in range(len(smag1)):
            m1 = mag1
            m2 = mag2
            if lf2:
                m1 = mag1[0]
                m2 = mag2[0]

            _mag1, _mag2, _smag1, _smag2 = limit_mags(m1, m2, smag1[i],
                                                      smag2[i], comp2)
            mag1s.append(_mag1)
            mag2s.append(_mag2)
            smag1s.append(_smag1)
            smag2s.append(_smag2)
        mag1s = np.array(mag1s)
        mag2s = np.array(mag2s)
        smag1 = np.array(smag1s)
        smag2 = np.array(smag2s)

    color = mag1s - mag2s
    scolor = smag1 - smag2

    # set the yaxis
    symag = smag2
    ymag = mag2s
    if yfilter.upper() != 'I':
        symag = smag1
        ymag = mag1s

    return color, ymag, scolor, symag, nmodels
