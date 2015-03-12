import ResolvedStellarPops as rsp
from TPAGBparams import snap_src
data_loc = os.path.join(snap_src, 'data', 'galaxies')

def load_obs(target, optfilter1=''):
    """load in NIR and OPT galaxy as StarPop objects"""
    from astropy.io import fits
    nirgalname, = rsp.fileio.get_files(data_loc, '*{}*fits'.format(target.upper()))
    optgalname, = rsp.fileio.get_files(data_loc, ('*{}*{}*fits'.format(target, optfilter1).lower()))
    nirgal = rsp.StarPop()
    nirgal.data = fits.getdata(nirgalname)

    optgal = rsp.StarPop()
    optgal.data = fits.getdata(optgalname)
    return optgal, nirgal