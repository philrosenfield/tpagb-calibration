import logging
import sys
import os

import matplotlib.pyplot as plt
import numpy as np

from astropy.io import ascii

from .fileio import replace_ext, ensure_dir, InputParameters
from .tracks import PadovaTrack
from .. import utils


logger = logging.getLogger(__name__)

__all__ = ['Trilegal_SFH', 'IsoTrack',
           'change_galaxy_input', 'cmd_input_dict', 'cmd_input_fmt',
           'find_mag_num', 'find_photsys_number', 'galaxy_input_dict',
           'galaxy_input_fmt', 'get_args', 'get_loop_data',
           'get_stage_label', 'read_isotrack_file',
           'run_trilegal',
           'write_spread', 'trilegal_sfh',
           'read_leos_tracks']

def trilegal2matchphot(trilegal_output, col1=None, col2=None, extra=''):
    from ..pop_synth.simgalaxy import SimGalaxy
    if col1 is None:
        col1 = trilegal_output.split('_')[2].upper()
    if col2 is None:
        col2 = trilegal_output.split('_')[3].upper()

    col1 += extra
    col2 += extra

    sgal = SimGalaxy(trilegal_output)
    outfile = replace_ext(trilegal_output, '.match')
    np.savetxt(outfile, np.column_stack((sgal.data[col1], sgal.data[col2])), fmt='%.3f')
    return outfile

def trilegal2hdf5(trilegal_output, overwrite=False, remove=True):
    """
    Convert a file to hdf5 using compression and path set to 'data'
    """
    new_out = replace_ext(trilegal_output, '.hdf5')
    tbl = ascii.read(trilegal_output)
    tbl.write(new_out, format='hdf5', path='data', compression=True,
              overwrite=overwrite)
    if remove:
        os.remove(trilegal_output)
    return new_out


def read_leos_tracks(fname):
     data = np.genfromtxt(fname, usecols=(1,2,3,4,5),
                          names=['age', 'LOG_L', 'LOG_TE', 'mass', 'stage'])
     return data.view(np.recarray)

def write_spread(sgal, outfile, overwrite=False, slice_inds=None):
    if overwrite or not os.path.isfile(outfile):
        # ast corrected filters are filter_cor, this may be changed... anyway
        # need someway to check if ast corrections were made.
        cors = [c for c in sgal.data.key_dict.keys() if '_cor' in c]
        if len(cors) == 0:
            print('error can not make spread file without ast_corrections')
            return -1

        filt1, filt2 = np.sort(cors)
        if hasattr(sgal, 'ast_mag1'):
            # this isn't only a trilegal catalog, it's already been corrected
            # with asts, and sliced for only recovered asts. see simgalaxy.
            cor_mag1 = sgal.ast_mag1[sgal.rec]
            cor_mag2 = sgal.ast_mag2[sgal.rec]
        else:
            # it's a trilegal catalog, now with ast corrections, though was not
            # loaded with them, perhaps they were just written to a new file.
            cor_mag1_full = sgal.data.get_col(filt1)
            cor_mag2_full = sgal.data.get_col(filt2)
            rec1, = np.nonzero(np.abs(cor_mag1_full) < 90.)
            rec2, = np.nonzero(np.abs(cor_mag2_full) < 90.)
            if slice_inds is not None:
                rec = list(set(rec1) & set(rec2) & set(slice_inds))
            else:
                rec = list(set(rec1) & set(rec2))
            cor_mag1 = cor_mag1_full[rec]
            cor_mag2 = cor_mag2_full[rec]

        cor_mags = np.column_stack((cor_mag1, cor_mag2))

        with open(outfile, 'w') as f:
            f.write('# %s %s \n' % (filt1, filt2))
            np.savetxt(f, cor_mags, fmt='%10.5f')
        print('wrote %s' % outfile)
    else:
        print('warning %s exists, send overwrite=True arg to overwrite' % outfile)
    return outfile


def change_galaxy_input(galaxy_input, **kwargs):
    '''
    if no kwargs are given, will write None as object_mass and object_sfr_file.
    see galaxy_input_dict()
    '''
    input_pars = InputParameters(galaxy_input_dict())
    input_pars.add_params(kwargs)
    input_pars.write_params(galaxy_input, galaxy_input_fmt())


def find_mag_num(file_mag, filter1):
    file_mag = os.path.join(os.environ['TRILEGAL_ROOT'], file_mag)
    with open(file_mag, 'r') as f:
        line = f.readlines()[1].strip().split()
    try:
        return line.index(filter1)
    except ValueError:
        print('%s not found in %s.' % (filter1, file_mag))


def galaxy_input_dict(photsys=None, filter1=None, object_mass=None,
                      object_sfr_file=None, aringer=False, **kwargs):
    photom_dir = 'odfnew'
    if aringer is True:
        photom_dir += 'bern'
    file_mag = 'tab_mag_%s/tab_mag_%s.dat' % (photom_dir, photsys)
    default = dict({'coord_kind': 1,
                    'coord1': 0.0,
                    'coord2': 0.0,
                    'field_area': 1.0,
                    'kind_mag': 3,
                    'file_mag': file_mag,
                    'file_bcspec': 'bc_odfnew/%s/bc_cspec.dat' % photsys,
                    'kind_dustM': 1,
                    'file_dustM': 'tab_dust/tab_dust_dpmod60alox40_%s.dat' % photsys,
                    'kind_dustC': 1,
                    'file_dustC': 'tab_dust/tab_dust_AMCSIC15_%s.dat' % photsys,
                    'mag_num': find_mag_num(file_mag, filter1),
                    'mag_limit_val': 20,
                    'mag_resolution': 0.1,
                    'r_sun': 8500.0,
                    'z_sun': 24.2,
                    'file_imf': 'tab_imf/imf_chabrier_lognormal.dat',
                    'binary_kind': 0,
                    'binary_frac': 0.0,
                    'binary_mrinf': 0.7,
                    'binary_mrsup': 1.0,
                    'extinction_kind': 0,
                    'extinction_rho_sun': 0.00015,
                    'extinction_infty': 0.045756,
                    'extinction_infty_disp': 0.0,
                    'extinction_h_r': 100000.0,
                    'extinction_h_z': 110.0,
                    'thindisk_kind': 0,
                    'thindisk_rho_sun': 59.0,
                    'thindisk_h_r': 2800.0,
                    'thindisk_r_min': 0.0,
                    'thindisk_r_max': 15000.0,
                    'thindisk_h_z0': 95.0,
                    'thindisk_hz_tau0': 4400000000.0,
                    'thindisk_hz_alpha': 1.6666,
                    'thindisk_sfr_file': 'tab_sfr/file_sfr_thindisk_mod.dat',
                    'thindisk_sfr_mult_factorA': 0.8,
                    'thindisk_sfr_mult_factorB': 0.0,
                    'thickdisk_kind': 0,
                    'rho_thickdisk_sun': 0.0015,
                    'thickdisk_h_r': 2800.0,
                    'thickdisk_r_min': 0.0,
                    'thickdisk_r_max': 15000.0,
                    'thickdisk_h_z': 800.0,
                    'thickdisk_sfr_file': 'tab_sfr/file_sfr_thickdisk.dat',
                    'thickdisk_sfr_mult_factorA': 1.0,
                    'thickdisk_sfr_mult_factorB': 0.0,
                    'halo_kind': 0,
                    'halo_rho_sun': 0.00015,
                    'halo_r_eff': 2800.0,
                    'halo_q': 0.65,
                    'halo_sfr_file': 'tab_sfr/file_sfr_halo.dat',
                    'halo_sfr_mult_factorA': 1.0,
                    'halo_sfr_mult_factorB': 0.0,
                    'bulge_kind': 0,
                    'bulge_rho_central': 76.0,
                    'bulge_am': 1900.0,
                    'bulge_a0': 100.0,
                    'bulge_eta': 0.5,
                    'bulge_csi': 0.6,
                    'bulge_phi': 20.0,
                    'bulge_cutoffmass': 0.8,
                    'bulge_sfr_file': 'tab_sfr/file_sfr_bulge.dat',
                    'bulge_sfr_mult_factorA': 1.0,
                    'bulge_sfr_mult_factorB': 0.0,
                    'object_kind': 1,
                    'object_mass': object_mass,
                    'object_dist': 10.,
                    'object_avkind': 1,
                    'object_av': 0.0,
                    'object_cutoffmass': 0.8,
                    'object_sfr_file': object_sfr_file,
                    'object_sfr_mult_factorA': 1.0,
                    'object_sfr_mult_factorB': 0.0,
                    'output_file_type': 1}.items() + kwargs.items())
    return default


def cmd_input_dict():
    return {'kind_tracks': 2,
            'file_isotrack': 'isotrack/parsec/CAF09_S12D_NS.dat',
            'file_lowzams': 'isotrack/bassazams_fasulla.dat',
            'kind_tpagb': 4,
            'file_tpagb': 'isotrack/isotrack_agb/tracce_CAF09_S_JAN13.dat',
            'kind_postagb': 1,
            'file_postagb': 'isotrack/final/pne_wd_test.dat',
            'ifmr_kind': 0,
            'file_ifmr': 'tab_ifmr/weidemann.dat',
            'kind_rgbmloss': 1,
            'rgbmloss_law': 'Reimers',
            'rgbmloss_efficiency': 0.2}


class Trilegal_SFH(object):
    def __init__(self, filename, galaxy_input=True):
        '''
        file can be galaxy input file for trilegal or trilegal age, sfr, z
        file.
        '''
        if galaxy_input is True:
            self.galaxy_input = filename
            self.current_galaxy_input = filename
        else:
            self.sfh_file = filename
            self.current_sfh_file = filename
        self.load_sfh()

    def load_sfh(self):
        if not hasattr(self, 'sfh_file'):
            with open(self.galaxy_input, 'r') as f:
                lines = f.readlines()
            self.sfh_file = lines[-3].split()[0]
            self.current_sfh_file = self.sfh_file[:]
            self.galaxy_input_sfh_line = ' '.join(lines[-3].split()[1:])
        try:
            self.age, self.sfr, z = np.loadtxt(self.sfh_file, unpack=True)
        except ValueError:
            self.age, self.sfr, z, self.zdisp = np.loadtxt(self.sfh_file, unpack=True)

        # should I do this with dtype?
        self.z_raw = z
        self.z = np.round(z, 4)

    def __format_cut_age(self, cut1_age):
        '''
        takes the > or < out of the string, and makes it in yrs.
        '''
        flag = cut1_age[0]
        yrfmt = 1.
        possible_yrmfts = {'Gyr': 1e9, 'Myr': 1e6, 'yr': 1.}
        for py, yrfmt in utils.sort_dict(possible_yrmfts, reverse=True):
            if py in str(cut1_age):
                import matplotlib
                if matplotlib.cbook.is_numlike(flag):
                    cut1_age = float(cut1_age.replace(py, ''))
                    flag = ''
                else:
                    cut1_age = float(cut1_age.replace(py, '').replace(flag, ''))
                cut1_age *= yrfmt
        return cut1_age, flag

    def increase_sfr(self, factor, cut_age, over_write_galaxy_input=True):
        '''
        cut_age is in Myr.
        '''
        new_fmt = '%s_inc%i.dat'
        new_file = new_fmt % (self.sfh_file.replace('.dat', ''), factor)
        if over_write_galaxy_input is True:
            galaxy_input = self.galaxy_input
        else:
            galaxy_input = new_fmt % (self.galaxy_input.replace('.dat', ''),
                                      factor)

        # copy arrays to not overwrite attributes
        sfr = self.sfr[:]
        age = self.age[:]
        z = self.z[:]

        # convert cut_age to yr
        cut_age *= 1e6

        inds, = np.nonzero(age <= (cut_age))
        sfr[inds] *= factor
        np.savetxt(new_file, np.array([age, sfr, z]).T)
        # update galaxy_input file
        print('this is broken!!!!!')
        #lines[-3] = '%s %s \n' % (new_file, self.galaxy_input_sfh_line)
        #logger.debug('new line: %s' % lines[-3])
        #with open(galaxy_input, 'w') as out:
        #    [out.write(l) for l in lines]

        self.current_galaxy_input = galaxy_input
        self.current_sfh_file = new_file
        return self.current_galaxy_input

    def adjust_value(self, val_str, str_operation, filename='default'):
        '''
        do some operation to a value.
        '''
        val = self.__getattribute__(val_str)
        if filename == 'default':
            base_dir = os.path.split(self.sfh_file)[0]
            new_dir = '_'.join([base_dir, 'adj/'])
            ensure_dir(new_dir)
            with open(os.path.join(new_dir, 'readme'), 'a') as out:
                out.write('SFH file %s adjusted from %s.' %
                          (os.path.split(self.sfh_file)[1], self.sfh_file))
                out.write('\n     operation: %s %s.\n' % (val_str,
                                                          str_operation))
            filename = os.path.join(new_dir, os.path.split(self.sfh_file)[1])

        newval = np.array([eval('%.4f %s' % (v, str_operation)) for v in val])
        self.write_sfh(filename, val_dict={val_str: newval})

    def write_sfh(self, filename, val_dict=None):
        '''
        write the sfh file either give age, sfr, or z or will use
        self.age self.sfr or self.z
        '''
        val_dict = val_dict or {}
        default_dict = {'age': self.age, 'sfr': self.sfr, 'z': self.z}
        new_dict = dict(default_dict.items() + val_dict.items())

        np.savetxt(filename, np.array((new_dict['age'],
                                       new_dict['sfr'],
                                       new_dict['z'])).T,
                   fmt='%.3f %g %.4f')

        print('wrote %s' % filename)

    def boost_split(self):
        boost_ages, = np.nonzero(1.4 <= self.age/1e9 <= 1.8)
        boost_z = 0.001
        boost_table = ''


def find_photsys_number(photsys, filter1):
    '''
    grabs the index of the filter in the tab mag file for this photsys.
    returns the index, and the file name
    '''
    mag_file = os.path.join(os.environ['BCDIR'],
                            'tab_mag_odfnew/tab_mag_%s.dat' % photsys)
    magline = open(mag_file, 'r').readlines()[1].strip().split()
    return magline.index(filter1), mag_file


def run_trilegal(cmd_input, galaxy_input, output, loud=False, dry_run=False):
    '''
    runs trilegal with os.system. might be better with subprocess? Also
    changes directory to trilegal root, if that's not in a .cshrc need to
    somehow tell it where to go.

    rmfiles: set to false if running with multiprocessing, destroying files,
    even if not necessary, seem to break .get()

    loud: after the trilegal run, prints trilegal messages

    to do:
    add -a or any other flag options
    possibly add the stream output to the end of the output file.
    '''
    import subprocess
    here = os.getcwd()
    os.chdir(os.environ['TRILEGAL_ROOT'])
    if here != os.getcwd():
        if not os.path.isfile(os.path.split(cmd_input)[1]):
            os.system('cp %s .' % cmd_input)
    cmd_input = os.path.split(cmd_input)[1]

    if loud:
        logging.info('running trilegal...')

    # trilegal 2.3 not working on my mac...
    if 'Linux' in os.uname():
        ver = 2.3
    else:
        ver = 2.2
    cmd = 'code_%.1f/main -f %s -a -l %s %s > %s.scrn' % (ver, cmd_input,
                                                          galaxy_input, output,
                                                          output)

    if dry_run is True:
        logger.info(cmd)
    else:
        try:
            logger.debug(cmd)
            retcode = subprocess.call(cmd, shell=True)
            if retcode < 0:
                logger.warning('TRILEGAL was terminated by signal', -retcode)
            else:
                logger.info('TRILEGAL was terminated successfully')
        except OSError:
            logger.error('TRILEGAL failed:', sys.exc_info()[1])

    if loud:
        logger.info('done.')

    os.chdir(here)
    return


def get_args(filename, ext='.dat'):
    filename = os.path.split(filename)[1]
    a = filename.split(ext)[0]
    a = a.replace('_', ' ').replace('R1', '').replace('isoch', '')
    s = ''.join(c for c in a if not c.isdigit())
    s = s.replace('.', ' ').split()
    d = {}
    x = a[:]
    s.append(' ')
    for i in range(len(s) - 1):
        if s[i] in a:
            x = x.replace(s[i], '')
            y = x.split(s[i + 1])[0]
            x = x.split(s[i + 1])[-1]
            d[s[i]] = float(y)
    try:
        del d['F']
    except KeyError:
        pass
    return d


def get_stage_label(region=None):
    # see parametri.h
    regions = ['PMS', 'MS', 'SUBGIANT', 'RGB', 'HEB', 'RHEB', 'BHEB', 'EAGB',
               'TPAGB', 'POSTAGB', 'WD']
    if region is None:
        return regions
    if type(region) == str:
        stage_lab = regions.index(region.upper())
    elif type(region) == int or type(region) == float:
        stage_lab = regions[int(region)]
    else:
        stage_lab = [regions[int(r)] for r in region]
    return stage_lab


def get_loop_data(cmd_input_file, metallicity):
    filename = read_cmd_input_file(cmd_input_file)['file_isotrack']
    file_isotrack = os.path.join(os.environ['ISOTRACK'],
                                 filename.replace('isotrack/', ''))
    z, y, mh, files = read_isotrack_file(file_isotrack)
    if not metallicity in z:
        print('warning %s: metallicity not found.' % get_loop_data.__name__)

    ptcri = [f[0] for f in files if str(metallicity) in f[0] and
             f[0].endswith('INT2')][0].replace('isotrack/', '')
    ptcri_file = os.path.join(os.environ['ISOTRACK'], ptcri)
    d = read_loop_from_ptrci(ptcri_file)
    return d


class IsoTrack(object):
    def __init__(self, filename):
        self.base, self.name = os.path.split(filename)
        self.Z, self.Y, self.MH, self.fnames = read_isotrack_file(filename)

    def load_int2(self):
        int2s = [f for f in self.fnames if f.endswith('INT2')]
        self.int2s = [PadovaTrack(i) for i in int2s]
        self.mhefs = np.array([pt.masses[0] for pt in self.int2s])


def read_isotrack_file(filename):
    with open(filename, 'r') as inp:
        isotf = inp.readlines()
    nmets = int(isotf[0])
    z = np.zeros(nmets)
    y = np.zeros(nmets)
    mh = np.zeros(nmets)
    files = []
    met_count = 0
    for j in range(len(isotf)):
        if j == 0:
            continue
        line = isotf[j].strip().split(' ')
        if len(line) == 3:
            z[met_count], y[met_count], mh[met_count] = map(float, line)
            met_count += 1
        else:
            files.append(line)
    files = np.concatenate(files)
    return z, y, mh, files



class trilegal_sfh(object):
    def __init__(self, filename, galaxy_input=True):
        '''
        file can be galaxy input file for trilegal or trilegal age, sfr, z
        file.
        '''
        if galaxy_input is True:
            self.galaxy_input = filename
            self.current_galaxy_input = filename
        else:
            self.sfh_file = filename
            self.current_sfh_file = filename
        self.load_sfh()

    def load_sfh(self):
        if not hasattr(self, 'sfh_file'):
            with open(self.galaxy_input, 'r') as f:
                lines = f.readlines()
            self.sfh_file = lines[-3].split()[0]
            self.current_sfh_file = self.sfh_file[:]
            self.galaxy_input_sfh_line = ' '.join(lines[-3].split()[1:])

        self.age, self.sfr, z = np.loadtxt(self.sfh_file, unpack=True)
        # should I do this with dtype?
        self.z_raw = z
        self.z = np.round(z, 4)

    def __format_cut_age(self, cut1_age):
        '''
        takes the > or < out of the string, and makes it in yrs.
        '''
        flag = cut1_age[0]
        yrfmt = 1.
        possible_yrmfts = {'Gyr': 1e9, 'Myr': 1e6, 'yr': 1.}
        for py, yrfmt in utils.sort_dict(possible_yrmfts, reverse=True):
            if py in str(cut1_age):
                import matplotlib
                if matplotlib.cbook.is_numlike(flag):
                    cut1_age = float(cut1_age.replace(py, ''))
                    flag = ''
                else:
                    cut1_age = float(cut1_age.replace(py, '').replace(flag, ''))
                cut1_age *= yrfmt
        return cut1_age, flag

    def increase_sfr(self, factor, cut_age, over_write_galaxy_input=True):
        '''
        cut_age is in Myr.
        '''
        new_fmt = '%s_inc%i.dat'
        new_file = new_fmt % (self.sfh_file.replace('.dat', ''), factor)
        if over_write_galaxy_input is True:
            galaxy_input = self.galaxy_input
        else:
            galaxy_input = new_fmt % (self.galaxy_input.replace('.dat', ''),
                                      factor)

        # copy arrays to not overwrite attributes
        sfr = self.sfr[:]
        age = self.age[:]
        z = self.z[:]

        # convert cut_age to yr
        cut_age *= 1e6

        inds, = np.nonzero(age <= (cut_age))
        sfr[inds] *= factor
        np.savetxt(new_file, np.array([age, sfr, z]).T)
        # update galaxy_input file
        print('this is broken!!!!!')
        #lines[-3] = '%s %s \n' % (new_file, self.galaxy_input_sfh_line)
        #logger.debug('new line: %s' % lines[-3])
        #with open(galaxy_input, 'w') as out:
        #    [out.write(l) for l in lines]

        self.current_galaxy_input = galaxy_input
        self.current_sfh_file = new_file
        return self.current_galaxy_input

    def adjust_value(self, val_str, str_operation, filename='default'):
        '''
        do some operation to a value.
        '''
        val = self.__getattribute__(val_str)
        if filename == 'default':
            base_dir = os.path.split(self.sfh_file)[0]
            new_dir = '_'.join([base_dir, 'adj/'])
            ensure_dir(new_dir)
            with open(os.path.join(new_dir, 'readme'), 'a') as out:
                out.write('SFH file %s adjusted from %s.' %
                          (os.path.split(self.sfh_file)[1], self.sfh_file))
                out.write('\n     operation: %s %s.\n' % (val_str,
                                                          str_operation))
            filename = os.path.join(new_dir, os.path.split(self.sfh_file)[1])

        newval = np.array([eval('%.4f %s' % (v, str_operation)) for v in val])
        self.write_sfh(filename, val_dict={val_str: newval})

    def write_sfh(self, filename, val_dict=None):
        '''
        write the sfh file either give age, sfr, or z or will use
        self.age self.sfr or self.z
        '''
        val_dict = val_dict or {}
        default_dict = {'age': self.age, 'sfr': self.sfr, 'z': self.z}
        new_dict = dict(default_dict.items() + val_dict.items())

        np.savetxt(filename, np.array((new_dict['age'],
                                       new_dict['sfr'],
                                       new_dict['z'])).T,
                   fmt='%.3f %g %.4f')

        print('wrote %s' % filename)


### input file formats


def galaxy_input_fmt():
    fmt = \
        """%(coord_kind)i %(coord1).1f %(coord2).1f %(field_area).1f # 1: galactic l, b (deg), field_area (deg2) # 2: ra dec in ore ( gradi 0..24)

%(kind_mag)i %(file_mag)s # kind_mag, file_mag
%(file_bcspec)s
%(kind_dustM)i %(file_dustM)s # kind_dustM, file_dustM
%(kind_dustC)i %(file_dustC)s # kind_dustC, file_dustC
%(mag_num)i %(mag_limit_val).1f %(mag_resolution).1f # Magnitude: num, limiting value, resolution

%(r_sun).1f %(z_sun).1f # r_sun, z_sun: sun radius and height on disk (in pc)

%(file_imf)s # file_imf
%(binary_kind)i # binary_kind: 0=none, 1=yes
%(binary_frac).2f # binary_frac: binary fraction
%(binary_mrinf).1f %(binary_mrsup).1f  # binary_mrinf, binary_mrsup: limits of mass ratios if binary_kind=1

%(extinction_kind)i  # extinction kind: 0=none, 1=exp with local calibration, 2=exp with calibration at infty
%(extinction_rho_sun)f  # extinction_rho_sun: local extinction density Av, in mag/pc
%(extinction_infty)f %(extinction_infty_disp).1f  # extinction_infty: extinction Av at infinity in mag, dispersion
%(extinction_h_r).1f %(extinction_h_z).1f  # extinction_h_r, extinction_h_z: radial and vertical scales

%(thindisk_kind)i  # thindisk kind: 0=none, 1=z_exp, 2=z_sech, 3=z_sech2
%(thindisk_rho_sun).1f  # thindisk_rho_sun: local thindisk surface density, in stars formed/pc2
%(thindisk_h_r).1f %(thindisk_r_min).1f %(thindisk_r_max).1f  # thindisk_h_r, thindisk_r_min,max: radial scale, truncation radii
%(thindisk_h_z0).1f %(thindisk_hz_tau0).1f %(thindisk_hz_alpha)%f # thindisk_h_z0, thindisk_hz_tau0, thindisk_hz_alpha: height now, increase time, exponent
%(thindisk_sfr_file)s %(thindisk_sfr_mult_factorA).1f %(thindisk_sfr_mult_factorB).1f # File with (t, SFR, Z), factors A, B (from A*t + B)

%(thickdisk_kind)i # thickdisk kind: 0=none, 1=z_exp, 2=z_sech2, 3=z_sech2
%(rho_thickdisk_sun)f  # rho_thickdisk_sun: local thickdisk volume density, in stars formed/pc3
%(thickdisk_h_r).1f %(thickdisk_r_min).1f %(thickdisk_r_max).1f # thickdisk_h_r, thickdisk_r_min,max: radial scale, truncation radii
%(thickdisk_h_z).1f # thickdisk_h_z: scale heigth (a single value)
%(thickdisk_sfr_file)s %(thickdisk_sfr_mult_factorA).1f %(thickdisk_sfr_mult_factorB).1f  # File with (t, SFR, Z), factors A, B

%(halo_kind)i # halo kind: 0=none, 1=1/r^4 cf Young 76, 2=oblate cf Gilmore
%(halo_rho_sun)f # 0.0001731 0.0001154 halo_rho_sun: local halo volume density, to be done later: 0.001 for 1
%(halo_r_eff).1f %(halo_q).2f #  halo_r_eff, halo_q: effective radius on plane (about r_sun/3.0), and oblateness
%(halo_sfr_file)s %(halo_sfr_mult_factorA).1f %(halo_sfr_mult_factorB).1f # File with (t, SFR, Z), factors A, B

%(bulge_kind)i  # bulge kind: 0=none, 1=cf. Bahcall 86, 2=cf. Binney et al. 97
%(bulge_rho_central).1f # bulge_rho_central: central bulge volume density, unrelated to solar position
%(bulge_am).1f %(bulge_a0).1f #  bulge_am, bulge_a0: scale length and truncation scale length
%(bulge_eta).1f %(bulge_csi).1f %(bulge_phi).1f #  bulge_eta, bulge_csi, bulge_phi0: y/x and z/x axial ratios, angle major-axis sun-centre-line (deg)
%(bulge_cutoffmass).1f # bulge_cutoffmass: (Msun) masses lower than this will be ignored
%(bulge_sfr_file)s %(bulge_sfr_mult_factorA).1f %(bulge_sfr_mult_factorB).1f # File with (t, SFR, Z), factors A, B

%(object_kind)i # object kind: 0=none, 1=at fixed distance
%(object_mass)g %(object_dist).1f # object_mass, object_dist: total mass inside field, distance
%(object_avkind)i %(object_av).2f # object_avkind, object_av: Av added to foregroud if =0, not added if =1
%(object_cutoffmass).1f # object_cutoffmass: (Msun) masses lower than this will be ignored
%(object_sfr_file)s %(object_sfr_mult_factorA).1f %(object_sfr_mult_factorB).1f # File with (t, SFR, Z), factors A, B # la vera eta' e' t_OK=A*(t+B)

%(output_file_type)i # output file: 1=data points"""
    return fmt


def cmd_input_fmt():
    fmt = \
        """%(kind_tracks)i %(file_isotrack)s %(file_lowzams)s # kind_tracks, file_isotrack, file_lowzams
%(kind_tpagb)i %(file_tpagb)s # kind_tpagb, file_tpagb
%(kind_postagb)i %(file_postagb)s # kind_postagb, file_postagb DA VERIFICARE file_postagb
%(ifmr_kind)i %(file_ifmr)s # ifmr_kind, file with ifmr
%(kind_rgbmloss)i %(rgbmloss_law)s %(rgbmloss_efficiency).2f # RGB mass loss: kind_rgbmloss, law, and its efficiency
################################explanation######################
kind_tracks: 1= normal file
file_isotrack: tracks for low+int mass
file_lowzams: tracks for low-ZAMS
kind_tpagb: 0= none
        1= Girardi et al., synthetic on the flight, no dredge up
        2= Marigo & Girardi 2001, from file, includes mcore and C/O
        3= Marigo & Girardi 2007, from file, includes period, mode and mloss
        4= Marigo et al. 2011, from file, includes slope
file_tpagb: tracks for TP-AGB

kind_postagb: 0= none
        1= from file
file_postagb: PN+WD tracks

kind_ifmr: 0= default
           1= from file

kind_rgbmloss: 0=off
               1=on (with law=Reimers for the moment)"""
    return fmt
