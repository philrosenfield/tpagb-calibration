#!/usr/bin/python
"""
prepare input file

sfh_file: hard coded
TRGB: from table or from Mtrgb
Av, Dmod: From match
RGB region: Color limits by hand, TRGB / completeness or offset
Write optical LF
write ratio table
"""
from __future__ import print_function

import argparse
import logging
import os
import sys

import ResolvedStellarPops as rsp

from ResolvedStellarPops.tpagb_path_config import tpagb_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    from pop_synth.stellar_pops import limiting_mag, rgb_agb_regions
else:
    from .pop_synth.stellar_pops import limiting_mag, rgb_agb_regions

angst_data = rsp.angst_tables.angst_data


def load_sim_masses(target):
    '''
    adapted from thesis spaz.

    the simulation should have more than 2.5 times the number of stars in
    the CMD as are in the data. Set here are object_mass that should give
    at least that number of stars based on the best fit sfh.
    '''
    if target in ['ngc3741', 'eso540-030', 'ugc-4305-1', 'kkh37', 'ugc-4305-2',
                  'ngc404', 'ngc2976-deep', 'ngc4163', 'ddo78', 'ngc2403-deep']:
        mass = 5e+08
    elif target in ['ddo82', 'ic2574-sgs']:
        mass = 2.5e+09
    elif target in ['ugc-5139']:
        mass = 1.0e+09
    else:
        logger.warning('no info on object mass for {}, assuming 1e8Msun'.format(target))
        mass = 5.0e+08
    return mass


def prepare_from_directory(directory, search_str, inp_extra):
    """
    Make a partial input file culled from information in several different files
    inp_extra should be used to specify filter1 if more than one filter is in
    a directory.
    """
    directory = os.path.abspath(directory)
    assert os.path.isdir(directory), 'Must supply valid directory name'

    # get matchphot, fake, sfh_file, Hybric MC file
    pars = {'sfh_file': rsp.fileio.get_files(directory,
                                             search_str.format('sfh'))[0]}

    try:
        pars['hmc_file'] = rsp.fileio.get_files(directory,
                                                search_str.format('zc'))[0]
    except:
        pass

    matchphot = rsp.fileio.get_files(directory,
                                     search_str.format('match'))[0]

    target, (filter1, filter2) = rsp.parse_pipeline(matchphot)
    target = target.lower()

    outfile_loc = os.path.join(tpagb_path, 'SNAP/varysfh', target)
    rsp.fileio.ensure_dir(outfile_loc)

    galaxy_input = os.path.join(outfile_loc, '{0}{1}.galinp'.format(target,
                                                                    inp_extra))
    pars.update({'outfile_loc': outfile_loc,
                 'target': target,
                 'filter1': filter1,
                 'filter2': filter2,
                 'galaxy_input': galaxy_input})
    inp = rsp.fileio.InputParameters()
    inp.add_params(pars)

    return inp


def prepare_galaxy_inputfile(inps, fake_file, inp_extra):
    """Make a galaxy input file for trilegal"""
    # If match was run with setz, this is the logz dispersion.
    # Only useful for clusters, also it is not saved in the match output files
    # Only set in the match parameter file.
    #matchphot = rsp.fileio.get_files(directory,
    #                                 search_str.format('match'))[0],
    # trilegal sfr filename
    gal_dict = {'photsys': 'wfc3snap',
                'object_cutoffmass': 0.8,
                'binary_frac': 0.35,
                'file_imf': 'tab_imf/imf_kroupa_match.dat',
                'object_mass': load_sim_masses(inps.target)}

    object_sfr_file = os.path.join(inps.outfile_loc,
                                   '{0}{1}.trisfr'.format(inps.target,
                                                          inp_extra))
    rsp.match.utils.process_match_sfh(inps.sfh_file,
                                      outfile=object_sfr_file,
                                      zdisp=0.00)

    msfh = rsp.match.utils.MatchSFH(inps.sfh_file)
    # filter1 is used here to find the mag depth for trilegal input.
    gal_dict.update({'mag_limit_val': limiting_mag(fake_file, 0.1)[1],
                     'object_av': msfh.Av,
                     'object_dist': 10 ** (msfh.dmod / 5. + 1.),
                     'object_sfr_file': object_sfr_file,
                     'filter1': inps.filter2})

    trigal_dict = rsp.trilegal.utils.galaxy_input_dict(**gal_dict)
    gal_inp = rsp.fileio.InputParameters(default_dict=trigal_dict)
    gal_inp.write_params(inps.galaxy_input,
                         rsp.trilegal.utils.galaxy_input_fmt())
    return


def main(argv):
    """
    need matchphot in the format of target filter1 filter2
    need match sfh file for Av and dmod
    col_min, col_max, mag_faint, mag_bright
    fakefile
    photsys
    'object_mass': inps.object_mass or 1e7,
    'object_sfr_file': inps.object_sfr_file,
    'file_imf': inps.file_imf or 'tab_imf/imf_salpeter.dat',
    'binary_frac': inps.binary_frac or 0.,
    'object_cutoffmass': inps.object_cutoffmass or 0.8}
    """

    parser = argparse.ArgumentParser(description="Create input file \
                                                  for VarySFH")

    parser.add_argument('-c', '--cmd_input_file', type=str,
                        default='cmd_input_parsecCAF09_V1.2S_M36_S12D2.dat',
                        help='create partial input file from \
                              specified directory')

    parser.add_argument('-m', '--object_mass', type=str, default=None,
                        help='simulation object mass')

    parser.add_argument('-v', '--pdb', action='store_true',
                        help='verbose mode')

    parser.add_argument('-f', '--filter', type=str, default=None,
                        help='V filter (if more than one in directory)')

    parser.add_argument('-n', '--nsfhs', type=str, default=1,
                        help='Number of sampled SFHs to run')

    parser.add_argument('name', type=str,
                        help='directory name')

    args = parser.parse_args(argv)

    # set up logging
    handler = logging.FileHandler('{}_prepare_data.log'.format(args.name))
    if args.pdb:
        handler.setLevel(logging.DEBUG)
    else:
        handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if args.filter is not None:
        fsearch = '*{}'.format(args.filter)
        inp_extra = '_{}'.format(args.filter)
    else:
        fsearch = ''
        inp_extra = ''
    search_str = fsearch + '*{}'

    inps = prepare_from_directory(args.name, search_str, inp_extra)

    inps.cmd_input_file = args.cmd_input_file
    inps.nsfhs = args.nsfhs
    inps.object_mass = args.object_mass or load_sim_masses(inps.target)
    outfile = os.path.join(inps.outfile_loc,
                           '{0}{1}.vsfhinp'.format(inps.target, inp_extra))
    inps.write_params(outfile)

    fake_file = rsp.fileio.get_files(args.name, search_str.format('fake'))[0]
    prepare_galaxy_inputfile(inps, fake_file, inp_extra)
    return


if __name__ == "__main__":
    main(sys.argv[1:])
