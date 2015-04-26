"""
Run many trilegal simulations and cull scaled LF functions to compare with data
"""
import argparse
import logging
import numpy as np
import os
import sys
import time

import ResolvedStellarPops as rsp

from .star_formation_histories import StarFormationHistories as SFH
from ..pop_synth.stellar_pops import limiting_mag

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

__all__ = ['VarySFHs']

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
        logger.warning('no info on object mass for {}, assuming 5e8Msun'.format(target))
        mass = 5.0e+08
    return mass

def jobwait(line=''):
    """add bash script to wait for current jobs to finish"""
    line += "\nfor job in `jobs -p`\ndo\n    echo $job\n    wait $job\ndone\n\n"
    return line

class VarySFHs(SFH):
    '''run several variations of the age sfr z from SFH'''
    def __init__(self, indict):
        """Vary the SFH from MATCH for a trilegal simulations
        
        Parameters
        ----------
        filter1, filter2 : str, str
        V, I filters. Used only in file name conventions
            
        outfile_loc : str
            path to put the trilegal output files
    
        target : str
            name of observation target for file name conventions
    
        hmc_file : str
            path to the Hybrid MCMC file
        
        sfh_file : str
            path to the MATCH SFH file
            
        cmd_input_file : str
            path to the cmd input file to run TRILEGAL
    
        object_mass : str, will be converted to float
            optional, overwrite the mass set in galaxy_input
    
        nsfhs : str, will be converted to int
            number of sfhs to sample
        """
        # load SFH instance to make lots of trilegal runs
        self.initialize_inputs(indict)

        # load in hmc data to self.data
        SFH.__init__(self, hmc_file=self.hmc_file, sfh_file=self.sfh_file)

        # setup file formats
        self.trilegal_file_fmt()

    def initialize_inputs(self, indict):
        """load input parameters needed for vary_sfh"""
        # parameters needed
        inputs = ['filter1', 'filter2', 'outfile_loc', 'target', 'hmc_file',
                  'cmd_input_file', 'fake_file', 'nsfhs', 'sfh_file']

        needed = [k for k in inputs if not k in indict.keys()]
        if len(needed) > 0:
            logger.error('missing needed input parameters: {}'.format(needed))

        unused = [k for k in indict.keys() if not k in inputs]
        if len(unused) > 0:
            logger.warning('not using {}'.format(unused))

        [self.__setattr__(k, v) for k, v in indict.items() if k in inputs]
        return

    def trilegal_file_fmt(self):
        """ file name formats for trilegal input and trilegal sfr-z """
        tmp = os.path.split(self.cmd_input_file)[1]
        agb_mod = tmp.lower().replace('.dat', '').replace('cmd_input_', '')
        # trilegal output format
        self.tname = \
            os.path.join(self.outfile_loc, 'out_%s_%s_%s_%s' % (self.target,
                                                                self.filter1,
                                                                self.filter2,
                                                                agb_mod))
        self.triout_fmt = self.tname + '_%003i.dat'

        sfr_fmt = '{}_{}_%003i.trisfr'.format(self.target, self.filter1)
        self.sfr_fmt = os.path.join(self.outfile_loc, sfr_fmt)
        
        galinp_fmt = '{}_{}_%003i.galinp'.format(self.target, self.filter1)
        self.galinp_fmt = os.path.join(self.outfile_loc, galinp_fmt)
        
    def prepare_galaxy_input(self, object_mass=None, overwrite=False,
                             file_imf=None, binary_frac=0.35,
                             object_cutoffmass=0.8):
        '''
        write the galaxy input file
        
        TO DO:
        BF could/should come from match param file... could make a mistake here
        also could do better with IMF...
        wfc3snap and filter1 are hard coded...
        '''
        self.galaxy_inputs = []
        msfh = rsp.match.utils.MatchSFH(self.sfh_file)
        
        if msfh.IMF == 0:
            file_imf = 'tab_imf/imf_kroupa_match.dat'
        
        gal_dict = \
            {'mag_limit_val': limiting_mag(self.fake_file, 0.1)[1],
             'object_av': msfh.Av,
             'object_dist': 10 ** (msfh.dmod / 5. + 1.),
             'photsys': 'wfc3snap',
             'object_mass': object_mass or load_sim_masses(self.target),
             'file_imf': file_imf,
             'filter1': 'F814W',
             'object_cutoffmass': object_cutoffmass}
    
        trigal_dict = rsp.trilegal.utils.galaxy_input_dict(**gal_dict)
        
        for i in range(len(self.sfr_files)):
            trigal_dict['object_sfr_file'] =  self.sfr_files[i]
            if len(self.sfr_files) == 1:
                new_out = os.path.join(self.outfile_loc,
                                       '{}_{}.galinp'.format(self.target,
                                                             self.filter1))
            else:
                new_out = self.galinp_fmt % i
            self.galaxy_inputs.append(new_out)
            if not os.path.isfile(new_out) or overwrite:
                gal_inp = rsp.fileio.InputParameters(default_dict=trigal_dict)
                gal_inp.write_params(new_out, rsp.trilegal.utils.galaxy_input_fmt())
            else:
                logger.info('not overwritting {}'.format(new_out))

    def prepare_trilegal_files(self, random_sfr=True, random_z=False,
                               zdisp=False, overwrite=False, object_mass=None):
        '''make the sfhs, make the galaxy inputs'''
        self.sfr_files = self.make_many_trilegal_sfhs(nsfhs=self.nsfhs,
                                                      outfile_fmt=self.sfr_fmt,
                                                      random_sfr=random_sfr,
                                                      random_z=random_z,
                                                      zdisp=zdisp,
                                                      dry_run=overwrite)

        self.prepare_galaxy_input(overwrite=overwrite, object_mass=object_mass)
        return

    def run_once(self, galaxy_input=None, triout=None, ite=0, overwrite=False):
        """write call to trilegal string"""
        flag = 0
        ver = 2.3
        call = ''

        #if os.path.isfile(triout) and not overwrite:
        #    logger.warning('{} exists, will overwrite if no hdf5 file found'.format(triout))
        #    flag += 1

        hdf5file = rsp.fileio.replace_ext(triout, 'hdf5')
        if os.path.isfile(hdf5file) and not overwrite:
            logger.warning('{} already exists, not calling trilegal'.format(hdf5file))
            flag += 1
        
        if flag < 1:
            call = 'nice -n +19 taskset -c {0} code_{1}/main'.format(ite, ver)
            call += ' -f {0} -a -l {1} {2} > {2}.scrn'.format(self.cmd_input_file,
                                                               galaxy_input,
                                                               triout)
        return call

    def call_run(self, dry_run=False, nproc=8, overwrite=False):
        """Call run_once or run_parallel depending on self.nsfh value"""
        if self.nsfhs <= 1:
            self.prepare_trilegal_files(random_sfr=False, random_z=False,
                                        zdisp=False, overwrite=overwrite)
            cmd = self.run_once(galaxy_input='{}.galinp'.format(self.target),
                                triout=self.tname + '_bestsfr.dat',
                                overwrite=overwrite)
            cmd += ' &\n'
        else:
            cmd = self.run_many(nproc=nproc, overwrite=overwrite)
        return cmd

    def run_many(self, nproc=8, overwrite=False):
        """Call self.run_once self.nsfh of times in iterations base on nproc"""
        self.prepare_trilegal_files(random_sfr=True, random_z=False,
                                    zdisp=False, overwrite=overwrite)

        # How many sets of calls to the max number of processors
        niters = np.ceil(self.nsfhs / float(nproc))
        sets = np.arange(niters * nproc, dtype=int).reshape(niters, nproc)

        line = ''
        for j, iset in enumerate(sets):
            # don't use not needed procs
            iset = iset[iset < self.nsfhs]
            for i in range(len(iset)):
                cmd = self.run_once(galaxy_input=self.galaxy_inputs[iset[i]],
                                    triout=self.triout_fmt % iset[i], ite=i,
                                    overwrite=overwrite)
                line += '{} &\n'.format(cmd)
            line += jobwait()
        return line

def call_VarySFH(inputs, loud=False, nproc=8, outfile='trilegal_script.sh',
                 overwrite=False):
    """
    write a script to run trilegal in parallel sampling the sfh
    
    overwrite may not work as expected:
    a) for trilegal input files (*.galinp, *.sfr): won't overwrite
    b) trilegal output: won't write the command to call trilegal if the .hdf5
       file already exists. (Will still overwrite the .dat file)
    """
    # set up logging
    handler = logging.FileHandler('vary_sfh.log')
    logger.setLevel(logging.DEBUG)
    if loud:
        handler.setLevel(logging.DEBUG)
    else:
        handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info('logger writing to vary_sfh.log')

    vsh = VarySFHs(inputs)

    line = vsh.call_run(nproc=nproc, overwrite=overwrite)
    if outfile is None:
        print(line)
    else:
        logger.info('output file: {}'.format(outfile))
        with open(outfile, 'a') as out:
            out.write(line)
    return

def main(argv):
    """main function to call_VarySFH"""
    parser = argparse.ArgumentParser(description="Run trilegal many times by \
                                     randomly sampling SFH uncertainies")

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='verbose mode')

    parser.add_argument('-n', '--nproc', type=int, default=8,
                        help='number of processors')

    parser.add_argument('-f', '--overwrite', action='store_true',
                        help='write call to trilegal even if output file exists')

    parser.add_argument('-s', '--nsfhs', type=int, default=1,
                        help='number of times to sample sfh')

    parser.add_argument('-c', '--cmd_input_file', type=str, default='cmd_input_CAF09_V1.2S_M36_S12D_NS_NAS.dat',
                        help='trilegal cmd input file')

    parser.add_argument('sfh_file', type=str,
                        help='MATCH SFH file')

    parser.add_argument('hmc_file', type=str,
                        help='MATCH HybridMC file')

    parser.add_argument('fake_file', type=str,
                        help='AST file (for stellar pop mag depth)')

    args = parser.parse_args(argv)
    target, filter1, filter2 = \
        os.path.split(args.sfh_file)[1].split('.')[0].split('_')
    
    agb_mod = args.cmd_input_file.replace('cmd_input_', '').replace('.dat', '').lower()
    outfile_loc = os.path.join(os.getcwd(), agb_mod)
    rsp.fileio.ensure_dir(outfile_loc)

    indict = {'filter1': filter1,
              'filter2': filter2,
              'outfile_loc': outfile_loc,
              'target': target,
              'hmc_file': args.hmc_file,
              'cmd_input_file': args.cmd_input_file,
              'fake_file': args.fake_file,
              'nsfhs': args.nsfhs,
              'sfh_file': args.sfh_file}

    call_VarySFH(indict, loud=args.verbose, nproc=args.nproc,
                 overwrite=args.overwrite)


if __name__ == '__main__':
    main(sys.argv[1:])
