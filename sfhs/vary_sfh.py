"""
Run many trilegal simulations and cull scaled LF functions to compare with data
"""
import argparse
import logging
import numpy as np
import os
import sys
import time

from ..trilegal.utils import galaxy_input_dict, galaxy_input_fmt
from dweisz.match import scripts as match
from .star_formation_histories import StarFormationHistories as SFH
from ..pop_synth.stellar_pops import limiting_mag
from .. import fileio

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
    if target in ['ddo71', 'hs117', 'kkh37',
                  'ddo78', 'scl-de1', 'ugc8508', 'ngc2403-halo-6', 'ngc2403-deep']:
        mass = 1.0e+08
    elif target in ['ic2574-sgs', 'ngc2976-deep', 'ugc5139', 'ngc300-wide1', 'ngc3741',
                    'ngc404-deep', 'ugc4459', 'm81-deep', 'ugca292']:
        mass = 2.5e+08
    elif target in ['ugc4305-1', 'ugc4305-2', 'ngc4163', 'ddo82', 'eso540-030',
                    'ngc3077-phoenix']:
        mass = 5.0e+08
    else:
        mass = 1.0e+07
        logger.warning('no info on object mass for {}, assuming {:.2e} Msun'.format(target, mass))

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
            inp_obj : fileio.InputParameters object
                input parameters object
            input_file : path to file that can be read into a dictionary via fileio.load_input

            Necessary contents of input_file/inp_obj
            ------------------
            file_origin : str
                what type of SFH file (match-grid, match-hmc)

            filter1, filter2 : str, str
                V, I filters. Used only in file name conventions

            galaxy_input : str
                template galaxy input. object_mass and sfr-z file will be adjusted

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
        [self.__setattr__(k, v) for k, v in indict.items()]
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
                             file_imf=None, object_cutoffmass=0.8):
        '''
        write the galaxy input file

        TO DO:
        BF could/should come from match param file... could make a mistake here
        also could do better with IMF...
        wfc3snap and filter1 are hard coded...
        '''
        self.galaxy_inputs = []
        msfh = match.sfh.SFH(self.sfh_file, meta_file=self.meta_file)

        if msfh.IMF == 0:
            file_imf = 'tab_imf/imf_kroupa02.dat'

        gal_dict = \
            {'mag_limit_val': limiting_mag(self.fake_file, 0.1)[1],
             'object_av': msfh.Av,
             'object_dist': 10 ** (msfh.dmod / 5. + 1.),
             'photsys': 'wfc3snap',
             'object_mass': object_mass or load_sim_masses(self.target),
             'file_imf': file_imf,
             'filter1': 'F814W',
             'object_cutoffmass': object_cutoffmass}

        trigal_dict = galaxy_input_dict(**gal_dict)

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
                gal_inp = fileio.InputParameters(default_dict=trigal_dict)
                gal_inp.write_params(new_out, galaxy_input_fmt())
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
                                                      overwrite=overwrite,
                                                      sample=self.burstsfh)

        self.prepare_galaxy_input(overwrite=overwrite, object_mass=object_mass)
        return

    def run_once(self, galaxy_input=None, triout=None, ite=0, overwrite=False):
        """write call to trilegal string"""
        ver = 2.3
        call = 'nice -n +19 taskset -c {0} code_{1}/main'.format(ite, ver)
        call += ' -f {0} -a -l {1} {2} > {2}.scrn'.format(self.cmd_input_file,
                                                          galaxy_input,
                                                          triout)
        return call

    def call_run(self, nproc=8, overwrite=False):
        """Call run_once or run_parallel depending on self.nsfh value"""
        if self.nsfhs <= 1:
            self.prepare_trilegal_files(random_sfr=False, random_z=False,
                                        zdisp=False, overwrite=overwrite)
            cmd = self.run_once(galaxy_input=self.galaxy_inputs[0],
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
        logger.info('appending to output file: {}'.format(outfile))
        with open(outfile, 'a') as out:
            out.write(line)
    return

def main(argv):
    """main function to call_VarySFH"""
    description="Run trilegal many times by randomly sampling SFH uncertainies."
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='verbose mode')

    parser.add_argument('-n', '--nproc', type=int, default=8,
                        help='number of processors')

    parser.add_argument('-b', '--burstsfh', action='store_true',
                        help='use Ben Johnson\'s scombine to sample the SFH')

    parser.add_argument('-f', '--overwrite', action='store_true',
                        help='write call to trilegal even if output file exists')

    parser.add_argument('-s', '--nsfhs', type=int, default=1,
                        help='number of times to sample sfh')

    parser.add_argument('-c', '--cmd_input_file', type=str, default='cmd_input_CAF09_V1.2S_M36_S12D_NS_NAS.dat',
                        help='trilegal cmd input file')

    parser.add_argument('-o', '--outfile', type=str, default='trilegal_script.sh',
                        help='Output file to run trilegal')

    parser.add_argument('-e', '--hmc_file', type=str,
                        help='MATCH HybridMC file')

    parser.add_argument('-m', '--meta_file', type=str,
                        help='MATCH output file with bestfit Av and dmod, and IMF if sfh_file is a zcmerge file.')

    parser.add_argument('sfh_file', type=str,
                        help='MATCH SFH file: must have the format target_filter1_filter2.extensions')

    parser.add_argument('fake_file', type=str,
                        help='AST file (for stellar pop mag depth)')

    args = parser.parse_args(argv)

    target, filter1, filter2 = \
        os.path.split(fileio.replace_ext(args.sfh_file, ''))[1].split('_')[:3]

    agb_mod = args.cmd_input_file.replace('cmd_input_', '').replace('.dat', '').lower()
    outfile_loc = os.path.join(os.getcwd(), agb_mod)
    fileio.ensure_dir(outfile_loc)

    indict = {'cmd_input_file': args.cmd_input_file,
              'fake_file': args.fake_file,
              'filter1': filter1,
              'filter2': filter2,
              'hmc_file': args.hmc_file,
              'nsfhs': args.nsfhs,
              'outfile_loc': outfile_loc,
              'sfh_file': args.sfh_file,
              'target': target,
              'meta_file': args.meta_file,
              'burstsfh': args.burstsfh}

    call_VarySFH(indict, loud=args.verbose, nproc=args.nproc,
                 overwrite=args.overwrite, outfile=args.outfile)


if __name__ == '__main__':
    main(sys.argv[1:])
