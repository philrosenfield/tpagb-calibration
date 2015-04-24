"""Run calcsfh or hybridMC in Parallel (using subprocess)"""
import argparse
import logging
import os
import subprocess
import sys

import numpy as np

from ResolvedStellarPops.match.utils import match_diagnostic
from ResolvedStellarPops.fileio.fileIO import get_files

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Could be in a config or environ
calcsfh = '$HOME/research/match2.5/bin/calcsfh'
zcombine = '$HOME/research/match2.5/bin/zcombine'
hybridmc = '$HOME/research/match2.5/bin/hybridMC'

def match_table(sfh_files):
    rows = []
    rowfmt = r'{0} & {1} & {2} & {3} & {4:.2f} & {5:.2f} & {6:.2f} & {7:.2f} & {8} \\'
    for filename in sfh_files:
        mcmd = rsp.match.utils.MatchSFH(filename)
        try:
            target, filters = rsp.asts.parse_pipeline(filename)
        except:
            target, filters = '...', ['...', '...']
        
        filters = ','.join(filters)
        
        agegyr = 10 ** (mcmd.data.lagef - 9)
        isf1, = np.nonzero(agegyr < 1)
        isf13, = np.nonzero((agegyr >= 1) & (agegyr <= 3))
        
        total_sfr = np.sum(mcmd.data.sfr)

        sfr1 = np.sum(mcmd.data.sfr[isf1]) / total_sfr
        sfr13 = np.sum(mcmd.data.sfr[isf13]) / total_sfr
        
        mh1 = np.sum(mcmd.data.mh[isf1])/float(len(np.nonzero(mcmd.data.mh[isf1])[0]))
        mh13 = np.sum(mcmd.data.mh[isf13])/float(len(np.nonzero(mcmd.data.mh[isf13])[0]))
        z1 = ResolvedStellarPops.convertz.convertz(mh=mh1)[1]
        z13 = ResolvedStellarPops.convertz.convertz(mh=mh13)[1]
        feh1 = ResolvedStellarPops.convertz.convertz(mh=mh1)[-2]
        feh13 = ResolvedStellarPops.convertz.convertz(mh=mh13)[-2]
        
        #print target, filters, mcmd.Av, mcmd.dmod, '%.2g' % sfr1, '%.3f' % z1, '%.2g' % sfr13, '%.3f' % z13, mcmd.bestfit
        row = rowfmt.format(target, filters, mcmd.Av, mcmd.dmod, sfr1, feh1,
                            sfr13, feh13, mcmd.bestfit)
        print row
        rows.append(row)
        return rows

def check_params(prefs):
    """
    make a diagnostic plot with the photometery and param file limits
    see ResolvedStellarPops.match.utils.match_diacnostic.__doc__
    """
    for pref in prefs:
        param, match, _ = calcsfh_existing_files(pref)
        match_diagnostic(param, match)

def test_files(prefs, run_calcsfh=True):
    """make sure match input files exist"""
    return_code = 0
    for pref in prefs:
        if run_calcsfh:
            pfiles = calcsfh_existing_files(pref)
        else:
            pfiles = [hybridmc_existing_files(pref)]
        test = [os.path.isfile(f) for f in pfiles]
        if False in test:
            logger.error('missing a file in {}'.format(pref))
            logger.error(pfiles)
            return_code += 1
    if return_code > 0:
        sys.exit(2)
    return

def uniform_filenames(prefs, dry_run=False):
    """
    make all fake match and par files in a directory follow the format
    target_filter1_filter2.gst.suffix all lower case
    use dry_run to print the mv command, or will call os.system.
    """
    for pref in prefs:
        dirname, p = os.path.split(pref)
        filters = '_'.join(p.split('_')[1:])
        print dirname, p, filters
        fake, = get_files(dirname, '*{}*fake'.format(filters))
        match, = get_files(dirname, '*{}*match'.format(filters))
        param, = get_files(dirname, '*{}*param'.format(filters))
        ufake = '_'.join(fake.split('_')[1:]).replace('_gst.fake1',
                                                      '.gst').lower()
        umatch = '_'.join(match.split('_')[1:]).lower()
        uparam = param.replace('.param', '.gst.param').lower()
        for old, new in zip([fake, match, param],[ufake, umatch, uparam]):
            cmd = 'mv {dir}/{old} {dir}/{new}'.format(dir=dirname, old=old,
                                                      new=new)
            logger.info(cmd)
            if not dry_run:
                os.system(cmd)

def calcsfh_existing_files(pref):
    """file formats for param match and matchfake"""
    param, = get_files(os.getcwd(), pref + '*.param')
    match, = get_files(os.getcwd(), pref + '*.match')
    fake, = get_files(os.getcwd(), pref + '*.matchfake')
    return (param, match, fake)


def calcsfh_new_files(pref):
    """file formats for match grid, sdout, and sfh file"""
    out =  pref + '.out'
    scrn = pref + '.scrn'
    sfh = pref + '.sfh'
    return (out, scrn, sfh)


def hybridmc_existing_files(pref):
    """file formats for the HMC, based off of calcsfh_new_files"""
    mcin, = get_files(os.getcwd(), pref + '*.out.dat')
    return mcin


def hybridmc_new_files(pref):
    """file formats for HybridMC output and the following zcombine output"""
    pref = pref.strip()
    mcmc = pref + '.mcmc'
    mcscrn = mcmc + '.scrn'
    mczc = mcmc + '.zc'
    return (mcmc, mcscrn, mczc)


def run_parallel(prefs, dry_run=False, nproc=8, run_calcsfh=True):
    """run calcsfh and zcombine in parallel, flags are hardcoded."""
    test_files(prefs, run_calcsfh)

    rdict = {'calcsfh': calcsfh, 'zcombine': zcombine,'hybridmc': hybridmc}
    # calcsfh
    # calcsfh, param, match, fake, out, scrn
    cmd1 = '{calcsfh} {param} {match} {fake} {out} -PARSEC -mcdata -kroupa -zinc -sub=v2 > {scrn}'
    # zcombine
    #zcombine, out, sfh
    cmd2 = '{zcombine} {out} -bestonly > {sfh}'
    # hybridmc
    #hybridmc, mcin, mcmc, mcscrn
    cmd3 = '{hybridmc} {mcin} {mcmc} -tint=2.0 -nmc=10000 -dt=0.015 > {mcscrn}'
    # zcombine w/ hybrid mc
    #zcombine, mcmc, mczc
    cmd4 = '{zcombine} {mcmc} -unweighted -medbest -jeffreys -best={sfh} > {mczc}'

    niters = np.ceil(len(prefs) / float(nproc))
    sets = np.arange(niters * nproc, dtype=int).reshape(niters, nproc)
    logging.debug('{} prefs, {} niters'.format(len(prefs), niters))

    for j, iset in enumerate(sets):
        # don't use not needed procs
        iset = iset[iset < len(prefs)]
        
        # run calcsfh
        procs = []
        for i in iset:
            if run_calcsfh:
                rdict['param'], rdict['match'], rdict['fake'] = calcsfh_existing_files(prefs[i])
                rdict['out'], rdict['scrn'], rdict['sfh'] = calcsfh_new_files(prefs[i])
                if os.path.isfile(rdict['mcmc']):
                    logger.error('{} exists. Not re-runing calcsfh.'.format(rdict['sfh']))
                    cmd = ''
                else:
                    cmd = cmd1.format(**rdict)
            else:
                rdict['sfh'] = calcsfh_new_files(prefs[i])[-1]
                rdict['mcin'] = hybridmc_existing_files(prefs[i])
                rdict['mcmc'], rdict['mcscrn'], rdict['mczc'] = hybridmc_new_files(prefs[i])
                if os.path.isfile(rdict['mcmc']):
                    logger.error('{} exists. Not re-runing hybridMC.'.format(rdict['mcmc']))
                    cmd = ''
                else:
                    cmd = cmd3.format(**rdict)
            if not dry_run:
                procs.append(subprocess.Popen(cmd, shell=True))
            logger.info(cmd)
        
        # wait for calcsfh
        if not dry_run:
            [p.wait() for p in procs]
            logger.debug('calcsfh or hybridMC set {} complete'.format(j))
        
        # run zcombine
        procs = []
        for i in iset:
            if run_calcsfh:
                rdict['out'], rdict['scrn'], rdict['sfh'] = calcsfh_new_files(prefs[i])
                zcom = cmd2.format(**rdict)
            else:
                zcom = cmd4.format(**rdict)
            if not dry_run:
                procs.append(subprocess.Popen(zcom, shell=True))
            logger.info(zcom)
        
        # wait for zcombine
        if not dry_run:
            [p.wait() for p in procs]
            logger.debug('zcombine set {} complete'.format(j))


def main(argv):
    """parse in put args, setup logger, and call run_parallel"""
    parser = argparse.ArgumentParser(description="Run calcsfh in parallel. Note: bg cmd, if in use, need to be in the current folder")

    parser.add_argument('-d', '--dry_run', action='store_true',
                        help='only print commands')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='set logging to debug')

    parser.add_argument('-n', '--nproc', type=int, default=8,
                        help='number of processors')

    parser.add_argument('-m', '--hmc',  action='store_false',
                        help='run hybridMC (must be after a calcsfh run)')

    parser.add_argument('-t', '--table',  action='store_true',
                        help='make a table of recent sfh')

    parser.add_argument('-f', '--logfile', type=str, default='calcsfh_parallel.log',
                        help='log file name')

    parser.add_argument('-s', '--simplify', action='store_true',
                        help='make filename uniform and exit (before calcsfh run)')

    parser.add_argument('-c', '--checkparam', action='store_true',
                        help='make a diagnostic plot with the phot and param file')

    parser.add_argument('pref_list', type=argparse.FileType('r'),
                        help="list of prefixs to run on. E.g., ls */*.match | sed 's/.match//' > pref_list")

    args = parser.parse_args(argv)
    prefs = [l.strip() for l in args.pref_list.readlines()]
    
    handler = logging.FileHandler(args.logfile)
    if args.verbose:
        handler.setLevel(logging.DEBUG)
    else:
        handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    logger.info('command:', argv)

    if args.simplify:
        uniform_filenames(prefs, dry_run=args.dry_run)
    elif args.checkparam:
        check_params(prefs)
    elif args.table:
        print 'should code here. I stoppped because I think match_table should go to rsp'
    else:
        logger.info('running on {}'.format(', '.join([p.strip() for p in prefs])))
        run_parallel(prefs, dry_run=args.dry_run, nproc=args.nproc, run_calcsfh=args.hmc)


if __name__ == '__main__':
    main(sys.argv[1:])
