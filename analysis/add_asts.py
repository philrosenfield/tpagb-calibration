"""
Add AST corrections to trilegal catalogs
"""
import argparse
import logging
import os
import sys
import numpy as np

from ..pop_synth.asts import ASTs, ast_correct_starpop
from ..pop_synth import SimGalaxy
from .. import fileio
# where the matchfake files live
from ..TPAGBparams import snap_src
matchfake_loc = os.path.join(snap_src, 'data', 'galaxies')

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def make_ast_corrections(trilegal_catalogs, filters, outfiles=None,
                         diag_plot=False, overwrite=True, hdf5=False,
                         filter1=None, filter2=None, fake=None, filterset=0):
    """
    see asts.ast_correct_starpop
    """
    if outfiles is None:
        outfmt = 'default'
    else:
        outfiles = np.atleast_1d(outfiles)
        outfmt = 'supplied'

    if len(filters) == 4:
        correct = 'all'
    elif len(filters) == 2:
        correct = 'both'
    else:
        print('Must provide 2 or 4 filters')
        sys.exit(1)

    trilegal_catalogs = np.atleast_1d(trilegal_catalogs)
    fakes = np.atleast_1d(fake)
    asts = [ASTs(f, filters=filters, filterset=filterset) for f in fakes]

    logger.debug('{}'.format(trilegal_catalogs))

    for i, trilegal_catalog in enumerate(trilegal_catalogs):
        logger.info('working on {}'.format(trilegal_catalog))

        sgal = SimGalaxy(trilegal_catalog, mode='update')
        # "overwrite" (append columns) to the existing catalog by default
        if outfmt == 'default':
            outfile = trilegal_catalog
        else:
            outfile = outfiles[i]
        # do the ast corrections
        msg = '{0:s}_cor and already in header'
        for ast in asts:
            ast_correct_starpop(sgal, asts_obj=ast, overwrite=overwrite,
                                outfile=outfile, diag_plot=diag_plot,
                                hdf5=hdf5, correct=correct,
                                filterset=filterset)
    return


def main(argv):
    """
    Make AST corrections to trilegal catalog(s)

    usage:
    python add_asts.py -vd ~/research/TP-AGBcalib/SNAP/varysfh/kkh37

    name:
    python add_asts.py -vd -t ugc4459 [path to]/ugc-04459
    """
    parser = argparse.ArgumentParser(description="Cull useful info from \
                                                  trilegal catalog")

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='verbose mode')

    parser.add_argument('-o', '--outfile', type=str, help='outfile name')

    parser.add_argument('-f', '--fake', type=str, help='fake file name')

    parser.add_argument('-s', '--filterset', type=int, default=0,
                        help='if 2 filters, and the fake file has 4, provide which filters to use 0: first two or 1: second two')

    parser.add_argument('filters', type=str,
                        help='comma separated list of filters in trilegal catalog')

    parser.add_argument('name', type=str, help='trilegal catalog')

    args = parser.parse_args(argv)

    if args.outfile is None:
        args.outfile = args.name

    # set up logging
    handler = logging.FileHandler('add_asts.log')
    if args.verbose:
        handler.setLevel(logging.DEBUG)
    else:
        handler.setLevel(logging.INFO)
    fmttr = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmttr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    filters = args.filters.split(',')
    make_ast_corrections(args.name, filters=filters, outfiles=args.outfile,
                         fake=args.fake, filterset=args.filterset)
    return

if __name__ == "__main__":
    main(sys.argv[1:])
