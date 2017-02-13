"""
Add AST corrections to trilegal catalogs
"""
import argparse
import logging
import os
import sys


from ..pop_synth.asts import ASTs, ast_correct_starpop
from ..pop_synth import SimGalaxy
from .. import fileio
# where the matchfake files live
from ..TPAGBparams import snap_src
matchfake_loc = os.path.join(snap_src, 'data', 'galaxies')

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def make_ast_corrections(trilegal_catalogs, target, outfiles='default',
                         diag_plot=False, overwrite=True, hdf5=False,
                         fake=None):
    """
    apply ast corrections from fake files found in matchfake_loc/*[target]*
    see asts.ast_correct_starpop
    """
    if type(outfiles) is str:
        outfmt = 'default'
    else:
        outfmt = 'supplied'

    if fake is None:
        # search string for fake files
        search_str = '*{}*.matchfake'.format(target.upper())

        fakes = fileio.get_files(matchfake_loc, search_str)
        logger.info('fake files found: {}'.format(fakes))
    else:
        fakes = [fake]

    asts = [ASTs(f) for f in fakes]
    logger.debug('{}'.format(trilegal_catalogs))

    for i, trilegal_catalog in enumerate(trilegal_catalogs):
        logger.info('working on {}'.format(trilegal_catalog))

        sgal = SimGalaxy(trilegal_catalog)
        # "overwrite" (append columns) to the existing catalog by default
        if outfmt == 'default':
            outfile = trilegal_catalog
        else:
            outfile = outfiles[i]
        # do the ast corrections
        msg = '{}_cor and already in header'
        for ast in asts:
            correct = 'both'
            header = open(trilegal_catalog, 'r').readline()
            if ast.filter1 + '_cor' in header.split():
                logger.warning(msg.format(ast.filter1))
                if correct == 'both':
                    correct = 'filter2'
            if ast.filter2 + '_cor' in header.split():
                logger.warning(msg.format(ast.filter2))
                if correct == 'both':
                    correct = 'filter1'
                else:
                    continue

            ast_correct_starpop(sgal, asts_obj=ast, overwrite=overwrite,
                                    outfile=outfile, diag_plot=diag_plot,
                                    hdf5=hdf5, correct=correct)
    return


def main(argv):
    """
    Make AST corrections to trilegal catalog(s)

    usage:
    python add_asts.py -vd ~/research/TP-AGBcalib/SNAP/varysfh/kkh37

    if the target directory name is different than it is in the matchfake file
    name:
    python add_asts.py -vd -t ugc4459 [path to]/ugc-04459
    """
    parser = argparse.ArgumentParser(description="Cull useful info from \
                                                  trilegal catalog")

    parser.add_argument('-d', '--directory', action='store_true',
                        help='opperate on *_???.dat files in a directory')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='verbose mode')

    parser.add_argument('-o', '--outfile', type=str, help='outfile name',
                        default=None)

    parser.add_argument('-t', '--target', type=str, help='target name')

    parser.add_argument('-f', '--fake', type=str, help='fake file name')

    parser.add_argument('name', type=str, nargs='*',
                        help='trilegal catalog or directory if -d flag')

    args = parser.parse_args(argv)

    if args.outfile is None:
        args.outfile = args.name

    if not args.target:
        if args.directory:
            target = os.path.split(args.name[0])[1]
        else:
            target = os.path.split(args.name[0])[1].split('_')[1]
    else:
        target = args.target

    # set up logging
    handler = logging.FileHandler('{}_analyze.log'.format(target))
    if args.verbose:
        handler.setLevel(logging.DEBUG)
    else:
        handler.setLevel(logging.INFO)
    fmttr = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmttr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info('using matchfake location: {}'.format(matchfake_loc))

    # assume trilegal was run with outfile ending with *_???.dat
    if args.directory:
        if args.name[0].endswith('/'):
            args.name[0] = args.name[0][:-1]
        tricats = fileio.get_files(args.name[0], '*_???.dat')
    else:
        tricats = args.name

    if args.verbose:
        logger.info('working on target: {}'.format(target))

    make_ast_corrections(tricats, target, outfiles=args.outfile,
                         fake=args.fake)
    return

if __name__ == "__main__":
    main(sys.argv[1:])
