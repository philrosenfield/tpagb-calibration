import argparse
import sys

from astropy.io import ascii


def replace_ext(filename, new_ext):
    '''
    replace a filename's extention with a new one
    '''
    if not new_ext.startswith('.'):
        new_ext = '.' + new_ext

    return '.'.join(filename.split('.')[:-1])  + new_ext

def ascii2hdf5(inputfile, outputfile, clobber=False, overwrite=True):
    """
    Convert a file to hdf5 using compression and path set to 'data'
    """
    tbl = ascii.read(inputfile)
    tbl.write(v, format='hdf5', path='data', compression=True,
              overwrite=overwrite)
    if clobber:
        os.remove(inputfile)
    return new_out
    
def main(argv):
    pass

    parser = argparse.ArgumentParser(description="Convert ascii file to hdf5")

    parser.add_argument('-f', '--force', action='store_true',
                        help='overwrite if hdf5 file exists')

    parser.add_argument('-c', '--clobber', action='store_true',
                        help='remove input ascii file')

    parser.add_argument('-o', '--output', type=str, default=None,
                        help='output file name: if not set, uses inputfile name')

    parser.add_argument('name', nargs='*', type=str, help='ascii file(s) to convert')

    args = parser.parse_args(argv)

    assert os.path.isfile(args.name), '{} not found'.format(args.name)

    if args.output is None:
        args.output = replace_ext(args.name, '.hdf5')
    
    ascii2hdf5(args.name, args.output, clobber=args.clobber, overwrite=args.force)

if __name__ == '__main__':
    main(sys.argv[1:])