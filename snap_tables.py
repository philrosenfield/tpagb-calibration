from __future__ import print_function
import argparse
import sys
import match

def main(argv):
    parser = argparse.ArgumentParser(description="Make some tables")

    parser.add_argument('-s', '--sfhtable', action='store_true',
                        help='make data table with sfh and feh')
     
    parser.add_argument('-o', '--outfile', type=str, default='snap_table.dat',
                        help='output file')
    
    parser.add_argument('files', nargs='*', type=str,
                        help='input files to read from')

    args = parser.parse_args(argv)
    
    if args.sfhtable:
        lines = ''
        for f in args.files:
            sfh = match.utils.MatchSFH(f)
            d = sfh.param_table()
            lines += d['fmt'].format(**d)
        lines = lines.replace('nan', '...')
        with open(args.outfile, 'w') as out:
            out.write(d['header'])
            out.write(lines)
        print('wrote {}'.format(args.outfile))
    else:
        print('nothing to do')

if __name__ == '__main__':
    main(sys.argv[1:])