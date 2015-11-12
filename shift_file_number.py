import sys
import argparse


def shift(fnames, off=0):
    """
    example:
    out_ugc4305-2_f555w_f814w_caf09_v1.2s_m36_s12d_ns_nas_017.dat
    off = 25
    new fomrat:
    out_ugc4305-2_f555w_f814w_caf09_v1.2s_m36_s12d_ns_nas_042.dat
    """
    line = ''
    for f in fnames:
            pref, idxext = '_'.join(f.split('_')[:-1]), f.split('_')[-1]
            idx, ext = idxext.split('.')
            nidx = int(idx) + off
            nf = '_'.join([pref, '{:03d}.{}'.format(nidx, ext)])
            line += 'mv {} {}\n'.format(f, nf)
    return line


def main(argv):
    description = "Write a scripts to shift filename numbers"

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-o', '--outfile', type=str, default='shift.sh',
                        help='output file')

    parser.add_argument('-s', '--offset', type=int, default=0,
                        help='numeric offset')

    parser.add_argument('files', type=str, nargs='*', help='input files')

    args = parser.parse_args(argv)

    lines = shift(args.files, off=args.offset)

    with open(args.outfile, 'w') as out:
        out.write(lines)

if __name__ == "__main__":
    main(sys.argv[1:])
