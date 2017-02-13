#!/usr/bin/python
import argparse
import sys

import matplotlib as mpl
mpl.use('Agg')

from .isochrones import Isochrones
import matplotlib.pyplot as plt

def main(argv):
    parser = argparse.ArgumentParser(description="Plot isochrones of a isochrone file")
    parser.add_argument('-o', '--outfile', type=str, default=None,
                        help='output file')

    parser.add_argument('fname', type=str, help='isochrones file')

    args = parser.parse_args(argv)

    isoc = Isochrones(args.fname)
    fig, ax = plt.subplots()
    isoc.plot_all_isochrones('LOG_TE', 'LOG_L', plot_isochrone_kw={'ax':ax})
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_title('{}'.format(isoc.name.replace('_', '\_')))

    if args.outfile is None:
        outfile = isoc.name + '.png'
    plt.savefig(outfile)
    print('write {}'.format(outfile))


if __name__ == '__main__':
    main(sys.argv[1:])
