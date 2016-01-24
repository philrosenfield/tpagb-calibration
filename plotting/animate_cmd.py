import argparse
import matplotlib
import sys

import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import numpy as np
import seaborn as sns

from ResolvedStellarPops import SimGalaxy
from tpagb_calibration.fileio import load_lf_file

matplotlib.use("Agg")

def setup(fps=1):
    """Setup FFMpegWriter"""
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='CMD Animation', artist='Phil Rosenfield')
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    return writer

def read_data(infile, lffile):
    """load SimGalaxy and lf_file"""
    return SimGalaxy(infile), load_lf_file(lffile)


def load_data(sgal, filter1, filter2, lfd=None):
    """load lage, mass, color, mag and slice by inx_norm from lfd"""
    if lfd is None:
        sinds = np.arange(sgal.data['logAge'])
    else:
        sinds = lfd['idx_norm'][0]
    lage = sgal.data['logAge'][sinds]
    color = sgal.data[filter1][sinds] - sgal.data[filter2][sinds]
    mag = sgal.data[filter2][sinds]
    mass = sgal.data['m_ini'][sinds]
    stage = sgal.data['stage'][sinds]
    return lage, mass, color, mag, stage

def plot_setup(filter1, filter2, fsize=20):
    """Setup subplots, axis limits hard coded... """
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 16))
    axs = np.concatenate(axs)
    plt.subplots_adjust(left=0.2, hspace=0.25)
    [ax.tick_params(labelsize=18) for ax in axs]

    for ax in [axs[0], axs[1]]:
        ax.set_xlim(-0.5, 3)
        ax.set_ylim(28.5, 16)
        ax.set_xlabel(r'${}-{}$'.format(filter1, filter2), fontsize=fsize)
        ax.set_ylabel(r'${}$'.format(filter2), fontsize=fsize)

    axs[2].set_xlim(10.2, 6.8)
    axs[2].set_ylim(0, 0.25)

    #axs[3].set_xlim(0, 6.8)
    #axs[3].set_ylim(0, 0.25)

    axs[2].set_xlabel(r'$\log \rm{Age}\ (\rm{yr})$', fontsize=fsize)
    axs[2].set_ylabel(r'$\rm{Relative\ Mass\ Formed}$', fontsize=fsize)
    axs[3].set_xlabel(r'$\rm{Evolutionary\ Stage}$', fontsize=fsize)
    sns.despine()
    sns.color_palette("bright")
    sns.set_context("talk")
    sns.set_style("whitegrid")
    sns.set_style("ticks")
    [ax.grid() for ax in axs]
    return fig, axs


def main(argv):
    parser = argparse.ArgumentParser(description="Animate CMD, and SFH")

    parser.add_argument('infile', type=str,
                        help='trilegal output file')

    parser.add_argument('lffile', type=str, help='lf_file with norm inds')

    parser.add_argument('-o', '--outfile', type=str, default="cmd_animate.mp4",
                        help='movie name')

    parser.add_argument('-d', '--dlage', type=float, default=0.1,
                        help='lage  step size')

    args = parser.parse_args(argv)

    writer = setup()
    #infile = 'out_ugc5139_f555w_f814w_caf09_v1.2s_m36_s12d_ns_nas_014.dat'
    #lffile = 'ugc5139_f814w_f160w_m36_lf.dat'
    sgal, lfd = read_data(args.infile, args.lffile)
    filter1 = args.infile.split('_')[2].upper()
    filter2 = args.infile.split('_')[3].upper()
    lage, mass, color, mag, stage = load_data(sgal, filter1, filter2, lfd=lfd)
    totmass = np.sum(mass)

    lages = np.arange(6.6, 10.21, args.dlage)[::-1]
    #import pdb; pdb.set_trace()
    stagess = np.unique(stage)[::-1]
    inds = [list(set(np.nonzero(lage < lages[i])[0]) &
                 set(np.nonzero(lage >= lages[i+1])[0]))
            for i in range(len(lages)-1)]
    smass =  [np.sum(mass[i]) / totmass for i in inds]

    fig, axs = plot_setup(filter1, filter2)
    axs[2].plot(lages[:-1], smass, linestyle='steps-mid', color='k', lw=2)

    ii = np.array([])
    with writer.saving(fig, args.outfile, 300):
        for i in range(len(lages) - 1):
            axs[0].plot(color[inds[i]], mag[inds[i]], '.')
            axs[1].scatter(color[inds[i]], mag[inds[i]], c=stage[inds[i]],
                           marker='o', s=15,
                           cmap=plt.cm.Spectral)
            # add a cmd plot colored by stage for same age binning
            # evolutionary stage histogram
            axs[2].plot(lages[i], smass[i], 'o')
            #axs[3].plot(lages[i], smass[i], 'o')
            writer.grab_frame()
        axs[2].plot(lages[i], smass[i], 'o')
        writer.grab_frame()
        writer.grab_frame()
    print('wrote {}'.format(args.outfile))

if __name__ == "__main__":
    main(sys.argv[1:])
