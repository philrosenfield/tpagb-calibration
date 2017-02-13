import os
import seaborn
import glob

import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits
from ezpadova import cmd

from ..fileio import get_files
from ..plotting.plot_tracks import AGBTracks

seaborn.set_style('whitegrid')

def load_tracks(z, src='/Users/rosenfield/research/AGBTracks/CAF09/S_NOV13/'):
    agbdir = os.path.join(src, glob.glob1(src, '*{}_*'.format(z))[0])
    agbfiles = get_files(agbdir, 'agb_*')
    return [AGBTrack(a) for a in agbfiles]


def age_interp():
    zs = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.014,
          0.017,  0.02, 0.03, 0.04, 0.05, 0.06]
    #zsprime = zs[:-1] + np.diff(zs) / 2.
    zsprime = [0.00011, 0.00024, 0.00059, 0.0012, 0.0021, 0.0043, 0.0065, 0.0089, 0.0195, 0.014,
               0.022]
    
    agemin = 5.5
    agemax = 10.15
    
    for z in zsprime:
        fig, ax = plt.subplots()
        ax.grid()
        outfile = 'iso_{}_{}_{}.fits'.format(agemin, agemax, z)
        #tracks = load_tracks(z)
        #for track in tracks[::8]:
        #    t = track.data
        #    ct = ax.plot(t['T_star'], t['L_star'], color='k', alpha=0.3, lw=0.8, zorder=1000)
        if not os.path.isfile(outfile):
            tab = cmd.get_t_isochrones(agemin, agemax, 0.05, z, model='parsec12s_r14')
            logT = 'logT'
            logA = 'logA'
            logL = 'logL'
        else:
            tab = fits.getdata(outfile)
            logT = 'logTe'
            logA = 'logageyr'
            logL = 'logLLo'
        
        inds, = np.nonzero(tab['slope'])
        cm = ax.scatter(tab[logT][inds], tab[logL][inds], c=tab[logA][inds],
                        edgecolor='None', cmap=plt.cm.Spectral,zorder=1)
        plt.colorbar(cm)
        ax.set_xlabel('log Te')
        ax.set_ylabel('log L')
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_title('Z={}'.format(z))
        if not os.path.isfile(outfile):
            tab.write(outfile)
        plt.savefig(outfile.replace('fits', 'png'))
        plt.close()