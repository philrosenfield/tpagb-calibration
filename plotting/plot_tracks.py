import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

from matplotlib.ticker import MaxNLocator

from ..TPAGBparams import EXT

sns.set()
sns.set_context('paper')
plt.style.use('paper')

def replace_(s, rdict):
    for k, v in rdict.items():
        s = s.replace(k, v)
    return s

class AGBTrack(object):
    """
    AGBTrack adapted from colibri2trilegal
    """
    def __init__(self, filename):
        """
        Read in track, set mass and period.
        """
        self.load_agbtrack(filename)
        # period is either P0 or P1 based on value of Pmod
        self.period = np.array([self.data['P{:.0f}'.format(p)][i]
                                for i, p in enumerate(self.data['Pmod'])])
        self.mass = float(filename.split('agb_')[1].split('_')[0])

    def load_agbtrack(self, filename):
        '''
        Load COLIRBI track and make substitutions to column headings.
        '''
        rdict = {'#': '', 'lg': '', '*': 'star'}
        with open(filename, 'r') as f:
            line = f.readline()
            self.data = np.genfromtxt(f, names=replace_(line, rdict).strip().split())
        return self.data

    def vw93_plot(self, agescale=1e5, outfile=None):
        """
        Make a plot similar to Vassiliadis and Wood 1993. Instead of Vesc,
        I plot C/O.
        """
        fig, axs = plt.subplots(nrows=6, sharex=True, figsize=(5.4, 10))
        fig.subplots_adjust(hspace=0.05, right=0.97, top=0.97, bottom=0.07,
                            left=0.2)

        ycols = ['T_star', 'L_star', 'period', 'CO', 'M_star', 'dMdt']

        for i in range(len(axs)):
            ycol = ycols[i]
            ax = axs[i]
            ax.grid(ls='-', color='k', alpha=0.1, lw=0.5)
            #ax.grid()
            try:
                ax.plot(self.data['ageyr'] / agescale, self.data[ycol], color='k')
            except:
                # period is not in the data but calculated in the init.
                ax.plot(self.data['ageyr'] / agescale, self.__getattribute__(ycol), color='k')
            if ycol == 'CO':
                ax.axhline(1, linestyle='dashed', color='k', alpha=0.5, lw=1)
            ax.set_ylabel(translate_colkey(ycol), fontsize=20)
            ax.yaxis.set_major_locator(MaxNLocator(5, prune='upper'))
        axs[0].yaxis.set_major_locator(MaxNLocator(5, prune=None))
        ax.set_xlabel(translate_colkey('ageyr', agescale=agescale), fontsize=20)
        ax.text(0.98, 0.05, r'$\rm{M}_i=%.2f\ \rm{M}_\odot$' % self.mass,
                transform=ax.transAxes, ha='right', fontsize=16)
        if outfile is not None:
            plt.savefig(outfile)
        return fig, axs


def translate_colkey(col, agescale=1.):
    """
    Turn COLIBRI column name into a axes label
    """
    def str_agescale(scale=1.):
        """
        Set the age unit string.
        """
        u = ''
        if scale == 1e9:
            u = 'G'
        elif scale == 1e6:
            u = 'M'
        elif np.log10(scale) >= 1.:
            u = '10^%i\ ' % int(np.log10(scale))
        return u

    tdict = {'T_star': r'$log\ \rm{T}_{\rm{eff}}\ \rm{(K)}$',
             'L_star': r'$log\ L\ (L_\odot)$',
             'period': r'$\rm{P}$',
             'CO': r'$\rm{C/O}$',
             'M_star': r'$\rm{M}\ (\rm{M}_\odot)$',
             'dMdt': r'$\dot{\rm{M}}$',
             'ageyr': r'$\rm{Age\ (%syr)}$' % str_agescale(agescale)}

    new_col = col
    if col in tdict.keys():
        new_col = tdict[col]

    return new_col

def main(argv):
    parser = argparse.ArgumentParser(description="Make a plot like Vassiliadis and Wood 1993")

    parser.add_argument('infiles', type=str, nargs='*',
                        help='COLIBRI track file(s)')

    args = parser.parse_args(argv)

    for infile in args.infiles:
        agb = AGBTrack(infile)
        outfile = infile.replace('.dat', EXT)
        agb.vw93_plot(outfile=outfile)
        plt.close()

if __name__ == "__main__":
    main(sys.argv[1:])
