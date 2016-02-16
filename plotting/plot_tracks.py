import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

from matplotlib.ticker import MaxNLocator
from palettable.wesanderson import Darjeeling2_5

from ..TPAGBparams import EXT
from tpagb_calibration.utils import minmax

sns.set()
sns.set_context('paper')
plt.style.use('paper')

def replace_(s, rdict):
    for k, v in rdict.items():
        s = s.replace(k, v)
    return s

def duration_masslost(agbs, justprint=False, norm=False):
    if justprint:
        aa = [3., 4., 5.]
        for a in aa:
            for agb in agbs:
                if agb.Z not in [0.001, 0.008]:
                    continue
                #plt.plot(agbs[i].data['ageyr'], agbs[i].data['L_star'])
                ind1, ind2 = agb.ml_regimes()
                if not None in [ind1, ind2] or ind1 != ind2:
                    if agb.mass != a:
                        continue
                    age = agb.data['ageyr'] / 1e5
                    mass = agb.data['M_star']
                    #print sum(agb.data['dt'][np.nonzero(agb.data['L_star'] < 3.4)[0]])
                    print(ind1, ind2)
                    print '{:g} {:g} {:.2f} {:.2f} {:.2f} {:.2f}'.format(agb.Z, agb.mass,
                            age[ind1], age[ind2]-age[ind1], age[-1]-age[ind2], age[-1])

    fig, axs = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(8, 4))
    #sns.despine()
    col1, col2 = axs.T[0], axs.T[1]
    colors = Darjeeling2_5.mpl_colors[1:-1]
    kw = {'align':'edge'}
    for agb in agbs:
        if agb.mass >= 3.2 and agb.mass <= 5.:
            for i, col in enumerate([col1, col2]):
                if agb.Z == 0.001:
                    ax = col[0]
                if agb.Z == 0.004:
                    continue
                if agb.Z == 0.008:
                    ax = col[1]
                ipd, = np.nonzero(agb.data['M_predust'] == agb.data['dMdt'])
                idd, = np.nonzero(agb.data['Mdust'] == agb.data['dMdt'])
                iall = np.arange(len(agb.data['dMdt']))
                isw = np.array(list(set(iall) - set(ipd) - set(idd)))
                ttp = 1e5
                if norm:
                    ttp = np.sum(agb.data['dt'])
                tpd = np.sum(agb.data['dt'][ipd]) / ttp
                tdd = np.sum(agb.data['dt'][idd]) / ttp
                tsw = np.sum(agb.data['dt'][isw]) / ttp
                if agb.Z == 0.001 and agb.mass == 5. and i == 0:
                    ax.barh(agb.mass, tpd, 0.2, color=colors[0], label=r'$\dot{M}_{pd}$', **kw)
                    ax.barh(agb.mass, tdd, 0.2, color=colors[1], label=r'$\dot{M}_{dd}$', left=tpd, **kw)
                    ax.barh(agb.mass, tsw, 0.2, color=colors[2], label=r'$\dot{M}_{sw}$', left=tdd+tpd, **kw)
                    if norm:
                        loc = 'upper left'
                        frameon = True
                    else:
                        loc='upper right'
                        frameon = False
                    ax.legend(labelspacing=0.02, loc=loc, frameon=frameon, fontsize=10,  handlelength=1)
                else:
                    if i > 0:
                        ttp = 1
                        if norm:
                            ttp = agb.mass
                        tpd = np.sum(agb.data['dt'][ipd] * agb.data['dMlost'][ipd]) / ttp
                        tdd = np.sum(agb.data['dt'][idd] * agb.data['dMlost'][idd]) / ttp
                        tsw = np.sum(agb.data['dt'][isw] * agb.data['dMlost'][isw]) / ttp
                        if norm:
                            ax.set_xlim(0, 1)
                        if agb.mass == 5.:
                            ax.text(0.98, 0.02, r'$\rm{Z}=%g$' % agb.Z, fontsize=16,
                                    transform=ax.transAxes, ha='right')

                    ax.barh(agb.mass, tpd, 0.2, color=colors[0], **kw)
                    ax.barh(agb.mass, tdd, 0.2, color=colors[1], left=tpd, **kw)
                    ax.barh(agb.mass, tsw, 0.2, color=colors[2], left=tdd+tpd, **kw)

    for ax in axs.flatten():
        ax.tick_params(direction='out', color='k', size=2.6, width=0.5)
        #ax.grid(lw=0.6, color='k')
        ax.grid()
        if not norm:
            ax.set_xlim(ax.set_xlim(0, 4.5))
        ax.set_ylim(3.2, 5.2)

    [ax.tick_params(labelbottom='off') for ax in axs.flatten()[:-2]]
    [ax.tick_params(labelright='on') for ax in col2]
    N = len(col2[-1].get_xticks())
    [ax.xaxis.set_major_locator(MaxNLocator(N, prune='lower')) for ax in col2]
    if norm:
        col1[-1].set_xlabel(r'$\rm{TP-AGB\ Lifetime}$')
        col1[-1].set_xlabel(r'$\rm{Fraction\ of\ Initial\ Mass\ Lost}$')
        [ax.set_xlim(0, 0.9) for ax in col2]
        [ax.set_xlim(0, 1) for ax in col1]
    else:
        col1[-1].set_xlabel(r'$\rm{TP-AGB\ Age\ (10^5\ yr)}$')
        col1[-1].set_xlabel(r'$\rm{Mass\ Lost\ (M_\odot)}$')

    fig.text(0.03, 0.58, r'$\rm{TP-AGB\ Initial\ Mass (M_\odot)}$', rotation='vertical', ha='center', va='center',
             fontsize=20)
    fig.subplots_adjust(left=0.1, hspace=0.05, wspace=0.05, right=0.92, bottom=0.2, top=0.98)
    if norm:
        plt.savefig('duration_masslost_norm.png')
    else:
        plt.savefig('duration_masslost.png')
    return fig, axs


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
        self.Z = float(filename.split('agb_')[1].split('_')[1].replace('Z',''))

    def ml_regimes(self):
        indi=None
        indf=None
        try:
            arr, = np.nonzero(self.data['Mdust'] == self.data['dMdt'])
            indi = arr[0]
            indf = arr[-1]
        except:
            pass
        return indi, indf

    def load_agbtrack(self, filename):
        '''
        Load COLIRBI track and make substitutions to column headings.
        '''
        rdict = {'#': '', 'lg': '', '*': 'star'}
        with open(filename, 'r') as f:
            line = f.readline()
            self.data = np.genfromtxt(f, names=replace_(line, rdict).strip().split())
        return self.data

    def vw93_plot(self, agescale=1e5, outfile=None, xlim=None, ylims=None,
                  fig=None, axs=None, annotate=True, annotation=None):
        """
        Make a plot similar to Vassiliadis and Wood 1993. Instead of Vesc,
        I plot C/O.
        """
        ylims = ylims or [None] * 6
        if axs is None:
            fig, axs = plt.subplots(nrows=6, sharex=True, figsize=(5.4, 10))
            fig.subplots_adjust(hspace=0.05, right=0.97, top=0.97, bottom=0.07,
                                left=0.2)

        ycols = ['T_star', 'L_star', 'period', 'CO', 'M_star', 'dMdt']

        for i in range(len(axs)):
            ycol = ycols[i]
            ax = axs[i]
            #ax.grid(ls='-', color='k', alpha=0.1, lw=0.5)
            ax.grid()
            try:
                ax.plot(self.data['ageyr'] / agescale, self.data[ycol], color='k')
            except:
                # period is not in the data but calculated in the init.
                ax.plot(self.data['ageyr'] / agescale, self.__getattribute__(ycol), color='k')
            if ycol == 'CO':
                ax.axhline(1, linestyle='dashed', color='k', alpha=0.5, lw=1)
            ax.set_ylabel(translate_colkey(ycol), fontsize=20)
            ax.yaxis.set_major_locator(MaxNLocator(5, prune='upper'))
            if ylims[i] is not None:
                ax.sey_ylim(ylims[i])
        if xlim is not None:
            ax.set_xlim()
        axs[0].yaxis.set_major_locator(MaxNLocator(5, prune=None))
        ax.set_xlabel(translate_colkey('ageyr', agescale=agescale), fontsize=20)
        [ax.get_yaxis().set_label_coords(-.16,0.5) for ax in axs]
        # doesn't work with latex so well...
        axs[3].get_yaxis().set_label_coords(-.165,0.5)
        [ax.get_yaxis().set_label_coords(-.17,0.5) for ax in [axs[-1], axs[2]]]

        indi, indf = self.ml_regimes()
        if not None in [indi, indf]:
            [[ax.axvline(self.data['ageyr'][i]/agescale, ls=':', color='k',
                         alpha=0.5, lw=0.8)
            for ax in axs] for i in [indi, indf]]
        if annotate:
            if annotation is None:
                annotation = r'$\rm{M}_i=%.2f\ \rm{M}_\odot$' % self.mass
            axs[4].text(0.02, 0.05, annotation, ha='left', fontsize=16,
                        transform=axs[4].transAxes)

        if outfile is not None:
            plt.tight_layout()
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
             'period': r'$\rm{P\ (days)}$',
             'CO': r'$\rm{C/O}$',
             'M_star': r'$\rm{M}\ (\rm{M}_\odot)$',
             'dMdt': r'$\dot{\rm{M}}\ (\rm{M}_\odot/\rm{yr})$',
             'ageyr': r'$\rm{TP-AGB\ Age\ (%syr)}$' % str_agescale(agescale)}

    new_col = col
    if col in tdict.keys():
        new_col = tdict[col]

    return new_col

def compare_vw93(agbs, outfile=None, xlim=None, ylims=None):
    if agbs[0].mass != agbs[1].mass:
        annotations = [None, None]
    else:
        fmt = r'$\rm{Z}=%g$'
        annotations = [fmt % agbs[0].Z, fmt % agbs[1].Z]
    # sharex is off because I want one column pruned.
    fig, axs = plt.subplots(nrows=6, ncols=2, figsize=(8.4, 10))
    col1 = axs.T[0]
    col2 = axs.T[1]

    fig, col1 = agbs[0].vw93_plot(fig=fig, axs=col1, annotation=annotations[0])
    fig, col2 = agbs[1].vw93_plot(fig=fig, axs=col2, annotation=annotations[1])

    for i in range(len(col1)):
        ax1 = col1[i]
        ax2 = col2[i]
        [ax.set_ylim(minmax(ax1.get_ylim(), ax2.get_ylim())) for ax in [ax1, ax2]]
        [ax.set_xlim(minmax(ax1.get_xlim(), ax2.get_xlim())) for ax in [ax1, ax2]]
        ax2.tick_params(labelleft=False, labelright=True)
        [ax.tick_params(labelbottom=False) for ax in [ax1, ax2]]
        ax2.set_ylabel('')

    [ax.tick_params(labelbottom=True) for ax in [col1[-1], col2[-1]]]
    # prune only the left column
    N = len(col1[0].get_xticks())
    col1[-1].xaxis.set_major_locator(MaxNLocator(N, prune='upper'))
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.05, wspace=0.02)
    if outfile is not None:
        plt.savefig(outfile)
        print('wrote {}'.format(outfile))
    return fig, (col1, col2)

def main(argv):
    parser = argparse.ArgumentParser(description="Make a plot like Vassiliadis and Wood 1993")

    parser.add_argument('infiles', type=str, nargs='*',
                        help='COLIBRI track file(s)')

    parser.add_argument('-x', '--xlim', type=str, default=None,
                        help='comma separated x axis limits')

    parser.add_argument('-t', '--tlim', type=str, default=None,
                        help='comma separated log teff axis limits')

    parser.add_argument('-l', '--llim', type=str, default=None,
                        help='comma separated log l axis limits')

    parser.add_argument('-p', '--plim', type=str, default=None,
                        help='comma separated period axis limits')

    parser.add_argument('-c', '--clim', type=str, default=None,
                        help='comma separated c/o axis limits')

    parser.add_argument('-m', '--mlim', type=str, default=None,
                        help='comma separated mass axis limits')

    parser.add_argument('-d', '--dmlim', type=str, default=None,
                        help='comma separated mass loss axis limits')

    parser.add_argument('-z', '--compare', action='store_true',
                        help='one plot')

    parser.add_argument('-f', '--dmplot', action='store_true',
                        help='duration mass lost plot')

    parser.add_argument('-n', '--norm', action='store_false',
                        help='with -f do not norm (use units)')

    args = parser.parse_args(argv)

    ylims = [args.tlim, args.llim, args.plim, args.clim, args.mlim, args.dmlim]
    xlim = args.xlim

    if args.dmplot:
        agbs = [AGBTrack(infile) for infile in args.infiles]
        duration_masslost(agbs, norm=args.norm)

    elif args.compare:
        agbs = [AGBTrack(infile) for infile in args.infiles]
        outfile = 'tpagb_comp'
        if agbs[0].mass == agbs[1].mass:
            outfile += '_m%g' % agbs[0].mass
        elif agbs[0].Z == agbs[1].Z:
            outfile += '_z%g' % agbs[0].mass
        outfile += EXT

        compare_vw93(agbs, outfile=outfile, xlim=xlim, ylims=ylims)
    else:
        for infile in args.infiles:
            agb = AGBTrack(infile)
            outfile = infile.replace('.dat', EXT)
            agb.vw93_plot(outfile=outfile, xlim=xlim, ylims=ylims)
            plt.close()

if __name__ == "__main__":
    main(sys.argv[1:])
