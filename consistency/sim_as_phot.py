import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

from ResolvedStellarPops import SimGalaxy
from dweisz.match.scripts.sfh import SFH

from ..TPAGBparams import EXT, snap_src
from ..plotting.plotting import emboss


def cull_data(simgalaxname):
    """
    load simgalaxy, recovered stars, and tpagb.
    """
    sgal = SimGalaxy(simgalaxname)
    good, = np.nonzero((sgal.data['F814W_cor'] < 99.) &
                       (sgal.data['F160W_cor'] < 99.))
    #itps, = np.nonzero(sgal.data['stage'] == 7)
    itps, = np.nonzero(sgal.data['Mcore'] > 0)
    inds = list(set(good) & set(itps))
    mass = sgal.data['m_ini'][good]
    tot_mass = np.sum(mass)
    tp_mass =  sgal.data['m_ini'][inds]
    return sgal, tot_mass, tp_mass, mass, good


def csfr_masshist():
    # use padua sfh from Dan Weisz and PARSEC sfh from me (third attempt)
    pd_loc = os.path.join(snap_src, 'varysfh/extpagb/sim_as_phot/padua/')
    pc_loc = os.path.join(snap_src, 'varysfh/extpagb/sim_as_phot/parsec/')
    # trilegal simulations from input sfh
    sims = ['out_eso540-030_f606w_f814w.mcmc.zc_caf09_v1.2s_m36_s12d_ns_nas_bestsfr.dat',
            'out_ugc5139_f555w_f814w.mcmc.zc_caf09_v1.2s_m36_s12d_ns_nas_bestsfr.dat']
    # zc merged sfh solutions
    sfhs= ['eso540-030_f606w_f814w.mcmc.zc.dat',
           'ugc5139_f555w_f814w.mcmc.zc.dat']
    # calc sfh output for Av, dmod, etc (not really needed for csfr plots)
    metas = ['eso540-030_f606w_f814w.sfh',
             'ugc5139_f555w_f814w.sfh']

    dm = 0.1  # could be input... mass step
    bins = np.arange(0.8, 20 + dm, dm)
    for i, sim in enumerate(sims):
        if i == 0:
            continue
        # make tp-agb and full mass histograms
        sgalpd, tot_masspd, tp_masspd, masspd, goodpd = \
            cull_data(os.path.join(pd_loc, 'caf09_v1.2s_m36_s12d_ns_nas', sim))
        hpd = np.histogram(tp_masspd, bins=bins)[0] / tot_masspd
        hfpd = np.histogram(masspd, bins=bins)[0] / tot_masspd

        sgalpc, tot_masspc, tp_masspc, masspc, goodpc = \
            cull_data(os.path.join(pc_loc, 'caf09_v1.2s_m36_s12d_ns_nas', sim))
        hpc = np.histogram(tp_masspc, bins=bins)[0] / tot_masspc
        hfpc = np.histogram(masspc, bins=bins)[0] / tot_masspc

        # compare tp-agb in both sfh solutions
        a, p = ks_2samp(hpd, hpc)
        target = sgalpd.name.split('_')[1]
        print('TPAGB {} KS: {:.4f} {:.4f}'.format(target, a, p))

        # load sfh for csfr plots
        sfhpd = SFH(os.path.join(pd_loc, sfhs[i]),
                    meta_file=os.path.join(pd_loc, metas[i]))
        sfhpc = SFH(os.path.join(pc_loc, sfhs[i]),
                    meta_file=os.path.join(pc_loc, metas[i]))

        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16, 8))
        fig.subplots_adjust(hspace=.4, top=0.97, bottom=0.11)
        # Padua
        ax1 = sfhpd.plot_csfr(ax=ax1, plt_kw={'color': 'k'},
                              fill_between_kw={'color': 'gray', 'alpha': 0.3})
        # Parsec
        ax1 = sfhpc.plot_csfr(ax=ax1, plt_kw={'color': '#30a2da'},
                              fill_between_kw={'color': '#30a2da', 'alpha': 0.3})

        # tpagb
        tpd, = ax2.plot(bins[:-1], hpd, linestyle='steps-mid', label=r'$\rm{Padua}$',
                 color='k')
        tpc, = ax2.plot(bins[:-1], hpc, linestyle='steps-mid', label=r'$\rm{PARSEC}$',
                 color='#30a2da')
        # full
        pd, = ax2.plot(bins[:-1], hfpd, linestyle='steps-mid', color='k', alpha=0.3)
        pc, = ax2.plot(bins[:-1], hfpc, linestyle='steps-mid', color='#30a2da',
                 alpha=0.3)

        ax2.set_yscale('log')
        ax2.set_xlim(0.8, 12.)
        ax2.set_ylim(1e-7, 1)
        ax2.set_xlabel(r'$\rm{Mass\ (M_\odot)}$')
        ax2.set_ylabel(r'$\rm{\#/Mass\ bin\ (%.1f\ M_\odot)}$' % dm)
        ax1.set_xlabel(r'$\rm{Time\ (Gyr)}$')
        ax1.set_ylabel(r'$\rm{Cumulative\ SF}$')
        ax2.legend(((tpd, pd), (tpc, pc)), (r'$\rm{Padua}$', r'$\rm{PARSEC}$'), loc='best')
        for ax in [ax1, ax2]:
            ax.grid()
            ax.tick_params(direction='in', which='both')
        lab = r'$\rm{{{}}}$'.format(target.upper().replace('-1', '').replace('-','\!-\!'))
        ax1.text(0.05, 0.05, lab, ha='right', fontsize=16, **emboss())
        outfile = '{}_csfr_mass{}'.format(target, EXT)
        import pdb; pdb.set_trace()
        plt.savefig(outfile)
        print('wrote {}'.format(outfile))

if __name__ == "__main__":
    csfr_masshist()
