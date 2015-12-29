from ResolvedStellarPops import SimGalaxy
from astropy.io import fits
import os
from dweisz.match.scripts.fake import make_fakeparam
from dweisz.match.scripts.fileio import get_files, read_fake, replace_ext
from dweisz.match.scripts.sfh import SFH
import numpy as np
import matplotlib.pyplot as plt
from ..TPAGBparams import EXT, snap_src
from scipy.stats import ks_2samp
from .plotting import emboss

def cull_data(simgalaxname):
    """
    load simgalaxy, recovered stars, and tpagb.
    """
    sgal = SimGalaxy(simgalaxname)
    good, = np.nonzero((sgal.data['F814W_cor'] < 99.) &
                       (sgal.data['F160W_cor'] < 99.))
    itps, = np.nonzero(sgal.data['stage'] == 7)
    inds = list(set(good) & set(itps))
    mass = sgal.data['m_ini'][good]
    tot_mass = np.sum(mass)
    tp_mass =  sgal.data['m_ini'][inds]
    return sgal, tot_mass, tp_mass, mass

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
    bins = np.arange(0.8, 12 + dm, dm)
    for i, sim in enumerate(sims):
        # make tp-agb and full mass histograms
        sgalpd, tot_masspd, tp_masspd, masspd = \
            cull_data(os.path.join(pd_loc, 'caf09_v1.2s_m36_s12d_ns_nas', sim))
        hpd = np.histogram(tp_masspd, bins=bins)[0] / tot_masspd
        hfpd = np.histogram(masspd, bins=bins)[0] / tot_masspd

        sgalpc, tot_masspc, tp_masspc, masspc = \
            cull_data(os.path.join(pc_loc, 'caf09_v1.2s_m36_s12d_ns_nas', sim))
        hpc = np.histogram(tp_masspc, bins=bins)[0] / tot_masspc
        hfpc = np.histogram(masspc, bins=bins)[0] / tot_masspc

        # compare tp-agb in both sfh solutions
        a, p = ks_2samp(hpd, hpc)
        target = sgalpd.name.split('_')[1]
        print('TPAGB {} KS: {:.2f} {:.2f}'.format(target, a, p))

        # load sfh for csfr plots
        sfhpd = SFH(os.path.join(pd_loc, sfhs[i]),
                    meta_file=os.path.join(pd_loc, metas[i]))
        sfhpc = SFH(os.path.join(pc_loc, sfhs[i]),
                    meta_file=os.path.join(pc_loc, metas[i]))

        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6))
        fig.subplots_adjust(hspace=.4, top=0.97, bottom=0.11)
        # Padua
        ax1 = sfhpd.plot_csfr(ax=ax1, plt_kw={'color': 'k'},
                              fill_between_kw={'color': 'gray', 'alpha': 0.3})
        # Parsec
        ax1 = sfhpc.plot_csfr(ax=ax1, plt_kw={'color': '#30a2da'},
                              fill_between_kw={'color': '#30a2da', 'alpha': 0.3})

        # tpagb
        ax2.plot(bins[:-1], hpd, linestyle='steps-mid', label=r'$\rm{Padua}$',
                 color='k')
        ax2.plot(bins[:-1], hpc, linestyle='steps-mid', label=r'$\rm{PARSEC}$',
                 color='#30a2da')
        # full
        ax2.plot(bins[:-1], hfpd, linestyle='steps-mid', color='k', alpha=0.5)
        ax2.plot(bins[:-1], hfpc, linestyle='steps-mid', color='#30a2da',
                 alpha=0.5)

        ax2.set_yscale('log')
        ax2.set_xlim(0.8, 12.)
        ax2.set_ylim(1e-7, 1)
        ax2.set_xlabel(r'$\rm{Mass\ (M_\odot)}$')
        ax2.set_ylabel(r'$\rm{\#/Mass\ bin\ (%.1f\ M_\odot)}$' % dm)
        ax1.set_xlabel(r'$\rm{Time\ (Gyr)}$')
        ax1.set_ylabel(r'$\rm{Cumulative\ SF}$')
        for ax in [ax1, ax2]:
            ax.legend(loc='best')
            ax.tick_params(direction='in', which='both')
        lab = r'$\rm{{{}}}$'.format(target.upper().replace('-1', '').replace('-','\!-\!'))
        ax1.text(0.05, 0.05, lab, ha='right', fontsize=16, **emboss())
        outfile = '{}_csfr_mass{}'.format(target, EXT)
        plt.savefig(outfile)
        print('wrote {}'.format(outfile))

if __name__ == "__main__":
    csfr_masshist()

# nothing below is used... kept just in case.

def csfr_masshistold():
    """ Do all (first attempt at this)
    sgal_loc = '/Volumes/tehom/research/TP-AGBcalib/SNAP/varysfh/extpagb/sim_as_phot/caf09_v1.2s_m36_s12d_ns_nas/match_run/'
    sgal_files = get_files(sgal_loc, '*cor*dat')
    fake_loc = '/Volumes/tehom/research/TP-AGBcalib/SNAP/varysfh/extpagb/sim_as_phot/caf09_v1.2s_m36_s12d_ns_nas/match_run/fake_run'
    fake_files = get_files(fake_loc, '*dat')
    targets = [os.path.split(p)[1].split('_')[0] for p in fakes]
    sgal_files = np.concatenate([[s for s in sgal_files if t in s] for t in targets])

    sgals = [SimGalaxy(s) for s in sgal_files]
    fakes = [read_fake(f) for f in fake_files]

    for i, (f, s) in enumerate(zip(fakes, sgals)):
        fig, ax = plt.subplots()
        inds, = np.nonzero((f['mag1'] < 99) & (f['mag2'] < 99))
        bins = np.arange(f['mass'].min(), f['mass'].max(), 0.5)
        h = np.histogram(f['mass'][inds], bins=bins)[0]

        bins1 = np.arange(s.data['m_ini'].min(), s.data['m_ini'].max(), 0.5)
        h1 = np.histogram(s.data['m_ini'], bins=bins1)[0]

        ax.plot(bins1[1:], h1 / np.sum(s.data['m_ini']), linestyle='steps-mid')
        ax.plot(bins[1:], h / np.sum(f['mass'][inds]), linestyle='steps-mid')
        ax.set_yscale('log')
        ax.set_title(targets[i])

    # trilegal as data (second attempt at this)
    data_sfh = '/Volumes/tehom/research/TP-AGBcalib/SNAP/varysfh/ugc8508_f475w_f814w.sfh'
    data_hmc = '/Volumes/tehom/research/TP-AGBcalib/SNAP/varysfh/ugc8508_f475w_f814w.mcmc.zc'
    sfh = SFH(data_sfh, hmc_file=data_hmc)

    test_base = '/Volumes/tehom/research/TP-AGBcalib/SNAP/varysfh/extpagb/sim_as_phot/caf09_v1.2s_m36_s12d_ns_nas/match_run/'
    model_sfh = test_base + 'ugc8508_f475w_f814w.sfh'
    tsfh = SFH(model_sfh)
    # cut and asts added to best fit MATCH-derived SFH
    sgal_file =  test_base + 'ugc8508_f475w_f814w_bestsfr_cor.dat'
    # cut and asts added to best fit MATCH-derived SFH used as input SFH to TRILEGAL.
    tsgal_file =  test_base + 'caf09_v1.2s_m36_s12d_ns_nas/out_ugc8508_f475w_f814w_bestsfr.dat'
    sgal = SimGalaxy(sgal_file)
    tsgal = SimGalaxy(tsgal_file)

    dm = 0.1
    bins = np.arange(0.8, 8 + dm, dm)
    good, = np.nonzero( (tsgal.data['F814W_cor'] < 99.) & (tsgal.data['F160W_cor'] < 99.))
    itps, = np.nonzero(tsgal.data['stage'] == 7)
    inds1 = list(set(good) & set(itps))
    tmass = tsgal.data[inds1]['m_ini']
    h1 = np.histogram(tmass, bins=bins)[0] / np.sum(tmass)

    good, = np.nonzero((sgal.data['F814W_cor'] < 99.) & (sgal.data['F160W_cor'] < 99.))
    itps, = np.nonzero(sgal.data['stage'] == 7)
    inds = list(set(good) & set(itps))
    mass = sgal.data['m_ini'][inds]
    h = np.histogram(mass, bins=bins)[0] / np.sum(mass)

    from scipy.stats import ks_2samp
    a, p = ks_2samp(h, h1)
    print 'KS:', a, p

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6))
    fig.subplots_adjust(hspace=.4, top=0.97, bottom=0.11)
    ax1 = sfh.plot_csfr(ax=ax1)
    ax1 = tsfh.plot_csfr(ax=ax1)
    ax2.plot(bins[:-1], h1, linestyle='steps-mid')
    ax2.plot(bins[:-1], h, linestyle='steps-mid')
    #ax2.errorbar(bins[:-1], h * np.sum(mass), yerr=np.sqrt(h), fmt='none')
    #ax2.errorbar(bins[:-1], h1 * np.sum(tmass), yerr=np.sqrt(h1), fmt='none')
    ax2.set_yscale('log')
    ax2.set_xlim(0.8, 6.5)
    ax1.tick_params(direction='in', which='both')
    ax2.tick_params(direction='in', which='both')
    ax2.set_xlabel(r'$\rm{TP\!-\!AGB\ Mass\ (M_\odot)}$')
    ax2.set_ylabel(r'$\rm{\#/Mass\ bin (0.1\ M_\odot)}$')
    ax1.set_xlabel(r'$\rm{Time\ (Gyr)}$')
    ax1.set_ylabel(r'$\rm{Cummulative\ SF}$')
    plt.savefig('ugc8505_csfr_mass{}'.format(EXT))
    """
    pass

def cut_sims():
    """
    match CMD space with data for first/second attempts
    """
    sims = ['ugc8508_f475w_f814w_bestsfr_cor.dat',
            'ngc3741_f475w_f814w_bestsfr_cor.dat',
            'ngc2403-halo-6_f606w_f814w_bestsfr_cor.dat',
            'ugc4459_f555w_f814w_bestsfr_cor.dat',
            'ugc5139_f555w_f814w_bestsfr_cor.dat',
            'ugc4305-1_f555w_f814w_bestsfr_cor.dat',
            'ngc2403-deep_f606w_f814w_bestsfr_cor.dat']

    gals = ['/Volumes/tehom/research/TP-AGBcalib/SNAP/data/galaxies/copy/ugc8508_f475w_f814w_v1_gst.fits',
            '/Volumes/tehom/research/TP-AGBcalib/SNAP/data/galaxies/copy/ngc3741_f475w_f814w_v1_gst.fits',
            '/Volumes/tehom/research/TP-AGBcalib/SNAP/data/galaxies/copy/ngc2403-halo-6_f606w_f814w_v1_gst.fits',
            '/Volumes/tehom/research/TP-AGBcalib/SNAP/data/galaxies/copy/ugc4459_f555w_f814w_v1_gst.fits',
            '/Volumes/tehom/research/TP-AGBcalib/SNAP/data/galaxies/copy/ugc5139_f555w_f814w_v1_gst.fits',
            '/Volumes/tehom/research/TP-AGBcalib/SNAP/data/galaxies/copy/ugc4305-1_f555w_f814w_v1_gst.fits',
            '/Volumes/tehom/research/TP-AGBcalib/SNAP/data/galaxies/copy/ngc2403-deep_f606w_f814w_v1_gst.fits']


    for sim, gal in zip(sims, gals):
        g = fits.getdata(gal)
        s = SimGalaxy(sim)
        filter1, filter2 = ['{}_cor'.format(f) for f in s.filters]
        try:
            mag1 = g['MAG1_ACS']
            mag2 = g['MAG2_ACS']
        except:
            mag1 = g['MAG1_WFPC2']
            mag2 = g['MAG2_WFPC2']
        color = mag1 - mag2

        smag1 = s.data[filter1]
        smag2 = s.data[filter2]
        scolor = smag1 - smag2

        inds, = np.nonzero((smag1 < mag1.max()) & (smag1 > mag1.min()) &
                           (smag2 < smag2.max()) &  (smag2 > smag2.min()) &
                           (scolor < scolor.max()) & (scolor > scolor.min()))
        np.savetxt(sim.replace('_cor.dat','_cor_cut.match'),
                   np.column_stack((smag1[inds], smag2[inds])), fmt='%.4f')
    return


def call_make_fakeparam():
    loc ='/Volumes/tehom/research/TP-AGBcalib/SNAP/varysfh/extpagb/sim_as_phot/caf09_v1.2s_m36_s12d_ns_nas/match_run/fake_run'
    params = get_files(loc, '*.param')
    targets = [os.path.split(p)[1].split('_')[0] for p in params]
    sfh_files = get_files(loc, '*sfh')
    sfhfiles =  np.concatenate([[s for s in sfh_files if t in s] for t in targets])
    [make_fakeparam(params[i], sfhfiles[i]) for i in range(len(params))]


# 1 Run best fit sfh (on andromeda)
# 2 Normalize to optical data
# /Volumes/tehom/research/TP-AGBcalib/SNAP/varysfh/extpagb/sim_as_phot/caf09_v1.2s_m36_s12d_ns_nas/match_run/normalize.sh
# 3 add asts
# add_asts.sh
# 4 Cut lower mag limits to match optical data
# 5 cull file to mag1, mag2 list
#cut_sims()
# 6 run calcsfh on those
# !make_calcsfh.sh
# NO -- SKIP 7. Run varysfh.sh on the trilegal-as-data and use that in (8)
# 7 make fake with the sfh
#call_make_fakeparam()
# ! bash make_fake.sh; ! bash fake.sh
# 8 compare csfrs and mass distibutions.
#csfr_masshist()

"""
To run lots of jobs ...

# I did this in ipython
# triouts = !! ls out*dat
# [trilegal.utils.trilegal2matchphot(t, extra='_cor') for t in triouts]
# then copied fake and param files to the same directory and uploaded to odyssey.
# targets = np.unique([t.split('_')[1] for t in triouts])

i = 1  # command counter
k = 0  # job array script counter
nproc = 32  # how mand commands per job array (itc_cluster = 64, conroy = 32, imac ~10)
slurm = True # write scripts for job array [True] write one big script with waits [False]

# where to find calcsfh binary
calcsfh = '/n/home01/prosenfield/match2.5/bin/calcsfh'

# calcsfh flags
flags = '-PARSEC -mcdata -kroupa -zinc'

# jobarray script format
sfmt = 'calcsfh_trilegal_script_{}.sh'

header = 'calcsfh="{}"\n'.format(calcsfh)
line = header

phots = [[o for o in triouts if t in o] for t in targets]

ntot = len(triouts)

for j, t in enumerate(targets):
    param  = get_files('.', '*{}*param'.format(t))[0][2:]
    fake  = get_files('.', '*{}*fake'.format(t))[0][2:]
    for phot in phots[j]:
        out = replace_ext(phot, '.out')
        scrn = replace_ext(phot, '.scrn')
        line += '$calcsfh {0} {1} {2} {3} {4} > {5}\n'.format(param, phot, fake, out, flags, scrn)
        if i % nproc == 0 or i == ntot:
            if slurm:
                # dump the commands to a file to be called by the job array
                with open(sfmt.format(k), 'w') as outp:
                    outp.write(line)
                # start over
                line = header
                k += 1
            else:
                # just put in a wait for the rest of the jobs to complete
                l += 'wait\n'
        i += 1

if not slurm:
    with open(smft.format(k), 'w') as outp:
        outp.write(line)
"""
