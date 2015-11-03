from ResolvedStellarPops import SimGalaxy
from astropy.io import fits
import os
from dweisz.match.scripts.fake import make_fakeparam
from dweisz.match.scripts.fileio import get_files, read_fake, replace_ext
from dweisz.match.scripts.sfh import SFH
import numpy as np
import matplotlib.pyplot as plt
from ..TPAGBparams import EXT

def csfr_masshist():
    """ Do all
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
    """
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


def cut_sims():
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

