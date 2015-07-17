"""
Utilities to analyze trilegal output catalogs of TPAGB models
"""
import argparse
import logging
import numpy as np
import os
import sys
import time

#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pylab as plt
import ResolvedStellarPops as rsp

from astropy.io import ascii
from IPython import parallel
from ..pop_synth.stellar_pops import normalize_simulation, rgb_agb_regions
from ..sfhs.star_formation_histories import StarFormationHistories
from ..fileio import load_obs, find_fakes

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

angst_data = rsp.angst_tables.angst_table.AngstTables()

__all__ = ['get_itpagb']

def tpagb_rheb_line(color, mag, dmod=0.):
    b = 1.17303
    m = -5.20269
    # redder than the line
    return np.nonzero(color > ((mag - b - dmod) / m))

def get_itpagb(target, color, mag, col):
    
    # careful! get_snap assumes F160W
    if '160' in col or '110' in col or 'IR' in col:
        try:
            mtrgb, Av, dmod = angst_data.get_snap_trgb_av_dmod(target.upper())
        except:
            return [np.nan]
        redward_of_rheb, = tpagb_rheb_line(color, mag, dmod=dmod)
    else:
        mtrgb, Av, dmod = angst_data.get_tab5_trgb_av_dmod(target.upper(),
                                                           filters=col)
        redward_of_rheb = np.arange(len(color))
    
    brighter_than_trgb, = np.nonzero(mag < mtrgb)
    itpagb = list(set(redward_of_rheb) & set(brighter_than_trgb))
    return itpagb

# Someday this will call all the codes in order
# add_asts
# normalize
# decontaminate
# plotting
# stats


def main(argv):
    """main function of analyze"""
    description="Run analysis routines, so far just match_stats."
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='does nothing')

    parser.add_argument('-f', '--overwrite', action='store_true',
                        help='does nothing.')

    parser.add_argument('hmc_file', type=str,
                        help='MATCH HybridMC file')

    parser.add_argument('cmd_file', type=str,
                        help='MATCH SFH file: must have the format target_filter1_filter2.extensions')

    args = parser.parse_args(argv)
    
    rsp.match.likelihood.match_stats(args.hmc_file, args.cmd_file, dry_run=False)


def fit_samples(samples):
    gmix = mixture.GMM(n_components=2, covariance_type='full', n_iter=1000)
    gmix.fit(samples)
    print gmix.means_
    #colors = ['r' if i==0 else 'g' for i in gmix.predict(samples)]
    #ax = plt.gca()
    #ax.scatter(samples[:,0], samples[:,1], c=colors, alpha=0.8)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.hist(samples[:,0], 100) # draw samples
    b = np.arange(samples[:,0].min(), samples[:,0].max(), 100)
    ax2.plot(b, np.exp(gmm.score_samples(b)[0]), 'r') # draw GMM
    plt.show()
    return gmix

def compute_GMM(N, covariance_type='full', n_iter=1000):
    models = [None for n in N]
    for i in range(len(N)):
        print N[i]
        models[i] = GMM(n_components=N[i], n_iter=n_iter,
                        covariance_type=covariance_type)
        models[i].fit(sample)
    AIC = [m.aic(sample) for m in models]
    BIC = [m.bic(sample) for m in models]

    i_best = np.argmin(BIC)
    gmm_best = models[i_best]
    print "best fit converged:", gmm_best.converged_
    print "BIC: n_components =  %i" % N[i_best]

    return models, gmm_best

#for mu, C, w in zip(gmm_best.means_, gmm_best.covars_, gmm_best.weights_):
#        draw_ellipse(mu, C, ax=ax, scales=[3], fc='none', ec='k')

def contamination(phot, faint_mag, bright_mag=21, mag_bins=None, cmin=1, cmax=2.5,
                  thresh=100):
    # 1 if fits file (observation):
    # read
    # 2 if stellar pop (model):
    # correct for ASTs
    # read
    mag1, mag2 = np.loadtxt(phot, unpack=True)
    color = mag1 - mag2

    # 3 get stars brighter than trgb
    inds, =  np.nonzero((color > cmin) & (mag2 < faint_mag) & (color < cmax) &
                        (mag2 > bright_mag))
    # 4 bin them
    
    # 5 fit double gaussian
    # a) at all mags (is there strong evidence for 2 gaussians?)
    # b) at some mag step
    if mag_bins is None:
        smag2 = np.argsort(mag2[inds])
        nstars = len(smag2)
        stars_per_bin = nstars / 3
        inds1 = smag2[:stars_per_bin]
        inds2 = smag2[stars_per_bin: 2 * stars_per_bin]
        inds3 = smag2[2 * stars_per_bin:]
        
    #dinds = np.digitize(mag2[inds], mag_bins)
    # some check here if the dinds are well populated...
    cseps = []
    mseps = []
    #halfs = np.diff(mag_bins) / 2.
    #halfs = np.append(halfs, 0)
    from sklearn.mixture import GMM
    fig, ax = plt.subplots()
    ax.plot(color[inds], mag2[inds], '.', color='gray')
    #for j, i in enumerate(np.unique(dinds)):
    for j, i in enumerate([inds1, inds2, inds3]):
        #isamp = inds[dinds==i]
        isamp = inds[i]
        if len(isamp) < thresh:
            continue
        #print mag_bins[i]
        #mseps.append(mag_bins[j] + halfs[j])
        mdiff = np.max(mag2[isamp]) - np.min(mag2[isamp])
        mseps.append(np.mean(mag2[isamp]))
        #sample = np.column_stack((color[isamp], mag2[isamp]))
        sample = color[isamp]
        #models = fit_samples(sample)
        #models = compute_GMM(N)
        gmix = GMM(n_components=2, covariance_type='full', n_iter=1000)
        gmix.fit(sample)
        print gmix.means_
        
        #ax2 = ax1.twinx()
        #ax1.hist(sample, 100) # draw samples
        
        x = np.linspace(sample.min(), sample.max(), 100)
        logprob, responsibilities = gmix.score_samples(x)
        pdf = np.exp(logprob)
        
        pdf_individual = responsibilities * pdf[:, np.newaxis]
        # each gaussian shifted to the proper mag bin (1 mag width wide)
        ax.plot(x, -1. * mdiff * pdf_individual[:, 1] / np.max(pdf) + mseps[j], '--g')
        ax.plot(x, -1. * mdiff * pdf_individual[:, 0] / np.max(pdf) + mseps[j], '--k')
        [ax.vlines(v, *ax.get_ylim()) for v in gmix.means_]
        #ax.plot(x, -1. * np.diff(mag_bins)[j] * pdf_individual[:, 1] / np.max(pdf) + mag_bins[i], '--g')
        #ax.plot(x, -1. * np.diff(mag_bins)[j] * pdf_individual[:, 0] / np.max(pdf) + mag_bins[i], '--k')
        #ax.set_title(mag_bins[i])
        # full probablity
        #ax.plot(x, -1. * np.exp(gmix.score_samples(x)[0])/np.max(np.exp(gmix.score_samples(x)[0])) + mag_bins[i], 'r') # draw GMM
        ax.plot(x, -1. * mdiff * pdf/np.max(pdf) + mseps[j], 'r') # draw GMM
        
        #isect = np.argmin(np.abs(pdf_individual[:, 1] - pdf_individual[:, 0]))
        isect = rsp.utils.find_peaks(pdf)['minima_locations']
        #ax.plot(x[pdf.argmax() + isect], pdf[pdf.argmax() + isect], 'o')
        if len(isect) > 0:
            cseps.append(x[isect])

    #mseps = np.array(mag_bins)[:-1] + np.diff(mag_bins)/2.
    ax.plot(cseps, mseps, 'o')
    #   i) fixed width
    #   ii) fixed width + min threshold of stars (per bin?) to combine mag bins
    #       ... need a minimum of 2 points, one on the trgb, one above.
    # 6 find "contamination" of each gaussian around the intersection points
    # 7 refit to minimize contamination?

# call contamination -- all galaxies, all old galaxies, one at a time
# could use same SFH and apply ast corrections over and over and see how it changes.    

if __name__ == "__main__":
    main(sys.argv[1:])


### Snippets below ###
def contamination_by_phases(sgal, srgb, sagb, filter2, diag_plot=False,
                            color_cut=None, target='', line=''):

    """
    contamination by other phases than rgb and agb
    """
    regions = ['MS', 'RGB', 'HEB', 'BHEB', 'RHEB', 'EAGB', 'TPAGB']
    if line == '':
        line += '# %s %s \n' % (' '.join(regions), 'Total')

    sgal.all_stages()
    indss = [sgal.__getattribute__('i%s' % r.lower()) for r in regions]
    try:
        if np.sum(indss) == 0:
            msg = 'No stages in StarPop. Run trilegal with -l flag'
            logger.warning(msg)
            return '{}\n'.format(msg)
    except:
        pass
    if diag_plot is True:
        fig, ax = plt.subplots()

    if color_cut is None:
        inds = np.arange(len(sgal.data[filter2]))
    else:
        inds = color_cut
    mag = sgal.data[filter2][inds]

    ncontam_rgb = [list(set(s) & set(inds) & set(srgb)) for s in indss]
    ncontam_agb = [list(set(s) & set(inds) & set(sagb)) for s in indss]

    rheb_eagb_contam = len(ncontam_rgb[4]) + len(ncontam_rgb[5])
    frac_rheb_eagb = float(rheb_eagb_contam) / \
        float(np.sum([len(n) for n in ncontam_rgb]))

    heb_rgb_contam = len(ncontam_rgb[2])
    frac_heb_rgb_contam = float(heb_rgb_contam) / \
        float(np.sum([len(n) for n in ncontam_rgb]))

    mags = [mag[n] if len(n) > 0 else np.zeros(10) for n in ncontam_rgb]

    mms = np.concatenate(mags)
    ms, = np.nonzero(mms > 0)
    bins = np.linspace(np.min(mms[ms]), np.max(mms[ms]), 10)
    if diag_plot is True:
        [ax.hist(mags, bins=bins, alpha=0.5, stacked=True,
                     label=regions)]

    nrgb_cont = np.array([len(n) for n in ncontam_rgb], dtype=int)
    nagb_cont = np.array([len(n) for n in ncontam_agb], dtype=int)

    line += 'rgb %s %i \n' % ( ' '.join(map(str, nrgb_cont)), np.sum(nrgb_cont))
    line += 'agb %s %i \n' % (' '.join(map(str, nagb_cont)), np.sum(nagb_cont))

    line += '# rgb eagb contamination: %i \n' % rheb_eagb_contam
    line += '# frac of total in rgb region: %.3f \n' % frac_rheb_eagb
    line += '# rc contamination: %i \n' % heb_rgb_contam
    line += '# frac of total in rgb region: %.3f \n' % frac_heb_rgb_contam

    logger.info(line)
    if diag_plot is True:
        ax.legend(numpoints=1, loc=0)
        ax.set_title(target)
        plt.savefig('contamination_%s.png' % target, dpi=150)
    return line

