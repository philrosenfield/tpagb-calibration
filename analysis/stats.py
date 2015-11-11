import argparse
import logging
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from astroML.plotting import hist as mlhist
from scipy.stats import ks_2samp, ttest_ind

import ResolvedStellarPops as rsp
from ResolvedStellarPops.angst_tables import angst_data

from ..analysis.analyze import get_itpagb
from ..TPAGBparams import EXT, data_loc
from ..fileio import load_lf_file, load_observation, find_fakes, get_files
from ..pop_synth.stellar_pops import completeness_corrections

logger = logging.getLogger()

def compare_lfs(lf_files, filter1='F814W_cor', filter2='F160W_cor',
                col1='MAG2_ACS', col2='MAG4_IR', dmag=0.2):
    """
    print 2 Sample KS-test results comparing model lfs to data
    """
    def print_stats(func, a, p):
        print('{} a, p: {:.3f} {:.3f}'.format(func.__name__, *map(func, [a, p])))

    def mean_median_std(alphas, pvalues):
        print_stats(np.median, alphas, pvalues)
        print_stats(np.mean, alphas, pvalues)
        print_stats(np.std, alphas, pvalues)

    ialphas = np.array([])
    ipvalues = np.array([])
    oalphas = np.array([])
    opvalues = np.array([])
    for i, lf_file in enumerate(lf_files):
        target = os.path.split(lf_file)[1].split('_')[0]
        observation, = rsp.fileio.get_files(data_loc + '/copy', '*{}*fits'.format(target))
        try:
            mag1, mag2 = load_observation(observation, col1, col2)
        except:
            mag1, mag2 = load_observation(observation, 'MAG2_WFPC2', col2)
        color = mag1 - mag2
        oinds = get_itpagb(target, color, mag2, 'F160W')
        tpmag1 = mag1[oinds]
        tpmag2 = mag2[oinds]
        nirfake, optfake = find_fakes(target)
        lfd = load_lf_file(lf_file)

        filt1 = filter1
        filt2 = filter2
        try:
            lfd[filter1]
            lfd[filter2]
        except:
            filt1 = filter1.replace('_cor', '')
            filt2 = filter2.replace('_cor', '')

        mag1s = np.array([lfd[filt1][i][lfd['idx_norm'][i]] for i in range(len(lfd[filt1]))])
        mag2s = np.array([lfd[filt2][i][lfd['idx_norm'][i]] for i in range(len(lfd[filt2]))])
        colors = mag1s - mag2s

        minds = [get_itpagb(target, colors[i], mag2s[i], 'F160W') for i in range(len(mag2s))]
        tpmag1s = np.array([mag1s[i][minds[i]] for i in range(len(minds))])
        tpmag2s = np.array([mag2s[i][minds[i]] for i in range(len(minds))])
        ntpagb = np.array([len(i) for i in lfd['sim_agb']], dtype=float)

        for mag, mags, fake, filt in zip([tpmag1, tpmag2], [tpmag1s, tpmag2s],
                                         [optfake, nirfake], [filter1, filter2]):

            mtrgb = angst_data.get_tab5_trgb_av_dmod(target,
                                                     filt.replace('_cor', ''))[0]
            bins = np.arange(16, mtrgb, dmag)
            #figx = plt.subplots()
            #ohist, bins, _ = mlhist(mag, bins='knuth')
            #plt.close()
            comp_corr = completeness_corrections(fake, bins)
            #ohist /= comp_corr[:-1]
            #print(len(bins))
            ohist = np.array(np.histogram(mag, bins=bins)[0], dtype=float) / comp_corr[:-1]
            mhists = np.array([np.histogram(m, bins=bins)[0] for m in mags])
            #figx = plt.subplots()
            #mhists, bins, _ = zip(*[mlhist(m, bins='knuth') for m in mags])
            #plt.close()
            #mhists = np.array(mhists)
            #print([len(b) for b in bins])

            mmhist = np.median(mhists.T, axis=1)
            am, pm = ks_2samp(ohist, mmhist)
            print('{} {} a < p: {}'.format(target, filt, am < pm))
            print('Mean: {} {}'.format(am, pm))
            if '814' in filt:
                oalphas = np.append(oalphas, am)
                opvalues = np.append(opvalues, pm)
            else:
                ialphas = np.append(ialphas, am)
                ipvalues = np.append(ipvalues, pm)
        print('number of model tpagb, tpagb:')
        mean_median_std(ntpagb, ntpagb)

    print('Totals NIR a < p: {}'.format(ialphas < ipvalues))
    mean_median_std(ialphas, ipvalues)
    print('Totals OPT a < p: {}'.format(oalphas < opvalues))
    mean_median_std(oalphas, opvalues)

    return


def narratio_table(nartables, verbose=False, latex=True, full=True):
    def quadriture(x):
        xx = x * x
        return np.sqrt(np.sum(xx))

    dratios = np.array([])
    mratios =  np.array([])
    pct_diffs = np.array([])
    dr_errs = np.array([])
    mr_errs = np.array([])
    pctd_errs =  np.array([])
    line = ''
    if latex:
        line += '# target data_ratio model_ratio frac_diff\n'
        fmt = r'${:.3f}\pm{:.3f}$'
        delimiter = ' & '
    else:
        line += '# target data_ratio data_ratio_err model_ratio model_ratio_err'
        line += ' frac_diff frac_diff_err\n'
        fmt = '{:.3f} {:.3f}'
        delimiter = ' '

    for i, nartable in enumerate(nartables):
        ratio_data = rsp.fileio.readfile(nartable, string_length=36,
                                         string_column=[0, 1, 2])
        try:
            targets = np.unique([t for t in ratio_data['target']
                                 if not 'data' in t])
        except:
            print('# Error: no data for {}.'.format(nartable))
            continue

        if verbose:
            line += '% {}\n'.format(nartable)

        for target in targets:
            dindx = np.nonzero(ratio_data['target'] == 'data'.format(target))[0][0]
            indx, = np.nonzero(ratio_data['target'] == target)

            nagb = float(ratio_data[dindx]['nagb'])
            nrgb = float(ratio_data[dindx]['nrgb'])

            dratio = nagb / nrgb
            dratio_err = rsp.utils.count_uncert_ratio(nagb, nrgb)

            mrgb = np.mean(map(float, ratio_data[indx]['nrgb']))
            magb = np.mean(map(float, ratio_data[indx]['nagb']))

            mratio = magb / mrgb
            mratio_err = rsp.utils.count_uncert_ratio(magb, mrgb)

            pct_diff = 1 - (mratio / dratio)
            pct_diff_err = np.abs(pct_diff * (mratio_err / mratio + dratio_err / dratio))

            if full:
                line += delimiter.join([target.upper(),
                                        fmt.format(dratio, dratio_err),
                                        fmt.format(mratio, mratio_err),
                                        fmt.format(pct_diff, pct_diff_err)])

            else:
                line += delimiter.join([fmt.format(mratio, mratio_err),
                                        fmt.format(pct_diff, pct_diff_err)])
            line += '\n'

            dratios = np.append(dratios, dratio)
            mratios = np.append(mratios, mratio)
            dr_errs = np.append(dr_errs, dratio_err)
            mr_errs = np.append(mr_errs, mratio_err)

    mmratios = np.mean(mratios)
    mdratios = np.mean(dratios)
    mmr_err = quadriture(mr_errs)
    mdr_err = quadriture(dr_errs)

    pct_diff_mean = 1 - (mmratios / mdratios)
    pct_diff_mean_err = np.abs(pct_diff_mean * (mmr_err / mmratios + mdr_err / mdratios))

    if full:
        line += delimiter.join(['Mean',
                                fmt.format(np.mean(dratios), mdr_err),
                                fmt.format(np.mean(mratios), mmr_err),
                                fmt.format(pct_diff_mean, pct_diff_mean_err)])
    else:
        line += delimiter.join([fmt.format(np.mean(mratios), mmr_err),
                                fmt.format(pct_diff_mean, pct_diff_mean_err)])
    line += '\n'

    print line



def main(argv):
    parser = argparse.ArgumentParser(description='make narratio table')

    parser.add_argument('-n', '--narratio', action='store_true',
                        help='make narratio table or if not set, do lf comparisons')

    parser.add_argument('files', type=str, nargs='*',
                        help='narratio files or lf_files depending on -n')

    args = parser.parse_args(argv)


    if args.narratio:
        narratio_table(args.files)
    else:
        compare_lfs(args.files)


if __name__ == "__main__":
    main(sys.argv[1:])
