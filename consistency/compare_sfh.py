import numpy as np
import from dweisz.match import scripts as match
import os
import ResolvedStellarPops as rsp

def within(val, val2, perr=0.0, merr=0.0,  perr2=0.0, merr2=0.0):
    frac_diff = (val - val2) / val2
    nerr = np.nan
    nerr2 = np.nan
    if perr2 is not None:
        dif = val2 - val
        if np.sign(dif) > 0:
            # how many error bars away
            if merr2 > 0:
                nerr2 = (val - val2) / merr2
            if merr > 0:
                nerr = (val2 - val) / merr
        else:
            if perr2 > 0:
                nerr2 = (val - val2) / perr2
            if perr > 0:
                nerr = (val2 - val) / perr

    return frac_diff, nerr, nerr2

def MelbourneTable1():
    table1 = '/Volumes/tehom/Dropbox/andromeda/research/TP-AGBcalib/SNAP/tables/melbourne2012_tab1.dat'
    tab1 =  rsp.fileio.readfile(table1, string_column=0)
    sfh_loc = '/Volumes/tehom/Dropbox/andromeda/research/TP-AGBcalib/SNAP/varysfh/extpagb'
    sfh_names = ['ddo71_f606w_f814w.sfh',
                 'ddo78_f475w_f814w.sfh',
                 #'ddo82_F606W_F814W.sfh',
                 #'eso540-030_F606W_F814W.sfh',
                 'hs117_f606w_f814w.sfh',
                 'ic2574-sgs_F555W_F814W.sfh',
                 'kdg73_f475w_f814w.sfh',
                 'kkh37_f475w_f814w.sfh',
                 #'m81-deep_f475w_f814w.sfh',
                 #'m81-deep_f606w_f814w.sfh',
                 'ngc2403-deep_f606w_f814w.sfh',
                 'ngc2403-halo-6_f606w_f814w.sfh',
                 'ngc2976-deep_f606w_f814w.sfh',
                 #'ngc300-wide1_F475W_F814W.sfh',
                 'ngc300-wide1_F606W_F814W.sfh',
                 'ngc3741_F475W_F814W.sfh',
                 #'ngc404-deep_F606W_F814W.sfh',
                 #'ngc4163_f475w_f814w.sfh',
                 #'ngc4163_f606w_f814w.sfh',
                 'scl-de1_f606w_f814w.sfh',
                 'ugc4305-1_f555w_f814w.sfh',
                 'ugc4459_f555w_f814w.sfh',
                 'ugc5139_f555w_f814w.sfh',
                 'ugc8508_f475w_f814w.sfh',
                 #'ugca292_f475w_f814w.sfh',
                 #'ugca292_f606w_f814w.sfh'
                 ]

    sfh_names = [os.path.join(sfh_loc, s) for s in sfh_names]
    sfhs = [match.utils.MatchSFH(s) for s in sfh_names]
    wdict = {}
    dmodfs = []
    fs = []
    for s in sfhs:
        s.target, s.filters = rsp.asts.parse_pipeline(s.name)
        wdict[s.target] = s.sfr_weighted_metallicity()
        dat = tab1[[tab1['target'] == s.target]]
        dmod_f, nnd1, nnd2 = within(s.dmod, dat['mM'], perr=s.dmod_perr, merr=s.dmod_merr)
        if np.abs(dmod_f) > 0.01:
            print s.target, dmod_f, s.dmod, dat['mM']

        if len(dmod_f) > 0:
            dmodfs.append(dmod_f[0])
            wdict[s.target]['fdmod'] = dmod_f
            wdict[s.target]['Mdmod'] = dat['mM']
            wdict[s.target]['sfr'] = s
        f, n, n2 = within(wdict[s.target]['mhw'], dat['MH'], perr2=dat['MH_err'], merr2=dat['MH_err'])
        if len(f) > 0:
            fs.append(f[0])
            wdict[s.target]['MMH'] = dat['MH']
        if np.abs(f) > 0.15:
            if n2 > 1:
                print s.target, f, n2
                fs.pop(-1)
            print s.name, f, n2

    print('dmod agrees to {:.2f}% with a mean fractional diff {:.4f}'.format(1-100*np.max(np.abs(dmodfs)), np.mean(np.abs(dmodfs))))
    print('M/H agrees to {:.2f}% with a mean fractional diff {:.4f}'.format(1-100*np.max(np.abs(fs)), np.mean(fs)))
    return wdict
