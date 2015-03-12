#!/usr/bin/python
"""prepare normalization table"""
from __future__ import print_function

import argparse
import difflib
import os
import sys
import time

from astropy.io import ascii, fits
import matplotlib.pyplot as plt
import numpy as np
import ResolvedStellarPops as rsp

from ..pop_synth.stellar_pops import limiting_mag, rgb_agb_regions

nirfilter1 = 'F110W'
nirfilter2 = 'F160W'
optfilter2 = 'F814W'

angst_data = rsp.angst_tables.angst_data

def test_prepared_table(filename, fakefiles, saveplot=True):
    plt.style.use('ggplot')
    tbl = ascii.read(filename)
    for fakefile in fakefiles:
        ast = rsp.ASTs(fakefile)
        ast.completeness(combined_filters=True, interpolate=True)
        ax = ast.completeness_plot()

        ifilts = list(np.nonzero((tbl['filter1'] == ast.filter1))[0])
        itargs = [i for i in range(len(tbl['target'])) if ast.target in tbl['target'][i]]
        indx = list(set(ifilts) & set(itargs))
        if len(indx) == 0:
            print('skipping {}'.format(fakefile))
            continue
        vdict = {}
        for i, k in enumerate(['comp50mag1', 'comp50mag2', 'comp90mag1', 'comp90mag2']):
            vdict[k] = tbl[indx][k]
            ax.vlines(tbl[indx][k], *ax.get_ylim(), label=k, color=next(ax._get_lines.color_cycle))
        [ax.hlines(f, *ax.get_xlim()) for f in [0.5, 0.9]]
        plt.legend(loc='best')
        if saveplot:
            plt.savefig('{}_complines.png'.format(fakefile))
        plt.close()
    return ax

def move_on(ok, msg='0 to move on: '):
    ok = int(raw_input(msg))
    time.sleep(1)
    return ok

def _plot_cmd(color, mag, xlim=None, ylim=None, hlines=None, vlines=None,
              ax=None):
    new = False
    if ax is None:
        fig, ax = plt.subplots()
        new = True
    ax.plot(color, mag, 'o', color='k', ms=3, alpha=0.3, mec='none')
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        if new:
            ax.set_ylim(ax.get_ylim()[::-1])
        
    if hlines is not None:
        for h in hlines:
            ax.hlines(h, *ax.get_xlim())
    
    if vlines is not None:
        for v in vlines:
            ax.vlines(v, *ax.get_ylim())
    ax.grid()
    return ax    

def interactive_cmd(color, mag, xlim=None, ylim=None, hlines=[],
                    vlines=[], ax=None, diag_plot=None, color_only=False):
    if ax is None:
        ax = _plot_cmd(color, mag, xlim=xlim, ylim=ylim, hlines=hlines,
                       vlines=vlines)

    ok = 1
    while ok == 1:
        ax.set_title('click two color limits')
        pts = plt.ginput(2, timeout=-1)
        colmin, colmax = [pts[i][0] for i in range(2)]
        if colmin > colmax:
            colmin, colmax = colmax, colmin
        ax.vlines(colmin, *ax.get_ylim(), color='red')
        ax.vlines(colmax, *ax.get_ylim(), color='red')
        plt.draw()
        ok = move_on(0)
    
    plt.cla()
    vlines.extend([colmin, colmax])
    ax = _plot_cmd(color, mag, xlim=xlim, ylim=ylim, hlines=hlines,
                   vlines=vlines, ax=ax)
    
    if not color_only:
        ok = 1
        while ok == 1:
            ax.set_title('click two mag limits')
            plt.draw()
            pts = plt.ginput(2, timeout=-1)
            magbright, magfaint = [pts[i][1] for i in range(2)]
            if magbright > magfaint:
                magbright, magfaint = magfaint, magbright
            ax.hlines(magfaint, *ax.get_xlim(), color='red')
            ax.hlines(magbright, *ax.get_xlim(), color='red')            
            plt.draw()
            ok = move_on(ok)
    
        plt.cla()
        hlines.extend([magfaint, magbright])

        ax = _plot_cmd(color, mag, xlim=xlim, ylim=ylim, hlines=hlines,
                       vlines=vlines, ax=ax)
    else:
        magbright = -99
        magfaint = 99

    if diag_plot is not None:
        plt.savefig(diag_plot)

    return (colmin, colmax, magbright, magfaint)

def loadmags(phot):
    phot_ext = 'acs'
    if 'wfpc2' in phot:
        phot_ext = 'wfpc2'
    elif 'IR' in phot:
        phot_ext = 'IR'
    
    tab = fits.getdata(phot)

    mag1 = tab['mag1_%s' % phot_ext]
    mag2 = tab['mag2_%s' % phot_ext]
    return mag1, mag2

def prepare(phot, fake_file, outfile, comp_frac=0.9, overwrite=False,
            color_only=False, click_on_cmd=True):
    """
    Make a line of a table culled from photometric and artificial star test
    information. The table will have 2 completeness fractions as well as
    mag and color limites chosen by eye. Mag limits are typically not important
    to chose by eye, and can simply be some mag offset from the trgb.
    Occasionally, there will be large reddening at the trgb or RHeB contamination
    near the RGB. For these galaxies, a note in the output file is necesssary.
    I added a mag_by_eye column that defaults to 0, and by hand changed it to 1
    for the cases where mag_by_eye would be better to normalize than offsets.
    """
    target, (filter1, filter2) = rsp.parse_pipeline(phot)
    lines = []
    
    if os.path.isfile(outfile):
        entries = np.genfromtxt(outfile, usecols=(0,), dtype=str)
        if target in entries:
            if overwrite:
                print('overwriting entry: {}'.format(target))
                lines = open(outfile).readlines()
                # + 1 accounts for the header
                lineidx = list(entries).index(target) + 1
            else:
                print('Appending {}, may cause dupilicate table entries. '
                      'Use -f to overwrite'.format(target))

    comp_mag1, comp_mag2 = limiting_mag(fake_file, comp_frac) 
    # I'm using angst tables:
    #comp50mag1, comp50mag2 = limiting_mag(fake_file, 0.5)

    angst_target = \
        difflib.get_close_matches(target.upper(),
                                  angst_data.targets)[0].replace('-', '_')
    print(target, angst_target)
    if filter1 == nirfilter1:
        trgb = angst_data.get_snap_trgb_av_dmod(angst_target)[0]
        comp50mag2 = angst_data.get_snap_50compmag(angst_target, nirfilter2)
        comp50mag1 = angst_data.get_snap_50compmag(angst_target, nirfilter1)
    else:
        target_row = angst_data.__getattribute__(angst_target)
        trgb = target_row['%s,%s' % (filter1, optfilter2)]['mTRGB']
        comp50mag1 = angst_data.get_50compmag(angst_target, filter1)
        comp50mag2 = angst_data.get_50compmag(angst_target, optfilter2) 

    if click_on_cmd:
        # do interactive plot
        mag1, mag2 = loadmags(phot)
        color = mag1 - mag2
        colmin, colmax, magbright, magfaint = \
            interactive_cmd(color, mag2, color_only=color_only,
                            hlines=[comp50mag2, comp_mag2, trgb])
    else:
        colmin, colmax, magbright, magfaint = -99., 99, -99, 99

    fmt = ('{target:25s} {filter1} {filter2} {comp50mag1: .4f} {comp50mag2: .4f} '
            '{comp_mag1: .4f} {comp_mag2: .4f} {mag2_trgb: .4f} '
            '{magbright: .4f}  {magfaint: .4f} {colmin: .4f} {colmax: .4f} 0 \n')

    outd = {'target': target,
            'filter1': filter1,
            'filter2': filter2,
            'comp50mag1': comp50mag1,
            'comp50mag2': comp50mag2,
            'comp_mag1': comp_mag1,
            'comp_mag2': comp_mag2,
            'mag2_trgb': trgb,
            'magbright': magbright,
            'magfaint': magfaint,
            'colmin': colmin,
            'colmax': colmax}
    
    if not os.path.isfile(outfile):
        header = ('# target filter1 filter2 comp50mag1 comp50mag2 '
                  'comp{0}mag1 comp{0}mag2 mag2_trgb magbright '
                  'magfaint colmin colmax mag_by_eye \n').format(int(comp_frac * 100))
    else:
        header = ''

    if len(lines) > 0:
        # replace existing line with new line, write out the entire file
        lines[lineidx] = fmt.format(**outd)
        with open(outfile, 'w') as out:
            [out.write(l) for l in lines]
    else:
        with open(outfile, 'a') as out:
            out.write(header)
            out.write(fmt.format(**outd))

    return

def main(argv):
    """
    """
    parser = argparse.ArgumentParser(description="Prepare for normalization schemes")

    parser.add_argument('-c', '--comp_frac', type=float, default=0.9,
                        help='completeness fraction to calculate')

    parser.add_argument('-f', '--overwrite', action='store_true',
                        help='overwrite table entry if it exists')
    
    parser.add_argument('-a', '--automatic', action='store_false',
                        help='do not do interactive plotting (by-eye values will be 99)')
    
    parser.add_argument('phot', type=str, help='photometry')

    parser.add_argument('fake', type=str, help='match AST file')

    parser.add_argument('outfile', type=str, help='output file name')

    args = parser.parse_args(argv)

    prepare(args.phot, args.fake, args.outfile, comp_frac=args.comp_frac,
            overwrite=args.overwrite, click_on_cmd=args.automatic)

    return


if __name__ == "__main__":
    main(sys.argv[1:])
