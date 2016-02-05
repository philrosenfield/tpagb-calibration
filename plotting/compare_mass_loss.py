
"""#step status  NTP           age/yr        M_*   lg dM/dt      lg L_*   lg L_H   lg T_*       M_c           Y                Z    PHI_TP       C/O      Tbot   Pmod        P1        P0  Idust  M_dustmin  M_predust      Mdust             H           He3           He4           Li7           Be7           C12           C13           N14           N15           O16           O17           O18           F19          Ne20          Ne21          Ne22          Na23          Mg24          Mg25          Mg26          Al27          Si28         Al26g         Mcmin          lambda           l_max            Mdup          dMlost           dMtot              dt             dmc       dmc_nodup     Lconv   Tbot_ov   lgrho_ph  lgptot_ph   lgkap_ph      mu_ph
"""


    age/yr
    lg T_*
    lg L_*
    P [0/1 depending on pmod]
    C/O
    M_*
    lg dM/dt


class AGBTrack(object):
    def __init__(self, filename):
        self.load_agbtrack(filename)
        self.period = np.array([data['P{:.0f}'.format(p)][i]
                                for i, p in enumerate(data['Pmod'])])


    def load_agbtrack(self, filename):
        '''
        made to read all of Paola's tracks. It takes away her "lg" meaning log.
        Returns an AGBTracks object. If there is a problem reading the data, all
        data are passed as zeros.
        '''
        f = open(filename, 'r')
        lines = f.readlines()
        f.close()
        line = lines[0]
        if len(lines) == 1:
            logger.warning('only one line in {}'.format(filename))
            return [np.nan]
        col_keys = line.replace('#', '').replace('lg', '').replace('*', 'star')
        col_keys = col_keys.strip().split()
        try:
            data = np.genfromtxt(filename, names=col_keys)
        except ValueError:
            print('problem with', filename)
            data = np.zeros(len(col_keys))
        return data

    def vw93_plot(self):
        fig, axs = plt.subsplot(nrows=6, sharex=True)
        ycols = ['T_star', 'L_star', 'period', 'CO', 'M_star', 'dMdt']
        scale = 1e5
        for i in range(len(axs)):
            try:
                ax[i].plot(data['ageyr'] / scale, data[ycols[i]])
            except:
                ax[i].plot(data['ageyr'] / scale, ycols[i])


def translate_colkey(col):
    new_col = col
    tdict = {'T_star': ,
            'L_star':,
            'period':,
            'CO':,
            'M_star':,
            'dMdt': }

    if col in tdict.keys():
        new_col = tdict[col]

    return new_col

def compare_mass_loss(masses=1.0, z=0.001, sets=['NOV13', 'OCT13', 'NOV13eta0'],
		      paola=False):
    '''
    made to plot a comparison between several mass prescriptions.
    Labels for the plot are set up stupidly, maybe in in_dict or labels arg...
    '''
    from matplotlib.ticker import NullFormatter

    teff_max = 3.5
    track_files = None
    if paola is True:
	# hack to use specific tracks from paola
	track_dir = research_path + '/TP-AGBcalib/AGBTracks/plots_for_paperI/'
	file_end = '_Mc0.00_dMc0.00_Tbd6.40_L0.00_dL0.00_C0.00_Nr3.00_rates0_KOPv_KMOLv.dat'
	if masses == 2.0:
	    track_files = \
	     [track_dir + 'agb_2.00_Z0.00100000_Mdot50_eta0.00' + file_end,
	      track_dir + 'agb_2.00_Z0.00100000_Mdot49_eta0.40' + file_end,
	      track_dir + 'agb_2.00_Z0.00100000_Mdot48_eta8.00' + file_end,
	      track_dir + 'agb_2.00_Z0.00100000_Mdot50_eta0.40' + file_end]
	    teff_max = 3.4
	if masses == 1.0:
	    track_files = \
	     [track_dir + 'agb_1.00_Z0.00100000_Mdot50_eta0.00' + file_end,
	      track_dir + 'agb_1.00_Z0.00100000_Mdot49_eta0.40' + file_end,
	      track_dir + 'agb_1.00_Z0.00100000_Mdot48_eta8.00' + file_end,
	      track_dir + 'agb_1.00_Z0.00100000_Mdot50_eta0.40' + file_end]
	    teff_max = 3.4

	labels = ['$\\dot{M}_{\\rm{pre-dust}}=0.0$','$\\rm{R75}$',
		  '$\\rm{SC05}$', '$\\rm{mSC05}$']

    if track_files is not None:
	nrows = len(track_files)
    else:
	nrows = len(sets)
    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(8, 8))
    anorm = 1e6
    xlab0 = '\\rm{Age}\ (10^6\ \\rm{yr})'
    ylab0 = '\log\ \dot{M}\ (\\rm{M_\odot\ yr}^{-1})'
    ylab1 = '\log\ L\ (L_\odot)'
    xlab1 = '\log\ T_{\\rm eff}\ (\\rm{K})'

    agb_tracks_dir = research_path + 'TP-AGBcalib/AGBTracks/CAF09'
    cols = ['#d73027', '#4575b4', '#fc8d59', '#91bfdb', '#fee090', '#e0f3f8']

    if type(masses) is not list:
        masses = [masses]
	cols = ['k']

    for j, mass in enumerate(masses):
	if track_files is None:
            tnames = []
            labels = []
            for tset in sets:
                label = translate_model_name(tset)
                direc = os.path.join(agb_tracks_dir, 'S_' + tset)
                direc, = [os.path.join(direc, d)
                        for d in os.listdir(direc) if str(z) in d]
                tname = rsp.fileIO.get_files(direc, 'agb_%.2f*' % mass)[0]
                tnames.append(tname)
                labels.append('$%s$' % label)
            tracks = [fileIO.get_numeric_data(t) for t in tnames]
	else:
	    tracks = [fileIO.get_numeric_data(t) for t in track_files]

        for i in range(len(tracks)):
            axs[i][0].plot(tracks[i].data_array['ageyr']/anorm,
                           tracks[i].data_array['dMdt'],
                           label='$M=%g\ M_\odot$' % mass, lw=1, color=cols[j])
            axs[i][0].plot(tracks[i].data_array['ageyr'][tracks[i].cstar]/anorm,
                           tracks[i].data_array['dMdt'][tracks[i].cstar],
                           lw=1, color='darkred')
            axs[i][1].plot(tracks[i].data_array['T_star'],
                           tracks[i].data_array['L_star'],
                           label='$M=%g\ M_\odot$' % mass, lw=1, color=cols[j])
            axs[i][1].plot(tracks[i].data_array['T_star'][tracks[i].cstar],
                           tracks[i].data_array['L_star'][tracks[i].cstar],
                           lw=1, color='darkred')
            axs[i][0].annotate(labels[i], (0.03, 0.96), fontsize=16,
                               xycoords='axes fraction', va='top')
    axs[-1, 0].set_xlabel('$%s$' % xlab0, fontsize=20)
    axs[-1, 1].set_xlabel('$%s$' % xlab1, fontsize=20)
    plt.annotate('$%s$' % ylab0, (0.05, 0.5), fontsize=20, va='center',
		       xycoords='figure fraction', rotation='vertical')
    plt.annotate('$%s$' % ylab1, (0.95, 0.5), fontsize=20, va='center',
		       xycoords='figure fraction', rotation='vertical')

    [ax.yaxis.tick_right() for ax in axs.flatten()[1::2]]
    [ax.xaxis.set_major_formatter(NullFormatter())
     for ax in axs.flatten()[:-2]]

    # mass loss
    [ax.set_ylim(-11.5, -4.5) for ax in axs[:, 0]]

    # log l
    [ax.set_ylim(2.81, 4.25) for ax in axs[:, 1]]

    # log te
    [ax.set_xlim(3.66, teff_max) for ax in axs[:, 1]]

    # age Myr
    [ax.set_xlim(0, 2.45) for ax in axs[:, 0]]

    # top left plot only
    if paola is False:
        [ax.legend(loc=4, fontsize=16, frameon=False)
	 for ax in [axs.flatten()[0]]]

    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.savefig('compare_massloss_M%g_Z%g.png' % (masses[0], z), dpi=150)
    return axs
