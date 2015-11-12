# Make the csfr figure and basic table
# cd /Users/rosenfield/research/TP-AGBcalib/SNAP/varysfh/extpagb
# python -m tpagb_calibration.plotting.csfr_plots -f

# add the \chi^2 column to the basic table
# cd /Users/rosenfield/research/TP-AGBcalib/SNAP/match_runs/extpagb
# e.g,
# python -m dweisz.match.scripts.likelihood ugc4305-2_f555w-f814w.sfh ugc4305-2_f555w-f814w.out.cmd

# make the TRILEGAL - MATCH test plots
# see tpagb_calibration.consistency.csfr_masshist

# make double gaussian fit plots and get stats for paper text
# cd /Users/rosenfield/research/TP-AGBcalib/SNAP/data/opt_ir_matched_v2/copy
# e.g.,
# python -m tpagb_calibration.contamination -pd ugc5139_f555w_f814w_f110w_f160w.fits

# make contamination plots (TPAGB in blue and RGB box) make TRGB color vs M_B_T plot
# cd /Users/rosenfield/research/TP-AGBcalib/SNAP/match_runs/extpagb
# python -m tpagb_calibration.contamination -f

# (re)make LF plots
# cd /Users/rosenfield/research/TP-AGBcalib/SNAP/match_runs/extpagb/keep/all_run/caf09_v1.2s_m36_s12d_ns_nas
# see tpagb_calibration/plotting.sh
# or go from scratch...
# 1. Run calcsfh, hybridMC, check outputs.
# 2. see tpagb_calibration/scripts/varysfh.sh; makes trilegal_script.sh
# 3. bash trilegal_scripts.sh, if you do 25 at a time, use utils.shift_file_number
# 4. Run asts: see tpagb_calibration/scripts/add_asts.sh
# 5. Normailze: See tpagb_calibration/scripts/plotting.sh: normit, give -p to plot or
# 6. Make plots: See tpagb_calibration/scripts/plotting.sh: runit

# make narratio table
# cd /Users/rosenfield/research/TP-AGBcalib/SNAP/match_runs/extpagb/keep/all_run/caf09_v1.2s_m36_s12d_ns_nas
# python -m tpagb_calibration.analysis.stats -n *nar*dat

# see tpagb_calibration.contamination -
# make figure comparing to M12 counts and flux
# cd /Users/rosenfield/research/TP-AGBcalib/SNAP/varysfh/extpagb
# python -m tpagb_calibration.analysis.agb_flux -f

# make tpagb mass and age distribution figure
# cd /Users/rosenfield/research/TP-AGBcalib/SNAP/match_runs/extpagb/keep/all_run/caf09_v1.2s_m36_s12d_ns_nas
# python -m tpagb_calibration.plotting.tpagb_histograms
