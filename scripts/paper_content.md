## With MATCH outputs:
e.g.,
```cd /Users/rosenfield/research/TP-AGBcalib/SNAP/match_runs/extpagb```
### Make the csfr figure and basic table
wherever the zc merge files are:
``` python -m tpagb_calibration.plotting.csfr_plots -f ```

### add the \chi^2 column to the basic table
e.g,
```python -m dweisz.match.scripts.likelihood ugc4305-2_f555w-f814w.sfh ugc4305-2_f555w-f814w.out.cmd ```

### make the TRILEGAL - MATCH test plots
see ```tpagb_calibration.consistency.csfr_masshist```

### make double gaussian fit plots and get stats for paper text
e.g.,
``` cd /Users/rosenfield/research/TP-AGBcalib/SNAP/data/opt_ir_matched_v2/copy ```
``` python -m tpagb_calibration.contamination -pd *.fits ```

### make contamination plots (TPAGB in blue and RGB box) make TRGB color vs M_B_T plot
See doc string to test rheb-tpagb line using trilegal outputs. 
``` python -m tpagb_calibration.contamination -f ```

## With TRILEGAL ouputs:
Where the trilegal outputs are stored, e.g.,
``` cd /Users/rosenfield/research/TP-AGBcalib/SNAP/match_runs/extpagb/keep/all_run/caf09_v1.2s_m36_s12d_ns_nas ```
### (re)make LF plots
see tpagb_calibration/plotting.sh or go from scratch...
* Run ```calcsfh```, ```hybridMC```, check outputs.
* see ```tpagb_calibration/scripts/varysfh.sh```; makes ```trilegal_script.sh```
* ```bash trilegal_scripts.sh```, if you do 25 at a time and want to add to 100, ```use utils.shift_file_number; bash shift.sh```
* Apply asts: see ```tpagb_calibration/scripts/add_asts.sh```
* Normailze: See ```tpagb_calibration/scripts/plotting.sh: normit```, give ```-p``` to plot or
* Make plots: See ```tpagb_calibration/scripts/plotting.sh: runit```

### make narratio table
``` python -m tpagb_calibration.analysis.stats -n *nar*dat > nartable.tex ```

### make figure comparing to M12 counts and flux
``` python -m tpagb_calibration.analysis.agb_flux -f ```

### make tpagb mass and age distribution figure
``` python -m tpagb_calibration.plotting.tpagb_histograms```
