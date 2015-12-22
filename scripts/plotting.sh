#!/bin/bash
LOC="/Users/rosenfield/research/TP-AGBcalib/SNAP/data/opt_ir_matched_v2/copy"

function doit {
if [ $1 == "normit" ]
then
    call="python -m tpagb_calibration.analysis.normalize $3 $4 $5 $6 $7"
    #call="python -m tpagb_calibration.analysis.normalize -b 0.9 $3 $4 $5 $6 $7"
    $call $LOC/${2}_f110w_f160w.fits out_${2}*_???.dat
elif [ $1 == "plotit" ]
then
    call="python -m tpagb_calibration.plotting.plotting $3 $4 $5 $6 $7"
    #call="python -m tpagb_calibration.plotting.plotting -b 0.9 $3 $4 $5 $6 $7"
    target=$(echo $2 | cut -d '_' -f 1)
    lf=$(ls ${target}*lf.dat)
    $call $LOC/${2}_f110w_f160w.fits $lf
else
    echo "usage: bash normalize.sh option"
    echo "option:"
    echo "normit ... do normalization"
    echo "plotit ... plot after normalization"
    exit
fi
}

doit $1 ddo82_f606w_f814w -c 1.59,0.40
doit $1 eso540-030_f606w_f814w -c 1.27,0.40
doit $1 kdg73_f475w_f814w -c 1.33,0.40
doit $1 ngc2403-deep_f606w_f814w -c 1.62,0.60 -q MAG2_WFPC2,MAG4_IR
doit $1 ngc2403-halo-6_f606w_f814w -c 1.59,0.60
doit $1 ngc300-wide1_f606w_f814w -c 1.74,0.60
doit $1 ngc3741_f475w_f814w -c 1.34,0.40
doit $1 ngc4163_f606w_f814w -c 1.49,0.40
doit $1 ugc4305-1_f555w_f814w -c 1.47,0.40
doit $1 ugc4305-2_f555w_f814w -c 1.46,0.40
doit $1 ugc4459_f555w_f814w -c 1.40,0.40
doit $1 ugc5139_f555w_f814w -c 1.37,0.40
doit $1 ugc8508_f475w_f814w -c 1.41,0.40
doit $1 ugca292_f475w_f814w -c 1.15,0.40



#$1 hs117_f606w_f814w -c 1.32,0.40
#$1 ngc300-wide1_f475w_f814w -c 1.74,0.60
#$1 ngc404-deep_f606w_f814w -c 1.74,0.40 -q MAG2_WFPC2,MAG4_IR
#$1 ddo71_f606w_f814w -c 1.37,0.40
#$1 ddo78_f475w_f814w -c 1.55,0.40
#$1 kkh37_f475w_f814w -c 1.38,0.40
#$1 m81-deep_f475w_f814w -c 1.80,0.60
#$1 m81-deep_f606w_f814w -c 1.80,0.60
#$1 ngc2976-deep_f606w_f814w -c 1.69,0.40
#$1 ngc404-deep_f606w_f814w -c 1.74,0.40 -q MAG2_WFPC2,MAG4_IR
#$1 ngc4163_f475w_f814w -c 1.49,0.40
#$1 scl-de1_f606w_f814w -c 1.32,0.40
#$1 ugca292_f606w_f814w -c 1.15,0.40


## get the prefixs:
## ls out*000.dat | cut -d '_' -f 2-4
## or
## ls caf09_v1.2s_m36_s12d_ns_nas/out*000.dat | cut -d '_' -f 7-9
