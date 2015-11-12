#! /bin/bash

if [ "$(uname)"  != 'Darwin' ]; then
    LOC="/home/rosenfield/research/TP-AGBcalib/SNAP/data/opt_ir_matched_v2/copy"
else
    LOC="/Volumes/tehom/research/TP-AGBcalib/SNAP/data/opt_ir_matched_v2/copy"
fi

function normit {
call="python -m tpagb_calibration.analysis.normalize -b 0.9 $2 $3 $4 $5 $6"
$call $LOC/${1}_f110w_f160w.fits out_${1}*_???.dat
}

function runit {
call="python -m tpagb_calibration.plotting.plotting -b 0.9 $2 $3 $4 $5 $6"
target=$(echo $1 | cut -d '_' -f 1)
lf=$(ls ${target}*lf.dat)
$call $LOC/${1}_f110w_f160w.fits $lf
}
normit ngc300-wide1_f606w_f814w -c 1.6,0.4
runit ngc300-wide1_f606w_f814w -c 1.6,0.4
#
##for PREF in ddo78_f475w_f814w kkh37_f475w_f814w ngc2403-halo-6_f606w_f814w ngc4163_f606w_f814w hs117_f606w_f814w ugc4305-1_f555w_f814w ugc4459_f555w_f814w ugc5139_f555w_f814w
#for PREF in ngc2403-halo-6_f606w_f814w ngc4163_f606w_f814w ugc4305-1_f555w_f814w ugc4459_f555w_f814w ugc5139_f555w_f814w
#do
#runit $PREF -c 1.4,0.4
##normit $PREF -c 1.4,0.4
#done
#
##for PREF in ngc3741_f475w_f814w kdg73_f475w_f814w scl-de1_f606w_f814w ugc8508_f475w_f814w
#for PREF in ngc3741_f475w_f814w kdg73_f475w_f814w ugc8508_f475w_f814w
#do
#runit $PREF -c 1.25,0.4
##normit $PREF -c 1.25,0.4
#done
#
#runit ngc2403-deep_f606w_f814w -c 1.6,0.4 -q MAG2_WFPC2,MAG4_IR
#runit eso540-030_f606w_f814w -c 0.25,3
##normit ngc2403-deep_f606w_f814w -c 1.6,0.4 -q MAG2_WFPC2,MAG4_IR
##normit eso540-030_f606w_f814w -c 0.25,3
#
#runit ugca292_f475w_f814w -c 1.25, 0.4
#runit ngc300-wide1_f606w_f814w -c 1.6, 0.4
#runit ugc4305-2_f555w_f814w -c 1.4, 0.4
#
#
##runit ngc2976-deep_f606w_f814w -c 1.6,0.4
