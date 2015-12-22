#! /bin/bash

LOC="/Users/rosenfield/research/TP-AGBcalib/SNAP/data/opt_ir_matched_v2/copy"

function normit {
call="python -m tpagb_calibration.analysis.normalize $2 $3 $4 $5 $6"
#call="python -m tpagb_calibration.analysis.normalize -b 0.9 $2 $3 $4 $5 $6"
$call $LOC/${1}_f110w_f160w.fits out_${1}*_???.dat
}

function runit {
call="python -m tpagb_calibration.plotting.plotting $2 $3 $4 $5 $6"
#call="python -m tpagb_calibration.plotting.plotting -b 0.9 $2 $3 $4 $5 $6"
target=$(echo $1 | cut -d '_' -f 1)
lf=$(ls ${target}*lf.dat)
$call $LOC/${1}_f110w_f160w.fits $lf
}


#normit ngc404-deep_f606w_f814w -c 1.74,0.40 -q MAG2_WFPC2,MAG4_IR
#
#normit ddo71_f606w_f814w -c 1.37,0.40
#normit ddo78_f475w_f814w -c 1.55,0.40
normit ddo82_f606w_f814w -c 1.59,0.40
normit eso540-030_f606w_f814w -c 1.27,0.40
#normit hs117_f606w_f814w -c 1.32,0.40
normit kdg73_f475w_f814w -c 1.33,0.40
#normit kkh37_f475w_f814w -c 1.38,0.40
#normit m81-deep_f475w_f814w -c 1.80,0.60
#normit m81-deep_f606w_f814w -c 1.80,0.60
normit ngc2403-deep_f606w_f814w -c 1.62,0.60 -q MAG2_WFPC2,MAG4_IR
normit ngc2403-halo-6_f606w_f814w -c 1.59,0.60
#normit ngc2976-deep_f606w_f814w -c 1.69,0.40
normit ngc300-wide1_f475w_f814w -c 1.74,0.60
normit ngc300-wide1_f606w_f814w -c 1.74,0.60
normit ngc3741_f475w_f814w -c 1.34,0.40
#normit ngc404-deep_f606w_f814w -c 1.74,0.40 -q MAG2_WFPC2,MAG4_IR
#normit ngc4163_f475w_f814w -c 1.49,0.40
normit ngc4163_f606w_f814w -c 1.49,0.40
#normit scl-de1_f606w_f814w -c 1.32,0.40
normit ugc4305-1_f555w_f814w -c 1.47,0.40
normit ugc4305-2_f555w_f814w -c 1.46,0.40
normit ugc4459_f555w_f814w -c 1.40,0.40
normit ugc5139_f555w_f814w -c 1.37,0.40
normit ugc8508_f475w_f814w -c 1.41,0.40
normit ugca292_f475w_f814w -c 1.15,0.40
#normit ugca292_f606w_f814w -c 1.15,0.40

#runit ddo71_f606w_f814w -c 1.37,0.40
#runit ddo78_f475w_f814w -c 1.55,0.40
runit ddo82_f606w_f814w -c 1.59,0.40
runit eso540-030_f606w_f814w -c 1.27,0.40
#runit hs117_f606w_f814w -c 1.32,0.40
runit kdg73_f475w_f814w -c 1.33,0.40
#runit kkh37_f475w_f814w -c 1.38,0.40
#runit m81-deep_f475w_f814w -c 1.80,0.60
#runit m81-deep_f606w_f814w -c 1.80,0.60
runit ngc2403-deep_f606w_f814w -c 1.62,0.60 -q MAG2_WFPC2,MAG4_IR
runit ngc2403-halo-6_f606w_f814w -c 1.59,0.60
#runit ngc2976-deep_f606w_f814w -c 1.69,0.40
runit ngc300-wide1_f475w_f814w -c 1.74,0.60
runit ngc300-wide1_f606w_f814w -c 1.74,0.60
runit ngc3741_f475w_f814w -c 1.34,0.40
#runit ngc404-deep_f606w_f814w -c 1.74,0.40 -q MAG2_WFPC2,MAG4_IR
#runit ngc4163_f475w_f814w -c 1.49,0.40
runit ngc4163_f606w_f814w -c 1.49,0.40
#runit scl-de1_f606w_f814w -c 1.32,0.40
runit ugc4305-1_f555w_f814w -c 1.47,0.40
runit ugc4305-2_f555w_f814w -c 1.46,0.40
runit ugc4459_f555w_f814w -c 1.40,0.40
runit ugc5139_f555w_f814w -c 1.37,0.40
runit ugc8508_f475w_f814w -c 1.41,0.40
runit ugca292_f475w_f814w -c 1.15,0.40
#runit ugca292_f606w_f814w -c 1.15,0.40

#
#
#
#
##normit ngc300-wide1_f606w_f814w -c 1.6,0.4
##runit ngc300-wide1_f606w_f814w -c 1.6,0.4
##
###for PREF in ddo78_f475w_f814w kkh37_f475w_f814w ngc2403-halo-6_f606w_f814w ngc4163_f606w_f814w hs117_f606w_f814w ugc4305-1_f555w_f814w ugc4459_f555w_f814w ugc5139_f555w_f814w
##for PREF in ngc2403-halo-6_f606w_f814w ngc4163_f606w_f814w ugc4305-1_f555w_f814w ugc4459_f555w_f814w ugc5139_f555w_f814w
##do
##runit $PREF -c 1.4,0.4
###normit $PREF -c 1.4,0.4
##done
##
###for PREF in ngc3741_f475w_f814w kdg73_f475w_f814w scl-de1_f606w_f814w ugc8508_f475w_f814w
##for PREF in ngc3741_f475w_f814w kdg73_f475w_f814w ugc8508_f475w_f814w
##do
##runit $PREF -c 1.25,0.4
###normit $PREF -c 1.25,0.4
##done
##
##runit ngc2403-deep_f606w_f814w -c 1.6,0.4 -q MAG2_WFPC2,MAG4_IR
##runit eso540-030_f606w_f814w -c 0.25,3
###normit ngc2403-deep_f606w_f814w -c 1.6,0.4 -q MAG2_WFPC2,MAG4_IR
###normit eso540-030_f606w_f814w -c 0.25,3
##
##runit ugca292_f475w_f814w -c 1.25, 0.4
##runit ngc300-wide1_f606w_f814w -c 1.6, 0.4
##runit ugc4305-2_f555w_f814w -c 1.4, 0.4
##
##
###runit ngc2976-deep_f606w_f814w -c 1.6,0.4
#
#
## get the prefixs:
## ls out*000.dat | cut -d '_' -f 2-4
## or
## ls caf09_v1.2s_m36_s12d_ns_nas/out*000.dat | cut -d '_' -f 7-9
