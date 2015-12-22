#AGB=$1
#CMD="python -m tpagb-calibration.sfhs.vary_sfh -vfc $AGB"
CMD="python -m tpagb-calibration.sfhs.vary_sfh -vs 25 -n 11"
ext=".mcmc.zc"
for p in $(ls *$ext)
do
    echo $p
    $CMD ${p/$ext}.sfh $p ${p/$ext}.matchfake
    # without hybridMC files be sure nsfhs = 1 (or default value no -s flag)
    #$CMD $p $p ${p/.sfh}.matchfake
done
