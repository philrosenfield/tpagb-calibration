for l in $(ls out*dat) 
do
python -m tpagb-calibration.analysis.add_asts -v $l
done
