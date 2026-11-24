#!/bin/bash

# run_apply_linreg_daily.sh

#d0=20240101
#d1=20241231

coef_calc=cold_median

d0="$1"
if [ -z $d0 ]; then
  echo "No first date YYYYMMDD given"
  exit
fi

d1="$2"
if [ -z $d1 ]; then
  d1=$d0
fi

d=$d0
while [[ "$d" -le "$d1" ]]; do

  # # This was a one time...
  # gridid=psn25
  # #coeftype=grd
  # coeftype=sfm
  # chan=h19
  # python apply_linreg.py ${gridid} ${coeftype} ${d} ${chan}

  for gridid in psn25 pss25; do
    #for coeftype in grd sfm; do
    coeftype=sfm
      for chan in h19 v19 v22 h37 v37; do
        #python apply_linreg.py ${gridid} ${coeftype} ${d} ${chan}
        python apply_linreg.py ${gridid} ${coeftype} ${d} ${chan} ${coef_calc}
      done
    #done
  done

  d=$(date +'%Y%m%d' -d "$d + 1 day")
done
