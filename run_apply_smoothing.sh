#!/bin/bash
#
# run_apply_smoothing.sh
#
# Usage:
#   ./run_apply_smoothing.sh <start_YYYYMMDD> [<end_YYYYMMDD>]
#
# Formerly hardcoded start and end dates,
#   encode all of the TB files for both hemispheres for those dates

#d0=20230101
#d1=20231231

#d0=20240101
#d1=20241231

grddir=/share/apps/nsidc0802/nsidc0802/dev/scotts/tb_gw1am2_grd

d0="$1"
if [ -z $d0 ]; then
  echo "no first YYYYMMDD given"
  exit
fi

d1="$2"
if [ -z $d1 ]; then
  echo "no second YYYYMMDD given"
  d1="$d0"
  echo "  using: $d1"
fi

ymd=$d0
while [[ "$ymd" -le "$d1" ]]; do
  echo
  echo "Current date: ${ymd}  [$(date)]"
  year=${ymd:0:4}
  month=${ymd:4:2}
  for gridid in psn25 pss25; do
    for fn in ${grddir}/${gridid}/${year}/${month}/*_${ymd}*.dat; do
      python apply_gauss_smooth.py $fn
    done
  done

  # Increment date
  ymd=$(date +'%Y%m%d' -d "$ymd + 1 day")
done
