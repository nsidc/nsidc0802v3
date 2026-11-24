"""Compute the RMS difference between two TB fields, including any regression values
Creates new files with suffix <blah>_lr.dat instead of <blah>.dat
  to indicate that the linear regression has occurred.

Usage:
    python apply_linreg.py <filename>
  eg:
    python apply_linreg.py /share/apps/nsidc0802/nsidc0802/dev/scotts/tb_gw1am2_grd/tb_gw1am2_psn25_h19r06_20240315.dat

Note: Expectation is that an un-gaussian-smoothed file will have dir/name like:
    tb_gw1am2_grd/psn25/2024/01/tb_gw1am2_grd_psn25_v37r06_20240101.dat
  and a gaussian-smoothed file will have a dir/name like:
    tb_gw1am2_sfm/psn25/2024/01/tb_gw1am2_sfm_psn25_h19r06_20240101.dat

templates:
  basedir (bdir) is: /home/scotts/Desktop/data
  coeftype is:
    grd:  gridded
    sfm:  spatial-frequency-matched
  fn_ = 
    grd_ = {bdir}/tb_gw1am2_{coeftype}/{gridid}/{year}/{month}/tb_gw1am2_{coeftype}_{gridid}_{chanres}_{ymd}.dat

  Corresponding "truth" file (F17 0001) is:
    srcdir = /home/scotts/Desktop/data/nsidc0001/
    ngrid:
      psn25: N25km
      pss25: S25km
    ver: v6.0
    ncfn_ = {bdir}/nsidc0001/{gridid}/{year}/NSIDC0001_TB_PS_{ngrid}_{ymd}_{ver}.nc
"""

import dateutil.parser
import sys
import numpy as np
from netCDF4 import Dataset
import os
import datetime as dt
import dateutil.parser
import re
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from apply_gauss_smooth import read_tb


chan_list = ('h19', 'v19', 'v22', 'h37', 'v37')
chan_varnames = {
    'h19': 'TB_F17_19H',
    'v19': 'TB_F17_19V',
    'v22': 'TB_F17_22V',
    'h37': 'TB_F17_37H',
    'v37': 'TB_F17_37V',
}
GROUP_NAME = 'F17'

NSIDC_GRIDNAME = {
    'psn25': 'N25km',
    'pss25': 'S25km',
}


#def get_ncfn_0001(gridid, date, ver='v6.0', bdir='/home/scotts/Desktop/data/'):
def get_ncfn_0001(gridid, date, ver='v6.0', bdir='/share/apps/nsidc0802/nsidc0802/dev/scotts'):
    """Return the name of the nsidc0001 netCDF file name"""
    year = date.strftime('%Y')
    ymd = date.strftime('%Y%m%d')
    ngrid = NSIDC_GRIDNAME[gridid]
    ncfn_ = '{bdir}/nsidc0001/{gridid}/{year}/NSIDC0001_TB_PS_{ngrid}_{ymd}_{ver}.nc'
    ncfn = ncfn_.format(
        bdir=bdir,
        gridid=gridid,
        year=year,
        ngrid=ngrid,
        ymd=ymd,
        ver=ver,
    )

    return ncfn


#def get_bin_fn(coeftype, gridid, date, chanres, bdir='/home/scotts/Desktop/data/'):
def get_bin_fn(coeftype, gridid, date, chanres, bdir='/share/apps/nsidc0802/nsidc0802/dev/scotts'):
    """Return the appropriate binary file name"""
    bin_fn_ = '{bdir}/tb_gw1am2_{coeftype}/{gridid}/{year}/{month}/tb_gw1am2_{coeftype}_{gridid}_{chanres}_{ymd}.dat'
    bin_fn = bin_fn_.format(
        bdir=bdir,
        coeftype=coeftype,
        gridid=gridid,
        year=date.strftime('%Y'),
        month=date.strftime('%m'),
        chanres=chanres,
        ymd=date.strftime('%Y%m%d'),
    )

    return bin_fn


def get_dims_from_fn(fn):
    """Get the dimensions from the file name"""
    bfn = os.path.basename(fn)
    if 'psn25' in bfn:
        xdim = 304
        ydim = 448
        dtype = np.float32
    elif 'pss25' in bfn:
        xdim = 316
        ydim = 332
        dtype = np.float32
    else:
        raise RuntimeError(f'Could not get dims from fn: {fn}')

    return xdim, ydim, dtype


def read_tb_field(fn, chan):
    """Read the TB field from this file"""
    tb = None
    bfn = os.path.basename(fn)
    if bfn[-3:] == '.nc':
        ds = Dataset(fn)
        groupname = GROUP_NAME
        varname = chan_varnames[chan]
        tb = np.squeeze(np.asarray(ds.groups[groupname].variables[varname]))
        is_missing = np.isnan(tb)
        n_missing = np.sum(np.where(is_missing, 1, 0))
        if n_missing > 0:
            print(f'Replacing {n_missing} missing value with zeros...')
            tb[is_missing] = 0
    elif bfn[-4:] == '.dat':
        xdim, ydim, dtype = get_dims_from_fn(fn)
        tb = np.fromfile(fn, dtype=dtype).reshape(ydim, xdim)

    if tb is not None:
        return tb
    else:
        raise RuntimeError(f'Could not get TB field from: {fn}')


def get_date_from_fn(fn):
    """Return the date from this file name"""
    bfn = os.path.basename(fn)

    re_8digits = re.compile(r'\d{8,8}')
    date_str = re_8digits.findall(fn)[0]

    date = dateutil.parser.parse(date_str).date()
    return date


def plot_xvals_and_yvals(xvals, yvals, m, b, r2):
    """Plot these two distributions..."""
    # plot the values
    # Set up the plot
    # Extract the crossing values
    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot the raw points (black)
    scatter_legend = f'TB: {comp_chan}'
    ax.scatter(xvals, yvals, s=0.02, color='black', label=scatter_legend)

    # Plot the comparison line
    xfit = np.asarray([xvals.min(), xvals.max()])
    #yfit = xfit * pvals[0] + pvals[1]
    yfit = xfit * m + b
    # fit_legend = f'm={pvals[0]:.3f} b={pvals[1]:.3f}'
    fit_legend = f'm={m:.4f} b={b:.4f}  R^2={r2:.4f}'
    ax.plot(xfit, yfit, color='red', label=fit_legend)

    # Plot 1-to-1 line
    x1to1 = np.asarray([xvals.min(), xvals.max()])
    y1to1 = x1to1
    xy1to1_legend = 'Slope=1, Intercept=0'
    ax.plot(x1to1, y1to1, color='blue', label=xy1to1_legend)

    # Add info
    ax.set_xlabel(f'{os.path.basename(xfn)}')
    ax.set_ylabel(f'{os.path.basename(yfn)}')
    ax_legend = ax.legend()
    ax_legend.legend_handles[0]._sizes = [10]
    plt.show()


def load_nearsie_mask(gridid, date):
    """Read the near-sea-ice-extent mask for this grid anddate"""
    year = date.strftime('%Y')
    ymd = date.strftime('%Y%m%d')
    mask_dir = f'/home/scotts/Desktop/data/nsidc0079_siem/{gridid}/{year}'
    bfn = f'nsidc0079_siemask_{gridid}_{ymd}.dat'
    mask_fn = f'{mask_dir}/{bfn}'
    nearsie_mask = np.fromfile(mask_fn, dtype=np.uint8).reshape(ydim, xdim)
    is_nearsie = nearsie_mask > 0

    return is_nearsie


def compute_rms_tbfields(xfn, mx, bx, yfn, my, by):
    """Compute the RMS of these fields"""
    date = get_date_from_fn(xfn)
    ymd = date.strftime('%Y%m%d')
    year = date.strftime('%Y')

    if 'psn25' in os.path.basename(xfn):
        gridid = 'psn25'
        xdim = 304
        ydim = 448
    elif 'pss25' in os.path.basename(xfn):
        gridid = 'pss25'
        xdim = 316
        ydim = 332
    else:
        raise RuntimeError(f'could not determine gridid from xfn: {xfn}')

    tbx = read_tb_field(xfn, chan)
    if (mx != 1.0) and (bx != 0):
        tbx = mx * tbx + bx

    tby = read_tb_field(yfn, chan)
    if (my != 1.0) and (by != 0):
        tbx = my * tby + by


    # Optional: Write scaled field?

    # Read masking field, eg
    #  /home/scotts/Desktop/data/nsidc0079_siem/2025/nsidc0079_siemask_psn25_20251231.dat
    mask_dir = f'/home/scotts/Desktop/data/nsidc0079_siem/{gridid}/{year}'
    bfn = f'nsidc0079_siemask_{gridid}_{ymd}.dat'
    mask_fn = f'{mask_dir}/{bfn}'
    nearsie_mask = np.fromfile(mask_fn, dtype=np.uint8).reshape(ydim, xdim)
    is_nearsie = nearsie_mask > 0

    # Determine where data exists in both fields
    is_both_have_tb = (tbx != 0) & (tby != 0)

    # Combine masks
    is_comparison_tb = is_both_have_tb & is_nearsie

    xvals = tbx[is_comparison_tb].flatten()
    yvals = tby[is_comparison_tb].flatten()

    rms = np.sqrt(np.mean(np.square(yvals - xvals)))
    print('RMS: {rms}')
    print('next up: plot...')
    breakpoint()

    if doplot:
        plot_xvals_and_yvals(xvals, yvals, m, b, r2)
        # # plot the values
        # # Set up the plot
        # # Extract the crossing values
        # fig, ax = plt.subplots(figsize=(16, 12))

        # # Plot the raw points (black)
        # scatter_legend = f'TB: {comp_chan}'
        # ax.scatter(xvals, yvals, s=0.02, color='black', label=scatter_legend)

        # # Plot the comparison line
        # xfit = np.asarray([xvals.min(), xvals.max()])
        # #yfit = xfit * pvals[0] + pvals[1]
        # yfit = xfit * m + b
        # # fit_legend = f'm={pvals[0]:.3f} b={pvals[1]:.3f}'
        # fit_legend = f'm={m:.4f} b={b:.4f}  R^2={r2:.4f}'
        # ax.plot(xfit, yfit, color='red', label=fit_legend)

        # # Plot 1-to-1 line
        # x1to1 = np.asarray([xvals.min(), xvals.max()])
        # y1to1 = x1to1
        # xy1to1_legend = 'Slope=1, Intercept=0'
        # ax.plot(x1to1, y1to1, color='blue', label=xy1to1_legend)

        # # Add info
        # ax.set_xlabel(f'{os.path.basename(xfn)}')
        # ax.set_ylabel(f'{os.path.basename(yfn)}')
        # ax_legend = ax.legend()
        # ax_legend.legend_handles[0]._sizes = [10]
        # plt.show()

    return m, b, r2


def get_file_info(fn, coeftype):
    """Get the relevant info for file"""
    bfn = os.path.basename(fn)
    bfn_noext = os.path.splitext(bfn)[0]
    #print(f'parse bfn:\n{bfn}')

    if coeftype == 'no_grd':
        # Note: I don't think no_grd exists anymore...replaced w grd
        _, _, gridid, chanres, datestr = bfn_noext.split('_')
    elif coeftype == 'grd':
        # Typical:
        #  dir: /home/scotts/Desktop/data//tb_gw1am2_grd/psn25/2024/01/
        #   fn: tb_gw1am2_grd_psn25_h19r06_20240101.dat
        _, _, fn_coeftype, gridid, chanres, datestr = bfn_noext.split('_')
    elif coeftype == 'sfm':
        _, _, fn_coeftype, gridid, chanres, datestr = bfn_noext.split('_')
    elif coeftype == 'grdlr':
        # Typical:
        #  dir: /home/scotts/Desktop/data/tb_gw1am2_grdlr/psn25/2024/01/
        #   fn: tb_gw1am2_grdlr_psn25_h19r06_20240101.dat
        _, _, fn_coeftype, gridid, chanres, datestr = bfn_noext.split('_')
    elif coeftype == 'sfmlr':
        _, _, fn_coeftype, gridid, chanres, datestr = bfn_noext.split('_')
    else:
        raise RuntimeError(f'unknown coeftype: {coeftype}')

    try:
        assert fn_coeftype == coeftype
    except AssertionError:
        print('**************')
        print('**************')
        print(f'WARNING: {fn_coeftype=} != {coeftype}')
        print('**************')
        print('**************')
    except NameError:
        pass

    date = dt.datetime.strptime(datestr, '%Y%m%d')

    file_info = {
        'gridid': gridid,
        'date': date,
        'chanres': chanres,
    }
    return file_info


def get_xdim_ydim(gridid):
    if gridid == 'psn25':
        xdim = 304
        ydim = 448
    elif gridid == 'pss25':
        xdim = 316
        ydim = 332
    else:
        raise RuntimeError(f'cant get xydim for: {gridid=}')

    return xdim, ydim


def get_fn_coefs_daily(gridid, date, coeftype):
    """Return the file name to use for these parameters"""
    year = date.year
    ymd = date.strftime('%Y%m%d')
    if coeftype == 'grd':
        fn_coefs = f'comps_{gridid}_{year}_grd_clean.csv'
    elif coeftype == 'no_grd':
        # Note: no_grd is the grd without the _grd in the file name
        fn_coefs = f'comps_{gridid}_{year}_grd_clean.csv'
    elif coeftype == 'sfm':
        # eg:        comps_psn25_2024_sfm_clean
        fn_coefs = f'comps_{gridid}_{year}_sfm_clean.csv'
    else:
        print(f' could not determine fn_coefs')
        print(f'  {gridid=}')
        print(f'  {date=}')
        print(f'  {chanres=}')
        print(f'  {coeftype=}')
        breakpoint()

    return fn_coefs


def get_fn_coefs_annual(gridid, date, coeftype):
    """Return the file name to use for these parameters"""
    year = date.year
    fn_coefs_ = 'sfm_coefs_{gridid}_{year}.csv'
    fn_coefs = fn_coefs_.format(
        gridid=gridid,
        year=year,
    )

    return fn_coefs


def get_coefs_daily(gridid, date, chanres, coeftype, coef_calc):
    """Read in the coefficients for this
    Format is something like:
        sfm_coefs_psn25_2024.csv:
            gridid,year,coeftype,chanres,all_mean,all_median,cold_mean,cold_median
            gridid,year,coeftype,chanres,m_all_mean,m_all_median,m_cold_mean,m_cold_median,...
              b_all_mean,b_all_median,b_cold_mean,b_cold_median
            psn25,2024,grd,h19r06,0.9504398837142966,0.9531989097595216,
               0.956029003833834, 0.9563093185424804,9.016848096660539,
               8.324813842773438,7.669409850820961,7.546600341796875
    coeftype is: grd or sfm
    coef_calc is:  all_mean, all_median, cold_mean, cold_median
    """
    fn_coefs = get_fn_coefs_daily(gridid, date, coeftype)

    df = pd.read_csv(fn_coefs)
    df_loc = df.loc[(df['gridid']==gridid) & (df['chanres']==chanres) & (df['date']==int(ymd))]
    m = float(df_loc[coef_calc].iloc[0])
    b = float(df_loc[coef_calc].iloc[0])

    return m, b


def get_coefs_annual(gridid, date, chanres, coeftype, coef_calc):
    """Read in the coefficients for this
    Note: coef_calc is the suffix for m or b
      eg all_mean =>
          m:  m_all_mean
          b:  b_all_mean
          """
    fn_coefs = get_fn_coefs_annual(gridid, date, coeftype)

    df = pd.read_csv(fn_coefs)
    df_loc = df.loc[
        (df['gridid']==gridid) &
        (df['chanres']==chanres) &
        (df['coeftype'] == coeftype)
    ]
    m_column = f'm_{coef_calc}'
    b_column = f'b_{coef_calc}'
    m = float(df_loc[m_column].iloc[0])
    b = float(df_loc[b_column].iloc[0])
    # print('have m,b')
    # print(f'  m: {m}')
    # print(f'  b: {b}')

    return m, b


def apply_linreg(field, m, b):
    """Apply the linear regression to this field"""
    is_missing = field ==0
    field_lr = m * field + b
    field_lr[is_missing] = 0

    return field_lr


# def apply_linreg(fn):
#     """Compute the linear regression for this file"""
#     if '_sfm' in fn:
#         coeftype = 'sfm'
#     elif '_grd' in fn:
#         coeftype = 'grd'
#     else:
#         coeftype = 'no_grd'
#         #print('assuming coeftype: {coeftype}')
# 
#     gridid, date, chanres = get_file_info(fn, coeftype)
#     xdim, ydim = get_xdim_ydim(gridid)
# 
#     m, b = get_coefs(gridid, date, chanres, coeftype)
# 
#     tb_orig = read_tb_field(fn, chanres[:3])
# 
#     is_missing = tb_orig == 0
#     tb_linreg = m * tb_orig + b
#     tb_linreg[is_missing] = 0
# 
#     ofn = get_linreg_output_fn(fn)
#     #is_nearsie = nearsie_mask > 0
#     #is_valid_tb = tb_orig != 0
#     #is_comparison_tb = is_valid_tb & is_nearsie
# 
#     print('have m and b and tb_orig')
#     breakpoint()


def get_fn_info():
    """Parse the command line for arguments"""
    # ncfn_ = '{bdir}/nsidc0001/{gridid}/{year}/NSIDC0001_TB_PS_{ngrid}_{ymd}_{ver}.nc'
    # bin_fn_ = '{bdir}/tb_gw1am2_{coeftype}/{gridid}/{year}/{month}/tb_gw1am2_{coeftype}_{gridid}_{chanres}_{ymd}.dat'
    gridid = sys.argv[1]
    coeftype = sys.argv[2]
    date = dateutil.parser.parse(sys.argv[3]).date()
    chan = sys.argv[4]
    coef_calc = sys.argv[5]

    return locals()


def get_chanres(chanstr):
    """Return the appropriate channel-resolution string
    Currently, this is <chan>r06 because all channels
    seem to have optimal L1R resolutions of res06
    """
    if len(chanstr) == 6:
        chanres = chanstr
    else:
        chanres = chanstr + 'r06'

    return chanres


def initial_file_name(fn_info):
    """call initial file name creator"""
    chanres = get_chanres(fn_info['chan'])

    ifn = get_bin_fn(
        coeftype=fn_info['coeftype'],
        gridid=fn_info['gridid'],
        date=fn_info['date'],
        chanres=chanres,
    )

    if not os.path.isfile(ifn):
        print(f'about to error because no ifn:\n{ifn}')
        raise RuntimeError(f'No such input file name: {ifn}')

    return ifn


def output_file_name(fn_info):
    """Get the linreg version of this file name"""
    chanres = get_chanres(fn_info['chan'])
    if fn_info['coeftype'] == 'sfm':
        coeftype = 'cal'
    elif fn_info['coeftype'] == 'grd':
        coeftype = 'glr'
    else:
        raise RuntimeError(f'cant compute output coeftype from: {fn_info["coeftype"]=}')
    #coeftype = fn_info['coeftype'] + 'lr'

    ofn = get_bin_fn(
        coeftype=coeftype,
        gridid=fn_info['gridid'],
        date=fn_info['date'],
        chanres=chanres,
    )

    outdir = os.path.dirname(ofn)
    if not os.path.isfile(outdir):
        os.makedirs(outdir, exist_ok=True)

    return ofn


def create_linreg_file(fn_info, ifn, ofn, verbose=True):
    """Use this file info to apply linear regression to ifn and write to ofn"""
    m, b = get_coefs_annual(
        fn_info['gridid'],
        fn_info['date'],
        file_info['chanres'],
        fn_info['coeftype'],
        fn_info['coef_calc']
    )

    chan = file_info['chanres'][:3]
    tb_orig = read_tb_field(ifn, chan)
    tb_linreg = apply_linreg(tb_orig, m, b)
    tb_linreg.tofile(ofn)
    if verbose:
        print(f'Wrote: {ofn}')


if __name__ == '__main__':
    fn_info = get_fn_info()
    ifn = initial_file_name(fn_info)
    ofn = output_file_name(fn_info)
    file_info = get_file_info(ifn, fn_info['coeftype'])
    create_linreg_file(fn_info, ifn, ofn)
    # try:
    #     create_linreg_file(fn_info, ifn, ofn)
    # except Exception as exc:
    #     # print(f'Exception trying to create:\n{exc}')
    #     print(f'Could not create: {ofn}')
