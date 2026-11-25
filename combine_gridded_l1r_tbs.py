"""Combine TB files into single grid

combine_gridded_l1r_tbs.py

See get_usage_string() for example usage
"""

import datetime as dt
import glob
import os
import warnings
from pathlib import Path
from pprint import pprint as pprint

import dateutil.parser
import numpy as np
import pyproj
import yaml
from pyresample.bilinear import NumpyBilinearResampler
from pyresample.geometry import AreaDefinition, SwathDefinition
from scipy import sparse

REF_DATETIME = dt.datetime(1993, 1, 1, 0, 0, 0)


'''Options for NumpyBilinearResampler:
def resample_bilinear(data, source_geo_def, target_area_def, radius=50e3,
                      neighbours=32, nprocs=1, fill_value=0,
                      reduce_data=True, segments=None, epsilon=0):
    """Resample using bilinear interpolation.

    data : numpy array
        Array of single channel data points or
        (source_geo_def.shape, k) array of k channels of datapoints
    source_geo_def : object
        Geometry definition of source data
    target_area_def : object
        Geometry definition of target area
    radius : float, optional
        Cut-off distance in meters
    neighbours : int, optional
        Number of neighbours to consider for each grid point when
        searching the closest corner points
    nprocs : int, optional
        Number of processor cores to be used for getting neighbour info
    fill_value : {int, None}, optional
        Set undetermined pixels to this value.
        If fill_value is None a masked array is returned with undetermined
        pixels masked
    reduce_data : bool, optional
        Perform initial coarse reduction of source dataset in order
        to reduce execution time
    segments : int or None
        Number of segments to use when resampling.
        If set to None an estimate will be calculated
    epsilon : float, optional
        Allowed uncertainty in meters. Increasing uncertainty
        reduces execution time
'''
# EXAMPLE_AM2MBT_FNAME = 'AMSR2-MBT_v2r2_GW1_s202501060005067_e202501060144046_c202501060156059.nc'
# EXAMPLE_CONFIG_FNAME = 'am2mbt_e2n10_config.yaml'

EXAMPLE_AMSR2L1R_FNAME = './l1r_20150315/GW1AM2_202503150022_177A_L1SGRTBR_2220220.h5'
EXAMPLE_AMSR2L1R_CONFIG_FNAME = 'amsr2l1r_e1n25_config.yaml'

DEFINED_AMSR2L1R_ERRORS = [
    'nofile',
]
DEFAULT_CONFIG_AMSR2L1R = {
    'var_fn_template': '{dirname}/{gridid}_{varname}_{tstamp}.dat',
    'gridid': 'e1n25',
    'output_dir': './output',
    'intermed_dir': './intermed',
    'overwrite': False,
    # swath_vars maps the to-be-gridded fields to their AMSR2-MBT file var names
    'swath_vars': {
        # tbs
        'v18': 'Brightness_Temperature_18_GHzV',
        'v23': 'Brightness_Temperature_23_GHzV',
        'h36': 'Brightness_Temperature_36_GHzH',
        'v36': 'Brightness_Temperature_36_GHzV',
        # other fields
        'landflag': 'Land_Ocean_Flag_6_to_36',
        # lats
        'latlow': 'Latitude_for_Low_Resolution',
        'lat18': 'Latitude_for_18',
        'lat23': 'Latitude_for_23',
        'lat36': 'Latitude_for_36',
        # lons
        'lonlow': 'Longitude_for_Low_Resolution',
        'lon18': 'Longitude_for_18',
        'lon23': 'Longitude_for_23',
        'lon36': 'Longitude_for_36',
    },
    # grid_vars specifies the data and lat/lon fields used to geolocate the data
    'grid_vars': {
        'v19': ['v18', 'lat18', 'lon18'],
        'v22': ['v23', 'lat23', 'lon23'],
        'h37': ['h36', 'lat36', 'lon36'],
        'v37': ['v36', 'lat36', 'lon36'],
    },
}

GRIDID_AREADEFS = {
    'latlon_5deg': {
        'area_id': 'latlon_5deg_grid',
        'description': 'Simple_latlon_grid_at_10_deg',
        'proj_id': 'sample_latlon',
        'projection': 'EPSG:4326',
        'width': 72,
        'height': 18,
        'area_extent': (-180, -90, 180, 90),
    },
    'e2n10': {
        'area_id': 'ease2_nh_arctic_10km',
        'description': 'Arctic Sea Ice subset of EASE2 Northern Hemisphere',
        'proj_id': 'ease2_nh_arctic_10km',
        'projection': 'EPSG:6931',
        'width': 1050,
        'height': 1050,
        'area_extent': (-5250000, -5250000, 5250000, 5250000),
    },
    'e2n20': {
        'area_id': 'ease2_nh_arctic_20km',
        'description': 'Arctic Sea Ice subset of EASE2 Northern Hemisphere',
        'proj_id': 'ease2_nh_arctic_20km',
        'projection': 'EPSG:6931',
        'width': 525,
        'height': 525,
        'area_extent': (-5250000, -5250000, 5250000, 5250000),
    },
    'e2s10': {
        'area_id': 'ease2_sh_antarctic_10km',
        'description': 'Antarctic Sea Ice subset of EASE2 Southern Hemisphere',
        'proj_id': 'ease2_sh_antarctic_10km',
        'projection': 'EPSG:6932',
        'width': 840,
        'height': 840,
        'area_extent': (-4200000, -4200000, 4200000, 4200000),
    },
    'e2s20': {
        'area_id': 'ease2_sh_antarctic_20km',
        'description': 'Antarctic Sea Ice subset of EASE2 Southern Hemisphere',
        'proj_id': 'ease2_sh_antarctic_20km',
        'projection': 'EPSG:6932',
        'width': 420,
        'height': 420,
        'area_extent': (-4200000, -4200000, 4200000, 4200000),
    },
    'e2n10_NP_4x4': {
        'area_id': 'E2N10_NP_test',
        'description': 'Test grid near North Pole',
        'proj_id': 'e2n20_np_test',
        'projection': 'EPSG:6931',
        'width': 4,
        'height': 4,
        'area_extent': (-20000, -20000, 20000, 20000),
    },
    'e2n10_NP_10x10': {
        'area_id': 'E2N10_NP_test',
        'description': 'Test grid near North Pole',
        'proj_id': 'e2n20_np_test',
        'projection': 'EPSG:6931',
        'width': 10,
        'height': 10,
        'area_extent': (-50000, -50000, 50000, 50000),
    },
    'e1n25': {
        'area_id': 'ease1_nh_25km',
        'description': 'EASE1 Northern Hemisphere',
        'proj_id': 'ease1_nh_25km',
        'projection': 'EPSG:3408',
        #'width': 721,
        #'height': 721,
        #'area_extent': (-9036842.76, -9036842.76, 9036842.76, 9036842.76),
        'width': 719,
        'height': 719,
        'area_extent': (-9011775.238, -9011775.238, 9011775.238, 9011775.238),
    },
    'e1s25': {
        'area_id': 'ease1_sh_25km',
        'description': 'EASE1 Southern Hemisphere',
        'proj_id': 'ease1_sh_25km',
        'projection': 'EPSG:3409',
        'width': 721,
        'height': 721,
        'area_extent': (-9036842.7625, -9036842.7625, 9036842.7625, 9036842.7625),
    },
}


def xwm(m='exiting in xwm()'):
    # Note: this is only for development and should be used in production
    raise SystemExit(m)  # pragma: no cover


def convert_tai93_time(time_arr):
    """Converts TAI93 time in seconds (eg JAXA data) to standard time
    in seconds since Jan 1, 1993.

    Note, this assumes data is at least Jul 1, 2012, where there are already
    8 leap seconds in TAI93 time"""

    # Times before July 1, 2015 which include 8 leap seconds
    tai93_offset = np.zeros(time_arr.shape, dtype=int)
    tai93_offset[:] = -8

    # Time between July 1, 2015 and Dec 31, 2016
    sub_09 = (time_arr > 709862408) & (time_arr <= 757382409)
    tai93_offset[sub_09] = -9

    sub_10 = (time_arr > 709862408) & (time_arr <= 757382409)
    tai93_offset[sub_10] = -10

    time_standard = time_arr + tai93_offset

    return time_standard


def get_pyresample_area_def(gridid):
    """Return the pyresample module's AreaDefinition for a specified grid"""
    try:
        assert gridid in GRIDID_AREADEFS.keys()
    except AssertionError:
        raise ValueError(f'No AreaDefinition information for: {gridid}')

    gridid_area_def = GRIDID_AREADEFS[gridid]
    area_def = AreaDefinition(
        gridid_area_def['area_id'],
        gridid_area_def['description'],
        gridid_area_def['proj_id'],
        gridid_area_def['projection'],
        gridid_area_def['width'],
        gridid_area_def['height'],
        gridid_area_def['area_extent'],
    )

    return area_def


def get_pyresample_swathdef(lons, lats):
    """Return the pyresample module's SwathDefinition for a set of lons, lats"""
    swathdef = SwathDefinition(lons=lons, lats=lats)

    return swathdef


def get_usage_string():
    bfn = os.path.basename(__file__)

    usage_str = f'''
    Usage:
        {bfn} <AMSR2_MBT_h5file> [<config_yaml_fname>]
    eg:
        {bfn} {EXAMPLE_AMSR2L1R_FNAME}
        {bfn} {EXAMPLE_AMSR2L1R_FNAME} {EXAMPLE_AMSR2L1R_CONFIG_FNAME}
    '''

    return usage_str


def get_error_message(grid_mbt_error):
    error_str = ''
    if grid_mbt_error == 'nofile':
        error_str += '\n'
        error_str += 'No input file provided'
        error_str += '\n'
        error_str += get_usage_string()

    return error_str


def parse_gw1am2_timestamp(ts):
    """Extract the timestamp from a std JAXA L1R file name part
    This part will be a string: YYYYmmddHHMM"""

    try:
        year = int(ts[0:4])
        month = int(ts[4:6])
        day = int(ts[6:8])
        hour = int(ts[8:10])
        minute = int(ts[10:12])
        timestamp_datetime = dt.datetime(year, month, day, hour, minute)
    except ValueError:
        raise ValueError(f'Could not parse GW1AM2 timestamp: {ts}')

    return timestamp_datetime


# def parse_swath_timestamp(ts, ts_type='start'):
#    """Extract the timestamp from a std AMSR2-MBT file name part
#    This part will be 1 letter followed by 15 digits"""
#    try:
#        assert ts_type in ('start', 'end', 'creation')
#    except AssertionError:
#        ts_type = 'unknown'
#
#    try:
#        if ts_type != 'unknown':
#            assert ts_type[0] == ts[0]
#    except AssertionError:
#        raise ValueError(f'timestamp type {ts_type} mismatch first char {ts[0]}')
#
#    try:
#        year=int(ts[1:5])
#        month=int(ts[5:7])
#        day=int(ts[7:9])
#        hour=int(ts[9:11])
#        minute=int(ts[11:13])
#        seconds=int(ts[13:15])
#        microseconds=int(ts[15:16]) * 100000
#        timestamp_datetime = \
#            dt.datetime(year, month, day, hour, minute, seconds, microseconds)
#    except ValueError:
#        raise ValueError(f'Could not parse {ts_type} timestamp: {ts}')
#
#    return timestamp_datetime


def parse_amsr2l1r_filename(fn):
    """Return dict with standard parts of AMSR2 MBT file name
    Note: A typical file is EXAMPLE_AMSR2L1R_FNAME:
    AMSR2-MBT_v2r2_GW1_s202501060005067_e202501060144046_c202501060156059.nc
    which has parts:
      AMSR2-MBT
      v2r2
      GW1
      s202501060005067
      e202501060144046
      c202501060156059
    """
    fn_path = Path(fn)
    fn_stem = fn_path.stem

    amsr2l1r_parts = {}
    amsr2l1r_parts['is_h5file'] = fn_path.suffix == '.h5'

    # For filname like:
    #   GW1AM2_202503150022_177A_L1SGRTBR_2220220.h5
    # fn_parts are:
    #  ['GW1AM2', '202503150022', '177A', 'L1SGRTBR', '2220220']
    fn_parts = fn_stem.split('_')

    amsr2l1r_parts['is_standard'] = True
    if len(fn_parts) != 5:
        amsr2l1r_parts['is_standard'] = False
    else:
        # Parse the filename as expected
        # prod_name, prod_ver, prod_plat, ts_start, ts_end, ts_gen = \
        amsr2l1r_parts['product_name'] = fn_parts[0]  # eg GW1AM2
        amsr2l1r_parts['product_version'] = fn_parts[4]  # eg 2220220
        try:
            amsr2l1r_parts['timestamp_start'] = parse_gw1am2_timestamp(fn_parts[1])
        except ValueError:
            print('Some timestamps not parsed')
            amsr2l1r_parts['is_standard'] = False

    if not amsr2l1r_parts['is_h5file']:
        print(f'WARNING: this is not an h5file:\n  {fn}')
        amsr2l1r_parts['is_standard'] = False

    return amsr2l1r_parts


def get_filename_timestamp(fn):
    fn_parts = parse_amsr2l1r_filename(fn)

    try:
        return fn_parts['timestamp_start']
    except KeyError:
        raise ValueError(f'No timestamp_start found in: {fn}')


def read_amsr2l1r_config(fn):
    from yaml.scanner import ScannerError

    try:
        with open(fn, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f'Could not file config file: {fn}')
    except ScannerError:
        raise ValueError(f'Bad yaml file: {fn}')

    if config is None:
        raise ValueError(f'config is None read from: {fn}')

    return config


# TODO: handle radius
def grid_swath_data_bilinear(
    vals, lons, lats, areadef, radius=20e3, nanval=0, label=None
):
    # Grid the data to an areadef

    swathdef = get_pyresample_swathdef(lons=lons, lats=lats)

    resampler = NumpyBilinearResampler(swathdef, areadef, radius)

    breakpoint()
    if label is not None:
        print(f'resampling swath data ({label})...', end='', flush=True)
    with warnings.catch_warnings(record=True) as warning_list:
        gridded = resampler.resample(vals).astype(np.float32)

        for warning in warning_list:
            message = warning.message

            # Ignore proj messages about using CRS codes
            if isinstance(
                message, UserWarning
            ) and 'converting to a PROJ string' in str(message):
                pass
            else:
                print(f'warning: {message}')

    if label is not None:
        print('done', flush=True)

    # Set NaN values
    if nanval is not None:
        gridded[gridded == nanval] = np.nan

    return gridded


def get_gridinfo(gridid):
    if gridid == 'e1n25':
        xdim = 721
        ydim = 721
        res = 25067.525
        x0 = -(xdim / 2) * res
        y0 = (ydim / 2) * res
        epsg = 'epsg:3408'
        latmin = 0
        latmax = 90
    elif gridid == 'e1s25':
        xdim = 721
        ydim = 721
        res = 25067.525
        x0 = -(xdim / 2) * res
        y0 = (ydim / 2) * res
        epsg = 'epsg:3409'
        latmin = -90
        latmax = 0
    elif gridid == 'psn25':
        xdim = 304
        ydim = 448
        res = 25000.
        # Note: NSIDC polar stereo grids are not centered on pole
        x0 = -3850000
        y0 = 5850000
        epsg = 'epsg:3411'
        latmin = 30.98
        latmax = 90
    elif gridid == 'pss25':
        xdim = 316
        ydim = 332
        res = 25000.
        # Note: NSIDC polar stereo grids are not centered on pole
        x0 = -3950000
        y0 = 4350000
        epsg = 'epsg:3413'
        latmin = -90.
        latmax = -39.23
    else:
        raise ValueError(f'Cannot get gridinfo for {gridid=}')

    return epsg, xdim, ydim, res, x0, y0, latmin, latmax


def grid_swath_data_ditb(vals, lons, lats, gridid, radius=20e3, nanval=0, label=None):
    # Unfortunately, I can't figure out how to get pyresample to work with
    #   drop in the buckey
    # Grid the data to an areadef

    # Modeled after code in grid_amsu_swaths.py

    epsg, xdim, ydim, res, x0, y0, latmin, latmax = get_gridinfo(gridid)

    ll_crs = pyproj.CRS('epsg:4326')
    grid_crs = pyproj.CRS(epsg)

    ll_to_grid = pyproj.Transformer.from_crs(ll_crs, grid_crs)
    # xvals, yvals = ll_to_grid.transform(lons, lats)
    xvals, yvals = ll_to_grid.transform(lats, lons)

    # ivals = (xvals - x0) // res
    # jvals = (y0 - yvals) // res

    is_valid = (
        np.isfinite(xvals) & np.isfinite(yvals) & (lats >= latmin) & (lats <= latmax)
    )

    ivals_float = np.zeros(is_valid.shape, dtype=np.float32)
    jvals_float = np.zeros(is_valid.shape, dtype=np.float32)

    np.divide((xvals - x0), res, out=ivals_float, where=is_valid)
    np.divide((y0 - yvals), res, out=jvals_float, where=is_valid)

    np.floor(ivals_float, out=ivals, where=is_valid)
    np.floor(jvals_float, out=jvals, where=is_valid)

    is_ongrid = (
        is_valid
        & (vals > 0)
        & ((ivals >= 0) & (ivals < xdim))
        & ((jvals >= 0) & (jvals < ydim))
    )

    grid_sum = sparse.coo_array(
        (vals[is_ongrid].flat, (jvals[is_ongrid].flat, ivals[is_ongrid].flat)),
        shape=(ydim, xdim),
    ).toarray()

    grid_num = (
        sparse.coo_array(
            (
                np.ones(vals.shape)[is_ongrid].flat,
                (jvals[is_ongrid].flat, ivals[is_ongrid].flat),
            ),
            shape=(ydim, xdim),
        )
        .toarray()
        .astype(int)
    )

    has_value = grid_num > 0

    grid_avg = np.divide(grid_sum, grid_num, where=has_value)

    grid_avg[~has_value] = np.nan
    grid_cnt = grid_num.copy()
    grid_cnt[~has_value] = 0

    return grid_avg.astype(np.float32), grid_cnt.astype(np.int16)


def get_runtime_info(indir, config, intermed_dir=None, output_dir=None, overwrite=None):
    """Determine the information needed to create the data files"""
    runtime_info = {}

    runtime_info['indir'] = indir

    # tb_fn_prefix: tb_gw1am2_grd
    runtime_info['tb_fn_prefix'] = config['tb_fn_prefix']

    # tb_fn_template: "{outputdir}/{prefix}_{gridid}_{tb}_{date}.dat"
    runtime_info['tb_fn_template'] = config['tb_fn_template']

    if overwrite is not None:
        runtime_info['overwrite'] = overwrite
    else:
        runtime_info['overwrite'] = config['overwrite']

    runtime_info['grid_vars'] = config['grid_vars']

    runtime_info['gridid'] = config['gridid']
    runtime_info['var_fn_template'] = config['var_fn_template']
    if intermed_dir is None:
        runtime_info['intermed_dir'] = config['intermed_dir']
    else:
        runtime_info['intermed_dir'] = intermed_dir

    if output_dir is None:
        runtime_info['output_dir'] = config['output_dir']
    else:
        runtime_info['output_dir'] = output_dir

    return runtime_info


def set_up_directories(runtime_info):
    # Ensure that directories specified in runtime_info exist
    os.makedirs(runtime_info['output_dir'], exist_ok=True)
    os.makedirs(runtime_info['intermed_dir'], exist_ok=True)


def get_intermed_var_filename(varname, runtime_info, calc):
    # Determine the intermediate filename for gridded swath variable
    fn_template = runtime_info['var_fn_template']
    intermed_dir = runtime_info['intermed_dir']
    timestamp_str = runtime_info['timestamp_str']
    gridid = runtime_info['gridid']

    #'var_fn_template': 'DIRNAME/VAR_TSTAMP.dat',
    # fn_template = '{dirname}/{varname}_{tstamp}.dat'
    intermed_fn = fn_template.format(
        dirname=intermed_dir,
        varname=varname,
        tstamp=timestamp_str,
        gridid=gridid,
        calc=calc,
    )

    return intermed_fn


def get_timecode(fn):
    """Assumes filename of form:
    {indir}/e1n25_h19_av_202503151740.dat
    """
    bfn = os.path.basename(fn)
    parts = bfn.split('_')
    last_part = parts[-1]
    timecode_str = os.path.splitext(last_part)[0]

    return timecode_str


def combine_tb_files(
    indir,
    config,
    date,
    intermed_dir,
    output_dir,
    all_av_files=None,
    all_ct_files=None,
):  # pragma: no cover
    """Combine series of avg and count (av and ct) raw binary files"""

    print(f'Combine files in directory: {indir}')
    runtime_info = get_runtime_info(
        indir,
        config,
        intermed_dir=intermed_dir,
        output_dir=output_dir,
    )
    # overwrite = runtime_info['overwrite']

    # epsg, xdim, ydim, res, x0, y0, latmin, latmax = get_gridinfo(gridid)
    _, xdim, ydim, _, _, _, _, _ = get_gridinfo(runtime_info['gridid'])

    tbs_to_grid = list(runtime_info['grid_vars'].keys())

    for tb in tbs_to_grid:
        # each tb will have files like:
        #   {indir}/e1n25_h19_av_202503151740.dat
        #   {indir}/e1n25_h19_ct_202503151740.dat

        # TODO: These should not be created independently, so I'm SURE they'll
        #       all have corresponding timecodes....
        if all_av_files is None:
            av_glob_pattern = f'{indir}/{runtime_info["gridid"]}_{tb}_av_*.dat'
            av_files = sorted(glob.glob(av_glob_pattern))
        else:
            av_files = [f for f in all_av_files if f'{indir}/{runtime_info["gridid"]}_{tb}_av_' in f]
            
        if all_ct_files is None:
            ct_glob_pattern = f'{indir}/{runtime_info["gridid"]}_{tb}_ct_*.dat'
            ct_files = sorted(glob.glob(ct_glob_pattern))
        else:
            ct_files = [f for f in all_ct_files if f'{indir}/{runtime_info["gridid"]}_{tb}_ct_' in f]

        print(f'Working on {len(av_files)} av and {len(ct_files)} ct files')
        tb_sum = np.zeros((ydim, xdim), dtype=np.float32)
        tb_num = np.zeros((ydim, xdim), dtype=np.int16)

        for av_fn, ct_fn in zip(av_files, ct_files):

            av_timecode = get_timecode(av_fn)
            ct_timecode = get_timecode(ct_fn)

            try:
                assert av_timecode == ct_timecode
            except AssertionError as e:
                print('Difference in timecodes:')
                print(f'  {av_timecode=}')
                print(f'  {ct_timecode=}')
                raise e

            av = np.fromfile(av_fn, dtype=np.float32).reshape(ydim, xdim)
            ct = np.fromfile(ct_fn, dtype=np.int16).reshape(ydim, xdim)

            is_valid = ct > 0

            # Add weighted average
            tb_sum[is_valid] += av[is_valid] * ct[is_valid]
            tb_num[is_valid] += ct[is_valid]

            # print(f'added: {av_fn}')
            # print(f'added: {ct_fn}')

        is_valid_tb = tb_num > 0
        tb_field = np.zeros((ydim, xdim), dtype=np.float32)

        combined_average = np.zeros(tb_sum.shape, dtype=np.float32)
        np.divide(
            tb_sum,
            tb_num,
            out=combined_average,
            where=is_valid_tb,
            dtype=np.float32,
        )

        tb_field = np.zeros((ydim, xdim), dtype=np.float32)
        tb_field[is_valid_tb] = combined_average[is_valid_tb]

        output_fn = runtime_info['tb_fn_template'].format(
            outputdir=runtime_info['output_dir'],
            tb_fn_prefix=runtime_info['tb_fn_prefix'],
            gridid=runtime_info['gridid'],
            tb=tb,
            date=date.strftime('%Y%m%d'),
        )

        tb_field.tofile(output_fn)
        print(f'  Wrote: {output_fn}')


def main():
    """
    checks for existence of AMSR2-MBT swath file
    if config file is specified,
      verifies existence of config file
      verifies valididy of config file
    Then calls the processing routine
    """

    import sys

    # Check for a second argument with name of YAML with config info
    try:
        cfn = sys.argv[2]
        assert os.path.isfile(cfn)
        config = read_amsr2l1r_config(cfn)
    except IndexError:
        # No config file specified
        # config = DEFAULT_CONFIG_AMSR2L1R
        raise RuntimeError('No config file specified')
    except AssertionError:
        raise ValueError(f'No such specified config file: {cfn}')
    except ValueError:
        raise ValueError(f'Could not parse specified config file: {cfn}')

    # Get name of directory to process from first cmdline arg
    #   and do some minimal validation of the file
    try:
        indir = sys.argv[1]
        assert os.path.isdir(indir)
    except IndexError:
        raise ValueError(f'{get_error_message("nodir")}')
    except AssertionError:
        raise FileNotFoundError(f'Input directory does not exist: {indir}')

    # Get UTC date
    try:
        date_arg = sys.argv[3]
        date = dateutil.parser.parse(date_arg)
    except ValueError:
        raise RuntimeError(f'Could not parse date: {date_arg}')

    # Optional: get intermediate directory
    try:
        intermed_dir = sys.argv[4]
    except ValueError:
        intermed_dir = None

    # Optional: get output directory
    try:
        output_dir = sys.argv[5]
    except ValueError:
        output_dir = None

    # NB: These are not tested by unit tests because tested separately
    combine_tb_files(indir, config, date, intermed_dir, output_dir)  # pragma: no cover


if __name__ == '__main__':
    main()  # pragma: no cover
