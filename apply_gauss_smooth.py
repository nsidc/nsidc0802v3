"""Apply the Gaussian smoothing to a field

Usage:
    python apply_gauss_smooth.py <filename>
  eg
    python apply_gauss_smooth.py /share/apps/nsidc0802/nsidc0802/dev/scotts/grd/psn25/2024/03/tb_gw1am2_psn25_v22r06_20240315.dat

Usage:  usually uses the default configuration file, which is initially:
    {'psn25': 
        {'h19': '7, 0.68',
         'h37': '7, 0.22',
         'v19': '7, 0.68',
         'v22': '7, 0.60',
         'v37': '7, 0.22'},
    'pss25':
        {'h19': '7, 0.68',
         'h37': '7, 0.22',
         'v19': '7, 0.68',
         'v22': '7, 0.60',
         'v37': '7, 0.22'}}

eg:
     python apply_gauss_smooth.py ~/Desktop/data/tb_gw1am2/psn25/2023/01/tb_gw1am2_psn25_h19r06_20230101.dat
"""

import numpy as np
import sys
import os
import yaml

from gauss_smooth import compute_gaussian_kernel, apply_gaussian_kernel


DEFAULT_CONFIG_FILE = 'gauss_smooth_coefs.yml'


def parse_tb_fname(fn):
    """Typical file name:
      # OLD: tb_gw1am2_psn25_h19r06_20240701_sfm.dat
      # NOW DIR: /share/apps/nsidc0802/nsidc0802/dev/scotts/tb_gw1am2_grd/psn25/2024/03/
      # NOW FN:  tb_gw1am2_grd_psn25_h19r06_20240315.dat
                  _    _   src gridid chanres ymdend
    """
    bfn = os.path.basename(fn)
    try:
        _, _, src, gridid, chanres, ymdend = bfn.split('_')
    except ValueError:
        raise RuntimeError(f'Could not parse_tb_fname(): {fn}')

    file_info = {
        'src': src,
        'gridid': gridid,
        'chan': chanres[:3],
        'res': chanres[3:],
        'ymd': ymdend[:8],
    }

    return file_info



def get_gauss_params(gridid, chan, config):
    """Extract gaussian smoothing params from config dict"""
    size_sigma_str = config[gridid][chan]
    size_str, sigma_str = size_sigma_str.split(',')
    size = int(size_str)
    sigma = float(sigma_str)

    return size, sigma


def read_tb(fn):
    """Read this TB file name"""
    if 'psn25' in os.path.basename(fn):
        xdim = 304
        ydim = 448
    elif 'pss25' in os.path.basename(fn):
        xdim = 316
        ydim = 332
    else:
        raise RuntimeError(f'Cannot determine xdim ydim of: {fn}')

    tb = np.fromfile(fn, dtype=np.float32).reshape(ydim, xdim)

    return tb


def create_sfm_field(ifn, ofn, size, sigma):
    """Implement resolution matching for this file"""
    kernel = compute_gaussian_kernel(size, sigma)
    tb = read_tb(ifn)
    sfm = apply_gaussian_kernel(tb, kernel)
    sfm.tofile(ofn)
    print(f'Wrote: {ofn}')


def read_config_file(cfn):
    """Read this configuration file"""
    with open(cfn) as f:
        config = yaml.safe_load(f)

    return config


def apply_gaussian_smoothing(config, ifn, ofn, tb_src='tb_gw1am2'):
    """Determine an output file name, location, and generate smoothed TB fields
    Note: typical input file names are something like:
        dir: /home/scotts/Desktop/data/tb_gw1am2_grd/{gridid}/{year}/{month}/
        bfn: tb_gw1am2_psn25_h19r06_20230101.dat
                      gridid chan   YYYY  DD
                                res     MM
    Output should mirror this, but for a different base dirname, eg:
        tb_gw1am2_sfm/  instead of  tb_gw1am2/
        in:  /home/scotts/Desktop/data/tb_gw1am2_grd/{gridid}/{year}/{month}/
        out: /home/scotts/Desktop/data/tb_gw1am2_sfm/{gridid}/{year}/{month}/
    """
    # # Calculate the output file name
    # indir = os.path.dirname(ifn)
    # outdir = indir.replace(tb_src, f'{tb_src}_sfm')

    # assert indir != outdir

    # os.makedirs(outdir, exist_ok=True)

    # Determine the file info
    finfo = parse_tb_fname(ifn)
    size, sigma = get_gauss_params(
        finfo['gridid'],
        finfo['chan'],
        config,
    )

    # Use file info and config to get filename suffix
    # Note: the first part is the shape of the kernel:
    #  b: box
    #  r: radius of a circle
    # Second part is the gaussian smoothing value "sigma"
    # So, a kernel that is a box of height/width 7 and a sigma of 0.68
    #   has a suffix of:
    #    'b7s068'
    #  suffix = f'b{size}s{int(sigma*100):03d}'

    #  bfn = os.path.basename(fn)
    #  ymd = finfo['ymd']
    #  obfn = bfn.replace(ymd, f'{suffix}_{ymd}')
    #  assert bfn != obfn

    #  ofn = f'{outdir}/{obfn}'
    assert ifn != ofn

    create_sfm_field(ifn, ofn, size, sigma)


def get_corresponding_filename(ifn, fieldtype_in, fieldtype_out, make_dir=True):
    """Get a subdir/filename that is analogous to the input file name"""
    in_str = f'_{fieldtype_in}'
    out_str = f'_{fieldtype_out}'
    ofn = ifn.replace(in_str, out_str)
    try:
        assert ifn != ofn
    except AssertionError as err:
        print(f'Tried to get corresponding filename: {fieldtype_in} to {fieldtype_out}')
        print(f'  {ifn}')
        raise err

    if make_dir:
        os.makedirs(os.path.dirname(ofn), exist_ok=True)
    return ofn


if __name__ == '__main__':
    ifn = sys.argv[1]

    # Read the configuration file
    try:
        fn_config = sys.argv[2]
    except IndexError:
        fn_config = DEFAULT_CONFIG_FILE

    try:
        config = read_config_file(fn_config)
    except FileNotFoundError as err:
        print(f'No such config file (arg2): {fn_config}')
        raise err


    # Here, we assume the input file is of type 'grd'
    #  and the output is of fieldtype 'sfm'
    ofn = get_corresponding_filename(ifn, 'grd', 'sfm')

    #apply_gaussian_smoothing(ifn, config)
    apply_gaussian_smoothing(config, ifn, ofn)
