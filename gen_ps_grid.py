"""Generate a polar stereo grid from JAXA AMSR2 L1R swath files

Note: This builds on grid_jaxa_l1r.py and combine_gridded_l1r_tbs.py
"""

import datetime as dt
from string import Template
import dateutil.parser
import os
import glob

from grid_jaxa_l1r import (
    get_gridinfo,
    read_jxam2l1r_config,
    verify_jxam2l1r_file,
    process_jxam2l1r_file,
)
from combine_gridded_l1r_tbs import (
    combine_tb_files,
)

swath_dir_t = {
    # This was on Scott's laptop
    #'jxam2l1r': Template('/home/scotts/Desktop/data/jaxa_amsr2/${year}/${month}'),

    # NSIDC has JAXA AMSR2 L1R in a dir on /disks/archives/
    #   Current year files are in that directory
    #   Prior year files are in subdirs by year
    'jxam2l1r': Template('/disks/archive/nsidc0743_AMSR2_L1R_swath_tbs/${year}'),
    'jxam2l1r_thisyear': Template('/disks/archive/nsidc0743_AMSR2_L1R_swath_tbs/'),
}

fn_globstr_t = {
    # Typical JAXA AMSR2 L1R file name:
    #   GW1AM2_202403030028_178A_L1SGRTBR_2220220.h5
    #          YYYYMMDDHHMM orbt datatype verstr
    'jxam2l1r': Template('GW1AM2_${year}${month}${day}*_*_L1SGRTBR_*.h5'),
}


def get_gridinfo_dict(gridid):
    """Return a dictionary with the gridinfo items"""
    epsg, xdim, ydim, res, x0, y0, latmin, latmax = get_gridinfo(gridid)
    return locals()


def get_filelist_src_date(src, date):
    """List all the swath files for this source and date"""
    date_dict = {
        'year': date.strftime('%Y'),
        'month': date.strftime('%m'),
        'day': date.strftime('%d'),
        'ymd': date.strftime('%Y%m%d'),
    }
    # This will fail for data from this calendar year (before L1R data
    #  is moved to subdirectories in the /disks/archives/<location>
    srcdir = swath_dir_t[src].safe_substitute(date_dict)
    print(f'    {srcdir=}')
    assert os.path.isdir(srcdir)

    file_list = sorted(glob.glob(
        f'{srcdir}/{fn_globstr_t[src].safe_substitute(date_dict)}'
    ))
    # print(f'file_list: {file_list}')
    
    # new_file_list = file_list[:1]
    # new_file_list.extend([f for f in file_list if '202403130741' in f])
    # print("WARNING>>>SELECTING ONLY ONE FILE!!!")
    # print("WARNING>>>SELECTING ONLY ONE FILE!!!")
    # print("WARNING>>>SELECTING ONLY ONE FILE!!!")
    # return new_file_list
    return file_list


def get_swathfile_list(src, date):
    """Return the swath file list for this date"""
    yesterday = date - dt.timedelta(days=1)
    print(f'  {date=}')
    print(f'  {yesterday=}')

    yesterday_files = get_filelist_src_date(src, yesterday)
    today_files = get_filelist_src_date(src, date)

    swath_file_list = []
    swath_file_list.append(yesterday_files[-1])
    swath_file_list.extend(today_files)

    return swath_file_list


def get_sample_outfile(fn, dn, match_substring):
    """Return a sample output file for this file name
         typical fn:  /home/scotts/Desktop/data/jaxa_amsr2/2024/03/GW1AM2_202403022338_178D_L1SGRTBR_2220220.h5
      typical outfn: ./intermed/psn25_h19r06_av_202403022338.dat
    """
    bfn = os.path.basename(fn)
    timestamp = bfn.split('_')[1]
    out_fn = f'{dn}/{gridid}_{match_substring}_av_{timestamp}.dat'

    print(f'  sample_outfile: {out_fn}')

    return out_fn


def grid_from_l1r(gridid, src, date):
    """Create grid from all swaths for this date"""

    print(f'{gridid=}')
    print(f'{date=}')
    print(f'{src=}')
    gridinfo_d = get_gridinfo_dict(gridid)
    print(f'{gridinfo_d=}')
    print('check gridinfo')

    swfile_list = get_swathfile_list(src, date)
    print(f'  swfile_list: {len(swfile_list)} files')

    #config_fn = './amsr2l1r_psn25_config.yaml'
    #config_fn = f'./amsr2l1r_{gridid}_config.yaml'
    config_fn = f'./amsr2l1r_{gridid}_extra_config.yaml'
    print(f'using special config file:\n  {config_fn}')
    config = read_jxam2l1r_config(config_fn)
    print(f'  config: {len(config)} items')

    intermed_dir = './intermed'
    output_dir = './output'

    overwrite = False
    all_av_files = []
    all_ct_files = []
    for fn in swfile_list:
        datetime = dt.datetime(date.year, date.month, date.day)
        match_substring = 'h19r06'
        sample_outfile = get_sample_outfile(fn, intermed_dir, match_substring)
        missing_sample_file = not os.path.isfile(sample_outfile)
        if overwrite or missing_sample_file:
            verify_jxam2l1r_file(fn, config=config)
            av_files, ct_files = process_jxam2l1r_file(fn, config, datetime, intermed_dir, output_dir)

            all_av_files.extend(av_files)
            all_ct_files.extend(ct_files)
        else:
            glob_matchstring_av = sample_outfile.replace(match_substring, '*')
            av_files = glob.glob(glob_matchstring_av)

            glob_matchstring_ct = glob_matchstring_av.replace('_av_', '_ct_')
            ct_files = glob.glob(glob_matchstring_ct)

            print(f'  Adding {len(av_files)} av and {len(ct_files)} ct files to list')

            all_av_files.extend(av_files)
            all_ct_files.extend(ct_files)

        print(f'Finished processing:\n  {fn}')
        #breakpoint()

    print('about to run combine_tb_files()...')
    print(f'  n av files: {len(all_av_files)}')
    print(f'  n ct files: {len(all_ct_files)}')

    indir = intermed_dir
    combine_tb_files(indir, config, date, intermed_dir, output_dir, all_av_files=all_av_files, all_ct_files=all_ct_files)
    print('after combine_tb_files()')


if __name__ == '__main__':
    import sys

    gridid = sys.argv[1]
    date = dateutil.parser.parse(sys.argv[2]).date()
    try:
        src = sys.argv[3]
    except IndexError:
        src = 'jxam2l1r'

    grid_from_l1r(gridid, src, date)
