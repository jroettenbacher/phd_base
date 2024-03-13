#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 12.03.2024

Python translation of ecrad_execute_IFS.sh
Runs ecRad for a certain campaign and date using a given input and namelist version.
Does all the postprocessing including:

- ecrad_merge_radiative_properties.py
- ecrad_merge_files.py for input
- ecrad_merge_files.py for output
- ecrad_processing.py

"""
import pylim.helpers as h
import argparse
import os
import subprocess
import glob


def parse_args():
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('-c', '--campaign', default='halo-ac3', help='Campaign (default: halo-ac3)')
    parser.add_argument('-d', '--date_var', default='20220411', help='Date (default: 20220411)')
    parser.add_argument('-i', '--input_version', default='v6', help='Input version (default: v6)')
    parser.add_argument('-t', '--t_option', action='store_true', help='Use t option')
    parser.add_argument('-v', '--version', default='v15', help='Version (default: v15)')
    parser.add_argument('-k', '--key', default='RF17', help='Key (default: RF17)')
    return parser.parse_args()


def main():
    args = parse_args()

    ecrad = '/projekt_agmwend/Modelle/ECMWF_ECRAD/src/ecrad-1.5.0/bin/ecrad'
    campaign = args.campaign
    key = args.key
    input_version = args.input_version
    version = args.version
    date_var = args.date_var
    base_dir = h.get_path('ecrad', campaign=campaign)
    # setup logging and return input to user
    log = h.setup_logging('./logs', __file__, f'{key}_{input_version}_{version}')
    log.info(f'The following options have been passed:\n'
             f'campaign: {campaign}\n'
             f'key: {key}\n'
             f'date: {date_var}\n'
             f'input version: {input_version}\n'
             f'version: {version}\n'
             f'time interpolated: {args.t_option}\n'
             f'base_dir: {base_dir}\n')

    inpath = os.path.join(base_dir, date_var, 'ecrad_input')
    outpath = os.path.join(base_dir, date_var, 'ecrad_output')
    rad_prop_outpath = os.path.join(base_dir, date_var, f'radiative_properties_{version}')
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(rad_prop_outpath, exist_ok=True)

    namelist = f'{base_dir}/{date_var}/IFS_namelist_jr_{date_var}_{version}.nam'

    reg_file = f'ecrad_input_*_sod_{input_version}.nc'
    if args.t_option:
        reg_file = f'ecrad_input_*_sod_inp_{input_version}.nc'

    os.chdir(inpath)

    # delete all radiative property files before running ecRad to avoid errors
    subprocess.run(['rm', 'radiative_properties*'])

    file_list = sorted(glob.glob(reg_file))
    n_files = len(file_list)
    log.info(f'Number of files to calculate: {n_files}')

    for i, file_path in enumerate(file_list):
        filename = os.path.basename(file_path)
        log.info(f'infile: {filename}')
        sod = filename.split('_')[3]
        outfilename = os.path.join(outpath,
                                   (filename
                                    .replace('input', 'output')
                                    .replace(input_version, version)))
        log.info(f'outfile: {outfilename}')
        log.info(f'Processing filenumber: {i + 1} of {n_files}')

        subprocess.run([ecrad, namelist, filename, outfilename])

        if os.path.isfile('radiative_properties.nc'):
            subprocess.run(['mv', 'radiative_properties.nc', f'{rad_prop_outpath}/radiative_properties_{sod}.nc'])
            log.info(f'Moved radiative_properties.nc to {rad_prop_outpath}')
        elif os.path.isfile('radiative_properties_*.nc'):
            subprocess.run(['mv'] + glob.glob('radiative_properties_*.nc') + [f'{rad_prop_outpath}/'])
            log.info(f'Moved all radiative_properties files to {rad_prop_outpath}')
        else:
            log.info('No radiative_properties files to move found!')

    log.info('> Done with ecRad simulations.')
    os.chdir('/projekt_agmwend/home_rad/jroettenbacher/phd_base/processing')  # cd to working directory
    log.info('> Merging radiative_properties files...')
    subprocess.run(['python', 'ecrad_merge_radiative_properties.py',
                    f'date={date_var}',
                    f'version={version}',
                    f'campaign={campaign}'])

    log.info('> Merging input files...')
    subprocess.run(['python', 'ecrad_merge_files.py',
                    f'date={date_var}',
                    'io_flag=input',
                    f'version={input_version}',
                    f'campaign={campaign}'])

    log.info('> Merging output files...')
    subprocess.run(['python', 'ecrad_merge_files.py',
                    f'date={date_var}',
                    'io_flag=output',
                    f'version={version}',
                    f'campaign={campaign}'])

    log.info('> Merging merged input and output file...')
    subprocess.run(
        ['python', 'ecrad_processing.py',
         f'date={date_var}',
         f'key={key}',
         f'ov={version}',
         f'iv={input_version}',
         f'campaign={campaign}'])


if __name__ == '__main__':
    main()
