#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 29.04.2024

Runs ecRad for certain input files given in one folder and a namelist version.
"""
import pylim.helpers as h
import argparse
import os
import subprocess
import glob


def parse_args():
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('-b', '--base', help='Base directory where the folder ecrad_input and the namelist can be found.')
    parser.add_argument('-v', '--version', default='v1', help='Namelist version (default: v1)')
    parser.add_argument('-k', '--key', default='RF17',
                        help='Flight key or any other string, which is placed before the version'
                             ' and appended to the input and output folder')
    return parser.parse_args()


def main():
    args = parse_args()

    ecrad = '/projekt_agmwend/Modelle/ECMWF_ECRAD/src/ecrad-1.5.0/bin/ecrad'
    base_dir = args.base
    version = args.version
    key = args.key
    # setup logging and return input to user
    log = h.setup_logging('./logs', __file__, f'{version}')
    log.info(f'The following options have been passed:\n'
             f'version: {version}\n'
             f'base_dir: {base_dir}\n'
             f'key: {key}\n'
             )

    inpath = os.path.join(base_dir, f'ecrad_input_{key}')
    outpath = os.path.join(base_dir, f'ecrad_output_{key}')
    rad_prop_outpath = os.path.join(base_dir, f'radiative_properties_{key}_{version}')
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(rad_prop_outpath, exist_ok=True)

    namelist = f'{base_dir}/IFS_namelist_{key}_{version}.nam'

    reg_file = f'ecrad_input_*.nc'

    os.chdir(inpath)

    # delete all radiative property files before running ecRad to avoid errors
    subprocess.run(['rm', 'radiative_properties*'])

    file_list = sorted(glob.glob(reg_file))
    n_files = len(file_list)
    log.info(f'Number of files to calculate: {n_files}')

    for i, file_path in enumerate(file_list):
        filename = os.path.basename(file_path)
        log.info(f'infile: {filename}')
        outfilename = os.path.join(outpath,
                                   (filename
                                    .replace('input', 'output')
                                    .replace('.nc', f'_{version}.nc')
                                    )
                                   )
        log.info(f'outfile: {outfilename}')
        log.info(f'Processing filenumber: {i + 1} of {n_files}')

        subprocess.run([ecrad, namelist, filename, outfilename])

        if os.path.isfile('radiative_properties.nc'):
            subprocess.run(['mv', 'radiative_properties.nc', f'{rad_prop_outpath}/radiative_properties_{i}.nc'])
            log.info(f'Moved radiative_properties.nc to {rad_prop_outpath}')
        elif os.path.isfile('radiative_properties_*.nc'):
            subprocess.run(['mv'] + glob.glob('radiative_properties_*.nc') + [f'{rad_prop_outpath}/'])
            log.info(f'Moved all radiative_properties files to {rad_prop_outpath}')
        else:
            log.info('No radiative_properties files to move found!')

    log.info('> Done with ecRad simulations.')
    os.chdir('/projekt_agmwend/home_rad/jroettenbacher/phd_base/processing')  # cd to working directory


if __name__ == '__main__':
    main()
