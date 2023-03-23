#!/bin/bash

cwd=$(pwd)
date="20220411"
input_version="v2"
version="v8"
if [[ ${cwd} =~ /home/jroettenbacher/.* ]]; then
	data_basedir="/mnt/e/HALO-AC3"
	ecrad_basedir="/usr/local"
else
	data_basedir="/projekt_agmwend/data/HALO-AC3"
	ecrad_basedir="/projekt_agmwend/Modelle/ECMWF_ECRAD"
fi

inpath="${data_basedir}/08_ecrad/${date}/ecrad_input"
outpath="${data_basedir}/08_ecrad/${date}/ecrad_output"
rad_prop_outpath="${data_basedir}/08_ecrad/${date}/radiative_properties_${version}"
mkdir -p "${outpath}"
#mkdir -p "${rad_prop_outpath}"
# change to where infiles are located
cd "${inpath}" || exit 1
ecrad="${ecrad_basedir}/src/ecrad-1.5.0/bin/ecrad"
namelist="../IFS_namelist_jr_${date}_${version}.nam"  # personal namelist
filename="ecrad_input_standard_32460.0_sod_${input_version}.nc" # new input (ecrad_read_ifs.py)
echo ecRad: processing file ${filename}
echo using namelist "${namelist}"

echo infile: ${filename}
outfilename="${outpath}/${filename/input/output}"
outfilename="${outfilename/${input_version}/${version}}"
echo outfile: ${outfilename}

sod=$(echo $filename | grep -oP '(\d{5}\.\d{1})')

echo "${ecrad}" "${namelist}" "${filename}" "${outfilename}"
"${ecrad}" "${namelist}" "${filename}" "${outfilename}"

if test -f "radiative_properties.nc"; then
  mv "radiative_properties.nc" "${rad_prop_outpath}/radiative_properties_${sod}.nc"
  echo "Moved radiative_properties.nc to ${rad_prop_outpath}"
elif test -f "radiative_properties_*.nc"; then
  # move the radiation properties intermediate output files to the output folder and add the current second of day to the filename
  ls radiative_properties_* | sed "s/\(radiative_properties\)_\([0-9]\{4\}-[0-9]\{4\}\)\.nc/mv & ${rad_prop_outpath//\//\\/}\/\1_${sod}_\2.nc/" | sh
  echo "Moved all radiative_properties files to ${rad_prop_outpath}"
else
  echo "No radiative_properties files to move found!"
fi

echo ">" Done.
