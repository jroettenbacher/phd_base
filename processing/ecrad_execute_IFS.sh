#!/usr/bin/bash

ecrad="/projekt_agmwend/Modelle/ECMWF_ECRAD/src/ecrad-1.4.1/bin/ecrad"

# standard options
date_var=20220411
reg_file="ecrad_input_*_sod.nc"
version="v1"

# read in command line args to overwrite standard options
while getopts ":d:tv:" opt; do
  case ${opt} in
  d )
    date_var="${OPTARG}"
    echo Date given: "${date_var}"
  ;;
  t )
    reg_file="ecrad_input_*_sod_inp.nc"
    echo Option -t set
  ;;
  v )
    version="${OPTARG}"
    echo Version selected: "${version}"
  ;;
  ?)
    echo "script usage: $(basename \$0) [-d yyyymmdd] [-t]" >&2
    exit 1
  esac
done

inpath="/projekt_agmwend/data/HALO-AC3/08_ecrad/${date_var}/ecrad_input"
outpath="/projekt_agmwend/data/HALO-AC3/08_ecrad/${date_var}/ecrad_output"
rad_prop_outpath="/projekt_agmwend/data/HALO-AC3/08_ecrad/${date_var}/radiative_properties_${version}"
mkdir -p "${outpath}"
mkdir -p "${rad_prop_outpath}"

# change to where infiles are located
cd "${inpath}" || exit 1
namelist="${inpath}/IFS_namelist_jr_${date_var}_${version}.nam"
echo "${inpath}"
n_files=$(find "${inpath}" -maxdepth 1 -type f -name "${reg_file}" | wc -l)
echo Number of files to calculate:
echo "${n_files}"
counter=1

# delete all radiative property files before running ecRad to avoid errors
rm radiative_properties*

for file in ${inpath}/${reg_file}
do
	filename=${file##*/}
	echo "infile: ${filename}"
	# retrieve second of day from filename
	sod=$(echo $filename | grep -oP '(\d{5}\.\d{1})')
	# generate the outfilename by replacing input with output
 	outfilename="${outpath}/${filename/input/output}"
 	outfilename="${outfilename/.nc/_${version}.nc}"
	echo "outfile: ${outfilename}"
 	echo Processing filenumber: "${counter}" of "${n_files}"
 	echo "${counter}"

 	# call ecRad
 	"${ecrad}" "${namelist}" "${filename}" "${outfilename}"

  # check if only one radiative property file was written (only one column in input file)
  if test -f "radiative_properties.nc"; then
    mv "radiative_properties.nc" "${outpath}/radiative_properties_${sod}.nc"
  else
    # move the radiation properties intermediate output files to the output folder and add the current second of day to the filename
    ls radiative_properties_* | sed "s/\(radiative_properties\)_\([0-9]\{4\}-[0-9]\{4\}\)\.nc/mv & ${rad_prop_outpath//\//\\/}\/\1_${sod}_\2.nc/" | sh
    # list all radiative_properties files and pipe them to sed
    # sed uses reg ex to search for the numbering in the original filename and saves it to a group which can be called in the replacement
    # the replacement is a complete command with mv, & (the original filename) and the outpath in which all / are replaced by \/
    # then the first group is called, the sod is called and the second group (the column numbers) are called to make the new filename
    # this command is then piped to a shell to be executed
    # reg ex: \ -> escape character, \(\) -> define group which can be accessed with \1, \2, etc. in the replacement
  fi
  echo "Moved all radiative_properties files"
 	counter=$((counter+1))

done
echo ">" Done.
