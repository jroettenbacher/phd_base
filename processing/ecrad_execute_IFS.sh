#!/usr/bin/bash

ecrad="/projekt_agmwend/Modelle/ECMWF_ECRAD/src/ecrad-1.5.0/bin/ecrad"
python="/home/jroettenbacher/.conda/envs/phd_base/bin/python"

# standard options
date_var=20220411
version="v1"
input_version="v1"
reg_file="ecrad_input_*_sod_${input_version}.nc"

# read in command line args to overwrite standard options
while getopts ":i:d:tv:" opt; do
  case ${opt} in
  d )
    date_var="${OPTARG}"
    echo Date given: "${date_var}"
  ;;
  i )
    input_version="${OPTARG}"
    echo Input version selected: "${input_version}"
    reg_file="ecrad_input_*_sod_${input_version}.nc"
  ;;
  t )
    reg_file="ecrad_input_*_sod_inp_${input_version}.nc"
    echo Option -t set
  ;;
  v )
    version="${OPTARG}"
    echo Version selected: "${version}"
  ;;
  ?)
    echo "script usage: $(basename \$0) [-i v1] [-d yyyymmdd] [-t] [-v v1]" >&2
    exit 1
  esac
done
shift $((OPTIND -1))

#reg_file=${reg_file/.nc/_${version}.nc}
inpath="/projekt_agmwend/data/HALO-AC3/08_ecrad/${date_var}/ecrad_input"
outpath="/projekt_agmwend/data/HALO-AC3/08_ecrad/${date_var}/ecrad_output"
rad_prop_outpath="/projekt_agmwend/data/HALO-AC3/08_ecrad/${date_var}/radiative_properties_${version}"
mkdir -p "${outpath}"
mkdir -p "${rad_prop_outpath}"

# change to where infiles are located
cd "${inpath}" || exit 1
namelist="../IFS_namelist_jr_${date_var}_${version}.nam"
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
 	outfilename="${outfilename/${input_version}/${version}}"
	echo "outfile: ${outfilename}"
 	echo Processing filenumber: "${counter}" of "${n_files}"
 	echo "${counter}"

 	# call ecRad
 	"${ecrad}" "${namelist}" "${filename}" "${outfilename}"

  # check if only one radiative property file was written (only one column in input file)
  if test -f "radiative_properties.nc"; then
    mv "radiative_properties.nc" "${rad_prop_outpath}/radiative_properties_${sod}.nc"
    echo "Moved radiative_properties.nc to ${rad_prop_outpath}"
  elif test -f "radiative_properties_*.nc"; then
    # move the radiation properties intermediate output files to the output folder and add the current second of day to the filename
    ls radiative_properties_* | sed "s/\(radiative_properties\)_\([0-9]\{4\}-[0-9]\{4\}\)\.nc/mv & ${rad_prop_outpath//\//\\/}\/\1_${sod}_\2.nc/" | sh
    # list all radiative_properties files and pipe them to sed
    # sed uses reg ex to search for the numbering in the original filename and saves it to a group which can be called in the replacement
    # the replacement is a complete command with mv, & (the original filename) and the outpath in which all / are replaced by \/
    # then the first group is called, the sod is called and the second group (the column numbers) are called to make the new filename
    # this command is then piped to a shell to be executed
    # reg ex: \ -> escape character, \(\) -> define group which can be accessed with \1, \2, etc. in the replacement
    echo "Moved all radiative_properties files to ${rad_prop_outpath}"
  else
    echo "No radiative_properties files to move found!"
  fi
 	counter=$((counter+1))

done
echo "> Done with ecRad simulations."
cd "/projekt_agmwend/home_rad/jroettenbacher/phd_base/processing" || exit 1  # cd to working directory
echo "> Merging radiative_properties files..."
${python} ecrad_merge_radiative_properties.py date="${date_var}" version="${version}"

echo "> Merging input files..."
${python} ecrad_merge_files.py date="${date_var}" io_flag=input version="${input_version}"

echo "> Merging output files..."
${python} ecrad_merge_files.py date="${date_var}" io_flag=output version="${version}"

echo "> Merging merged input and output file..."
${python} ecrad_processing.py date="${date_var}" ov="${version}" iv="${input_version}"