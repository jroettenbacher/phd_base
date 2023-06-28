#!/bin/bash
#SBATCH --chdir=/ec/res4/scratch/gdmw/scratch_jr
set -f
cd /ec/res4/scratch/gdmw/scratch_jr/cams || exit


# download files for the complete area and multi level files
latlon_area='90/-60/67/30'  # HALO-(AC)3 area
declare -a years=($(seq 2003 1 2020))
declare -a months=(01 02 03 04 05 06 07 08 09 10 11 12)
declare -a months=(03 04)  # HALO-(AC)3 months

for year in "${years[@]}"
do
    mkdir -p "${year}"
    cd "${year}" || exit

    for m in "${months[@]}"
    do
        # write MARS retrievals for multi level files
        cat > mars_cams_ml_halo_ac3_"${m}" <<EOF
retrieve,
class    =mc,
date     =${year}-${m}-01/to/${year}-${m}-31,
levtype  =ml,
levelist =1/to/60,
expver   =egg4,
area     =${latlon_area},
param    =203/210061/210062,
accuracy =av,
padding  =0,
stream   =oper,
time     =00:00:00/03:00:00/06:00:00/09:00:00/12:00:00/15:00:00/18:00:00/21:00:00,
type     =an,
target   =${year}_${m}_cams_ml_halo_ac3.grb
EOF

    # write bash script to call with sbatch which will then execute the mars request
    # due to a new version of slurm calling mars with sbatch is no longer possible...
        cat > mars_cams_ml_halo_ac3_"${m}".sh << EOF
#!/bin/bash
#SBATCH --chdir=/ec/res4/scratch/gdmw/scratch_jr/cams/${year}
cd /ec/res4/scratch/gdmw/scratch_jr/cams/${year}
mars mars_cams_ml_halo_ac3_${m}
EOF
        # start sbatch job for multi levels
        sbatch --job-name="${year}"_"${m}"_cams_ml_halo_ac3 --time=03:00:00 --dependency=singleton mars_cams_ml_halo_ac3_"${m}".sh

    done  # months

    cd ..

done  # years

