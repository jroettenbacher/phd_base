#!/bin/bash
#SBATCH --chdir=/ec/res4/scratch/gdmw/scratch_jr/cams_data
set -f
cd /ec/res4/scratch/gdmw/scratch_jr/cams_data/monthly_mean || exit

# download monthly mean files from the greenhouse gas reanalysis (methane, CO2) and from the global reanalysis (11 aerosol species + ozone)
# download files for the complete area and multi level files
latlon_area='90/-60/67/30'  # HALO-(AC)3 area
declare -a years=($(seq 2019 1 2020))
#grid='F1280'  # regular (lat lon) gaussian grid (IFS resolution)
grid='O1280'  # octahedral reduced gaussian grid (original IFS grid)
#grid='N128'  # reduced gaussian grid (original CAMS resolution)

for year in "${years[@]}"
do
    # global greenhouse gas reanalysis (methane and CO2)
    mars_file=mars_cams_gghg_pl_halo_ac3_"${year}"_"${grid}"
    sbatch_file=mars_cams_gghg_pl_halo_ac3_"${year}"_"${grid}".sh
    # write MARS retrievals for multi level files
    cat > "${mars_file}" <<EOF
retrieve,
class    =mc,
date     =${year}0101/${year}0201/${year}0301/${year}0401/${year}0501/${year}0601/${year}0701/${year}0801/${year}0901/${year}1001/${year}1101/${year}1201,
levtype  =pl,
levelist =1/2/3/5/7/10/20/30/50/70/100/150/200/250/300/400/500/600/700/800/850/900/925/950/1000,
expver   =egg4,
area     =${latlon_area},
grid     =${grid},
param    =61.210/62.210,
accuracy =av,
padding  =0,
stream   =mnth,
time     =00:00:00/03:00:00/06:00:00/09:00:00/12:00:00/15:00:00/18:00:00/21:00:00,
type     =an,
target   =halo_ac3_cams_global_ghg_reanalysis_mm_pl_${grid}_${year}.grb
EOF

    # write bash script to call with sbatch which will then execute the mars request
    # due to a new version of slurm calling mars with sbatch is no longer possible...
    cat > "${sbatch_file}" << EOF
#!/bin/bash
#SBATCH --chdir=/ec/res4/scratch/gdmw/scratch_jr/cams_data/monthly_mean
cd /ec/res4/scratch/gdmw/scratch_jr/cams_data/monthly_mean
mars ${mars_file}
EOF
    # start sbatch job for multi levels
    sbatch --job-name="${year}"_cams_mm_pl_halo_ac3 --time=01:00:00 --dependency=singleton "${sbatch_file}"

# global reanalysis (aerosols + ozone)
    mars_file=mars_cams_eac4_pl_halo_ac3_"${year}"_"${grid}"
    sbatch_file=mars_cams_eac4_pl_halo_ac3_"${year}"_"${grid}".sh
    # write MARS retrievals for multi level files
    cat > "${mars_file}" <<EOF
retrieve,
class    =mc,
date     =${year}0101/${year}0201/${year}0301/${year}0401/${year}0501/${year}0601/${year}0701/${year}0801/${year}0901/${year}1001/${year}1101/${year}1201,
levtype  =pl,
levelist =1/2/3/5/7/10/20/30/50/70/100/150/200/250/300/400/500/600/700/800/850/900/925/950/1000,
expver   =eac4,
area     =${latlon_area},
grid     =${grid},
param    =1.210/2.210/3.210/4.210/5.210/6.210/7.210/8.210/9.210/10.210/11.210/203.210,
accuracy =av,
padding  =0,
stream   =mnth,
time     =00:00:00/03:00:00/06:00:00/09:00:00/12:00:00/15:00:00/18:00:00/21:00:00,
type     =an,
target   =halo_ac3_cams_eac4_global_reanalysis_mm_pl_${grid}_${year}.grb
EOF

    # write bash script to call with sbatch which will then execute the mars request
    # due to a new version of slurm calling mars with sbatch is no longer possible...
    cat > "${sbatch_file}" << EOF
#!/bin/bash
#SBATCH --chdir=/ec/res4/scratch/gdmw/scratch_jr/cams_data/monthly_mean
cd /ec/res4/scratch/gdmw/scratch_jr/cams_data/monthly_mean
mars ${mars_file}
EOF
    # start sbatch job for multi levels
    sbatch --job-name="${year}"_cams_mm_pl_halo_ac3 --time=01:00:00 --dependency=singleton "${sbatch_file}"


done  # years
