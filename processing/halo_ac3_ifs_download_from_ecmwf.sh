#!/bin/bash
#SBATCH --chdir=/ec/res4/scratch/gdmw/scratch_jr
cd /ec/res4/scratch/gdmw/scratch_jr || exit


# download surface files for the complete area and multi level files
latlon_area='90/-60/67/30'  # HALO-(AC)3 area
declare -a dates=(20220312 20220313 20220314 20220315 20220316 20220320 20220321 20220328 20220329 20220330 20220404 20220407 20220408 20220410 20220411 20220412)
#declare -a dates=(20220411)  # if you only want to run one date
declare -a times=(00 12)
grid='F1280'  # regular (lat lon) gaussian grid
grid='O1280'  # octahedral reduced gaussian grid (original grid)

# loop over dates if needed
for date in "${dates[@]}"
do
    date_12=$(date +%Y%m%d -d "${date} - 1 day")  # define date before
    mkdir -p "${date}"
    cd "${date}" || exit

    for time in "${times[@]}"
    do
        # set end_step according to time
        if [[ ${time} == 12 ]]; then
            end_step=12  # select only 12 steps when starting at 12Z
        else
            end_step=24  # select 24 steps when starting at 00Z
        fi
        # write MARS surface retrievals for 00Z and 12Z
        cat > mars_ifs_surface_"${date}"_"${time}" << EOF
retrieve,
target   =ifs_${date}_${time}_sfc_${grid}.grb,
levtype  =sfc,
date     =${date},
time     =${time},
step     =0/to/${end_step}/by/1,
grid     =${grid},
accuracy =av,
area     =${latlon_area},
class    =od,
padding  =0,
param    =31/32/34/134/136/137/151/164/165/166/172/235/243/cbh/169/175/176/177,
stream   =oper,
type     =fc
EOF

        # write bash script to call with sbatch which will then execute the mars request
        # due to a new version of slurm calling mars with sbatch is no longer possible...
       cat > mars_ifs_surface_"${date}"_"${time}".sh << EOF
#!/bin/bash
#SBATCH --chdir=/ec/res4/scratch/gdmw/scratch_jr/${date}
cd /ec/res4/scratch/gdmw/scratch_jr/${date}
mars mars_ifs_surface_${date}_${time}
EOF
        # start sbatch job for surface
        sbatch --job-name=mars_ifs_surface_"${date}"_"${time}" --time=05:00:00 --dependency=singleton mars_ifs_surface_"${date}"_"${time}".sh

        # write MARS retrievals for multi level files for 00Z and 12Z
        cat > mars_ifs_ml_"${date}"_"${time}" <<EOF
retrieve,
target   =ifs_${date}_${time}_ml_${grid}.grb,
levtype  =ml,
levellist=1/to/137,
date     =${date},
time     =${time},
grid     =${grid},
step     =0/to/${end_step}/by/1,
param    =75/76/130/131/132/133/135/152/203/246/247/248,
accuracy =av,
area     =${latlon_area},
class    =od,
padding  =0,
stream   =oper,
type     =fc
EOF

        # write bash script to call with sbatch which will then execute the mars request
        # due to a new version of slurm calling mars with sbatch is no longer possible...
        cat > mars_ifs_ml_"${date}"_"${time}".sh << EOF
#!/bin/bash
#SBATCH --chdir=/ec/res4/scratch/gdmw/scratch_jr/${date}
cd /ec/res4/scratch/gdmw/scratch_jr/${date}
mars mars_ifs_ml_${date}_${time}
EOF
        # start sbatch job for multi levels
        sbatch --job-name=mars_ifs_ml_"${date}"_"${time}" --time=05:00:00 --dependency=singleton mars_ifs_ml_"${date}"_"${time}".sh

    done

cd ..

done