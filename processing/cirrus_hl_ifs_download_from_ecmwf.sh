#!/bin/bash
#SBATCH --chdir=/ec/res4/scratch/gdmw/scratch_jr/ifs_data
cd /ec/res4/scratch/gdmw/scratch_jr/ifs_data || exit


# download surface files for the complete area and multi level files
latlon_area='75/-35/45/30'  # CIRRUS-HL  northern area
declare -a dates=(20210629)
# declare -a dates=(20220410 20220411 20220412)  # if you only want to run one date
declare -a times=(00)
# grid='F1280'  # regular (lat lon) gaussian grid
grid='O1280'  # octahedral reduced gaussian grid (original grid)

# loop over dates if needed
for date in "${dates[@]}"
do
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
        # define file names
        mars_sfc_file=mars_ifs_surface_"${date}"_"${time}"_"${grid}"
        sbatch_sfc_file=mars_ifs_surface_"${date}"_"${time}"_"${grid}".sh
        mars_ml_file=mars_ifs_ml_"${date}"_"${time}"_"${grid}"
        sbatch_ml_file=mars_ifs_ml_"${date}"_"${time}"_"${grid}".sh
        # write MARS surface retrievals for 00Z and 12Z
        cat > "${mars_sfc_file}" << EOF
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
param    =31/32/34/47/78/79/134/136/137/141/151/164/165/166/172/175/176/177/178/179/186/187/188/206/208/209/210/211/212/228/235/238/243/174098/228021/228022/228023/228088/228089/228090/228045/260048/260109,
stream   =oper,
type     =fc
EOF

        # write bash script to call with sbatch which will then execute the mars request
        # due to a new version of slurm calling mars with sbatch is no longer possible...
       cat > "${sbatch_sfc_file}" << EOF
#!/bin/bash
#SBATCH --chdir=/ec/res4/scratch/gdmw/scratch_jr/ifs_data/${date}
cd /ec/res4/scratch/gdmw/scratch_jr/ifs_data/${date}
mars ${mars_sfc_file}
EOF
        # start sbatch job for surface
        sbatch --job-name="${date}"_"${time}"_"${grid}"_ifs_surface --time=01:00:00 --dependency=singleton "${sbatch_sfc_file}"

        # write MARS retrievals for multi level files for 00Z and 12Z
        cat > "${mars_ml_file}" <<EOF
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
        cat > "${sbatch_ml_file}" << EOF
#!/bin/bash
#SBATCH --chdir=/ec/res4/scratch/gdmw/scratch_jr/ifs_data/${date}
cd /ec/res4/scratch/gdmw/scratch_jr/ifs_data/${date}
mars ${mars_ml_file}
EOF
        # start sbatch job for multi levels
         sbatch --job-name="${date}"_"${time}"_"${grid}"_ifs_ml --time=05:00:00 --dependency=singleton "${sbatch_ml_file}"

    done

    cd ..  # move out of daily folder

done