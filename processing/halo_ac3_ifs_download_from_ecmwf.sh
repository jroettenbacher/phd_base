#!/bin/bash
#SBATCH --chdir=/ec/res4/scratch/gdmw/scratch_jr
cd /ec/res4/scratch/gdmw/scratch_jr || exit


# download surface files for the complete area and multi level files
latlon_area='90/-60/67/30'

# loop over dates if needed
for date in 20220312 20220313 20220314 20220315 20220316 20220320 20220321 20220328 20220329 20220330 20220404 20220407 20220408 20220410 20220411 20220412;
do
    date_12=$(date +%Y%m%d -d "${date} - 1 day")  # define date before
    mkdir -p ${date}
    cd ${date} || exit

# write MARS surface retrievals for 12Z, 00Z and 12Z the day before
# 12Z
    cat > mars_ifs_surface_${date}_12 << EOF
retrieve,
        padding  = 0,
        accuracy = 16,
        class    = od,
        expver   = 1,
        stream   = oper,
        domain   = g,
        type     = fc,
        date     = ${date},
        time     = 12,
        step     = 0/to/12/by/1,
        target   = ifs_${date}_12_sfc.grb,
        param    = 31/32/34/134/136/137/151/164/165/166/172/235/243/cbh/169/175/176/177,
        repres   = sh,
        area     = ${latlon_area},
        resol    = 1279,
        grid     = 1280,
        gaussian = regular,
        levtype  = sfc
EOF

# write bash script to call with sbatch which will then execute the mars request
# due to a new version of slurm calling mars with sbatch is no longer possible...

   cat > mars_ifs_surface_${date}_12.sh << EOF
#!/bin/bash
#SBATCH --chdir=/ec/res4/scratch/gdmw/scratch_jr/${date}
cd /ec/res4/scratch/gdmw/scratch_jr/${date}
mars mars_ifs_surface_${date}_12
EOF

   sbatch --job-name=mars_ifs_surface_012 --time=05:00:00 --dependency=singleton mars_ifs_surface_${date}_12.sh

# 00Z
    cat <<EOF > mars_ifs_surface_${date}_00
retrieve,
        padding  = 0,
        accuracy = 16,
        class    = od,
        expver   = 1,
        stream   = oper,
        domain   = g,
        type     = fc,
        date     = ${date},
        time     = 00,
        step     = 0/to/24/by/1,
        target   = ifs_${date}_00_sfc.grb,
        param    = 31/32/34/134/136/137/151/164/165/166/172/235/243/cbh/169/175/176/177,
        repres   = sh,
        area     = ${latlon_area},
        resol    = 1279,
        grid     = 1280,
        gaussian = regular,
        levtype  = sfc
EOF

# write bash script to call with sbatch which will then execute the mars request
# due to a new version of slurm calling mars with sbatch is no longer possible...

   cat > mars_ifs_surface_${date}_00.sh << EOF
#!/bin/bash
#SBATCH --chdir=/ec/res4/scratch/gdmw/scratch_jr/${date}
cd /ec/res4/scratch/gdmw/scratch_jr/${date}
mars mars_ifs_surface_${date}_00
EOF

    sbatch --job-name=mars_ifs_surface_00 --time=05:00:00 --dependency=singleton mars_ifs_surface_${date}_00.sh

# 12Z the day before
    cat <<EOF > mars_ifs_surface_${date_12}_12
retrieve,
        padding  = 0,
        accuracy = 16,
        class    = od,
        expver   = 1,
        stream   = oper,
                domain   = g,
        type     = fc,
        date     = ${date_12},
        time     = 12,
        step     = 12/to/36/by/1,
        target   = ifs_${date_12}_12_sfc.grb,
        param    = 31/32/34/134/136/137/151/164/165/166/172/235/243/cbh/169/175/176/177,
        repres   = sh,
        area     = ${latlon_area},
        resol    = 1279,
        grid     = 1280,
        gaussian = regular,
        levtype  = sfc
EOF

# write bash script to call with sbatch which will then execute the mars request
# due to a new version of slurm calling mars with sbatch is no longer possible...

   cat > mars_ifs_surface_${date_12}_12.sh << EOF
#!/bin/bash
#SBATCH --chdir=/ec/res4/scratch/gdmw/scratch_jr/${date}
cd /ec/res4/scratch/gdmw/scratch_jr/${date}
mars mars_ifs_surface_${date_12}_12
EOF

    sbatch --job-name=mars_ifs_surface_12 --time=05:00:00 --dependency=singleton mars_ifs_surface_${date_12}_12.sh

# write MARS retrievals for multi level files for 12Z, 00Z and 12Z the day before
# 12Z

    cat <<EOF > mars_ifs_ml_${date}_12
retrieve,
        padding  = 0,
        accuracy = 16,
        class    = od,
        expver   = 1,
        stream   = oper,
        domain   = g,
        type     = fc,
        date     = ${date},
        time     = 12,
        step     = 0/to/12/by/1,
        target   = ifs_${date}_12_ml.grb,
        param    = 75/76/130/131/132/133/135/152/203/246/247/248,
        repres   = sh,
        area     = ${latlon_area},
        resol    = 1279,
        grid     = 1280,
        gaussian = regular,
        levtype  = ml,
        levellist = 1/to/137
EOF

# write bash script to call with sbatch which will then execute the mars request
# due to a new version of slurm calling mars with sbatch is no longer possible...

   cat > mars_ifs_ml_${date}_12.sh << EOF
#!/bin/bash
#SBATCH --chdir=/ec/res4/scratch/gdmw/scratch_jr/${date}
cd /ec/res4/scratch/gdmw/scratch_jr/${date}
mars mars_ifs_ml_${date}_12
EOF

    sbatch --job-name=mars_ifs_ml_012 --time=05:00:00 --dependency=singleton mars_ifs_ml_${date}_12.sh

# 00Z
    cat <<EOF > mars_ifs_ml_${date}_00
retrieve,
        padding  = 0,
        accuracy = 16,
        class    = od,
        expver   = 1,
        stream   = oper,
        domain   = g,
        type     = fc,
        date     = ${date},
        time     = 00     ,
        step     = 0/to/24/by/1,
        target   = ifs_${date}_00_ml.grb,
        param    = 54/75/76/130/131/132/133/135/152/246/247/248,
        repres   = sh,
        area     = ${latlon_area},
        resol    = 1279,
        grid     = 1280,
        gaussian = regular,
        levtype  = ml,
        levellist = 1/to/137
EOF

   cat > mars_ifs_ml_${date}_00.sh << EOF
#!/bin/bash
#SBATCH --chdir=/ec/res4/scratch/gdmw/scratch_jr/${date}
cd /ec/res4/scratch/gdmw/scratch_jr/${date}
mars mars_ifs_ml_${date}_00
EOF

   sbatch --job-name=mars_ifs_ml_00 --time=05:00:00 --dependency=singleton mars_ifs_ml_${date}_00.sh

# 12Z the day before
    cat <<EOF > mars_ifs_ml_${date_12}_12
retrieve,
        accuracy = 16,
        area     = ${latlon_area},
        class    = od,
        date     = ${date_12},
        domain   = g,
        expver   = 1,
        grid     = 1280,
        type     = fc,
        levellist = 1/to/137
        levtype  = ml,
        padding  = 0,
        param    = 54/75/76/130/131/132/133/135/152/246/247/248,
        repres   = sh,
        resol    = 1279,
        stream   = oper,
        step     = 12/to/36/by/1,
        target   = ifs_${date_12}_12_ml.grb,
        time     = 12,
EOF

   cat > mars_ifs_ml_${date_12}_12.sh << EOF
#!/bin/bash
#SBATCH --chdir=/ec/res4/scratch/gdmw/scratch_jr/${date}
cd /ec/res4/scratch/gdmw/scratch_jr/${date}
mars mars_ifs_ml_${date_12}_12
EOF

    sbatch --job-name=mars_ifs_ml_12 --time=05:00:00 --dependency=singleton mars_ifs_ml_${date_12}_12.sh

cd ..

done