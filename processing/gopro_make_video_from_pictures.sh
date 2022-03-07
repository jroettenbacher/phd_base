#!/usr/bin/env bash

# create a stop motion video from GoPro time laps pictures
flight="HALO-AC3_20220225_HALO_RF00"
date="20220225"
#base_dir="/mnt/d/CIRRUS-HL/Gopro"
base_dir="/mnt/e/HALO-AC3/04_Gopro"
#base_dir="/mnt/c/Users/Johannes/Videos"
inpath="${base_dir}/${flight}_map"
#inpath="/mnt/c/Users/Johannes/Pictures/GoPro/${date}"
outfile="${flight}_Gopro_video_fast.mp4"  # change name according to framrate (24=fast, 12=slow)
outpath="${base_dir}/${outfile}"
#outpath="${inpath}/../${outfile}"
framerate="24"
start_number="199"
ffmpeg -framerate ${framerate} -f image2 \
  -start_number ${start_number} \
  -i ${inpath}/${date}_Gopro_%04d.JPG \
  -vcodec libx264 -b:v 50000k \
  -r ${framerate} ${outpath}

# options: -f -> filetype, -i -> filename, %%04d -> 0001 ... 9999, -b:v -> video bitrate, -r 6 -> frames per second