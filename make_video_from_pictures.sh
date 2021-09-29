#!/usr/bin/env bash

# create a stop motion video from GoPro time laps pictures
date="20210729"
number="a"
#base_dir="/mnt/d/CIRRUS-HL/Gopro"
base_dir="/mnt/e/CIRRUS-HL/04_Gopro"
inpath="${base_dir}/Flight_${date}${number}"
outfile="${date}${number}_Gopro_video_fast.mp4"  # change name according to framrate (24=fast, 12=slow)
outpath="${base_dir}/${outfile}"
framerate="24"
start_number="102"
ffmpeg -framerate ${framerate} -f image2 \
  -start_number ${start_number} \
  -i ${inpath}/${date}_Gopro_%04d.JPG \
  -vcodec libx264 -b:v 50000k \
  -r ${framerate} ${outpath}

# options: -f -> filetype, -i -> filename, %%04d -> 0001 ... 9999, -b:v -> video bitrate, -r 6 -> frames per second