#!/usr/bin/env bash

# create a stop motion video from GoPro time laps pictures
date="20210708"
#base_dir="/mnt/d/CIRRUS-HL/Gopro"
base_dir="/mnt/c/Users/Johannes/Documents/Gopro"
inpath="${base_dir}/${date}_short2"
outfile="${date}_short2_Gopro_video.mp4"
outpath="${base_dir}/${outfile}"
framerate="12"
start_number="720"
ffmpeg -framerate ${framerate} -f image2 \
  -start_number ${start_number} \
  -i ${inpath}/${date}_Gopro_%04d.JPG \
  -vcodec libx264 -b:v 50000k \
  -r ${framerate} ${outpath}

# options: -f -> filetype, -i -> filename, %%04d -> 0001 ... 9999, -b:v -> video bitrate, -r 6 -> frames per second