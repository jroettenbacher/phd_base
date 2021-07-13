#!/usr/bin/env bash

# create a stop motion video from GoPro time laps pictures
date="20210708"
base_dir="/mnt/d/CIRRUS-HL/Gopro"
#inpath="/mnt/c/Users/Johannes/Documents/Gopro/${date}_2"
inpath="${base_dir}/${date}"
outfile="${date}_Gopro_video.mp4"
outpath="${base_dir}/${outfile}"

ffmpeg -framerate 12 -f image2 -i ${inpath}/${date}_Gopro_%04d.JPG -vcodec libx264 -b:v 50000k -r 12 ${outpath}

# options: -f -> filetype, -i -> filename, %%04d -> 0001 ... 9999, -b:v -> video bitrate, -r 6 -> frames per second