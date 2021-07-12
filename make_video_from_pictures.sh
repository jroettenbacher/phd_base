#!/usr/bin/env bash

# create a stop motion video from GoPro time laps pictures
date="20210623"
inpath="/mnt/c/Users/Johannes/Documents/Gopro/${date}"
outfile="${date}_Gopro_video.mp4"
outpath="/mnt/c/Users/Johannes/Documents/Gopro/${outfile}"

ffmpeg -i ${inpath}/${date}_Gopro_%%04d.JPG -vcodec libx264 -b:v 50000k -r 6 ${outpath}

# options: -i -> filename, %%04d -> 0001 ... 9999, -b:v -> video bitrate, -r 6 -> frames per second