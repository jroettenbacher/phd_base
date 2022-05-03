#!/usr/bin/env bash

# create a stop motion video from GoPro time laps pictures
flight="HALO-AC3_20220407_HALO_RF14"
# extract date and flight key
date=${flight:9:8}
key=${flight:23:6}
#base_dir="/mnt/d/CIRRUS-HL/Gopro"
base_dir="/mnt/e/HALO-AC3/04_Gopro"
#base_dir="/mnt/c/Users/Johannes/Videos"
ending="map"
inpath="${base_dir}/${flight}_${ending}"
#inpath="/mnt/c/Users/Johannes/Pictures/GoPro/${date}"
outfile="HALO-AC3_HALO_Gopro_video_fast_${date}_${key}.mp4"  # change name according to framrate (24=fast, 12=slow)
outpath="${base_dir}/${outfile}"
#outpath="${inpath}/../${outfile}"
framerate="24"
start_number="312"
ffmpeg -framerate ${framerate} -f image2 \
  -start_number ${start_number} \
  -i ${inpath}/HALO-AC3_HALO_Gopro_${date}_${key}_%04d.JPG \
  -vcodec libx264 -b:v 50000k \
  -r ${framerate} ${outpath}

# loop music over the movie https://audiotrimmer.com/royalty-free-music/
#ffmpeg -i ${outfile} -stream_loop -1 -i /mnt/c/Users/Johannes/Music/Funky-Chase.mp3 -shortest -map 0:v:0 -map 1:a: -c:v copy ${outfile/%.mp4/_music.mp4}

# options: -f -> filetype, -i -> filename, %%04d -> 0001 ... 9999, -b:v -> video bitrate, -r 6 -> frames per second