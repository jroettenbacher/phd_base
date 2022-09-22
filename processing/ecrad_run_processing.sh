#!/usr/bin/env bash
# run ecrad_processing.py with commandline keywords and output to a log file
today=$(date +"%Y%m%d")  # get current date
# set options
io_flag=input
t_inp=T
date=20210629

# call script, write stderror to stdout and pipe output to tee which prints to console and saves to a log file
python ecrad_processing.py io_flag=${io_flag} t_interp=${t_inp} date=${date} 2>&1 | tee ./log/${today}_ecrad_processing_${io_flag}_tinp-${t_inp}_${date}.log

# call script to merge merged input and output file only
# python ecrad_processing.py merge_io=T date=${date}
