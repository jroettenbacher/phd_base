#!/usr/bin/env python
"""Run uvspec
author: Johannes Röttenbacher
"""
# %% module import
from smart import get_path
import os
from subprocess import Popen
from tqdm import tqdm
from joblib import cpu_count

# %% set options and get files
flight = "Flight_20210715a"
uvspec_exe = "/opt/libradtran/2.0.4/bin/uvspec"
libradtran_dir = get_path("libradtran", flight)
input_files = [os.path.join(libradtran_dir, "wkdir", f) for f in os.listdir(f"{libradtran_dir}/wkdir")
               if f.endswith(".inp")]
input_files.sort()  # sort input files -> output files will be sorted as well
output_files = [f.replace(".inp", ".out") for f in input_files]
error_logs = [f.replace(".out", ".log") for f in output_files]

# %% call uvspec for one file
# index = 3
# with open(input_files[index], "r") as ifile, open(output_files[index], "w") as ofile, open(error_logs[index], "w") as log:
#     Popen([uvspec_exe], stdin=ifile, stdout=ofile, stderr=log)

# %% call uvspec for all files

processes = set()
max_processes = cpu_count() - 4
for infile, outfile, log_file in zip(tqdm(input_files, desc="libRadtran simulations"), output_files, error_logs):
    with open(infile, "r") as ifile, open(outfile, "w") as ofile, open(log_file, "w") as log:
        processes.add(Popen([uvspec_exe], stdin=ifile, stdout=ofile, stderr=log))
    if len(processes) >= max_processes:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])
