#!\usr\bin\env python
"""Add the timstamp to a picture (run on Ubuntu)
corrects the meta data timestamp in an image for a
author: Johannes Röttenbacher
"""
if __name__ == "__main__":
    # %% import libraries and set paths
    import pylim.helpers as h
    from pylim.halo_ac3 import gopro_offsets
    import os
    import datetime
    from tqdm import tqdm
    from subprocess import run, Popen

    # user input
    campaign = "halo-ac3"
    flight = "HALO-AC3_20220225_HALO_RF00"
    flight_key = flight[-4:] if campaign == "halo-ac3" else flight
    date = flight[9:17]
    correct_time = True
    start_file = 0
    # file = f"/mnt/c/Users/Johannes/Pictures/GoPro/{date}/{date}_Gopro_0001.JPG"

    path = f"{h.get_path('gopro', campaign='halo-ac3')}/{flight}"  # path to all files
    # path = f"/mnt/c/Users/Johannes/Pictures/GoPro/{date}"
    sync_to_bahamas = True if flight_key in gopro_offsets else False
    # needed for CIRRUS-HL when camera switched to local time
    # LT_to_UTC = gopro_lt[flight] if flight in gopro_lt else False
    LT_to_UTC = False
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".JPG")][start_file:]
    processes = set()
    max_processes = 10

# %% update meta data time stamp to set to UTC
    if correct_time:
        # GoPro switched to LT on 28.06 due to WiFi connection
        utc_correction = 2 if LT_to_UTC else 0  # convert local time to UTC
        # GoPro is not synced to BAHAMAS
        bahamas_correction = gopro_offsets[flight_key] if sync_to_bahamas else 0
        correction = utc_correction + bahamas_correction / 60 / 60  # convert seconds to hours
        # format correction
        delta = datetime.timedelta(hours=abs(correction))
        sign = "-" if correction > 0 else "+"
        cor_str = str(delta)
        # either give single file or path to all files
        run(['exiftool', '-m', '-progress', '-overwrite_original', f'-DateTimeOriginal{sign}={cor_str}', path])

# # %% test one file
#     f = file  # test one file
#     processes.add(Popen(['convert', f, '-fill', 'white', '-pointsize', '72', '-annotate', '+3100+2900',
#                          '%[exif:DateTimeOriginal] UTC', f]))
#     if len(processes) >= max_processes:
#         os.wait()
#         processes.difference_update([p for p in processes if p.poll() is not None])

# %% add the time stamp from the exif meta data in the right lower corner
    # set fill to white or black depending on background
    for f in tqdm(files, desc="Add Time Stamp"):
        processes.add(Popen(['convert', f, '-fill', 'black', '-pointsize', '72', '-annotate', '+3100+2900',
                             '%[exif:DateTimeOriginal] UTC', f]))
        if len(processes) >= max_processes:
            os.wait()
            processes.difference_update([p for p in processes if p.poll() is not None])

    print(f"Done with all files in {path}")
