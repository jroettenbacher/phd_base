#!\usr\bin\env python
"""Add the bahamas map to a picture

**Run on Linux!**

**Required User Input:**

* campaign
* flight
* maps
* map numbers from csv file (output of :ref:`processing:gopro_plot_maps.py`)

**Output:** new GoPro picture with map in the upper right corner and timestamp in the lower right corner

**Saves to a new directory!**

Adds the BAHAMAS plot onto the GoPro picture but only for pictures, which were taken in flight, according to a csv file.
Saves those pictures in a new folder: ``{flight}_map``.

*author*: Johannes Roettenbacher
"""
if __name__ == "__main__":
    # %%
    import pylim.helpers as h
    import os
    import pandas as pd
    from tqdm import tqdm
    from subprocess import Popen

    # %% set paths
    campaign = "halo-ac3"
    flight = "HALO-AC3_20220412_HALO_RF18"
    date = flight[9:17]
    flight_key = flight[-4:] if campaign == "halo-ac3" else flight
    gopro_dir = f"{h.get_path('gopro', campaign=campaign)}"
    gopro_path = f"{gopro_dir}/{flight}"
    map_path = f"{h.get_path('bahamas', flight, campaign)}/plots/time_lapse"
    maps = [os.path.join(map_path, f) for f in os.listdir(map_path) if f.endswith(".png")]
    logo = "/mnt/c/Users/Johannes/Pictures/logos/AC3-Logo-komplett_small.png"
    map_numbers = pd.read_csv(f"{gopro_dir}/{flight}_timestamps_sel.csv", index_col="datetime", parse_dates=True)
    f1, f2 = map_numbers.number.iloc[0], map_numbers.number.iloc[-1]
    files = [os.path.join(gopro_path, f) for f in os.listdir(gopro_path) if f.endswith(".JPG")][f1:f2-1]
    outpath = f"{gopro_dir}/{flight}_map"
    h.make_dir(outpath)
    processes = set()
    max_processes = 10
    # select single files to combine
    # maps = ["HALO-AC3_20220313_HALO_RF03_map_4224.png"]
    # maps = [os.path.join(map_path, f) for f in maps]
    # files = ["HALO-AC3_HALO_Gopro_20220313_RF03_4224.JPG"]
    # files = [os.path.join(gopro_path, f) for f in files]

    # %% add map to the right upper corner of the picture
    # for picture, map in zip(files[:1], maps[:1]):  # for testing
    # for picture in tqdm(files, desc="Add Map"):
    for picture, map in zip(tqdm(files, desc="Add Map"), maps):
        outfile = picture.replace(gopro_path, outpath)
        processes.add(Popen(['convert', picture,
                             map, '-geometry', '+3300+0',
                             '-composite', outfile]))
        if len(processes) >= max_processes:
            os.wait()
            processes.difference_update([p for p in processes if p.poll() is not None])

    print(f"Done with all files for {flight}")
