#!/usr/bin/env python

import os
import matplotlib.pyplot as plt
import pandas as pd

date = 20210625
path = f"C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/ASP04/HARPDATA/{date}"

file = [f for f in os.listdir(path) if f.endswith('dat')]

df = pd.read_csv(f"{path}/{file[0]}", sep="\t", skipinitialspace=True, index_col="PCTIME", parse_dates=True)
start_id = 8000
end_id = start_id + 3600
fig, ax = plt.subplots()
df.iloc[start_id:end_id].plot(y="TARGET3", ax=ax, label="Roll Target")
df.iloc[start_id:end_id].plot(y="POSN3", ax=ax, label="Roll Position")
plt.grid()
plt.savefig(f"{path}/../plots/{date}_Roll_target-position.png", dpi=100)
plt.show()
plt.close()

fig, ax = plt.subplots()
df.iloc[start_id:end_id].plot(y="TARGET4", ax=ax, label="Pitch Target")
df.iloc[start_id:end_id].plot(y="POSN4", ax=ax, label="Pitch Position")
plt.grid()
plt.savefig(f"{path}/../plots/{date}_Pitch_target-position.png", dpi=100)
plt.show()
plt.close()