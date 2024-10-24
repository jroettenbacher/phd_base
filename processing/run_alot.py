#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 25.03.2024

Run a certain script for a lot of different inputs

"""
import os
import subprocess
import numpy as np
from pylim import ecrad
import re

keys = ['RF17']
namelists = [f for f in os.listdir('/projekt_agmwend/data/HALO-AC3/08_ecrad/20220411') if '.nam' in f]
matches = [re.search(r"v\d{2}(\.[1,2])?", s) for s in namelists]
ecrad_versions = [m[0] for m in matches if m is not None]
ecrad_versions.remove('v10')
ecrad_versions.remove('v11')
ecrad_versions.remove('v12')
ecrad_versions.remove('v14')
for key in keys:
    date = '20220411' if key == 'RF17' else '20220412'
    for version in ecrad_versions:
        input_version = ecrad.get_input_version(version)
        print(os.getcwd())
        subprocess.run(
            ['python', 'ecrad_execute_IFS.py',
             '-k', key,
             '-v', version,
             '-i', input_version,
             '-d', date, ])
