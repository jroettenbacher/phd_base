# PhD Base Project

*author: Johannes Röttenbacher*

Here I code all the stuff I need for my PhD on Arctic cirrus. 
It includes processing for measurement data gathered by the HALO aircraft.
During the first year a new python package was created which provides useful functions used in the processing and analysis of the data.
The package is called `pylim` and can be found on [GitHub](https://github.com/radiation-lim/LIM-pylim).
Further there are folders for different purposes:

* `analysis`: case studies or general campaign analysis to answer scientific questions
* `experiments`: scripts for ecRad/libRadtran experiments
* `processing`: scripts to generate error/bias corrected, calibrated and shareable/publishable files from raw files
* `quicklooks`: quicklooks of measurements (raw and calibrated) and of calibrations

The data is sorted by flight.
The exact paths are defined in `config.toml` and can be adjusted according to your setup.
More information on how to adjust this file can be found in [Setup](./docs/setup.rst).

Documentation can be found [here](https://jroettenbacher.github.io/phd_base/).


## FAQ

* What do negative measurements in the SMART dark current mean?

Answer: The conversion of the analog signal to a digital can lead to this.

# License

The content and the documentation of this project itself is licensed under the Creative Commons Attribution 4.0 International License, and the underlying source code used to format and display that content is licensed under the MIT license.


