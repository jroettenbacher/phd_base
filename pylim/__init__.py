#!/usr/bin/env python
"""Init file for pylim a package designed to work with measurement data for the radiation group at LIM
author: Johannes Röttenbacher
"""
from pathlib import Path
# check if the config file can be found in the project directory
project_dir = Path(__file__).resolve().parent.parent
config_file = project_dir / "config.toml"
assert config_file.is_file(), f"No config.toml file found in {project_dir}! Needs to be provided"
