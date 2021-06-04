# PALM_static

Scripts to create static files for PALM simulation.

1. Change parameters in `namelist.static`.
2. Run `create_cfg_static.py`
3. Run `visualise_PALM_domains.py` to visualise PALM domains. 

Note that the tiff files (put in `raw_static`) must be in WGS84 projection (ESPG:4326) with the desired resolution of PALM domains. 

Static files can be find in `static_files`.

Remember to change filenames etc when visulising. 
