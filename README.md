# PALM_static

The geotiff files are important input for static driver. However, it is impossible to process all the geoinformation in a standard way. Here we present scripts to prepare tif files (`prep_static`) and create static files (`create_static`) for PALM simulation. Hopefully these tools can make PALM users' life easier.

## create PALM static driver
In `create_static` folder, the main script is `run_config_static.py`. To run this script, these input files are required:  
1. namelist.static
2. geotiff files for each domain

### namelist 
The namelist requires PALM domain configuration and geotiff filenames from users. The domain configuration is similar to variables used in PALM.  

Users must specify:
```
case_name         -  case names for all domains  
ndomain           -  maximum number of domains, when >=2, domain nesting is enabled  
centlat, centlon  -  centre latitude and longitude of the first domain. Note this is not required for nested domains  
nx                -  number of grid points along x-axis  
ny                -  number of grid points along y-axis  
nz                -  number of grid points along z-axis  
dx                -  grid spacing in meters along x-axis  
dy                -  grid spacing in meters along y-axis  
dz                -  grid spacing in meters along z-axis  
z_origin          -  elevated terrain mean grid position in meters (leave as 0.0 if unknown)  
ll_x              -  lower left corner distance to the first domain in meters along x-axis   
ll_y              -  lower left corner distance to the first domain in meters along y-axis   

dem               -  digital elevation model tif file name (for topography)  
bldh              -  building height tif file name  
bldid             -  building ID tif file name  
lu                -  land use classification tif file name  
sfch              -  surface objects height (excluding buildings) tif file name  
pavement          -  pavement type tif file name  
street            -  street type tif file name  
```

The **required** fields for tif files are `dem` and `lu`. A lookup table (in `raw_static` folder) is required to convert land use information to PALM recognisable types. Here we used New Zealand Land Cover Data Base (LCDB) v5.0. Our lookup table `nzlcdb_2_PALM_num.csv` is available in `raw_static` folder. 

For other tif file fileds, if users do not have files available, they should leave the file names empty as `"",`. The script will automatically read the "empty" tif file (`empty.tif`) provided in `raw_static`. 

Note that if the provided `empty.tif` causes any error (usually due to insufficient grid cells). Users may create their own empty tif file based on their own tif files using `create_empty.py`:
```
python create_empty.py [input tif file]
```

#### input tif files explained
We processed our own geotiff files using the GIS tools before using the python scripts here.  
- `bldh` refers to building height. This is calculated using the difference between digital surface model (DSM) and DEM. The building height is extracted using OpenStreet Map (OSM) building outlines.  
- `bldid` refers to buliding ID (available in OSM).   
- `street`refers to street type (available in OSM).  
- `sfch` refers to surface object height excluding buildings. This is calculated using the difference between digital surface model (DSM) and DEM. Buildings are excluded using building outlines available in OSM.  

Note that building type information is not available in New Zealand, and hence one building type is assigned for all buildings.   
  
Variables in the static driver here are not inclusive. Users may refer to PALM input data standard or Heldens et al. (2020).

_Heldens, W., Burmeister, C., Kanani-Sühring, F., Maronga, B., Pavlik, D., Sühring, M., Zeidler, J., and Esch, T.: Geospatial input data for the PALM model system 6.0: model requirements, data sources and processing, Geosci. Model Dev., 13, 5833–5873, https://doi.org/10.5194/gmd-13-5833-2020, 2020._


### geotiff files requirements
- Users may put their geotiff files in `create_static/raw_static`. 
- The geotiff files must have the same projection. 
- The geotiff files must have the same resolution as desired in PALM simulation, e.g. for a 10 m simulation, the geotiff files resolution must be 10 m. 

Users have their own geotiff files ready but the resolution and/or projection do not satisfy the requirements. We provide a python script `prep_tif.py` to reproject and resample geotiff files in `prep_static` folder.   
Users may provide their own tif files in `prep_static/tiff/` and run `prep_tif.py` for repreojection and resample:  
```
python prep_tif.py [infile] [out EPSG projection] [outfile prefix] [resolution list]
```

Once all geotiff files are ready, they can be linked from `prep_static/tiff` into `create_static/raw_static`:
```
ln -sf prep_static/tiff/*.tif create_static/raw_static/.
```

### run the main script
Now if users have all geotiff files ready, they may run the main script:
```
python run_config_static.py
```

The script should print some processing information and create the desired static files, which can be found in `static_files`. Each domain will also have 
1. its own geotiff file created in `static_files` for georeferences.
2. its own cfg file created in `cfg_files` for future reference, e.g. in WRF4PALM.

### visualise domain on OSM
Users may visualise domain by running `visualise_PALM_domains.py`:
```
python visulalise_PALM_domains.py
```

### flat terrain and precursor run
Once a static driver is used, all the PALM domains in the simulation requires static drivers. In case a flat terrain static driver and/or precursor run static driver are required, users may run `static_to_flat.py`. 
```
python static_to_flat.py [static_file] [nx,ny]
```

Note that this requires no urban variables (e.g. buildings and streets) in the input static driver. If precursor run is not required, users do not need to specify `nx` and `ny`.

#--------------------------------------------------------------------------------------------#
We have been trying to add more comments and more instructions of the scripts. However, if there is anything unclear, please do not hesitate to contact us. 

Dongqi Lin (dongqi.lin@pg.canterbury.ac.nz)
Jiawei Zhang (jiawei.zhang@canterbury.ac.nz)

@ Centre for Atmospheric Research, University of Canterbury

#--------------------------------------------------------------------------------------------#
# End of README












