# PALM_static

The geotiff files are important input for static driver. However, it is impossible to process all the geoinformation in a standard way. Here we present scripts to prepare tif files (`prep_static`) and create static files (`create_static`) for PALM simulation. Hopefully these tools can make PALM users' life easier.

## create PALM static driver
In `create_static` folder, the main script is `run_config_static.py`. To run this script, these input files are required:  
1. namelist.static
2. geotiff files for each domain

### namelist 
The namelist requires PALM domain configuration and geotiff filenames from users. The domain configuration is similar to variables used in PALM.  

Users must specify:

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

The **required** fields for tif files are `dem` and `lu`. A lookup table (in `raw_static` folder) is required to convert land use information to PALM recognisable types. Here we used New Zealand Land Cover Data Base (LCDB) v5.0. Our lookup table `nzlcdb_2_PALM_num.csv` is available in `raw_static` folder.

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

Once all geotiff files are ready, they can be linked into `create_static/raw_static`. 


This part requires users to have all tif files ready. The tif files must have the same projection 

1. Change parameters in `namelist.static`.
2. Run `create_cfg_static.py`
3. Run `visualise_PALM_domains.py` to visualise PALM domains. 

Note that the tiff files (put in `raw_static`) must be in WGS84 projection (ESPG:4326) with the desired resolution of PALM domains. 

Static files can be find in `static_files`.

Remember to change filenames etc when visulising. 

## Part 1: prepare tif files



