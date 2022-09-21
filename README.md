# PALM_static
**This documentation is now under development (20/09/2022)**

The geotiff files are important input for static driver. However, it is impossible to process all the geoinformation in a standard way. Here we present scripts to genearte static drivers for PALM simulation. We provide an interface for users to download geospatial data globally, while users can also provide their own geospatial data in `tif` format. The script will prepare all input files for the configured simulation domains and then generate static drivers. Hopefully these tools can make PALM users' lives easier.


Note: Users need to be registered to download data from NASA Earthdata Enter NASA Earthdata 
(Register here if you haven't, https://www.earthdata.nasa.gov/eosdis/science-system-description/eosdis-components/earthdata-login)


## How to run?
The main script is `run_config_static.py`. To run this script, a namelist file is required. The namelist for each case should be  `JOBS/case_name/INPUT/namelist.static-case_name`.

### namelist 
The namelist requires PALM domain configuration and geotiff filenames from users. The domain configuration is similar to variables used in PALM. 

Users must specify:
```
[case]
case_name         -  name of the case 
origin_time       -  date and time at model start*
default_proj      -  default is EPSG:4326. This projection uses lat/lon to locate domain. This may not be changed.
config_proj          -  projection of input tif files. We recommend users use local projection with units in metre, e.g. for New Zealand users, EPSG:2193 is a recommended choice.

[domain]
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

[geotif]          -  required input from user; can be provided by users in the INPUT folder or "online"
sst               -  input for water temperature
dem               -  digital elevation model input for topography
lu                -  land use classification  
resample_method   -  method to resample geotiff files for interpolation/extrapolation

# if NASA API is used format in YYYY-MM-DD
# SST date should be the same as the orignin_time

## No need to change start/end dates for NASA SRTMGL1_NC.003 
dem_start_date = '2000-02-12',  
dem_end_date = '2000-02-20',
## start/end dates for land use data set
lu_start_date = '2020-10-01',
lu_end_date = '2020-10-30',

[urban]             - input for urban canopy model; can leave as "" if this feature is not included in the simulations, or provided by user; or online from OSM
bldh                - input for building height 
bldid               - input for building ID
pavement            - input for pavement type
street              - input for building ID

[plant]           - input for plant canopy model; can leave as "" if this feature is not included in the simulations, or provided by user
sfch              - input for plant height; this is for leave area density (LAD)
```

**below needs to be edited (20/09/2022)**

To convert land use classifcation to PALM-recognisable types, a lookup table (in `util/lu_csv` folder) is required. Here we provided the lookup tables for 
- New Zealand Land Cover Data Base (LCDB) v5.0: `nzlcdb_2_PALM_num.csv` 
- Collection 6 MODIS Land Cover Type 1: `NASA_LC_type1_PALM_num.csv`

Before running the code (hereafter GEO4PALM?), link the corresponding csv file to `util/lu_2_PALM_num.csv`:
```
# In util/
ln -sf lu_csv/your_csv lu_2_PALM_num.csv
```

_The `origin_time` setting is similar to `origin_date_time` in [PALM documentation](https://palm.muk.uni-hannover.de/trac/wiki/doc/app/initialization_parameters#origin_date_time). This variable is required in static drivers, but will not be used in PALM simulation. Rather the date time should be specified in PALM's p3d namelist. The sunset/sunrise time is affected by lat/lon attributes in the static driver._

**Note: when no urban input is used, the vegetation type is set to 18 and the albedo type is set to 33 for urban area specified in land use files.**

For urban and plant canopy tif file fileds, if users do not have files available, they should leave the file names empty as `"",`. If a user desires to use data from OSM (OpenStreetMap), please leave the field as "online". Building footprint, building height, building ID, pavement type, and street type will be derived from OSM data. For buildings with no height information available, a dummy value of 3 m is given.


A namelist example is given in `JOBS/prefix/INPUT/` folder [To Do: probably need to give two examples - with/without urban canopy]

#### input tif files explained
GEO4PALM only supports input files in tif format. We provide a small tool to convert shp files to tif files `shp2tif.py`. 

Users do not have to provide tif files with specific resolution for the configured domains. We have a prepareation interface that will process all INPUT tif and store temporary tif files for each simulation domain in TMP. All static driver files will be stored in OUTPUT.

**Note: at present one building type is assigned for all buildings.**   
  
Variables in the static driver here may not be inclusive. Users may refer to PALM input data standard or Heldens et al. (2020).

_Heldens, W., Burmeister, C., Kanani-Sühring, F., Maronga, B., Pavlik, D., Sühring, M., Zeidler, J., and Esch, T.: Geospatial input data for the PALM model system 6.0: model requirements, data sources and processing, Geosci. Model Dev., 13, 5833–5873, https://doi.org/10.5194/gmd-13-5833-2020, 2020._


### run the main script
Once the namelist and all tif input from users are ready. One can run the script:
```
python run_config_static.py case_name
```
If "online" is used for `dem` and/or `lu`, the script will guide the user through the NASA AρρEEARS API. [To Do: users should be able to provide some auth file so they don't have to type username and password every time] 



### visualise domain on OSM 
Users may visualise domain by running `visualise_domains.py`:
```
python visulalise_PALM_domains.py [namelist_file]
```
This can be done before static files are created.

### flat terrain and precursor run 
Once a static driver is used, all the PALM domains in the simulation requires static drivers. In case a flat terrain static driver and/or precursor run static driver are required, users may run `static2flat.py`. 
```
python static_to_flat.py [static_file] [nx,ny]
```
Note that this requires no urban variables (e.g. buildings and streets) in the input static driver. If precursor run is not required, users do not need to specify `nx` and `ny`.


### water temperature
If "online" is used for `sst`, the water temperature is derived from UKMO daily SST data downloaded from OPeNDAP. The nearest SST will be used for water temperature. The day of the year is derived from `origin_time` in the namelist. The location to take SST data depends on `centlat` and `centlon` in the namelist.



--------------------------------------------------------------------------------------------  
We have been trying to add more comments and more instructions of the scripts. However, if there is anything unclear, please do not hesitate to contact us. 

Dongqi Lin (dongqi.lin@canterbury.ac.nz)  
Jiawei Zhang (Jiawei.Zhang@scionresearch.com)  

@ Centre for Atmospheric Research, University of Canterbury

--------------------------------------------------------------------------------------------
## End of README












