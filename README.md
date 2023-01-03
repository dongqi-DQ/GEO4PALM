# GEO4PALM

GEO4PALM is a Python tool that lets PALM users to download and preprocess geospatial data easier. GEO4PALM accepts all geospatial input files in geotiff or shp format. Once users have their own input data ready, GEO4PALM can convert such input data into PALM static driver. 

## Getting Started

### Don't have your own data sets? 

GEO4PALM provides several interfaces for the basic features of PALM static driver including:

1. NASA Earthdata digital elevation model (DEM; 30 m resolution; global)
2. NASA land use classification data sets (resolution may vary depeonding on the data set selected)
3. ESA WorldCover land use classification (10 m resolution; global)
4. OpenStreetMap (OSM) buildings and pavements/streets

### How do I download data using GEO4PALM?

In the GEO4PALM input namelist, users can either specify the input geospatial data filename or specify:
1. `"nasa",` to download and process data via NASA Earthdata interface
3. `"esa",` to donwload and process data via ESA WorldCover interface
4. `"osm",` to download and process data via OSMnx

**Note:**
1. Register to download data from NASA Earthdata [here](https://www.earthdata.nasa.gov/eosdis/science-system-description/eosdis-components/earthdata-login) if you haven't
2. Register to download data from ESA WorldCover [here](https://esa-worldcover.org/en!) if you haven't
3. Registration not required for OSM data. We use [OSMnx](https://github.com/gboeing/osmnx) package

### Have questions or issues?

You are welcome to ask it on the GitHub issue system. 

### How to run?

Download the entire code to your local directory.

In the master directory, you will find the main script `run_config_static.py`. To run this script, a namelist file is required. The namelist for each case should be `$master_directory/JOBS/case_name/INPUT/namelist.static-case_name`.

#### Preparing namelist 

The namelist requires PALM domain configuration and geotiff filenames from users. The domain configuration is similar to variables used in PALM. 

Users must specify:
```
[case]
case_name         -  name of the case 
origin_time       -  date and time at model start*
default_proj      -  default is EPSG:4326. This projection uses lat/lon to locate domain. This may not be changed.
config_proj       -  projection of input tif files. GEO4PALM will automatically assign the UTM zone if not provided.
                     We recommend users use local projection with units in metre, e.g. for New Zealand users, EPSG:2193 is a recommended choice.

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

[urban]           - input for urban canopy model; can leave as "" if this feature is not included in the simulations, or provided by user; or online from OSM
bldh              - input for building height 
bldid             - input for building ID
pavement          - input for pavement type
street            - input for building ID

[plant]           - input for plant canopy model; can leave as "" if this feature is not included in the simulations, or provided by user
sfch              - input for plant height; this is for leave area density (LAD)
```
_**Note**: The `origin_time` setting is similar to `origin_date_time` in [PALM documentation](https://palm.muk.uni-hannover.de/trac/wiki/doc/app/initialization_parameters#origin_date_time). This variable is required in static drivers, but will not be used in PALM simulation. Rather the date time should be specified in PALM's p3d namelist. The sunset/sunrise time is affected by lat/lon attributes in the static driver._

#### Preparing lookup table for land use typology

To convert land use classifcation to PALM-recognisable types, a lookup table (see files in `util/lu_csv` folder) is required. Here we provided the lookup tables for 
- New Zealand Land Cover Data Base (LCDB) v5.0: `nzlcdb_2_PALM_num.csv` 
- Collection 6 MODIS Land Cover Type 1: `NASA_LC_type1_PALM_num.csv`
- ESA WorldCover 2020 v1: `esa_2020v1_lu.csv`

Before running GEO4PALM, link the corresponding csv file to `util/lu_2_PALM_num.csv`:
```
# In util/
ln -sf lu_csv/your_csv lu_2_PALM_num.csv
```

#### Urban surface and plant canopy

For urban and plant canopy tif file fileds, if users do not have files available, they should leave the file names empty as `"",`. If a user desires to use data from OSM (OpenStreetMap), please leave the field as `"osm",`. Building footprint, building height, building ID, pavement type, and street type will be derived from OSM data. For buildings with no height information available, a dummy value of 3 m is given.

A namelist example is given in `JOBS/Christchurch/INPUT/` folder 

#### Input tif files explained

GEO4PALM only supports input files in tif format. All tif files must be put in `$master_directory/INPUT/` with filename specified in the namelist for the desired field and simulation domain. GEO4PALM has no requirements on data source, projection, and file size. Users do not need to preprocess tif files into specific resolution or projection for the configured domains. GEO4PALM will process all INPUT tif and store temporary tif files for each simulation domain in `$master_directory/TMP`. All static driver files will be stored in `$master_directory/OUTPUT`. All the input files specified in the namelist will be processed by GEO4PALM into PALM static driver based on the projection and grid spacing given in the namelist. 

For those who have shp files, we provide a small tool to convert shp files to tif files `shp2tif.py`. 

_How to use `shp2tif.py`?_
```
python shp2tif.py  [case_name] [shp file path] [variable_name]
```

`shp2tif.py` converts shp file into the finest resolution configured in the namelist.
**Note:** 
1. Converting big shp files may require a large amount of RAM.
2. At present only one building type is assigned for all buildings. Users are welcome to modify GEO4PALM if various building types are required.   
3. Variables in the static driver here may not be inclusive. Users may refer to PALM input data standard or Heldens et al. (2020).

_Heldens, W., Burmeister, C., Kanani-Sühring, F., Maronga, B., Pavlik, D., Sühring, M., Zeidler, J., and Esch, T.: Geospatial input data for the PALM model system 6.0: model requirements, data sources and processing, Geosci. Model Dev., 13, 5833–5873, https://doi.org/10.5194/gmd-13-5833-2020, 2020._


### Run the main script

Once the namelist and all tif input from users are ready. One can run the script:
```
python run_config_static.py case_name
```

_If "nasa" is used for `dem` and/or `lu`, the script will guide the user through the NASA AρρEEARS API. If "esa" is included for `dem` and/or `lu`, then the script will guide the user through ESA's Terrascope API._

### Visualise domain on OSM 
Users may visualise domain by running `visualise_domains.py` or `visualise_terrain.py`:
```
python visulalise_domains.py [namelist_file]
```
or 
```
python visulalise_terrain.py [namelist_file]
```
This can be done before static files are created. The two scripts are similar, while the former displays domains using `IPython` (pop-up Python image window) and the later displays domains in a web browser. As the domain visualisation images are downloaded from two different online sources, the downloading speed may vary between the two scripts. Users can opt between the two based on their own preferences. 

### Flat terrain and precursor run 
Once a static driver is used, all the PALM domains in the simulation requires static drivers. In case a flat terrain static driver and/or precursor run static driver are required, users may run `static2flat.py`. 
```
python static2flat.py [static_file] [nx,ny]
```
Note that this requires no urban variables (e.g. buildings and streets) in the input static driver. If precursor run is not required, users do not need to specify `nx` and `ny`. This script can be found in `$master_directory/util/tools/`


### Water temperature
If "online" is used for `sst`, the water temperature is derived from UKMO daily SST data downloaded from OPeNDAP. The SST at the nearest grid point will be used for water temperature. The day of the year is derived from `origin_time` in the namelist. The location to take SST data depends on `centlat` and `centlon` in the namelist.


--------------------------------------------------------------------------------------------  
We have been trying to add more comments and more instructions of the scripts. However, the documentation may not be sufficiently inclusive. If there is anything unclear, please do not hesitate to contact us. 

Dongqi Lin (dongqi.lin@canterbury.ac.nz)  
Jiawei Zhang (Jiawei.Zhang@scionresearch.com)  

@ Centre for Atmospheric Research, University of Canterbury

--------------------------------------------------------------------------------------------
## End of README


