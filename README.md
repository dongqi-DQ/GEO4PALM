# GEO4PALM


[![DOI](https://zenodo.org/badge/327139863.svg)](https://zenodo.org/badge/latestdoi/327139863)


GEO4PALM is a Python tool that lets PALM users to download and preprocess geospatial data easier. GEO4PALM accepts all geospatial input files in geotiff or shp format. Once users have their own input data ready, GEO4PALM can convert such input data into PALM static driver. The instruction below works for Linux operation system. For Windows and MacOS, some minor adjustments may need to be done by users themselves.

**Citation:**  
Lin, D., Zhang, J., Khan, B., Katurji, M., and Revell, L. E.: GEO4PALM v1.1: an open-source geospatial data processing toolkit for the PALM model system, Geosci. Model Dev. Discuss. [preprint], https://doi.org/10.5194/gmd-2023-150, in review, 2023.

# Table of Contents
1. [Python environment](#python-environment)
2. [PALM domain utility](#palm-domain-utility)
3. [Online data sets](#dont-have-your-own-data-sets)
4. [How to run GEO4PALM?](#how-to-run-geo4palm)

## Getting Started

### Python environment

As Python packages can have mismatches between each other, we provide a `geo4palm_env.yml` file so that users can create an environemnt for GEO4PALM as follows:

`conda env create -f geo4palm_env.yml`

Activate the environment with:

`conda activate geo4palm`

To install `terracatalogueclient`, users need to follow the following steps:  
1. locate the `pip` tool in the geo4palm environment.Try `which -a pip` which gives something like:
```
/usr/bin/pip
/home/user/miniconda3/envs/geo4palm/bin/pip
/bin/pip
```
2. Copy the pip path for the geo4palm environment (namely, `/home/user/miniconda3/envs/geo4palm/bin/pip` in this example). Then try:

`<pip path> install terracatalogueclient -i https://artifactory.vgt.vito.be/api/pypi/python-packages/simple`

e.g. 

```
/home/user/miniconda3/envs/geo4palm/bin/pip install terracatalogueclient -i https://artifactory.vgt.vito.be/api/pypi/python-packages/simple
```

TerraCatalogue client is for ESA land use API. More information can be found [here](https://vitobelgium.github.io/terracatalogueclient/installation.html).

## PALM domain utility
Users may visualise domain by running `palm_domain_utility.py` via `panel`. This can be done on a local machine and/or a remote server.

### Local machine

On local machines, use the command below:

```
panel serve --port 8081 palm_domain_utility.py --show
```
The terminal will return information as follows:

```
2023-06-02 14:15:42,168 Starting Bokeh server version 3.1.1 (running on Tornado 6.2)
2023-06-02 14:15:42,334 User authentication hooks NOT provided (default user enabled)
2023-06-02 14:15:42,336 Bokeh app running at: http://localhost:8081/palm_domain_utility
```

The port number (8081) can be any number assigned by useres. If the port option `--port xxxx` is emitted, the program will choose a random port number to use. With the `--show` option, the web-based GUI will automatically pop up via the default web browser in the user's environment. Alternatively, users can emit the `--show` optoin and copy and paste the url `http://localhost:xxxx/palm_domain_utility` in their prefered browser to access the GUI.

### Remote server
To access the PALM domain utility via a remote server, users need to first assign a port number for the purpose of **port forwarding**. For example, if one aims to use the port 9821 to communicate, then the user wants to connect to a remote server that is listening on port 9821 via ssh using this command:
```
ssh -L 9821:localhost:9821 [ssh_host]
```
`[ssh_host]` is the address/name of the remote server. Note that if you are using port forwarding on a regular basis, and don't want the hassle of opening a new tunnel every time, you can include a port forwarding line in your ssh config file ~/.ssh/config on your local machine. Under the alias for the server, add the following lines (full tutorial refer to [here](https://support.nesi.org.nz/hc/en-gb/articles/360001523916-Port-Forwarding)):
```
LocalForward <local_port> <host_alias>:<remote_port>
ExitOnForwardFailure yes
```

After setting up the port forwarding, use the command below:

```
panel serve --port 9821 palm_domain_utility.py
```
Remember to change the port number that assigned earlier in ssh. To access the GUI, copy and paste the url `http://localhost:xxxx/palm_domain_utility` in a web browser.

*NB: For advanced users, who want to use a local port number different to the remote port number in port forwarding, you need to add `--allow-websocket-origin localhost:xxxx` in the panel command line, where "xxxx" is the local port number.*

### How to use the GUI?
![domain_utility_screenshot](domain_utility_screenshot.png)

#### Create domains. 
1. In order to create a domain, you can prescribe `center lat`, `center lon`, grid numbers (`nx`,`ny`,`nz`) and grid resolution (`dx`,`dy`,`dz`). You can either enter the local epsg code (4 digits) in `local projection (epsg)` manually or leave it empty (the utility can automatically find/fill the best suitable UTM coordinate for you).

2. Click `Add to map` button to add the domain onto the interactive map widget (`Domain Vis`).
    > * The utility automatically does overlap-check when adding a new domain. To prevent the check, you can tick **Allow overlap** option. This function is only provided so you can better see how much you need to change your domain to avoid the overlap. The utility won't be able to generate configure file when there are overlaps between any of the domains.
    > * **undo** button, you can undo/remove the last added domain. Only works before you click "Get domain configure". Once you have generated domain configure, you need to use `Remove domain` function instead to remove any domains.

3. Repeat these two steps to create as many domains as you need.

4. `Get domain configure`. Once you have created some domains, click this button to get the configure text needed for both the GEO4PALM config file (`Static namelist Config`) and  &nesting_parameters section in the PALM namelist (`PALM namlist`).

5. Copy and paste the configuration lines for GEO4PALM and PALM.
    > * When you click `Get domain configure`, the utility will move the domains slightly if needed to comply with the parent-child domain boundary requirement.
    > * A domain list table will also be created under `Domain list` tab to show the information of all domains including the domain number and parent domain number.
#### Modify domains.
1. Move the domain. Use `Move Domain` tab if you want to move any domain. Simply enter the `domain number` (displayed on the interactive map) you want to move, and distance you want to move in meter in `east_west_move (m)` or/and `south_north_move (m)`. Positive values means move from east (south) to west (north), vice versa for the negative values. Click the `move domain` button to make the change.
2. Remove the domain. Use `Remove domain` tab if you want to remove any domain. Simply enter the domain number in `Remove Domain number` and click `Remove` button.
    > You can still use the functions mentioned in the previous section to add more domains. Just remember to click "Get domain configure" to allow the utility to adjust/finalize the domain configure text before you use it.
#### Import domain configure.
If you already have an existing GEO4PALM configure file and would like to visualize the domain. Simply copy the [domain] section to the `Static namelist Congfig` and click `check configuration`. This will import the domain setup into the utility. Below is an example of a [domain] section.
```
[domain]
ndomain = 1,
centlat = -43.00000, 
centlon = 172.00000, 
nx = 100, 
ny = 100, 
nz = 100, 
dx = 10.0, 
dy = 10.0, 
dz = 10.0, 
ll_x = 0.0, 
ll_y = 0.0, 
z_origin  = 0.0, 
```
b. Click `check configuration` to visualise the domains.

> * You should enter the local projection in `local projection (epsg)`as well. Otherwise, the utility will automatically find the most suitable UTM for you.
> * After import, you will also be able to use all the functionalities to add/modify the domains.

## Don't have your own data sets? 

### GEO4PALM downloads online data for you

GEO4PALM provides several interfaces for the basic features of PALM static driver including:

1. NASA Earthdata digital elevation model (DEM; 30 m resolution; global)
2. NASA land use classification data sets (resolution may vary depeonding on the data set selected)
3. ESA WorldCover land use classification (10 m resolution; global)
4. OpenStreetMap (OSM) buildings and pavements/streets
5. GHRSST Level 4 MUR product for sea surface temperatuer (SST)

### How do I download data using GEO4PALM?

In the GEO4PALM input configuration file, users can either specify the input geospatial data filename or specify:
1. `"nasa",` to download and process data via NASA Earthdata interface
3. `"esa",` to donwload and process data via ESA WorldCover interface
4. `"osm",` to download and process data via OSMnx
5. `"online",` to downlaod and process SST data via NASA Earthdata and OPeNDAP interface

**Note:**
1. Register to download data from NASA Earthdata [here](https://www.earthdata.nasa.gov/eosdis/science-system-description/eosdis-components/earthdata-login) if you haven't

2. Register to download data from ESA WorldCover [here](https://esa-worldcover.org/en!) if you haven't. If you have trouble finding where to register, it might be easier to register from this link [here](https://terrascope.be/en)

3. Registration not required for OSM data. We use [OSMnx](https://github.com/gboeing/osmnx) package


## How to run GEO4PALM?

Download the entire code to your local directory.

In the master directory, you will find the main script `run_config_static.py`. To run this script, a configuration file is required. The configuration file for each case should be `$master_directory/JOBS/case_name/INPUT/config.static-case_name`.

#### Preparing the configuration file 

The configuration file requires PALM domain configuration and geotiff filenames from users. The domain configuration is similar to variables used in PALM. 

Users must specify:
```
[case]
case_name         -  name of the case 
origin_time       -  date and time at model start*
default_proj      -  default is EPSG:4326. This projection uses lat/lon to locate domain. This may not be changed.
config_proj       -  projection of input tif files. GEO4PALM will automatically assign the UTM zone if not provided.
                     We recommend users use local projection with units in metre, e.g. for New Zealand users, EPSG:2193 is a recommended choice.

lu_table          -  land use look up table to convert land use classification to PALM recognisable

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

[settings]
water_temperature      -  user input water temperature values when no water temperature data is available
building_height_dummy  -  user input dummy height for buildings where building heights are missing in the 
                          OSM data set or if building heights are 0.0 m in the input data 
tree_height_filter     -  user input to filter small objects, i.e., if object height is smaller than this value
                          then this object is not included in the LAD estimation


[geotif]          -  required input from user; can be provided by users in the INPUT folder or "online"
water             -  input for water temperature
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
tree_lad_max      - input value for maximum leaf area density (LAD)
lad_max_height    - input value for the height where the leaf area index (LAI) reaches leave area density (LAD) 
sfch              - input for plant height; this is for leave area density (LAD)
```
_**Note**: The `origin_time` setting is similar to `origin_date_time` in [PALM documentation](https://palm.muk.uni-hannover.de/trac/wiki/doc/app/initialization_parameters#origin_date_time). This variable is required in static drivers, but will not be used in PALM simulation. Rather the date time should be specified in PALM's p3d namelist. The sunset/sunrise time is affected by lat/lon attributes in the static driver._

#### Preparing lookup table for land use typology

To convert land use classifcation to PALM-recognisable types, a lookup table (see files in `util/lu_csv` folder) is required. Here we provided the lookup tables for 
- New Zealand Land Cover Data Base (LCDB) v5.0: `nzlcdb_2_PALM_num.csv` 
- Collection 6 MODIS Land Cover Type 1: `NASA_LC_type1_PALM_num.csv`
- ESA WorldCover 2020 v1: `esa_2020v1_lu.csv`
- German Space Agency (DLR) data sets: `dlr_lu.csv`

Before running GEO4PALM, the corresponding csv file for land use type conversion should be put in the `INPUT` folder. Otherwise, GEO4PALM uses the default csv file `util/lu_2_PALM_num.csv`.

#### Urban surface and plant canopy

For urban and plant canopy tif file fileds, if users do not have files available, they should leave the file names empty as `"",`. If a user desires to use data from OSM (OpenStreetMap), please leave the field as `"osm",`. Building footprint, building height, building ID, pavement type, and street type will be derived from OSM data. For buildings with no height information available, a dummy value `building_height_dummy` should be given in the configuration file.

Configuration file examples are given in `JOBS/Chch_online/INPUT/` and  `JOBS/Berlin_DLR/INPUT/`

#### Water temperature
If `"online"` is used for `water`, the water temperature is derived from GHRSST Level 4 MUR product downloaded via NASA Earthdata. The SST at the nearest grid point will be used for water temperature. The day of the year is derived from `origin_time` in the configuration file. The location to take SST data depends on `centlat` and `centlon` in the namelist.

Users can provide a prescribed water temperature using `water_temperature` in `[settings]` for each simulation domain or provide a spatial tif file with water temperature for water bodies. 

#### Input tif files explained

GEO4PALM only supports input files in tif format. All tif files must be put in `$master_directory/INPUT/` with filename specified in the configuration file for the desired field and simulation domain. GEO4PALM has no requirements on data source, projection, and file size. Users do not need to preprocess tif files into specific resolution or projection for the configured domains. GEO4PALM will process all INPUT tif and store temporary tif files for each simulation domain in `$master_directory/TMP`. All static driver files will be stored in `$master_directory/OUTPUT`. All the input files specified in the configuration file will be processed by GEO4PALM into PALM static driver based on the projection and grid spacing given in the configuration file. 

For those who have shp files, we provide a small tool to convert shp files to tif files `shp2tif.py`. 

_How to use `shp2tif.py`?_
```
python shp2tif.py  [case_name] [shp file path] [variable_name]
```

`shp2tif.py` converts shp file into the finest resolution configured in the configuration file.
**Note:** 
1. Converting big shp files may require a large amount of RAM.
2. LAD calculation only allows fixed values of `tree_lai_max ` and `lad_max_height` at the moment due to limited data availability. Future development will include spatial variation in LAD. Usres are welcome to modify the code based on their data availability.
3. At present only one building type is assigned for all buildings. Users are welcome to modify GEO4PALM if various building types are required.   
4. Variables in the static driver here may not be inclusive. Users may refer to PALM input data standard or Heldens et al. (2020).

_Heldens, W., Burmeister, C., Kanani-Sühring, F., Maronga, B., Pavlik, D., Sühring, M., Zeidler, J., and Esch, T.: Geospatial input data for the PALM model system 6.0: model requirements, data sources and processing, Geosci. Model Dev., 13, 5833–5873, https://doi.org/10.5194/gmd-13-5833-2020, 2020._


### Run the main script

Once the configuration file and all tif input from users are ready. One can run the script in the main GEO4PALM directory:
```
python run_config_static.py case_name
```

_If "nasa" is used for `dem` and/or `lu` and/or "online" is used for `water`, the script will guide the user through the NASA AρρEEARS API. If "esa" is included for `dem` and/or `lu`, then the script will guide the user through ESA's Terrascope API._

### Flat terrain and precursor run 
Once a static driver is used, all the PALM domains in the simulation requires static drivers. In case a flat terrain static driver and/or precursor run static driver are required, users may run `static2flat.py`. 
```
python static2flat.py [static_file] [nx,ny]
```
Note that this requires no urban variables (e.g. buildings and streets) in the input static driver. If precursor run is not required, users do not need to specify `nx` and `ny`. This script can be found in `$master_directory/util/tools/`


## Do you have any questions or issues? 

You are welcome to raise them in the GitHub issue system.


--------------------------------------------------------------------------------------------  
We have been trying to add more comments and more instructions of the scripts. However, the documentation may not be sufficiently inclusive. If there is anything unclear, please do not hesitate to contact us. 

Dongqi Lin (dongqi.lin@canterbury.ac.nz)  
Jiawei Zhang (Jiawei.Zhang@scionresearch.com)  

@ Centre for Atmospheric Research, University of Canterbury

--------------------------------------------------------------------------------------------
## End of README


