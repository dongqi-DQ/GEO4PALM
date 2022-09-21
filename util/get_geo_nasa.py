import requests
import getpass, pprint, time, os, cgi, json
import geopandas as gpd
import os
from datetime import datetime, timezone
import pandas as pd
import sys
from pyproj import Transformer, CRS
import warnings
from shapely.geometry import box, Point
from pathlib import Path
import configparser
import ast
import warnings
## supress warnings
## switch to other actions if needed
warnings.filterwarnings("ignore")

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def date_validate(date_text,fmt='%Y-%m-%d'):
    try:
        datetime.strptime(date_text, fmt)
    except ValueError:
        print("Date format is not correct!")
        return False
    return True
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def login_earthdata(api):
    # login to the APPEEARS, return the response token
    user_credential = False
    while not user_credential:
        user = input('Enter NASA Earthdata Login Username: \n(Register here if you haven\'t, '\
                     'https://www.earthdata.nasa.gov/eosdis/science-system-description/eosdis-components/earthdata-login)'\
                     '\nUsername:')      # Input NASA Earthdata Login Username
        password = getpass.getpass(prompt = 'Enter NASA Earthdata Login Password: ')  # Input NASA Earthdata Login Password
        res_token = requests.post('{}login'.format(api), auth=(user, password)).json() # Insert API URL, call login service, provide credentials & return json
        del user, password                                                           # Remove user and password information
        ### check if the token is valid
        try:
            utc_dt = datetime.now(timezone.utc) # UTC time to check if the token hasn't expired
            token_exp_dt = pd.to_datetime(res_token["expiration"])
            if (res_token['token_type'] == 'Bearer') and (token_exp_dt>utc_dt):
                user_credential = True
                print("User credental successfully verified!")
            else:
                print("User credential not valide. Please re-entere.")
        except:
            if "message" in res_token:
                print(res_token["message"])
                print("Please re-enter.")
            else:
                print("Something went wrong. Please check your internet connection and your username/password.")
    return res_token
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def check_config_dataset(geodata_name_dict,product_response,full_product_code_list):
    ### checking if all the data required exists
    for geodata_type in geodata_name_dict:

        print("Checking {} dataset ({})".format(geodata_type, geodata_name_dict[geodata_type][0]))

        #in case the dataset requested is not avaialbe, check the suitable ones and ask the user to input
        if geodata_name_dict[geodata_type][0] not in full_product_code_list:
            available_product_list = [product["ProductAndVersion"] for product in product_response if geodata_type in product["Description"]]
            new_product_name=input("The dataset ({}) for {} is not available.".format(geodata_name_dict[geodata_type][0],geodata_type)+\
                                   "Please either terminate this program and check your configuration"+\
                                   "or choose and type in one of the avaiable datasets below.\n{}\n".format(available_product_list)+\
                                   "You can check the product infomration by searching the product names here https://lpdaac.usgs.gov/search/.\n")
            new_product_name = new_product_name.strip("'")
            while geodata_name_dict[geodata_type][0] not in full_product_code_list:
                if new_product_name in available_product_list:
                    #replace the unavailable dataset with the user selected one
                    geodata_name_dict[geodata_type][0] = new_product_name
                else:
                    new_product_name=input("Can't find that dataset. Please check and re-enter the dataset name:\n")
                    new_product_name = new_product_name.strip("'")
        print(geodata_name_dict[geodata_type][0]+" in the datasets")
    print("Dataset check successful.")
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def check_config_datalayers(geodata_name_dict,api):
    #get the layers needed for the data requests
    # check each requested data for the layer information
    # if no layer provided, check and assign layer info
    for geodata_type in geodata_name_dict:
        print("Checking layer information for the {} dataset ({})".format(geodata_type, geodata_name_dict[geodata_type][0]))
        #tmp_layer_info_dict is a dictionary of dictionaries with each dictionary item containing information of a layer
        tmp_layer_info_dict = requests.get('{}product/{}'.format(api, geodata_name_dict[geodata_type][0])).json() 
        if len(geodata_name_dict[geodata_type]) == 1:
            # in case of no layer name inputed
            if len(tmp_layer_info_dict) == 1:
                # automatic append layer info to the  geodata_name_dict if there is only one layer
                geodata_name_dict[geodata_type].append(list(tmp_layer_info_dict.keys())[0])
            elif len(tmp_layer_info_dict) > 1:
                # if there are more than one layers, let the user choose which one to add
                while len(geodata_name_dict[geodata_type]) < 2:
                    # only stop the loop when the layer info is added/appended successfully
                    tmp_layer_info = input("There are multipe layers in the dataset of {} ({}).".format(geodata_type,geodata_name_dict[geodata_type][0])+\
                                           "Please choose one of the layers below:\n"+str(list(tmp_layer_info_dict.keys()))+"\n")
                    tmp_layer_info = tmp_layer_info.strip("'")
                    if tmp_layer_info in tmp_layer_info_dict:
                        geodata_name_dict[geodata_type].append(tmp_layer_info)
                    else:
                        print("There might be some typo in your input. Please check and re-enter.")

            else:
                sys.exit("There is no layers avaiable for this dataset. Please check the information and restart the program.")

        elif len(geodata_name_dict[geodata_type]) > 1:
            for i in range(1,len(geodata_name_dict[geodata_type])):
                while geodata_name_dict[geodata_type][i] not in tmp_layer_info_dict:
                    tmp_layer_info = input("Can't find the requested layer ({}) in the ({}) dataset ({})."\
                                           .format(geodata_name_dict[geodata_type][i],geodata_type,geodata_name_dict[geodata_type][0])+\
                                           "Please choose one of the layers below to replace:\n"+str(list(tmp_layer_info_dict.keys()))+"\n")
                    tmp_layer_info = tmp_layer_info.strip("'")
                    if tmp_layer_info in tmp_layer_info_dict:
                        geodata_name_dict[geodata_type][i]=tmp_layer_info
                    else:
                        print("There might be some typo in your input. Please check and re-enter.")                
        else:
            sys.exit("One of your requested geo dataset is empty. Please check the information and restart the program.")
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def convert_geodict_jasoncompat(geodata_name_dict):
    ### convert geodata_name_dict to fully dictionary and compatible to jason format needed later
    geo_layer_jasoncompat_list = list()
    for geodata_type in geodata_name_dict:
        for i in range(1,len(geodata_name_dict[geodata_type])):
            geo_layer_jasoncompat_list.append({
                "layer": geodata_name_dict[geodata_type][i],
                "product": geodata_name_dict[geodata_type][0]
              })
    return geo_layer_jasoncompat_list
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def get_req_bbox(cent_lon,cent_lat,area_radius,default_proj,default_buffer_ratio):
    # get the area bbox based on cent_lon, cent_lat and area_radius
    # the default_buffer_ratio is used to crop a slightly larger area
    # default projection is the one that available from earthdata and will be requested in that proj.
    center_point = Point(cent_lon, cent_lat)
    center_point_gpd = gpd.GeoSeries(center_point,crs=default_proj)
    utm_crs=center_point_gpd.estimate_utm_crs() #automatically finds the utm coords based on the center lat and lon
    circle_utm = center_point_gpd.to_crs(utm_crs).buffer(area_radius*default_buffer_ratio)
    circle_org_proj = circle_utm.to_crs(default_proj)
    bbox=box(minx=circle_org_proj.bounds["minx"].values,miny=circle_org_proj.bounds["miny"].values,\
             maxx=circle_org_proj.bounds["maxx"].values,maxy=circle_org_proj.bounds["maxy"].values)
    return json.loads(gpd.GeoSeries(bbox).to_json())
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def check_req_projection(in_prjection,api):
    #check if the requested projection is available from the dataset
    #if so, return the projection name used in the dataset
    
    ### get the available EPSG projection list from the dataset
    projections = requests.get('{}spatial/proj'.format(api)).json()  # Call to spatial API, return projs as json
    projection_name_dict_list = [{p["Name"]:p["EPSG"]}\
                                 for p in projections \
                                 if p["EPSG"] and CRS.from_epsg(p["EPSG"]) == CRS.from_string(in_prjection)]

    if len(projection_name_dict_list) == 0:
        sys.exit("Something went wrong. defualt projection ({})".format(default_proj)+\
                 "is required but can't be found from the EarthData system.")
    elif len(projection_name_dict_list) == 1:
        in_proj_name = list(projection_name_dict_list[0].keys())[0]  # this will be used in the json task request
    else:
        sys.exit("Multiple projection matches found. This shouldn't happen. Please report to the developer.")
    return in_proj_name
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def check_time_range(geodata_name_dict,product_response,start_date_dict,end_date_dict):
    ### check time range
    for geodata_type in geodata_name_dict:
        dataset_info = [product for product in product_response \
                        if product['ProductAndVersion'] == geodata_name_dict[geodata_type][0]][0]
        ### check the start and end date
        avail_start_time_tmp = dataset_info['TemporalExtentStart']
        avail_end_time_tmp   = dataset_info['TemporalExtentEnd']

        req_start_time_tmp   = start_date_dict[geodata_type]
        req_end_time_tmp     = end_date_dict[geodata_type]
        ## check if no start(end)_date input
        while len(req_start_time_tmp) ==0 or (not date_validate(req_start_time_tmp)) \
        or (req_start_time_tmp < avail_start_time_tmp) or (req_start_time_tmp > avail_end_time_tmp):
            print("No valid start date provided for {} ({}).".format(geodata_type,geodata_name_dict[geodata_type]))
            req_start_time_tmp = input("Enter a start_time (YYYY-MM-DD) (between {} and {}) for {} ({}).\n".\
                                       format(avail_start_time_tmp,avail_end_time_tmp,geodata_type,geodata_name_dict[geodata_type]))

        while (len (req_end_time_tmp) ==0) or (not date_validate(req_end_time_tmp)) \
        or (req_end_time_tmp < avail_start_time_tmp) or (req_end_time_tmp > avail_end_time_tmp)\
        or (req_end_time_tmp < req_start_time_tmp):
            print("No valid end date provided for {} ({}).".format(geodata_type,geodata_name_dict[geodata_type]))
            req_end_time_tmp = input("Enter a end_time (YYYY-MM-DD) (between {} and {}) for {} ({}).\n".\
                                       format(req_start_time_tmp,avail_end_time_tmp,geodata_type,geodata_name_dict[geodata_type]))

        ### assign all validate values back to start_date_dict and end_date_dict
        start_date_dict[geodata_type] = req_start_time_tmp
        end_date_dict[geodata_type]   = req_end_time_tmp
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#        
def check_download_folder(path_dir):
    # check if the download folder exists
    # if so, ask the user if need to continue new download.
    # returns download_task_bool to indicate weather or not continue the download.
    download_task_bool = True  # True by default unless the folder is not empty and user confirm not to continue download.
    if os.path.isdir(path_dir):
        # if folder exists, check if the folder is empty
        if os.listdir(path_dir):
            # if folder exits and not empty, check if the user still wants to download the task
            while True:
                tmp_input_str=input("Files already exists in {}. Enter 1 to continue task downloading or 0 to skip this task."\
                                    .format(path_dir))
                tmp_input_str = tmp_input_str.lower().strip()
                if tmp_input_str == "0":
                    #user confirm skip this task.
                    download_task_bool = False
                    break
                elif tmp_input_str == "1":
                    # user confirm continue downloading which might overwrite the previous files.
                    download_task_bool = True
                    break
                else:
                    print("Wrong input, please check and re-enter.")        
    return download_task_bool
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def submit_tasks(geodata_name_dict,start_date_dict,end_date_dict,output_format_dict,task_name_prefix, task_type, output_dir,req_proj_name,crop_area_jason_input,api,head):
    ###preparing and submitting tasks
    task_response_list = list()
    for geodata_type in geodata_name_dict:
        output_format_tmp=output_format_dict[geodata_type]
        ### check time range
        start_time_tmp = datetime.strptime(start_date_dict[geodata_type], '%Y-%m-%d').strftime('%m-%d-%Y')
        end_time_tmp   = datetime.strptime(end_date_dict[geodata_type], '%Y-%m-%d').strftime('%m-%d-%Y')
        dataset_tmp = geodata_name_dict[geodata_type][0]
        task_name_tmp = task_name_prefix+"_"+geodata_type+"_"+dataset_tmp 

        ### check if the task data already exists
        path_dir_tmp = os.path.join(output_dir,task_name_tmp)
        download_task_bool=check_download_folder(path_dir_tmp)
        if download_task_bool:
            #only download the data when download_task_bool is true.

            req_layer_list_tmp = list()
            for i in range(1,len(geodata_name_dict[geodata_type])):
                req_layer_list_tmp.append({
                    "layer": geodata_name_dict[geodata_type][i],
                    "product": geodata_name_dict[geodata_type][0]
                  })
            task_tmp = {
                'task_type': task_type,
                'task_name': task_name_tmp,
                'params': {
                     'dates': 
                     [{
                         'startDate': start_time_tmp,
                         'endDate': end_time_tmp,
                     }],
                     'layers': req_layer_list_tmp,
                     'output': {
                             'format': {
                                     'type': output_format_tmp}, 
                                     'projection': req_proj_name},
                     'geo':crop_area_jason_input,
                }
            }
            task_response_tmp = requests.post('{}task'.format(api), json=task_tmp, headers=head).json()
            try:
                task_response_tmp["task_id"]
            except ValueError:
                raise ValueError(str(task_response_tmp))
            task_response_tmp["task_name"] = task_name_tmp
            task_response_list.append(task_response_tmp) 
    # in case no tasks need to be submit, terminate
    if len(task_response_list) == 0:
        warnings.warn("No tasks to submit. Please check your configurations.")
    return task_response_list
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def check_tasks_status(task_response_list,api,head,interval=20.0):
    # Ping API until request is complete, modified from the offical tutorial APPEEARS_API_Area.ipynb
    #interval of the pings are controlled by the interval variable
    starttime = time.time()
    for task_dict in task_response_list:
        task_id = task_dict['task_id']
        while requests.get('{}task/{}'.format(api, task_id), headers=head).json()['status'] != 'done':
            print("Waiting for the processing to finish. Status: "+\
                  requests.get('{}task/{}'.format(api, task_id), headers=head).json()['status'])
            time.sleep(interval - ((time.time() - starttime) % interval))
        print(requests.get('{}task/{}'.format(api, task_id), headers=head).json()['status'])
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
def data_download(task_response_list,output_dir,api,head):
    ### Downloading data
    output_dir =Path(output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for task_reponse in task_response_list:
        output_dir_subdataset = output_dir.joinpath(task_reponse["task_name"])
        if not os.path.exists(output_dir_subdataset):os.makedirs(output_dir_subdataset)
        bundle_tmp = requests.get('{}bundle/{}'.format(api,task_reponse["task_id"]), headers=head).json()  # Call API and return bundle contents for the task_id as json
        files_tmp_dict = dict()
        for files_info in bundle_tmp["files"]:
            files_tmp_dict[files_info['file_id']] = files_info['file_name']
        for files_tmp in files_tmp_dict:
            dl = requests.get('{}bundle/{}/{}'.format(api, task_reponse["task_id"], files_tmp), headers=head, stream=True, allow_redirects = 'True')                                # Get a stream to the bundle file
            if files_tmp_dict[files_tmp].endswith('.tif'):
                filename = files_tmp_dict[files_tmp].split('/')[1]
            else:
                filename = files_tmp_dict[files_tmp]
            filepath = output_dir_subdataset.joinpath(filename)                                                       # Create output file path
            with open(filepath, 'wb') as f:                                                                  # Write file to dest dir
                for data in dl.iter_content(chunk_size=8192): f.write(data)
        print('Downloaded files can be found at: {}'.format(output_dir_subdataset))

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#        
def download_nasa_main(api, geodata_name_dict, centlon, centlat, area_radius, default_proj, task_type,\
                       default_buffer_ratio, start_date_dict,end_date_dict, output_format_dict,case_name,static_tif_path):
    res_token=login_earthdata(api)

    product_response = requests.get('{}product'.format(api)).json()                         # request all products in the product service
    print('AρρEEARS currently supports {} products.'.format(len(product_response)))  # Print no. products available in AppEEARS
    full_product_code_list = [p["ProductAndVersion"] for p in product_response]  # get all products' name ("ProductAndVersion")

    check_config_dataset(geodata_name_dict,product_response,full_product_code_list)

    check_config_datalayers(geodata_name_dict,api=api)

    req_geodata_layer_list=convert_geodict_jasoncompat(geodata_name_dict)

    ### get token info to submit the jobs
    token = res_token['token']                      # Save login token to a variable
    head = {'Authorization': 'Bearer {}'.format(token)}  # Create a header to store token information, needed to submit a request

    crop_area_jason_input = get_req_bbox(centlon,centlat,area_radius,default_proj,default_buffer_ratio)

    ### Preparing and submitting tasks

    req_proj_name = check_req_projection(default_proj,api)
    print("Projection check done. Start submitting jobs.")

    check_time_range(geodata_name_dict,product_response,start_date_dict,end_date_dict)

    task_response_list=submit_tasks(geodata_name_dict,start_date_dict,end_date_dict,output_format_dict,\
                                    case_name, task_type, static_tif_path,req_proj_name,crop_area_jason_input,api,head)

    check_tasks_status(task_response_list,api,head,60.0)

    data_download(task_response_list,static_tif_path,api,head)
