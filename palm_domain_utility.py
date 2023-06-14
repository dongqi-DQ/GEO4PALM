#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# Fucntions to:
# - draw a domain based on the center lat lon and grid size and numbers,
#   and visually see where they are 
# - generate configuration lines for the p3d namelist 
# 
# @author: Jiawei Zhang 
#--------------------------------------------------------------------------------#

import panel as pn
pn.extension(notifications=True)
import geoviews as gv
import holoviews as hv
import numpy as np
gv.extension('bokeh')
from pyproj import Transformer
import cartopy.crs as ccrs
import pandas as pd
import sys
import configparser
import io
import pyproj


centlat_input=pn.widgets.FloatInput(start=-90,end=90,value=0, name="center lat",width=90)
centlon_input=pn.widgets.FloatInput(start=-180,end=180,value=0,name="center lon",width=90)
dx_input= pn.widgets.FloatInput(start=0,value=0,name="dx",width=90)
dy_input= pn.widgets.FloatInput(start=0,value=0,name="dy",width=90)
dz_input= pn.widgets.FloatInput(start=0,value=0,name="dz",width=90)
nx_input= pn.widgets.IntInput(start=1,value=1,name="nx",width=90)
ny_input= pn.widgets.IntInput(start=1,value=1,name="ny",width=90)
nz_input= pn.widgets.IntInput(start=1,value=1,name="nz",width=90)
z_origin_input = pn.widgets.FloatInput(start=0,value=0,name="z_origin",width=90)

crs_loc_input = pn.widgets.IntInput(start=0,value=0,name="local projection (epsg)",max_width=100,width_policy='fit')
add_to_map_button=pn.widgets.Button(name="Add to map", margin=(10, 10 ,10, 20), button_type="primary",width=100)
calculate_namelist_button=pn.widgets.Button(name="Get domain configure", margin=(10, 10 ,10 , 400), button_type="success",width=100)
undo_button = pn.widgets.Button(name="undo",margin=(10, 10 ,10 , 30),button_type="default",width=100)

# domain_input_box=pn.WidgetBox(crs_loc_input,pn.Row(centlat_input,centlon_input,nx_input,dx_input,ny_input,dy_input),pn.Row(nz_input,dz_input,add_to_map_button,calculate_namelist_button),width=900)
domain_input_box=pn.WidgetBox(pn.WidgetBox(crs_loc_input,pn.Row(centlat_input,centlon_input,nx_input,dx_input\
                                                   ,ny_input,dy_input,nz_input,dz_input,z_origin_input,width=1100),\
                                                   pn.Row(pn.Spacer(width=720),add_to_map_button,undo_button)),calculate_namelist_button,width=1100)

# use this dataframe to save all the numbers
# p_num is the parent number, num is the actual number
boundary_value_df = pd.DataFrame(columns=["centlon","centlat","centx","centy","nx","ny","nz","dx","dy","dz","xmin","ymin","xmax","ymax","p_num","num","z_origin"])


# map display
sate_image=gv.tile_sources.EsriImagery.opts(width=600,height=600)
disp_map=pn.panel(sate_image)

#create static config text box for output configure text
static_config_text=pn.widgets.TextAreaInput(height=250,width=460)
copy_static_config_button = pn.widgets.Button(name="Copy static domain configure", button_type="default",width=100)
draw_config_button = pn.widgets.Button(name="check configuration",button_type="primary",width=100) #button to plot domain based on the config text

copy_source_code = "navigator.clipboard.writeText(source.value);"
copy_static_config_button.js_on_click(args={"source": static_config_text}, code=copy_source_code)

static_config_box=pn.WidgetBox(static_config_text,pn.Row(copy_static_config_button,pn.layout.Spacer(width=100),draw_config_button))

#create palm model namelist text box for output configure text
namelist_text=pn.widgets.TextAreaInput(height=250,width=460)
copy_namelist_text_button = pn.widgets.Button(name="Copy namelist domain configure", button_type="default",width=100)
copy_source_code = "navigator.clipboard.writeText(source.value);"
copy_namelist_text_button.js_on_click(args={"source": namelist_text}, code=copy_source_code)

namelist_config_box=pn.WidgetBox(namelist_text,copy_namelist_text_button)

def boundary_df_disp_columns(df,column_list=["centlon","centlat","nx","ny","nz","dx","dy","dz","p_num","num","z_origin"]):
    # simple function to choose which columns in the dataframe to display
    return df[column_list].sort_values("num").reset_index(drop=True)
#dataframe table display
df_pane = pn.pane.DataFrame(boundary_df_disp_columns(boundary_value_df), width=600)
df_pane.float_format='{:,.3f}'.format

def create_domain_boundary(df,centlon,centlat,nx,ny,nz,dx,dy,dz,z_origin,crs_loc,crs_in="EPSG:4326",crs_wgs="EPSG:4326",add_to_df=True):
    # df is to save all the boundary data
    # nz,dz, z_origin is only used here as input to the dataframe, no calculations regarding these here.
    check_crs_loc_input(crs_loc_input)
    in_to_loc = Transformer.from_crs(crs_in, crs_loc)
    loc_to_wgs = Transformer.from_crs(crs_loc,crs_wgs)
    
    centy_loc,centx_loc = in_to_loc.transform(centlat,centlon)
    ymin_loc = centy_loc-dy*ny/2
    xmin_loc = centx_loc-dx*nx/2
    ymax_loc =  ymin_loc+ dy*ny
    xmax_loc =  xmin_loc +dx*nx
    
    ymin_wgs,xmin_wgs = loc_to_wgs.transform(ymin_loc,xmin_loc)
    ymax_wgs,xmax_wgs = loc_to_wgs.transform(ymax_loc,xmax_loc)
    # this is only calculated to store in the dataframe so it can be displayed in the app.
    centy_wgs,centx_wgs = loc_to_wgs.transform(centy_loc,centx_loc)
    
    domain_boundary = gv.Rectangles([(xmin_wgs,ymin_wgs, xmax_wgs, ymax_wgs)]).opts(fill_alpha=0,line_width=2,line_color="r")
    if add_to_df:
        #save all the data, needed for adjust/generate final boundary configure
        df.loc[len(df)] = {"centlon":centx_wgs,"centlat":centy_wgs,"centx":centx_loc,"centy":centy_loc,"nx":nx,"ny":ny,"nz":nz,"dx":dx,"dy":dy,"dz":dz,"xmin":xmin_loc,\
                           "ymin":ymin_loc,"xmax":xmax_loc,"ymax":ymax_loc,"z_origin":z_origin}
    return domain_boundary

bd_create_domain_boundary = pn.bind(create_domain_boundary,centlat=centlat_input,centlon=centlon_input,\
                                    nx=nx_input,ny=ny_input,nz=nz_input,dx=dx_input,dy=dy_input,dz=dz_input,z_origin=z_origin_input,crs_loc=crs_loc_input)

def check_crs_loc_input(crs_loc_input):
    if (crs_loc_input.value < 1024) or(crs_loc_input.value > 32767):
        pn.state.notifications.position = 'bottom-center'
        pn.state.notifications.info("EPSG code of the UTM coordinate matches the entered lat, lon is used since no valid EPSG Code number is found in the input.", duration=0)        
        utm_zone = get_utm_zone(centlat_input.value, centlon_input.value)
        crs_loc_input.value = get_epsg_code(utm_zone)

def on_button_click(event):
    # add domain boundary to the map
    pn.state.notifications.clear()
    check_crs_loc_input(crs_loc_input)
    new_domain =  bd_create_domain_boundary(df=boundary_value_df)
    if not check_domain_draw_validation(boundary_value_df):
        pn.state.notifications.position = 'bottom-center'
        pn.state.notifications.info("The new domain will be overlapping with exising ones. Please check and change your settings.", duration=5000)
        boundary_value_df.drop(boundary_value_df.tail(1).index,inplace=True)
    else:
        disp_map.object = disp_map.object * new_domain
    df_pane.object = boundary_df_disp_columns(boundary_value_df)

def create_rectangles_from_df(df):
    domain_all = None
    for i in range(0,df.shape[0]):
        tmp_boundary = df.iloc[i]
        tmp_domain_boundary=create_domain_boundary(df,tmp_boundary.centx,tmp_boundary.centy,\
                                                   tmp_boundary.nx,tmp_boundary.ny,tmp_boundary.nz,tmp_boundary.dx,\
                                                   tmp_boundary.dy,tmp_boundary.dz,tmp_boundary.z_origin,crs_loc=crs_loc_input.value,\
                                                   crs_in=crs_loc_input.value,add_to_df=False)
        if domain_all is None:
            domain_all = tmp_domain_boundary
        else:
            domain_all = domain_all*tmp_domain_boundary
    return domain_all

def on_configure_button_click(event):
    pn.state.notifications.clear()
    sort_domain_num(boundary_value_df)
    adjust_domains(boundary_value_df)
    domain_boundries_all = create_rectangles_from_df(boundary_value_df)
    disp_map.object = sate_image * domain_boundries_all
    df_pane.object = boundary_df_disp_columns(boundary_value_df.sort_values('num',ascending=True))
    grid_resolution_check(boundary_value_df)
    static_config_text.value,namelist_text.value=bd_generate_config_text(df_in=boundary_value_df)

def on_undo_button_click(event):
    # remove the latest boudaries
    # this behaviour will be different after click get domain configure! 
    # df will be sorted different after that. 
    pn.state.notifications.clear()
    if boundary_value_df.shape[0] > 1:
        boundary_value_df.drop(boundary_value_df.tail(1).index,inplace=True)
        rectangles=create_rectangles_from_df(boundary_value_df)
        disp_map.object = sate_image * rectangles
    elif boundary_value_df.shape[0] == 1:
        boundary_value_df.drop(boundary_value_df.tail(1).index,inplace=True)
        disp_map.object = sate_image
    df_pane.object = boundary_df_disp_columns(boundary_value_df)

def compare_grid_resolution(pdomain,cdomain):
    grid_res_valid = False
    if (pdomain.dx%cdomain.dx == 0) and ((cdomain.dx*cdomain.nx)%pdomain.dx == 0) \
    and (pdomain.dy%cdomain.dy == 0) and ((cdomain.dy*cdomain.ny)%pdomain.dy == 0) \
    and (pdomain.dz%cdomain.dz == 0) and ((cdomain.dz*cdomain.nz)%pdomain.dz == 0):
        grid_res_valid = True
    if grid_res_valid == False:
        pn.state.notifications.position = 'bottom-center'
        pn.state.notifications.info("The domain num {}'s grid resolution and grid number is not compatable with its parent domain (num: {}).".format(int(cdomain.num),int(pdomain.num)), duration=0)
    return grid_res_valid

# Define a function to get UTM zone from latitude and longitude
def get_utm_zone(lat, lon):
    # Calculate the zone number based on longitude, by chatgpt
    zone_number = int((lon + 180) // 6 + 1)
    # Calculate the zone letter based on latitude
    if lat >= 72:
        zone_letter = 'X'
    elif lat < -80:
        zone_letter = 'C'
    else:
        letters = 'CDEFGHJKLMNPQRSTUVWXX'
        index = int((lat + 80) // 8)
        zone_letter = letters[index]
    # Return the UTM zone as a string
    return str(zone_number) + zone_letter


# Define a function to get EPSG coordinate number from UTM zone
def get_epsg_code(utm_zone):
    # Get the zone number and letter from the UTM zone string, by chatgpt
    zone_number = int(utm_zone[:-1])
    zone_letter = utm_zone[-1]
    # Check if the zone is in the northern or southern hemisphere
    if zone_letter >= 'N':
        hemisphere = 'north'
        epsg_code = 32600 + zone_number
    else:
        hemisphere = 'south'
        epsg_code = 32700 + zone_number
    # Return the EPSG code as an integer
    return epsg_code

    
def grid_resolution_check(df):
    grid_res_valid = True
    if df.shape[0] == 1:
        grid_res_valid = True
    elif df.shape[0] > 1:        
        for i in range(0,df.shape[0]):
            if int(df.iloc[i].p_num) == -1:
                continue
            else:
                for j in range(0,df.shape[0]):
                    if df.iloc[j].num == df.iloc[i].p_num:
                        grid_res_valid = grid_res_valid*compare_grid_resolution(df.iloc[j],df.iloc[i])
    if grid_res_valid == False:
        pn.state.notifications.position = 'bottom-center'
        pn.state.notifications.info("One ore more domains' grid resolution and grid number combination is not compatible with it's parent domain. You should not use the generated namelist directly.", duration=0)        
    return grid_res_valid
                
    
def domain_domain_relation(ps1,ps2,min_gap_grid_num=1):
    # return the horizontal relationship between two domains
    # 1 means domain one (ps1) is the parent domain
    # 2 means domnain two is the parent domain
    # -1 means domain one is the parent domain but is too close to domain2
    # -2 means domain two is the parent domain but is too close to domain1
    # 0 means domain one and two are not connected in any ways
    # -3 means domain one and two are overlapping
    # -4 means domnain one and two are not compatible in vertical vertical
    # min_gap_grid_num defines the minimum number of the outer grid space needed in order to be not too close 
    relation_num = 0
    if (ps1.xmin > ps2.xmax) or (ps2.xmin > ps1.xmax) or (ps1.ymin > ps2.ymax) or (ps2.ymin > ps1.ymax):
        # not connected at all
        relation_num = 0
    if (ps1.xmin < ps2.xmin) and (ps1.ymin < ps2.ymin) and (ps1.xmax > ps2.xmax) and (ps1.ymax > ps2.ymax):
        # ps1 is the parent domain
        if (ps1.dz*ps1.nz) > (ps2.dz*ps2.nz):
            # ps1 is the parent domain from vertical as well
            if ((ps2.xmin - ps1.xmin) // ps1.dx >= min_gap_grid_num) and ((ps2.ymin - ps1.ymin) // ps1.dy >=min_gap_grid_num) \
            and ((ps1.xmax - ps2.xmax) // ps1.dx >= min_gap_grid_num) and ((ps1.ymax - ps2.ymax) // ps1.dy >= min_gap_grid_num):
                relation_num = 1
            else:
                relation_num = -1
        else:
            # ps2 has a higher domain top than ps1 which is not valid
            relation_num = -4
    elif (ps1.xmin > ps2.xmin) and (ps1.ymin > ps2.ymin) and (ps1.xmax < ps2.xmax) and (ps1.ymax < ps2.ymax):
        # ps 2 is the parent domain
        if (ps2.dz*ps2.nz) > (ps1.dz*ps1.nz):
            # ps2 is the parent domain from vertical as well
            if ((ps1.xmin - ps2.xmin) // ps2.dx >= min_gap_grid_num) and ((ps1.ymin - ps2.ymin) // ps2.dy >=min_gap_grid_num) \
            and ((ps2.xmax - ps1.xmax) // ps2.dx >= min_gap_grid_num) and ((ps2.ymax - ps1.ymax) // ps2.dy >= min_gap_grid_num):
                relation_num = 2
            else:
                relation_num = -2
        else:
            relation_num = -4
    else:
        relation_num = -3
    
    return relation_num


def switch_domain_num(df,old_num,new_num,column_idx=11,keep_old_num=True,row_idx=None):
    # when keep old num, this function will swtich both number
    # when keep_old_num ==Fals, this function will simply replace the old number with the new one
    #column_idx make this one can be used for both domain column ('num') and parent domain column ('p_num')
    tmp_num = -999
    if row_idx is None:
        if keep_old_num == True:
            df.iloc[:,column_idx].where(df.iloc[:,column_idx]!=old_num, tmp_num,inplace=True)
            df.iloc[:,column_idx].where(df.iloc[:,column_idx]!=new_num, old_num,inplace=True)
            df.iloc[:,column_idx].where(df.iloc[:,column_idx]!=tmp_num, new_num,inplace=True)
        else:
            df.iloc[:,column_idx].where(df.iloc[:,column_idx]!=old_num, new_num,inplace=True)
            
    else:
        df.iloc[row_idx,column_idx] = new_num
    return df

def sort_domain_num(df):
    # df.iloc[0].p_num = parent_num
    # df.iloc[0].num = domain_num
    df['p_num'] = -1
    df['num'] = range(0,df.shape[0])
    df['p_num'] = df['p_num'].astype(int)
    df['num'] = df['num'].astype(int)
    parent_column_number = list(df.columns).index('p_num')
    domain_column_number = list(df.columns).index('num')
    for i in range(0,df.shape[0]):
        for j in range(i,df.shape[0]):
            relation_num = domain_domain_relation(df.iloc[i],df.iloc[j])
            if relation_num == 1:
                if df.iloc[i].num > df.iloc[j].num:
                    #switch the numbers to make sure parent domain has smaller numbers
                    
                    switch_domain_num(df,df.iloc[i].num,df.iloc[j].num,column_idx=domain_column_number,keep_old_num=True)
                    #update the domains which has current domain as it's parent domain to the new parent domain number
                    #using df.iloc[j].num below as the old parent number since df.iloc[j].num is now the old df.iloc[i].num after switch
                    switch_domain_num(df,df.iloc[j].num,df.iloc[i].num,column_idx=parent_column_number,keep_old_num=False)
                if (df.iloc[j].p_num == -1) or (domain_domain_relation(df.iloc[int(df.iloc[j].p_num)],df.iloc[i]) == 1):
                    df.iloc[j,parent_column_number] = df.iloc[i].num
                    
            elif relation_num == 2:
                # if df.iloc[j] is a parent domain
                if df.iloc[i].num < df.iloc[j].num:
                    # switch the numbers to make sure parent domain has smaller numbers
                    switch_domain_num(df,df.iloc[i].num,df.iloc[j].num,column_idx=domain_column_number,keep_old_num=True)
                    #change all domain whose parent domain number is the df.iloc[j] to df.iloc[j]'s new domain number
                    switch_domain_num(df,df.iloc[j].num,df.iloc[i].num,column_idx=parent_column_number,keep_old_num=False)
                if (df.iloc[i].p_num == -1) or (domain_domain_relation(df.iloc[int(df.iloc[i].p_num)],df.iloc[j]) == 1):
                    # if df.iloc[i] has no parent, set it to df.iloc[j] or
                    # if df.iloc[j] is within the current domain's present parent domain, 
                    # set df.iloc[j] as df.iloc[i]'s new parent domain
                    df.iloc[i,parent_column_number] = df.iloc[j].num
                    #switch_domain_num(df,df.iloc[i].num,df.iloc[j].num,column_idx=parent_column_number,keep_old_num=False,row_idx=i)
    return df

def check_domain_draw_validation(df,row=-1):
    # check if the ith row is validate with other rows
    # default is the last (newest) row
    #return true if the new domain is validate, False elsewise.
    validate = True
    if row == -1:
        row = df.shape[0]-1
    elif row < 0 or row >= df.shape[0]:
        sys.exit("Wroing row number.Shouldn't happen. Check the code.")
    if df.shape[0] > 1:
        for i in range(0,df.shape[0]):
            if (i != row) and (domain_domain_relation(df.iloc[row],df.iloc[i]) < 0):
                validate = False
    return validate

def adjust_domains(df):
    # use the sorted df as input
    # adjust the cent lat lon and distance of each domain to ouput final domain config
    df.sort_values('num',ascending=False,inplace=True)
    if df.iloc[-1].p_num != -1:
        sys.exit("The first parent domoain should have p_num as -1. Report to the developer.")
    for i in range(0,df.shape[0]-1):
        #use this two temperary vairables for calculation, easy to read
        current_domain = df.iloc[i]
        tmp_parent = df[df["num"] == current_domain.p_num].squeeze()
        
        
        tmp_parent.xmin = current_domain.xmin - np.ceil((current_domain.xmin - tmp_parent.xmin) / tmp_parent.dx).astype(int)*tmp_parent.dx
        tmp_parent.ymin = current_domain.ymin - np.ceil((current_domain.ymin - tmp_parent.ymin) / tmp_parent.dy).astype(int)*tmp_parent.dy
        tmp_parent.centx = tmp_parent.xmin + tmp_parent.nx*tmp_parent.dx/2
        tmp_parent.centy = tmp_parent.ymin + tmp_parent.ny*tmp_parent.dy/2
        tmp_parent.xmax = tmp_parent.xmin + tmp_parent.nx*tmp_parent.dx
        tmp_parent.ymax = tmp_parent.ymin + tmp_parent.ny*tmp_parent.dy
        
        ## assign the values back to the actual df
        df[df["num"] == df.iloc[i].p_num] = tmp_parent
    return df



def generate_config_text(df_in,crs_loc,crs_wgs="EPSG:4326",format_digit=1):
    # format_digit can be used to control the output digit of dx,dy and ll_x,lly
    df = df_in.copy(deep=True)
    df=df.sort_values('num',ascending=True)
    df.reset_index(drop=True,inplace=True)
    df["xmin"]= df["xmin"]- df["xmin"][0]
    df["ymin"]= df["ymin"]- df["ymin"][0]
    loc_to_wgs = Transformer.from_crs(crs_loc,crs_wgs)
    
    # static_config_output for the static driver
    static_config_output = "[domain]\n"
    
    # namelist_config_output for the model namelist
    namelist_config_output =" &nesting_parameters \n"
    
    static_config_output += "ndomain = {},\n".format(df.shape[0])
    cent_lat,cent_lon = loc_to_wgs.transform(df.iloc[0].centy,df.iloc[0].centx)
    static_config_output += "centlat = {:0.5f}, \n".format(cent_lat)+"centlon = {:0.5f}, \n".format(cent_lon)
    nx = "nx = "
    ny = "ny = "
    nz = "nz = "
    dx = "dx = "
    dy = "dy = "
    dz = "dz = "
    ll_x = "ll_x = "
    ll_y = "ll_y = "
    z_origin = "z_origin  = "
    domain_layouts = "        domain_layouts = "  #
    for i in range(0,df.shape[0]):
        nx += "{:0.0f}, ".format(df.iloc[i].nx)
        ny += "{:0.0f}, ".format(df.iloc[i].ny)
        nz += "{:0.0f}, ".format(df.iloc[i].nz)
        dx += "{:0.{}f}, ".format(df.iloc[i].dx,format_digit)
        dy += "{:0.{}f}, ".format(df.iloc[i].dy,format_digit)
        dz += "{:0.{}f}, ".format(df.iloc[i].dz,format_digit)
        ll_x += "{:0.{}f}, ".format(df.iloc[i].xmin,format_digit)
        ll_y += "{:0.{}f}, ".format(df.iloc[i].ymin,format_digit)
        z_origin += "{:0.{}f}, ".format(df.iloc[i].z_origin,format_digit)
        if i == 0:
            tmp_domain_layout = "palm_d" + "{:0.0f}, ".format(i+1)
        else:
            tmp_domain_layout = " "*38 + "palm_d" + "{:0.0f}, ".format(i+1)
        tmp_domain_layout += "{:0.0f},  {:0.0f}, process_num, {:0.1f}, {:0.1f},\n".format(df.iloc[i].num,df.iloc[i].p_num,df.iloc[i].xmin,df.iloc[i].ymin)
        domain_layouts += tmp_domain_layout
    nx += "\n"
    ny += "\n"
    nz += "\n"
    dx += "\n"
    dy += "\n"
    dz += "\n"
    ll_x += "\n"
    ll_y += "\n"
    z_origin += "\n"
    static_config_output = static_config_output+nx+ny+nz+dx+dy+dz+ll_x+ll_y+z_origin
    nest_string = "         nesting_mode   = 'one-way',\n"
    namelist_config_output += domain_layouts + nest_string +"/\n"
    return static_config_output, namelist_config_output

def on_click_draw_config(event):
    # draw domain on the map based on the text in the static_config_text box
    pn.state.notifications.clear()
    check_crs_loc_input(crs_loc_input)
    bd_construct_df_from_static()
    sort_domain_num(boundary_value_df)
    domain_boundries_all = create_rectangles_from_df(boundary_value_df)
    disp_map.object = sate_image * domain_boundries_all
    df_pane.object = boundary_df_disp_columns(boundary_value_df.sort_values('num',ascending=True))
    grid_resolution_check(boundary_value_df)

def convert_confs_values(con_string,d_type):
    # convert numerical values from the config string
    con_string_list = con_string.split(",")
    if d_type.lower() == "int":
        value_list = [int(con_s.strip()) for con_s in con_string_list if con_s != '']
    elif d_type.lower() == "float":
        value_list = [float(con_s.strip()) for con_s in con_string_list if con_s != '']
    return value_list

def construct_df_from_static(static_config_input,crs_loc,crs_in="EPSG:4326",crs_wgs="EPSG:4326"):
    #construct boundary_value_df from static_config_namelist
    in_to_loc = Transformer.from_crs(crs_in, crs_loc)
    loc_to_wgs = Transformer.from_crs(crs_loc,crs_wgs)
    
    #read the text from the configure input widget
    config = configparser.ConfigParser(inline_comment_prefixes=("#","!"))
    static_tmp = io.StringIO(static_config_input)
    config.read_file(static_tmp)
    if not config.has_section("domain"):
        static_config_input = "[domain]\n" + static_config_input
        static_tmp = io.StringIO(static_config_input)
        config.read_file(static_tmp)
    ndomain = convert_confs_values(config.get("domain","ndomain"),"int")[0]
    centlat_root = convert_confs_values(config.get("domain","centlat"),"float")[0]
    centlon_root = convert_confs_values(config.get("domain","centlon"),"float")[0]
    nx = convert_confs_values(config.get("domain","nx"),"int")[:ndomain]
    ny = convert_confs_values(config.get("domain","ny"),"int")[:ndomain]
    nz = convert_confs_values(config.get("domain","nz"),"int")[:ndomain]
    dx = convert_confs_values(config.get("domain","dx"),"float")[:ndomain]
    dy = convert_confs_values(config.get("domain","dy"),"float")[:ndomain]
    dz = convert_confs_values(config.get("domain","dz"),"float")[:ndomain]
    ll_x = convert_confs_values(config.get("domain","ll_x"),"float")[:ndomain]
    ll_y = convert_confs_values(config.get("domain","ll_y"),"float")[:ndomain]
    z_origin = convert_confs_values(config.get("domain","z_origin"),"float")[:ndomain]
    

    #empty the boundary_value_df and then add the data from the config
    boundary_value_df.drop(boundary_value_df.index,inplace=True)
    for i in range(ndomain):
        if i == 0:
            #get the local coordinate of the(0,0), aka xmin_loc and ymin_loc for the root domain
            centy_loc_tmp,centx_loc_tmp = in_to_loc.transform(centlat_root,centlon_root)
            xmin_loc_tmp = centx_loc_tmp - nx[i]*dx[i]/2 + ll_x[i]
            ymin_loc_tmp = centy_loc_tmp - ny[i]*dy[i]/2 + ll_y[i]
            xmax_loc_tmp = centx_loc_tmp + nx[i]*dx[i]/2 + ll_x[i]
            ymax_loc_tmp = centy_loc_tmp + ny[i]*dy[i]/2 + ll_y[i]
            centlat_tmp = centlat_root
            centlon_tmp = centlon_root
        else:
            xmin_loc_tmp = boundary_value_df.iloc[0].xmin + ll_x[i]
            ymin_loc_tmp = boundary_value_df.iloc[0].ymin + ll_y[i]
            xmax_loc_tmp = xmin_loc_tmp + nx[i]*dx[i]
            ymax_loc_tmp = ymin_loc_tmp + ny[i]*dy[i]
            centx_loc_tmp = xmin_loc_tmp + nx[i]*dx[i]/2
            centy_loc_tmp = ymin_loc_tmp + ny[i]*dy[i]/2
            centlat_tmp, centlon_tmp = loc_to_wgs.transform(centy_loc_tmp,centx_loc_tmp)

        boundary_value_df.loc[i] = {"centlon":centlon_tmp,"centlat":centlat_tmp,"centx":centx_loc_tmp,"centy":centy_loc_tmp,\
                                     "nx":nx[i],"ny":ny[i],"nz":nz[i],"dx":dx[i],"dy":dy[i],\
                                     "dz":dz[i],"xmin":xmin_loc_tmp,\
                                    "ymin":ymin_loc_tmp,"xmax":xmax_loc_tmp,"ymax":ymax_loc_tmp,"z_origin":z_origin[i]}
    
    return boundary_value_df
                               
bd_generate_config_text = pn.bind(generate_config_text,crs_loc=crs_loc_input)

bd_construct_df_from_static=pn.bind(construct_df_from_static,static_config_text,crs_loc_input)

undo_button.on_click(on_undo_button_click)

calculate_namelist_button.on_click(on_configure_button_click)

draw_config_button.on_click(on_click_draw_config)

add_to_map_button.on_click(on_button_click)

palm_domain_config_tool = pn.Column(domain_input_box,pn.Row(disp_map,pn.Column(static_config_box,pn.Spacer(width=10),namelist_config_box)),df_pane).servable(title="PALM toolkit")
