# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:07:42 2024

@author: hbehrooz
Read input files from 5 diffrent sources and creat basic pipe fialure dataset containg various pipe features and also climate information
it merge the service pipe and main pipes as one integrated dataset. 
"""
"""
Public Water Main

Water main assets owned and operated by the City of Calgary for the distribution of drinking water.

https://data.calgary.ca/Services-and-Amenities/Public-Water-Main/w6h9-w33i/about_data

P_ZONE: pressure zone  text
STATUS_IND status of asset Text
LENGTH: length of water main in m, Number
DIAM: diameter of water main in mm, Number
MATERIAL: material of water main, Text
YEAR: install year, Number
GLOBALID: Global Unique identifier, Text
MULTILINESTRING, water main geometry, MultiLine

"""
public_water_file="input/calgary/Public_Water_Main_20240621.csv"

"""
Public Water Service Lines

A water service pipe, also known as a water service line, is the way drinking water is delivered to each home in Calgary.
The water service pipe can be thought of as two sections:
(1) From the water main to the property line: This part of the pipe is owned and maintained by The City of Calgary.
(2) From the property line to your house: This part of the pipe is owned and maintained by the homeowner.
The data shown here only relates to the portion of the water service pipe on public property, owned and operated by The City. 
The material on private property can be a different material, and could have been installed at a different time.

https://data.calgary.ca/Health-and-Safety/Public-Water-Service-Lines/ta76-7bfx/about_data

BUILDING_TYPE:Text
WATER_SERVICE_ADDRESS:Text
MATERIAL_TYPE: Text
PIPE_DIAMETER (mm):Text
INSTALLED_DATE: Fixed Timestamp
line: MultiLine

"""
public_water_service_file="input/calgary/Public_Water_Service_Lines_20240621.csv"

"""
Water Main Breaks

Main breaks happen year-round. There are many contributing factors including: Pipe Age, Pipe Material and Soil Conditions, 
and Temperature. See Water outages and main breaks for more information.

https://data.calgary.ca/Environment/Water-Main-Breaks/dpcu-jr23/about_data

BREAK_DATE: Floating Timestamp
BREAK_TYPE: A - Full Circular B - Split C - Corrosion D - Fitting E - Joint F - Diagonal Crack G - Hole S - Saddle
STATUS:Text
point: Point in longitude latitude format
"""
water_main_break_file="input/calgary/Water_Main_Breaks_20240621.csv"

"""
Water Pressure Zones
Water pressure zone boundaries are subject to change due to new development in green field areas. 
For more information on water pressure zones, please visit: Water pressure in Calgary

https://data.calgary.ca/Environment/Water-Pressure-Zones/xn3q-y49u/about_data

ZONE: Text
MODIFIED_DT: Fixed Timestamp
MULTIPOLYGON: MultiPolygon format

"""
water_pressure_zone_file="input/calgary/Water_Pressure_Zones_20240621.csv"

"""
The historical weather data, forecast and current conditions graphics are courtesy of Environment and Climate Change Canada. 
The information presented is combined from multiple Environment and Climate Change Canada data sources 
and all effort is made to be accurate. However, if you find something missing or incorrect please send your feedback. 

Column labels for normals and extremes:

Suffix	Meaning
_v	Calculated value (max, min or mean)
_s	Standard deviation of mean
_c	Count of (number of) values included
_d	Date range for values
_y	Years where extreme occurred (limited to first 40)
For monthly normal and extremes, the dates are always listed as the first day of the month. However, the data is for the first until the last day of the monthly (or until the current day for the ongoing month).


https://calgary.weatherstats.ca/download.html


"""
weather_file="input/calgary/weatherstats_calgary_daily.csv"

import numpy as np
import pandas as pd
#from   ydata_profiling import ProfileReport
import geopandas as gpd
from shapely import LineString
from scipy.stats import mode
# Function to transform the point string
def transform_point(point_str):
    coords_str = point_str.replace('POINT (', '').replace(')', '')
    longitude_str, latitude_str = coords_str.split()
    return float(longitude_str), float(latitude_str)

# Function to transform the multiline string
def transform_multiline(multiline_str):
    coords_str = multiline_str.replace('MULTILINESTRING ((', '').replace('))', '')
    start_str, end_str = coords_str.split(', ')
    start_longitude_str, start_latitude_str = start_str.split()
    end_longitude_str, end_latitude_str = end_str.split()
    start_longitude = float(start_longitude_str)
    start_latitude = float(start_latitude_str)
    end_longitude = float(end_longitude_str)
    end_latitude = float(end_latitude_str)
    return start_longitude, start_latitude, end_longitude, end_latitude


def extract_lines(df):
# Extract individual LineString geometries and attributes
    lines = []
    for idx, row in df.iterrows():
        multiline = row['geometry']
        # If the geometry is a MultiLineString, iterate over each LineString segment
        if multiline.geom_type == 'MultiLineString':
            for line in multiline.geoms:
                cord=list(line.coords)
                for kk in range(len(cord)-1):  
                    row_tmp=row.copy()
                    row_tmp['geometry']=LineString(cord[kk:kk+2])
                    lines.append(row_tmp)
                    # lines.append({
                    #     'P_ZONE': row['P_ZONE'],
                    #     'STATUS_IND': row['STATUS_IND'],
                    #     'LENGTH': row['LENGTH'],
                    #     'DIAM': row['DIAM'],
                    #     'MATERIAL': row['MATERIAL'],
                    #     'YEAR': row['YEAR'],
                    #     'GLOBALID': row['GLOBALID'],
                    #     'geometry': LineString(cord[kk:kk+2])  # Add the LineString geometry
                    # })
        else :
            print("OOPS non multiline")
    # Create a new GeoDataFrame from the list of dictionaries
    dff = gpd.GeoDataFrame(lines, crs=df.crs)
    dff=dff.reset_index(drop=True)
    # Drop duplicate rows based on geometry (optional)
   # dff = dff.drop_duplicates(subset=['geometry'])
    return(dff)
# Initial CRS: Specify crs='EPSG:4326' when creating the GeoDataFrame to indicate that the coordinates are in 
#the WGS84 geographic CRS (latitude and longitude).

## Use gdf.to_crs('EPSG:3762') to convert the GeoDataFrame to the EPSG:3762 CRS, which is a suitable projected CRS 
#for Calgary. Convert to EPSG:3762 (NAD83(CSRS) / UTM zone 11N)

public_water_main_df = gpd.read_file(public_water_file, GEOM_POSSIBLE_NAMES="MULTILINESTRING")
public_water_main_df=public_water_main_df.drop_duplicates() #drop duplicate records
public_water_main_df.DIAM=public_water_main_df.DIAM.astype(int)

#fill zero diameter with most popular diameter
cat,freq=np.unique(public_water_main_df.DIAM,return_counts=True)
public_water_main_df.loc[public_water_main_df['DIAM']==0,'DIAM']=cat[freq.argmax()]
#set CRS to longitude latitude format type
public_water_main_df.crs = 'EPSG:4326'
#transform longitude and latitude to cartesian format (x,y)
public_water_main_df = public_water_main_df.to_crs(epsg=3762)

#split the multlines of one type pipe to several singel pipelines with same features 
public_water_main_lines=extract_lines(public_water_main_df)
public_water_main_lines['TYPE']='MAIN' #set an MIAN as main type of pipe
public_water_main_lines['ind']=public_water_main_lines.index
public_water_main_lines['YEAR']=pd.to_datetime(public_water_main_lines['YEAR'],errors='coerce')
public_water_main_lines['YEAR']=public_water_main_lines['YEAR'].dt.year
################
public_water_service_df=gpd.read_file(public_water_service_file, GEOM_POSSIBLE_NAMES="line")
public_water_service_df=public_water_service_df.drop_duplicates()
cat,freq=np.unique(public_water_service_df.BUILDING_TYPE,return_counts=True)
public_water_service_df.loc[public_water_service_df['BUILDING_TYPE']=='','BUILDING_TYPE']=cat[freq.argmax()]
public_water_service_df['INSTALLED_DATE']=pd.to_datetime(public_water_service_df['INSTALLED_DATE'],errors='coerce')
public_water_service_df['INSTALLED_DATE']=public_water_service_df['INSTALLED_DATE'].dt.year

public_water_service_df.crs = 'EPSG:4326'
public_water_service_df = public_water_service_df.to_crs(epsg=3762)
public_water_service_lines_df=extract_lines(public_water_service_df)
public_water_service_lines_df['ind']=public_water_service_lines_df.index
public_water_service_lines_df['LENGTH']=public_water_service_lines_df['geometry'].length
public_water_service_lines_df=public_water_service_lines_df.rename(columns={'MATERIAL_TYPE':'MATERIAL','INSTALLED_DATE':'YEAR','PIPE_DIAMETER (mm)':'DIAM','BUILDING_TYPE':'TYPE'})

#public_water_service_lines_df.DIAM=public_water_service_lines_df.DIAM.astype(int)
public_water_service_lines_df.DIAM=pd.to_numeric(public_water_service_lines_df.DIAM,errors='coerce')
#fill less than 10mm diameter with most popular diameter
cat,freq=np.unique(public_water_service_lines_df.DIAM,return_counts=True)
public_water_service_lines_df.loc[public_water_service_lines_df['DIAM']<10,'DIAM']=cat[freq.argmax()]

m_pipes=pd.concat([public_water_main_lines[['MATERIAL','YEAR','DIAM','TYPE','LENGTH','ind','geometry']],
          public_water_service_lines_df[['MATERIAL','YEAR','DIAM','TYPE','LENGTH','ind','geometry']]],axis=0,ignore_index=True)

m_pipes['PIPE_ID']=m_pipes.index
m_pipes.to_csv("ALL_PIPE_GEOMETRY.csv",index=False)


#transform YEAR to have only year part of date


public_water_main_break_df=gpd.read_file(water_main_break_file, GEOM_POSSIBLE_NAMES="point")
public_water_main_break_df=public_water_main_break_df.drop_duplicates()


"""
BREAK_TYPE
A - Full Circular B - Split C - Corrosion D - Fitting E - Joint F - Diagonal Crack G - Hole S - Saddle
"""
public_water_main_break_df['A']=public_water_main_break_df['BREAK_TYPE'].str.contains('A')
public_water_main_break_df['B']=public_water_main_break_df['BREAK_TYPE'].str.contains('B')
public_water_main_break_df['C']=public_water_main_break_df['BREAK_TYPE'].str.contains('C')
public_water_main_break_df['D']=public_water_main_break_df['BREAK_TYPE'].str.contains('D')
public_water_main_break_df['E']=public_water_main_break_df['BREAK_TYPE'].str.contains('E')
public_water_main_break_df['F']=public_water_main_break_df['BREAK_TYPE'].str.contains('F')
public_water_main_break_df['G']=public_water_main_break_df['BREAK_TYPE'].str.contains('G')
public_water_main_break_df['S']=public_water_main_break_df['BREAK_TYPE'].str.contains('S')

public_water_main_break_df.crs = 'EPSG:4326'
public_water_main_break_df = public_water_main_break_df.to_crs(epsg=3762)
max_distance=20 #maximum distance for finding the nearest pipe to break point in m
nearest_main = gpd.sjoin_nearest( public_water_main_break_df,m_pipes,distance_col='distance'
                                 ,how='inner',lsuffix='left', rsuffix='right',max_distance=max_distance)
print('Distance between break points and nearest pipelines calcualted for matching the break with pipes\nSummary of distance mesures are: ')
print(nearest_main['distance'].describe(percentiles=[.25, .5, .75,.95,.99,.999]))
nearest_main=nearest_main.drop(columns=['distance'])

#set the center of pipe as a georefrence for the pipe
nearest_main['X']=[(m_pipes.loc[i].geometry.xy[0][0]+m_pipes.loc[i].geometry.xy[0][1])/2 for i in nearest_main['index_right'].values]
nearest_main['Y']=[(m_pipes.loc[i].geometry.xy[1][0]+m_pipes.loc[i].geometry.xy[1][1])/2 for i in nearest_main['index_right'].values]
water_pressure_df=pd.read_csv(water_pressure_zone_file)
water_pressure_df=gpd.read_file(water_pressure_zone_file,GEOM_POSSIBLE_NAMES="MULTIPOLYGON")
water_pressure_df.crs = 'EPSG:4326'
water_pressure_df = water_pressure_df.to_crs(epsg=3762)

df= gpd.sjoin_nearest( nearest_main,water_pressure_df,distance_col='distance',how='inner',lsuffix='l', rsuffix='r')#, op='nearest')
df=df[['PIPE_ID','BREAK_DATE', 'A', 'B',
       'C', 'D', 'E', 'F', 'G', 'S', 'MATERIAL', 'YEAR', 'DIAM',
       'TYPE', 'LENGTH', 'ZONE','X','Y']]
df['BREAK_DATE']=pd.to_datetime(df['BREAK_DATE'],errors='coerce')
df['BREAK_DATE']=df['BREAK_DATE'].dt.year
df=df.rename(columns={'YEAR':'INSTALLED_YEAR'})
df=df.sort_values(by=['PIPE_ID', 'BREAK_DATE']) 
df['BREAKED']=1   #number of the current year breaks
df['BREAKS']=0   #total number of breaks since installation

#find most poupular pipe diameter for each group of usage TYPE
most_p=df.groupby('TYPE')['DIAM'].apply(lambda x: x.mode())
mp_dic=dict(zip([x[0] for x in most_p.index],most_p.values))

df.loc[df.DIAM.isna(),'DIAM']=df[df.DIAM.isna()]['TYPE'].map(mp_dic)
#unify the Matrial field as some materials have 2 or more detials 
material_descriptions = [
    'Asbestos Cement (AC)',
    'Bonded Ductile Iron (BDI)',
    'Cast Iron (CI)',
    'Cured-In-Place Pipe (CIPP)',
    'Concrete (CON)',
    'Copper (CU)',
    'Cast Iron (CI)',  # Note: Same as 'Cast Iron' above
    'Copper (CU)',  # Note: Same as 'Copper' above
    'Cross-linked Polyethylene (PEX)',
    'Ductile Iron (DI)',
    'Ductile Iron (DI)',  # Note: Same as 'Ductile Iron' above
    'Enamel-Coated Iron (ECI)',
    'Fiberglass Reinforced Polyvinyl Chloride (FPVC)',
    'Polyvinyl Chloride (PCI)',
    'Polyethylene Wrapped Ductile Iron (PDI)',
    'Polyethylene (PE)',
    'Primed (PRIM)',
    'Polyvinyl Chloride (PVC)',
    'Polyvinyl Chloride with Glass Reinforcement (PVCG)',
    'Polyethylene Wrapped Ductile Iron (PDI)',  # Note: Same as 'Polyethylene Wrapped Ductile Iron' above
    'Polyvinyl Chloride (PVC)',  # Note: Same as 'Polyvinyl Chloride' above
    'Steel (ST)',
    'Trenchless Yellow Jacketed Ductile Iron (TUDI)',
    'Trenchless Ductile Iron (TWD)',
    'Unknown',
    'Yellow Jacketed Ductile Iron (YDI)',
    'Yield Strength (YST)',
    'Yellow Jacketed Urethane Ductile Iron (YUDI)',
    'Yellow Jacketed Ductile Iron (YDI)'  # Note: Same as 'Yellow Jacketed Ductile Iron' above
]


material_long_descriptions = [
    'Asbestos Cement: A type of cement made with asbestos fibers, used for its strength and durability but now less common due to health concerns.',
    'Bonded Ductile Iron: Ductile iron pipes that have been bonded or coated with a protective layer to enhance their durability and resistance to corrosion.',
    'Cast Iron: A type of iron with a high carbon content that is cast into molds, known for its strength and brittleness.',
    'Cured-In-Place Pipe: A method of rehabilitating existing pipelines by inserting a liner that is cured in place to form a new pipe within the old one.',
    'Concrete: A construction material composed of cement, aggregate, and water, known for its strength and durability.',
    'Copper: A highly conductive metal used in pipes for its resistance to corrosion and ability to handle high pressures.',
    'Cast Iron: Same as above, referring to pipes made from cast iron.',
    'Copper: Same as above, referring to pipes made from copper.',
    'Cross-linked Polyethylene: A type of polyethylene with cross-linked molecular structure, making it more resistant to temperature and pressure changes.',
    'Ductile Iron: A type of iron with added elements to improve its ductility, making it more flexible and resistant to breakage.',
    'Ductile Iron: Same as above, referring to pipes made from ductile iron.',
    'Enamel-Coated Iron: Iron pipes coated with an enamel layer to protect against corrosion and wear.',
    'Fiberglass Reinforced Polyvinyl Chloride: A type of pipe made from PVC reinforced with fiberglass, providing extra strength and durability.',
    'Polyvinyl Chloride: A type of plastic pipe used for its durability and resistance to corrosion, commonly used in water systems.',
    'Polyethylene Wrapped Ductile Iron: Ductile iron pipes wrapped with a layer of polyethylene for additional protection against corrosion.',
    'Polyethylene: A flexible plastic material used in pipes, known for its resistance to chemicals and ease of installation.',
    'Primed: Pipes that have been treated with a primer to improve adhesion of coatings or linings.',
    'Polyvinyl Chloride: Same as above, referring to pipes made from PVC.',
    'Polyvinyl Chloride with Glass Reinforcement: PVC pipes reinforced with glass fibers for enhanced strength and durability.',
    'Polyethylene Wrapped Ductile Iron: Same as above, referring to pipes wrapped with polyethylene.',
    'Polyvinyl Chloride: Same as above, referring to pipes made from PVC.',
    'Steel: A strong metal alloy used in pipes, known for its strength and ability to handle high pressures.',
    'Trenchless Yellow Jacketed Ductile Iron: Ductile iron pipes with a yellow jacket used in trenchless installation methods.',
    'Trenchless Ductile Iron: Ductile iron pipes used in trenchless installation methods, allowing for less invasive pipe replacement or repair.',
    'Unknown: Pipes with an unspecified or unknown material type.',
    'Yellow Jacketed Ductile Iron: Ductile iron pipes coated with a yellow jacket for added protection and visibility.',
    'Yield Strength: A measure of the strength of a material, indicating the maximum stress that can be applied without causing permanent deformation.',
    'Yellow Jacketed Urethan Ductile Iron: Same as above, referring to pipes with a yellow jacket coating.',
    'Yellow Jacketed Ductile Iron: Same as above, referring to pipes with a yellow jacket coating.'
]

mat,mat_c=np.unique(df.MATERIAL,return_counts=True)
mat_pd=pd.DataFrame({'Material':mat,'count':mat_c,'desc':material_descriptions,'long_des':material_long_descriptions})
mat_dic=dict(zip(mat_pd['Material'],mat_pd['desc']))
df['MATERIAL']=df['MATERIAL'].map(mat_dic)

#%% add climate features to dataset
max_year=df['BREAK_DATE'].max()
min_year=df['BREAK_DATE'].min()
weather_df=pd.read_csv(weather_file,low_memory=False)
# Convert 'Date' to datetime format
weather_df['date'] = pd.to_datetime(weather_df['date'])
weather_df=weather_df[weather_df['date']>=pd.to_datetime('%4d/01/01 00:00:00'%min_year)]
# Set 'Date' as the index for time series operations
weather_df.set_index('date', inplace=True)

# Resample to annual average temperature
annual_average_temp = weather_df['avg_temperature'].resample('A').mean()

# Resample to annual sum of temperatures
annual_std_temp = weather_df['avg_temperature'].resample('A').std()

# Resample to annual sum of HDD and CDD
annual_sum_hdd = weather_df['heatdegdays'].resample('A').sum()
annual_sum_cdd = weather_df['cooldegdays'].resample('A').sum()

# Resample to annual sum of precipitation
annual_sum_precipitation = weather_df['precipitation'].resample('A').sum()

# Resample to annual sum of rain
annual_sum_rain = weather_df['rain'].resample('A').sum()
# Resample to annual sum of snow on the ground
annual_sum_snow = weather_df['snow_on_ground'].resample('A').sum()

wdf=pd.DataFrame(data={'year':range(min_year,max_year+1),'avg_temp':annual_average_temp.values,'std_tmp':annual_std_temp.values,
                       'HDD':annual_sum_hdd.values,'CDD':annual_sum_cdd.values,'precipitation':annual_sum_precipitation.values,
                       'rain':annual_sum_rain.values,'snow':annual_sum_snow.values})
#set wether condition for current year equal to last year
wdf.iloc[-1,1:]=wdf.iloc[-2,1:]
wdf.set_index('year', inplace=True)
#%%FOr each pie add record from 1956 till 2023 and those records befor instalaation of each 
#pippe will be et to have zero age until installation year


#df['Category'] = df['Category'].cat.add_categories(['D'])

num_pipes=len(np.unique(df['PIPE_ID']))

i=0
tsfl_l=[]
for p,pipe in df.groupby('PIPE_ID'):
    df_seq = pipe.copy()
    brk,cnt=np.unique(df_seq['BREAK_DATE'],return_counts=True)
    brk_cnt=dict(zip(brk,cnt)) 
    df_seq['BREAKED']=[brk_cnt[d] for d in df_seq['BREAK_DATE']]
    df_seq.drop_duplicates(subset=['BREAK_DATE'],  keep='first', inplace=True, ignore_index=True)

    df_seq=df_seq.set_index('BREAK_DATE')
    df_seq_sample=df_seq.iloc[0].copy()
    df_seq_sample[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'S', 'BREAKED']]=0
    #df_seq['BREAK_DATE'] = pd.to_datetime(df_seq['BREAK_DATE'], format='%Y')
    
    # Get installed year (assuming pipe['INSTALLED_YEAR'] is a single value or list)
    installed_year =df_seq['INSTALLED_YEAR'].values[0] # pd.to_datetime(pipe['INSTALLED_YEAR'].values[0], format='%Y')
    df_seq=df_seq[df_seq.index>=installed_year]
    indexes=df_seq.index
    
    # Reindex to include missing years
    df_seq = df_seq.reindex(range(min_year,max_year+1))
    df_seq.loc[ ~df_seq.index.isin(indexes)]=df_seq_sample.values
    df_seq['BREAKS']=df_seq['BREAKED'].cumsum()
    # df_seq['X']=df_seq['geometry'].map(lambda c: c.x)
    # df_seq['Y']=df_seq['geometry'].map(lambda c: c.y)
    
    df_seq['TSLF'] = df_seq.groupby(np.cumsum(df_seq['BREAKED']>0)).cumcount()

# Adjust TSLF for the first occurrence in each group (where Breaks == 1)
    df_seq.loc[df_seq['BREAKED'] >0, 'TSLF'] = 0
    
    df_seq=df_seq.reset_index()
    #df_seq=df_seq.drop(columns=['geometry'])
    

    df_seq['year']=df_seq['BREAK_DATE']
    df_seq.set_index('year',inplace=True)
    # Now df_seq will have all years from installed_y to current_year
    df_seq=pd.concat([df_seq,wdf],axis=1,join='inner').reset_index()

    
    df_seq=df_seq.astype({'BREAK_DATE':'int', 'PIPE_ID':'int', 'A':'int', 'B':'int', 'C':'int', 'D':'int', 'E':'int', 'F':'int'
                   , 'G':'int', 'S':'int','MATERIAL':'category', 'INSTALLED_YEAR':'int'
                   , 'DIAM':'int', 'TYPE':'category', 'LENGTH':'float32', 'ZONE':'category','BREAKED':'int',
                   'BREAKS':'int', 'X':'float32', 'Y':'float32'})
        
    if i==0:
        dff=df_seq.copy()
    else:
        dff=pd.concat([dff,df_seq],axis=0)
    i+=1
    tsfl_l=tsfl_l+list(df_seq.loc[df_seq[(df_seq['TSLF']==0) & (df_seq.index!=0)].index-1,'TSLF'].values)
    if i%10==0:print(i,"/",num_pipes,'Pipes processed',p)
print("Summary of time since last failure fo %d records:min=%f, max=%f, mean=%f, std=%f, mode=%f"%(\
       len(tsfl_l),np.min(tsfl_l),np.max(tsfl_l),np.mean(tsfl_l), np.std(tsfl_l), mode(tsfl_l)[0]))
age=dff['INSTALLED_YEAR']-dff['year']
age[age<0]=0
dff['age']=age    
dff.drop(columns=['BREAK_DATE'],inplace=True)   
dff.to_csv("breakdfv2.csv",index=False)

#%% update the MATERIAL field 
import numpy as np
import pandas as pd
#Some MAterila type fields are identical with diffrent abbreviation. her I update them 
material_descriptions = [
    'Asbestos Cement (AC)',
    'Bonded Ductile Iron (BDI)',
    'Cast Iron (CI)',
    'Cured-In-Place Pipe (CIPP)',
    'Concrete (CON)',
    'Copper (CU)',
    'Cast Iron (CI)',  # Note: Same as 'Cast Iron' above
    'Copper (CU)',  # Note: Same as 'Copper' above
    'Cross-linked Polyethylene (PEX)',
    'Ductile Iron (DI)',
    'Ductile Iron (DI)',  # Note: Same as 'Ductile Iron' above
    'Enamel-Coated Iron (ECI)',
    'Fiberglass Reinforced Polyvinyl Chloride (FPVC)',
    'Polyvinyl Chloride (PCI)',
    'Polyethylene Wrapped Ductile Iron (PDI)',
    'Polyethylene (PE)',
    'Primed (PRIM)',
    'Polyvinyl Chloride (PVC)',
    'Polyvinyl Chloride with Glass Reinforcement (PVCG)',
    'Polyethylene Wrapped Ductile Iron (PDI)',  # Note: Same as 'Polyethylene Wrapped Ductile Iron' above
    'Polyvinyl Chloride (PVC)',  # Note: Same as 'Polyvinyl Chloride' above
    'Steel (ST)',
    'Trenchless Yellow Jacketed Ductile Iron (TUDI)',
    'Trenchless Ductile Iron (TWD)',
    'Unknown',
    'Yellow Jacketed Ductile Iron (YDI)',
    'Yield Strength (YST)',
    'Yellow Jacketed Urethane Ductile Iron (YUDI)',
    'Yellow Jacketed Ductile Iron (YDI)'  # Note: Same as 'Yellow Jacketed Ductile Iron' above
]


material_long_descriptions = [
    'Asbestos Cement: A type of cement made with asbestos fibers, used for its strength and durability but now less common due to health concerns.',
    'Bonded Ductile Iron: Ductile iron pipes that have been bonded or coated with a protective layer to enhance their durability and resistance to corrosion.',
    'Cast Iron: A type of iron with a high carbon content that is cast into molds, known for its strength and brittleness.',
    'Cured-In-Place Pipe: A method of rehabilitating existing pipelines by inserting a liner that is cured in place to form a new pipe within the old one.',
    'Concrete: A construction material composed of cement, aggregate, and water, known for its strength and durability.',
    'Copper: A highly conductive metal used in pipes for its resistance to corrosion and ability to handle high pressures.',
    'Cast Iron: Same as above, referring to pipes made from cast iron.',
    'Copper: Same as above, referring to pipes made from copper.',
    'Cross-linked Polyethylene: A type of polyethylene with cross-linked molecular structure, making it more resistant to temperature and pressure changes.',
    'Ductile Iron: A type of iron with added elements to improve its ductility, making it more flexible and resistant to breakage.',
    'Ductile Iron: Same as above, referring to pipes made from ductile iron.',
    'Enamel-Coated Iron: Iron pipes coated with an enamel layer to protect against corrosion and wear.',
    'Fiberglass Reinforced Polyvinyl Chloride: A type of pipe made from PVC reinforced with fiberglass, providing extra strength and durability.',
    'Polyvinyl Chloride: A type of plastic pipe used for its durability and resistance to corrosion, commonly used in water systems.',
    'Polyethylene Wrapped Ductile Iron: Ductile iron pipes wrapped with a layer of polyethylene for additional protection against corrosion.',
    'Polyethylene: A flexible plastic material used in pipes, known for its resistance to chemicals and ease of installation.',
    'Primed: Pipes that have been treated with a primer to improve adhesion of coatings or linings.',
    'Polyvinyl Chloride: Same as above, referring to pipes made from PVC.',
    'Polyvinyl Chloride with Glass Reinforcement: PVC pipes reinforced with glass fibers for enhanced strength and durability.',
    'Polyethylene Wrapped Ductile Iron: Same as above, referring to pipes wrapped with polyethylene.',
    'Polyvinyl Chloride: Same as above, referring to pipes made from PVC.',
    'Steel: A strong metal alloy used in pipes, known for its strength and ability to handle high pressures.',
    'Trenchless Yellow Jacketed Ductile Iron: Ductile iron pipes with a yellow jacket used in trenchless installation methods.',
    'Trenchless Ductile Iron: Ductile iron pipes used in trenchless installation methods, allowing for less invasive pipe replacement or repair.',
    'Unknown: Pipes with an unspecified or unknown material type.',
    'Yellow Jacketed Ductile Iron: Ductile iron pipes coated with a yellow jacket for added protection and visibility.',
    'Yield Strength: A measure of the strength of a material, indicating the maximum stress that can be applied without causing permanent deformation.',
    'Yellow Jacketed Urethan Ductile Iron: Same as above, referring to pipes with a yellow jacket coating.',
    'Yellow Jacketed Ductile Iron: Same as above, referring to pipes with a yellow jacket coating.'
]
import pandas as pd 
dff= pd.read_csv("breakdfv1.csv")


mat,mat_c=np.unique(dff.MATERIAL,return_counts=True)
mat_pd=pd.DataFrame({'Material':mat,'count':mat_c,'desc':material_descriptions,'long_des':material_long_descriptions})
mat_dic=dict(zip(mat_pd['Material'],mat_pd['desc']))
dff['MATERIAL']=dff['MATERIAL'].map(mat_dic)
dff.to_csv("breakdfv2.csv",index=False)
m_pipes=pd.read_csv("ALL_PIPE_GEOMETRY.csv")
m_pipes['MATERIAL']=m_pipes['MATERIAL'].map(mat_dic)
dff.to_csv("ALL_PIPE_GEOMETRY.csv",index=False)


#%%
# run the profile report 
datatype={'BREAK_DATE':'int', 'PIPE_ID':'int', 'A':'int', 'B':'int', 'C':'int', 
          'D':'int', 'E':'int', 'F':'int', 'G':'int', 'S':'int','MATERIAL':'category',
          'INSTALLED_YEAR':'int', 'DIAM':'int', 'TYPE':'category', 
          'LENGTH':'float32', 'ZONE':'category','BREAKED':'int','BREAKS':'int',
          'X':'float32', 'Y':'float32', 'TSLF':np.int32, 'avg_temp':np.float64, 
          'std_tmp':np.float64, 'HDD':np.float64,
          'CDD':np.float64, 'precipitation':np.float64, 'rain':np.float64, 'snow':np.float64}


dff=pd.read_csv("breakdf.csv",dtype=datatype)


#pd.rename(columns={'BREAK_DATE':'year'})#dff['year']=dff['BREAK_DATE']
first_year=dff['year'].min()
last_year=dff['year'].max()
dff.set_index('year',inplace=True)
dff.reindex()
cat_columns=dff.select_dtypes(['category']).columns
dff[cat_columns] = dff[cat_columns].apply(lambda x: x.cat.codes)

#%%
profile = dff.profile_report(title=' composed df Pandas Profiling Report')

profile.to_file("compossed_df.html")


#%%
profile = public_water_main_break_df[['BREAK_DATE', 'BREAK_TYPE', 'STATUS', 'point']].profile_report(title='Pandas Profiling Report')

profile.to_file("public_water_main_break.html")

profile = public_water_main_df[['P_ZONE', 'STATUS_IND', 'LENGTH', 'DIAM', 'MATERIAL', 'YEAR',
       'GLOBALID', 'MULTILINESTRING']].profile_report(title=' public_water_main_df Pandas Profiling Report'
                                                      ,correlations={"auto": {"calculate": False}})

profile.to_file("public_water_main.html")

profile = public_water_service_df[['BUILDING_TYPE', 'WATER_SERVICE_ADDRESS', 'MATERIAL_TYPE',
       'PIPE_DIAMETER (mm)', 'INSTALLED_DATE', 'line']].profile_report(title=' public_water_service_df Pandas Profiling Report')

profile.to_file("public_water_service.html")
"""
BREAK_TYPE
A - Full Circular B - Split C - Corrosion D - Fitting E - Joint F - Diagonal Crack G - Hole S - Saddle
"""