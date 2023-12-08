import numpy as np
import pandas as pd
from tqdm import tqdm
# from netCDF4 import Dataset
import dill

# # Open the netCDF file
# nc_file = Dataset('/home/allen/data/constants_5.625deg.nc', mode='r')
# # Print out the first couple of values for each variable
# for variable in nc_file.variables:
#     data = nc_file.variables[variable][:]
#     # Assuming the data is at least 1D, get the first two values
#     print(f"{variable} first 360 values:", data.flat[:360])
# # Loop through all variables and print their names and details
# for var_name in nc_file.variables:
#     var = nc_file.variables[var_name]
#     print(var_name, var)
# # Close the file
# nc_file.close()

# existing_mmap = np.memmap('/home/allen/data/rainbench/imerg5625/imerg_5625__imerg5625bi.mmap',\
#      dtype='float32', mode='r+')
# print(existing_mmap.shape)

# existing_mmap = np.memmap('/home/allen/data/rainbench/era5625_aaai/era5625_aaai__era5625_const.mmap',\
#      dtype='float32', mode='r+')

# a = np.memmap('/home/allen/data/rainbench/era5625_aaai/era5625_us.mmap',\
#      dtype='float32', mode='r+', shape=(1, 1, 5, 12))
# a = np.memmap('/home/allen/data/rainbench/era5625_aaai/era5625_us_const.mmap',\
#                dtype='float32', mode='w+', shape=(359400, 18, 5, 12)) 
# a[:] = existing_mmap[:,:,20:25,9:21] 



# storm_dill_dict = {'variables': {'imerg5625/precipitationcal': {'name': 'precipitationcal', 'mmap_name': 'storm_data_us.mmap', 'type': 'temp', 'dims': (5, 12), 'offset': 0, 'first_ts': 283996800.0, 'last_ts': 1577833200.0, 'tfreq_s': 3600, 'levels': [0]}}, 'memmap': {'storm_data_us.mmap': {'dims': (359400, 1, 5, 12), 'dtype': 'float32', 'daterange': (283996800.0, 1577833200.0), 'tfreq_s': 3600}}}

# # Path for the dill file
# dill_file_path = '/home/allen/data/rainbench/imerg5625/storm_data_us.dill'

# # Serialize and save the content to a dill file
# with open(dill_file_path, 'wb') as file:
#     dill.dump(storm_dill_dict, file)

# a.flush()

# print(existing_mmap[0, 0, 20, 8])

# print(a[0, 0, 0, 0])

# existing_mmap = np.memmap('/home/allen/data/rainbench/era5625_aaai/era5625_us.mmap',\
#      dtype='float32', mode='r+', shape=(359400, 18, 5, 11))
# print(existing_mmap.shape)

# # Contiguous US Bound 
# 90 - 5.625/2 = 87.1875 
# lat first 360 values: [-87.1875 -81.5625 -75.9375 -70.3125 -64.6875 -59.0625 -53.4375 -47.8125
#  -42.1875 -36.5625 -30.9375 -25.3125 -19.6875 -14.0625  -8.4375  -2.8125
#    2.8125   8.4375  14.0625  19.6875  25.3125  30.9375  36.5625  42.1875
#   47.8125  53.4375  59.0625  64.6875  70.3125  75.9375  81.5625  87.1875]

# lon first 360 values: [  0.      5.625  11.25   16.875  22.5    28.125  33.75   39.375  45.
#   50.625  56.25   61.875  67.5    73.125  78.75   84.375  90.     95.625
#  101.25  106.875 112.5   118.125 123.75  129.375 135.    140.625 146.25
#  151.875 157.5   163.125 168.75  174.375 180.    185.625 191.25  196.875
#  202.5   208.125 213.75  219.375 225.    230.625 236.25  241.875 247.5
#  253.125 258.75  264.375 270.    275.625 281.25  286.875 292.5   298.125
#  303.75  309.375 315.    320.625 326.25  331.875 337.5   343.125 348.75
#  354.375]

# Latitude: 24.5 N to 49.5 N -> index 20 - 24 //5 ~4.4444
# Longitude: -66.5 E to -125.0 E index 9 - 20 //11  ~10.4
#             55 TO 113.5 
# 


# existing_mmap = np.memmap('/home/allen/data/rainbench/era5625_aaai/era5625_aaai__era5625_const.mmap',\
#      dtype='float32', mode='r+', shape=(359400, 18, 32, 64))
# print(existing_mmap.shape)


# file = dill.load(open('/home/allen/data/rainbench/simsat5625/simsat5625.dill', 'rb'))
# print(file)

# d = {'variables': {'imerg5625/precipitationcal': {'name': 'precipitationcal', 'mmap_name': 'storm_data_us_flood_2000.mmap', 'type': 'temp', 'dims': (5, 12), 'offset': 0, 'first_ts': 946684800.0, 'last_ts': 1577833200.0, 'tfreq_s': 3600, 'levels': [0]}}, 'memmap': {'storm_data_us_flood_2000.mmap': {'dims': (175320, 1, 5, 12), 'dtype': 'float32', 'daterange': (946684800.0, 1577833200.0), 'tfreq_s': 3600}}}

# with open('storm_data_us_flood_2000.dill', 'wb') as file:
#     dill.dump(d, file)

d = {'variables': {'era5625/nino34': {'name': 'nino34', 'mmap_name': 'nino34_5625.mmap', 'type': 'temp', 'dims': (5, 12), 'offset': 0, 'first_ts': 283996800.0, 'last_ts': 1577833200.0, 'tfreq_s': 3600, 'levels': None}}, 'memmap': {'nino34_5625.mmap': {'dims': (359400, 1, 5, 12), 'dtype': 'float32', 'daterange': (283996800.0, 1577833200.0), 'tfreq_s': 3600}}}
with open('nino34_5625.dill', 'wb') as file:
    dill.dump(d, file)

# df = pd.read_csv('/home/allen/data/hourly_elnino_34.csv')
# hour_data = df.iloc[955464:1314864]['Value'].to_numpy()

# x1 = [np.full((1,5,12), i) for i in tqdm(hour_data)]
# y1 = np.array(x1)

# df = pd.read_csv('/home/allen/data/hourly_elnino_12.csv')
# hour_data = df.iloc[955464:1314864]['Value'].to_numpy()

# x2 = [np.full((1,5,12), i) for i in tqdm(hour_data)]
# y2 = np.array(x2)

# df = pd.read_csv('/home/allen/data/hourly_elnino_3.csv')
# hour_data = df.iloc[955464:1314864]['Value'].to_numpy()

# x3 = [np.full((1,5,12), i) for i in tqdm(hour_data)]
# y3 = np.array(x3)

# df = pd.read_csv('/home/allen/data/hourly_elnino_4.csv')
# hour_data = df.iloc[955464:1314864]['Value'].to_numpy()

# x4 = [np.full((1,5,12), i) for i in tqdm(hour_data)]
# y4 = np.array(x4)

# print(y1.mean(), y1.std())
# print(y2.mean(), y2.std())
# print(y3.mean(), y3.std())
# print(y4.mean(), y4.std())


# # Step 1: Convert hourly_data to memmap
# hourly_data_memmap = np.memmap('nino34_5625.mmap', dtype='float32', mode='w+', shape=(359400, 1, 5, 12))
# hourly_data_memmap[:,0:,:,:] = y1
# hourly_data_memmap[:,1:,:,:] = y2
# hourly_data_memmap[:,2:,:,:] = y3
# hourly_data_memmap[:,3:,:,:] = y4

# hourly_data_memmap.flush()
# Step 2: Concatenate the two memmaps
# concatenated_memmap = np.concatenate((existing_memmap, hourly_data_memmap))

# Step 3: Save the concatenated memmap if needed
# To save the concatenated memmap, you can simply overwrite the existing_memmap file
# with the new data if that's your intention.
# Make sure to handle file I/O appropriately for your use case.

# Close the memmaps when done
# hourly_data_memmap.close()


# hourly_data_memmap = np.memmap('hourly_data.mmap', dtype='float64', mode='r', shape=(359400, 1, 32, 64))
# # Determine the shape of the concatenated array
# concatenated_shape = (existing_memmap.shape[0] + hourly_data_memmap.shape[0])
# # Create a new memmap array for the concatenated data
# concatenated_memmap = np.memmap('concatenated_memmap.mmap', dtype=np.float64, mode='w+', shape=concatenated_shape)
# # Copy data from existing memmap to the concatenated memmap
# concatenated_memmap[:existing_memmap.shape[0]] = existing_memmap[:]
# # Copy data from hourly_data memmap to the concatenated memmap
# concatenated_memmap[existing_memmap.shape[0]:] = hourly_data_memmap[:]

