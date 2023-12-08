import dill
import os

dc = dill.load(open('/home/allen/data/rainbench/era5625_aaai/era5625_us.dill', "rb"))
print(dc)
for k, v in dc["variables"].items():
    print('hi')