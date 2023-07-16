# plot meteonet rainmaps (todo: scores and inference)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from loader.utilities import load_map, next_date, split_date
from os.path import isfile

def get_data( ddir, date, num):
    y,M,d,h,m = date
    f = f'y{y}-M{M}-d{d}-h{h}-m{m}.npz'
    maps = []
    dates  = [] 
    while num > 0:
        maps.append(load_map(ddir+f))
        dates.append( f'{y}/{M}/{d} {h}:{m}')        
        f = next_date(f)
        y,M,d,h,m = split_date(f)
        num += -1
    return np.array(maps), dates

def plot_meteonet_rainmaps( data, dates, lon, lat, zone, title):
    # source: https://github.com/meteofrance/meteonet/blob/master/notebooks/radar/open_rainfall.ipynb
    fig, ax = plt.subplots(2, 2,figsize=(9,9))
    fig.suptitle(title, fontsize=16)

    # Choose the colormap
    cmap = colors.ListedColormap(['silver','white', 'darkslateblue', 'mediumblue','dodgerblue', 
                                  'skyblue','olive','mediumseagreen','cyan','lime','yellow',
                                  'khaki','burlywood','orange','brown','pink','red','plum'])
    bounds = [-1,0,2,4,6,8,10,15,20,25,30,35,40,45,50,55,60,65,75]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    pl=ax[0,0].pcolormesh(lon, lat, data[0,:,:],cmap=cmap, norm=norm)
    ax[0,0].set_ylabel('latitude (degrees_north)')
    ax[0,0].set_title(str(dates[0]) + " - "+  zone + " zone")

    pl=ax[0,1].pcolormesh(lon, lat, data[1,:,:],cmap=cmap, norm=norm)
    ax[0,1].set_title(str(dates[1]) + " - "+  zone + " zone")

    pl=ax[1,0].pcolormesh(lon, lat, data[2,:,:],cmap=cmap, norm=norm)
    ax[1,0].set_xlabel('longitude (degrees_east)')
    ax[1,0].set_ylabel('latitude (degrees_north)')
    ax[1,0].set_title(str(dates[2]) + " - "+  zone + " zone")

    pl=ax[1,1].pcolormesh(lon, lat, data[3,:,:],cmap=cmap, norm=norm)
    ax[1,1].set_xlabel('longitude (degrees_east)')
    ax[1,1].set_title(str(dates[3]) + " - "+  zone + " zone")

    # Plot the color bar
    cbar = fig.colorbar(pl,ax=ax.ravel().tolist(),cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, 
                    orientation= 'vertical').set_label('Rainfall (in 1/100 mm) / -1 : missing values')
    plt.show()


if isfile('data/rainmaps/y2018-M2-d5-h18-m55.npz'):
    print('reduced dataset')
    from data.constants import *
    datadir = "data"
elif isfile('meteonet/rainmaps/y2018-M11-d9-h0-m0.npz'):
    print('full dataset')
    datadir = "meteonet"
    from meteonet.constants import *
else:
    print('No dataset found. Please download one with download-meteonet-*.sh scripts.')

coord = np.load(f'{datadir}/radar_coords_NW.npz',allow_pickle=True)

lon = coord['lons'][lat_extract_start:lat_extract_end, lon_extract_start:lon_extract_end]
lat = coord['lats'][lat_extract_start:lat_extract_end, lon_extract_start:lon_extract_end]

data,dates = get_data(f'{datadir}/rainmaps/', (2017,3,1,12, 10), 4)

plot_meteonet_rainmaps(data,dates,lon,lat, zone, 'Rainmaps with Meteonet style')

# autre colormap pluie: https://unidata.github.io/python-gallery/examples/Precipitation_Map.html#sphx-glr-download-examples-precipitation-map-py
