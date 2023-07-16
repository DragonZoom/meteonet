# plot meteonet rainmaps (todo: scores and inference)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from loader.utilities import load_map, next_date, split_date


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


coord = np.load('data/radar_coords_NW.npz',allow_pickle=True)
from data.constants import *

lon = coord['lons'][lat_extract_start:lat_extract_end, lon_extract_start:lon_extract_end]
lat = coord['lats'][lat_extract_start:lat_extract_end, lon_extract_start:lon_extract_end]

data,dates = get_data('data/rainmaps/', (2017,1,28,12, 10), 4)

plot_meteonet_rainmaps(data,dates,lon,lat, zone, 'Rainmaps with Meteonet style')


# autre colormap pluie: https://unidata.github.io/python-gallery/examples/Precipitation_Map.html#sphx-glr-download-examples-precipitation-map-py
