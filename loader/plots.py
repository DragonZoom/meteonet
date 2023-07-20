# plot routines

import matplotlib.pyplot as plt
from matplotlib import colors
from loader.utilities import get_item_by_date, load_map, split_date
from os.path import basename

# autre colormap pluie: https://unidata.github.io/python-gallery/examples/Precipitation_Map.html#sphx-glr-download-examples-precipitation-map-py

def plot_meteonet_rainmaps( ds, date, lon, lat, zone, title=None):
    """ plot rainfaill inputs of an element chosen by date from a Meteonoet dataset"""
    # inspired from https://github.com/meteofrance/meteonet/blob/master/notebooks/radar/open_rainfall.ipynb

    input_len = ds.params['input_len']
    files = ds.params['files']
    items = ds.params['items']
    idx = get_item_by_date(ds, date)
    
    if idx == None: return None
    
    fig, ax = plt.subplots(input_len//2, 2,figsize=(10,20))
    if title: fig.suptitle(title, fontsize=16)

    # Choose the colormap
    cmap = colors.ListedColormap(['silver','white', 'darkslateblue', 'mediumblue','dodgerblue', 
                                  'skyblue','olive','mediumseagreen','cyan','lime','yellow',
                                  'khaki','burlywood','orange','brown','pink','red','plum'])
    bounds = [-1,0,2,4,6,8,10,15,20,25,30,35,40,45,50,55,60,65,75]
    norm = colors.BoundaryNorm(bounds, cmap.N)


    for i in range(input_len//2):
        file = files[items[idx][2*i]]
        data = load_map(file)
        ax[i,0].pcolormesh(lon, lat, data, cmap=cmap, norm=norm)
        ax[i,0].set_ylabel('latitude (degrees_north)')
        y,M,d,h,m = split_date( basename(file))
        ax[i,0].set_title( f'{y}/{M}/{d} {h}:{m} - {zone} zone')
        
        file = files[items[idx][2*i+1]]
        data = load_map(file)
        pl = ax[i,1].pcolormesh(lon, lat, data, cmap=cmap, norm=norm)
        y,M,d,h,m = split_date( basename(file))
        ax[i,1].set_title( f'{y}/{M}/{d} {h}:{m} - {zone} zone')
        
    ax[input_len//2-1,0].set_xlabel('longitude (degrees_east)')
    ax[input_len//2-1,1].set_xlabel('longitude (degrees_east)')

    # Plot the color bar
    cbar = fig.colorbar(pl,ax=ax.ravel().tolist(),cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, 
                    orientation= 'vertical').set_label('Rainfall (in 1/100 mm) / -1 : missing values')
    plt.show()
