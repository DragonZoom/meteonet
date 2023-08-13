# plot routines

import matplotlib.pyplot as plt
from matplotlib import colors
from loader.utilities import get_item_by_date, load_map, split_date, map_to_classes
import torch

# autre colormap pluie: https://unidata.github.io/python-gallery/examples/Precipitation_Map.html#sphx-glr-download-examples-precipitation-map-py

def plot_meteonet_rainmaps( ds, date, lon=None, lat=None, zone=None, title=None, n=2, size=5):
    """ plot rainfaill inputs of an element chosen by date from a Meteonoet dataset
       ds: a meteonet dataset
       date: date to display
       lon, lat, zone: provided by data.constants
       n: number of maps per line (2 default)
       size: size of map (5 default)
    """
    # inspired from https://github.com/meteofrance/meteonet/blob/master/notebooks/radar/open_rainfall.ipynb

    idx = get_item_by_date(ds, date)    
    if idx == None: return None

    p = ds.params['input_len'] // n
    files = ds.params['files']
    items = ds.params['items']
    
    fig, ax = plt.subplots( p, n, figsize=(size*n,size*p))
    if title: fig.suptitle(title, fontsize=16)

    # Choose the colormap
    cmap = colors.ListedColormap(['silver','white', 'darkslateblue', 'mediumblue','dodgerblue', 
                                  'skyblue','olive','mediumseagreen','cyan','lime','yellow',
                                  'khaki','burlywood','orange','brown','pink','red','plum'])
    bounds = [-1,0,2,4,6,8,10,15,20,25,30,35,40,45,50,55,60,65,75]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    zone = '' if zone is None else f' - {zone} zone' 
    print_lat_lon = lon is not None and lat is not None

    for i in range(p):
        for j in range(n):
            file = files[items[idx][2*i+j]]
            if print_lat_lon:
                pl = ax[i,j].pcolormesh(lon, lat, load_map(file), cmap=cmap, norm=norm)
            else:
                pl = ax[i,j].imshow(load_map(file), cmap=cmap, norm=norm)
            if j==0 and print_lat_lon: ax[i,0].set_ylabel('latitude (degrees_north)')
            y,M,d,h,m = split_date( file)
            ax[i,j].set_title( f'{y}/{M}/{d} {h}:{m}{zone}')
    if print_lat_lon:
        for j in range(n):
            ax[p-1,j].set_xlabel('longitude (degrees_east)')

    # Plot the color bar
    fig.colorbar( pl ,ax=ax.ravel().tolist(),cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, 
                  orientation= 'vertical').set_label('Rainfall (in 1/100 mm) / -1 : missing values')
    plt.show()

def plot_inference(ds, date, model, thresholds, lon, lat, zone, title):
    """ plot an inference for a given date and compare to truth """
    idx = get_item_by_date(ds, date)    
    if idx == None: return None

    item = ds[idx]
    model.eval()
    with torch.no_grad():
        pred = model(item['inputs'].unsqueeze(0))

    y,M,d,h,m = split_date(item['target_name'])

    pred = 1*(torch.sigmoid(pred[0,0])>.5) + (torch.sigmoid(pred[0,1])>.5) + (torch.sigmoid(pred[0,2])>.5)
    pred[0,0] = 3
    true = 1*(item['target']>thresholds[0]) + (item['target']>thresholds[1]) + (item['target']>thresholds[2])
    
    cmap = colors.ListedColormap(['white', 'mediumblue','skyblue','cyan'])

    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    fig.suptitle(title) 
    
    ax[0].set_title(f'{y}/{M}/{d} {h}:{m} Prediction')
    ax[0].set_ylabel('latitude')
    ax[0].set_xlabel('longitude')
    ax[1].set_xlabel('longitude')
    
    fig.text(0.3,-0.05,'thresholds: mediumblue >= 0.8, skyblue >= 8.3, cyan >=20.3', )
    
    ax[0].pcolormesh(lon, lat, pred, cmap=cmap) 
    ax[1].set_title(f'{y}/{M}/{d} {h}:{m} Truth')
    ax[1].pcolormesh(lon, lat, true, cmap=cmap) 

    plt.show()

def plot_CT( dataset, date, model, thresholds, c):
    """
        for a given date in a dataset and a class c, plot prediction and ground-truth, true positives, false positives 
        and false negative
    """
    idx = get_item_by_date(dataset, date)
    if idx == None: return None

    item = dataset[idx]
    print(item['inputs'].unsqueeze(0).shape)
    model.eval()
    with torch.no_grad():
        y_hat = model(item['inputs'].unsqueeze(0))

    true = map_to_classes(item['target'].unsqueeze(0), thresholds)
    pred = (torch.sigmoid(y_hat) > 0.5)
    
    plt.figure(figsize=(10,10))
    plt.suptitle( f'class {c}')
    plt.subplot(2,2,1)
    plt.imshow(pred[0,c]*1 + 2*true[0,c])
    plt.subplot(2,2,2)
    TP = pred[0,c]*true[0,c]
    plt.title(f'TP: {int(TP.sum())}')
    plt.imshow(TP)
    plt.subplot(2,2,3)
    FP = pred[0,c]*(true[0,c]==False)
    plt.title(f'FP: {int(FP.sum())}')
    plt.imshow(FP)
    plt.subplot(2,2,4)
    FN = (pred[0,c] == False)*(true[0,c])
    plt.title(f'FN: {int((FN).sum())}')
    plt.imshow(FN)
    plt.show()
