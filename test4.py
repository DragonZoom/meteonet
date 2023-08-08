# test inference

from loader.meteonet import MeteonetDataset
from pred_example.constants import *
import numpy as np
from glob import glob
from loader.plots import plot_meteonet_rainmaps

predex_ds = MeteonetDataset( glob("pred_example/rainmaps/*.npz"), 12, 18, 12,
                             wind_dir='pred_example/windmaps')

plot_meteonet_rainmaps( predex_ds, (2018, 3, 12, 5, 0), longitudes, latitudes, zone, n=4, size=4)
