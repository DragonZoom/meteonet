# Test inference

from glob import glob
from loader.meteonet import MeteonetDataset
from pred_example.constants import *
from loader.plots import plot_meteonet_rainmaps, plot_inference, plot_CT
from loader.utilities import map_to_classes
from models.unet import UNet
import torch

predex_ds = MeteonetDataset( glob("pred_example/rainmaps/*.npz"), 12, 18, 12, wind_dir='pred_example/windmaps')

# These normalisation constraints come from the train set used to generate weights
predex_ds.norm_factors = [ 8.903679332926599, 
                           159.08984183466058, 451.1743940059786,
                           19.10841423360815, 417.6967931487354]

plot_meteonet_rainmaps( predex_ds, (2018, 3, 12, 5, 0), longitudes, latitudes, zone,
                        'The 12 rainmaps as input to our model', n=4, size=4)

model = UNet(12*3, 3)
model.load_state_dict(torch.load('weights/model_dom_30m_wind.pt',  map_location=torch.device('cpu')))

thresholds   = [0.1, 1, 2.5]  # series of thresholds (unit: mm/h)
thresholds   = [100*k/12 for k in thresholds]

plot_inference( predex_ds, (2018, 3, 12, 5, 0), model.to('cpu'), thresholds, longitudes, latitudes, zone,
                'model rainmaps + windmaps')

plot_CT( predex_ds, (2018, 3, 12, 5, 0), model.to('cpu'), thresholds, 0)
plot_CT( predex_ds, (2018, 3, 12, 5, 0), model.to('cpu'), thresholds, 1)
plot_CT( predex_ds, (2018, 3, 12, 5, 0), model.to('cpu'), thresholds, 2)
