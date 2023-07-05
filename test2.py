from loader.meteonet import MeteonetDataset
from loader.utilities import get_files, map_to_classes
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

thresholds_in_mmh = [0, 0.1, 1, 2.5] 
thresholds_in_cent_mm = [100*k/12 for k in thresholds_in_mmh] #CRF sur 5 minutes en 1/100 de mm

files = get_files( "data/rainmaps/y2016*.npz")

dataset = MeteonetDataset( { 'train': files, 'max': 1.0 }, 'train', 6, 10, 6)

data = DataLoader( dataset, batch_size = 1, shuffle = True)

batch = next(iter(data))

# map = map[15:20,0:5]

# colormap pluie: https://unidata.github.io/python-gallery/examples/Precipitation_Map.html#sphx-glr-download-examples-precipitation-map-py


def view_inference( inputs, target, estimated, thresholds, nlines=2):
    n = inputs.shape[0]
    if n % nlines:
        print(f'{nlines=} should divide {inputs.shape[0]=}')
        return
    
    for i in range(n):
        plt.subplot( nlines, n//nlines, i+1)
        plt.axis('off')
        plt.imshow(inputs[i])
        
    # classes = map_to_classes( target, thresholds_in_cent_mm)
    plt.show()


view_inference( batch['inputs'][0],
                batch['target'], None,
                thresholds_in_cent_mm
               )
