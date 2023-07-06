from loader.meteonet import MeteonetDataset
from loader.utilities import get_files, map_to_classes
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from  model.unet import UNet

thresholds_in_mmh = [0, 0.1, 1, 2.5] 
thresholds_in_cent_mm = [100*k/12 for k in thresholds_in_mmh] #CRF sur 5 minutes en 1/100 de mm

files = get_files( "data/rainmaps/y2016*.npz")

dataset = MeteonetDataset( { 'train': files, 'max': 1.0 }, 'train', 6, 10, 6)
data = DataLoader( dataset, batch_size = 1, shuffle = True)
batch = next(iter(data))

# il faudra prendre un modèle entraîné
model = UNet( 6, 3)


# colormap pluie: https://unidata.github.io/python-gallery/examples/Precipitation_Map.html#sphx-glr-download-examples-precipitation-map-py


def view_inference( inputs, target, estimated, thresholds, nlines=2):
    n = inputs.shape[0]
    if n % nlines:
        print(f'{nlines=} should divide {inputs.shape[0]=}')
        return

    # obs
    for i in range(n):
        plt.subplot( nlines+2, n//nlines, i+1)
        plt.axis('off')
        plt.imshow(inputs[i])

        
    # target (ground truth)
    classes = map_to_classes( target, thresholds_in_cent_mm)
    i = len(inputs) 
    for c in classes:
        plt.subplot( nlines+2, n//nlines, i+1)
        plt.axis('off')
        plt.imshow(c[0])
        i += 1

    # estimated 
    print(estimated.shape)
    for j in range(estimated.shape[1]):
        plt.subplot( nlines+2, n//nlines, i+1)
        plt.axis('off')
        plt.imshow(estimated[0][j]>0.5)
        i += 1
    plt.show()


view_inference( batch['inputs'][0],
                batch['target'],
                model(batch['inputs']).detach().numpy(),
                thresholds_in_cent_mm
               )
