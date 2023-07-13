# Over sampling for unbalanced classes

from torch.utils.data import WeightedRandomSampler

def items_to_oversample( meteonet_ds, threshold):
    """ Get the items from dataset should be oversampled (1) / undersampled (0)
        invalides samples are tagged to -1
        Is this method go to meteonetdataset class?
    """
    w = []
    items = meteonet_ds.params['items']
    maxs = meteonet_ds.params['maxs']
    meteonet_ds.do_not_read_map = True
    for data, item in zip(meteonet_ds,items):
        if data:
            w.append( 1*((maxs[item[-1]]>threshold).sum()>0))
        else:
            w.append( -1)
    meteonet_ds.do_not_read_map = False
    return torch.tensor(w)

def meteonet_random_oversampler( w , p=.8):
    """ Oversample of factor resp. p, 1-p, and 0 items flagged resp. 1, 0, -1 (see items_to_oversample())
    """    
    NR = (w==1).sum().item()
    NN = (w==0).sum().item()
    pR = p/NR   
    pN = (1-p)/NN
    weights = (w==1)*pR + (w==0)*pN + (w==-1)*0
    return WeightedRandomSampler(weights, NR+NN, replacement=True)

def meteonet_sequential_sampler( meteonet_ds):
    """ get the available items from dataset """
    meteonet_ds.do_not_read_map = True
    L = [ i for i,d in enumerate(meteonet_ds) if d]
    meteonet_ds.do_not_read_map = False
    return L
