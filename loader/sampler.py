# Over sampling for unbalanced classes

from torch.utils.data import WeightedRandomSampler

def items_to_oversample( dataset, threshold, tqdm = None):
    """ Get the items from dataset should be oversampled (1) / undersampled (0)
        invalides samples are tagged to -1
        Is this method go to meteonetdataset class?
    """
    weights = []
    for d in tqdm(dataset) if tqdm else dataset:
        if d: # données valides (les fichiers existent)
            weights.append( 1*(d['target']>threshold).sum() > 0)
        else:
            weights.append( -1)
    return torch.tensor(weights)

def meteonet_random_oversampler( items , p=.8):
    """ Oversample of factor resp. p, 1-p, and 0 items flagged resp. 1, 0, -1 (see items_to_oversample())
    """    
    NR = (items==1).sum().item()
    NN = (items==0).sum().item()
    pR = p/NR   
    pN = (1-p)/NN
    w = (items==1)*pR + (items==0)*pN + (items==-1)*0
    #    w = (pR-pN)*weights+pN # weights = 1 -> pR, weights = 0 -> pN
    return WeightedRandomSampler(w, NR+NN, replacement=True)


def meteonet_sequential_sampler( dataset, tqdm = None):
    """ get the available items from dataset """
    return [i for i,d in enumerate(tqdm(dataset, unit=' items') if tqdm else dataset) if d]
