# Over sampling for unbalanced classes


from torch.utils.data import WeightedRandomSampler


    
def downsample_to_classes_above_classth_sampler(dataset, class_th):
    
    weights = [ ] # liste de poids

    # Usage 1:
    
    # on peut se servir du sampler pour éliminer les
    # données absentes (mon dataloader retourne None si
    # une acquisition manquante)
    #
    # vincent s'en sert pour éliminer les données qui
    # contiennent -1 et les classes sur représenté
    #
    # pour éliminer ces données il suffit de tirer avec 0 (éléments à éliminer)
    # ou 1 (éléments à garder)
    # remplacement est mis a True pour ne pas avoir a tirer plusieurs un même exemple.

    weights = [1, 1, 0, 1, 0, 1, 1]
    idxs = WeightedRandomSampler( weights,weight.sum(), remplacement=True)

