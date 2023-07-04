# Meteonet dataloader
import torch
from torch.utils.data import Dataset
from loader.utilities import next_date, load_map, map_to_classes
from os.path import dirname, basename, join, isfile

class MeteonetDataset(Dataset):
    """ A class to load Meteonet data
        V1 limitations:
         - no windmaps
         - no log transformation, to discuss
    """
    def __init__(self, params, tset="train", input_len = 12, target_pos = 18, stride = 12, thresholds = False):
        """
        params: a dictionnary containing sorted lists of train and set files
                and normalisation parameters, created by initparams() function
        tset: 'train' or 'val'
        input_len: number of maps to read as input of the model
                   recommended: 12 (stands for 1 hour)
        target_pos: position of target starting from the first map read as input
                    recommended: 18 = 12+6 (prevision horizon time at 6 = 30 minutes)
        stride: offset between each input sequence
                recommended: input_len (for no overlapping)
        thresholds: if non False, a list of thresholds for a classification task
        """
        self.input_len = input_len
        self.target_pos = target_pos
        self.stride = stride
        self.params = params
        self.thresh = thresholds
        self.files = params[tset]
        self.maps = []
        self.missing_dates = []
        
    def __len__(self):
        return (len(self.files) - self.target_pos)// self.stride

    def read(self, file):
        # log ??
        rainmap = torch.unsqueeze(torch.tensor(load_map(file), dtype=torch.float32), dim=0)
        # normalisation (map between 0 and 1)
        rainmap = (rainmap + 1)/self.params['max']
        return rainmap
    
    def __getitem__(self, i):
        j = i*self.stride

        curdate = basename(self.files[j])
        maps = self.read(self.files[j])
        
        # check if the next input_len-1 files have correct dates
        k = 1
        while k < self.input_len:
            nextdate = next_date(curdate)
            tmpdate = curdate
            curdate = basename(self.files[j+k])
            if nextdate != curdate:
                print( f'warning: {i}/{len(self)}  {tmpdate} is missing')
                self.missing_dates.append( (i,tmpdate) )
                return None
                # we use a map with no signal
                #_,n,m = maps.shape
                #maps = torch.cat((maps,torch.zeros(1,n,m,dtype=torch.float32)), dim=0)
            
        # go to the target date
        while k < self.target_pos:
            curdate = next_date(curdate)
            k += 1

        target_file = join(dirname(self.files[j]),curdate)
        if not target_file in self.files[j:j+k]:
            # print( f'SKIP (as target): {i}/{len(self)} {curdate} is missing')
            self.missing_dates.append( (i, curdate) )
            return None

        target = load_map(target_file)
        # print(f'DEBUG: input: {i}/{len(self)} {self.files[j]}+{self.input_len} target: {target_file}')

        return { 'inputs': maps,
                 'target': torch.from_numpy(map_to_classes(target, self.thresh)) if self.thresh else torch.from_numpy(target),
                 'name': target_file}

    
class Test(Dataset, load=False):
    """ A class to check the time cost of loading 200000 files """
    def __init__(self, files):
        self.files = files
        self.load = load
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        file = self.files[i]
        if self.load:
            return torch.unsqueeze(torch.tensor(load_map(file), dtype=torch.float32), dim=0)
        return isfile(file)
