# Meteonet dataloader
# (c) 2023 D.Bereziat, Sorbonne Université

import torch
import numpy as np
from torch.utils.data import Dataset
from meteonet.utilities import next_date, load_map, map_to_classes, split_date
from os.path import dirname, basename, join, isfile
import time
import logging

logging.basicConfig(level=logging.ERROR)

from tqdm import tqdm
from datetime import datetime

epsilon = 1e-3 # As Antonia

class MeteonetDataset(Dataset):
    """ A class to load Meteonet data
        Limitations:
         - no multiple targets
        Difference with the vanilla dataloader
         - maps having missing values are no more ignored

      Stratégie:
         l'indexation a lieu pendant l'instantiation de la classe.
         on peut la sauvegarder dans un cache
         il n'y a plus de calcul dans getitem(), ce dernier se contente de
         charger directement les fichiers grâce à l'index.
         Quand une date manque, getitem() retourne None
         les dates manquantes sont consultats dans la variable missing_date de la classe.
    """
    def __init__(self, rainmaps, input_len = 12, target_pos = 18, stride = 12,
                 wind_dir=None, cached=None, tqdm=None, logging=False):
        """
        files: list of file name paths (will be sorted),

        windir: directory where wind maps are stored
                should contain two subdirectories, U/ and V/,

        input_len: number of maps to read as input of the model
                   recommended: 12 (stands for 1 hour),

        target_pos: position of target starting from the first map read as input
                    recommended: 18 = 12+6 (prevision horizon time at 6 = 30 minutes),

        stride: offset between each input sequence
                recommended: input_len (for no overlapping),

        cached: store dataset indexation in a cache file
        """
        files = sorted(rainmaps, key=lambda f:split_date(f))
        recalculate = True
        if cached and isfile(cached):
            obj = np.load(cached,allow_pickle=True)
            params = obj['arr_0'].reshape(-1)[0]
            if params['input_len'] == input_len and  params['stride'] == stride and \
               params['target_pos'] == target_pos and params['files'] == files and \
               (wind_dir == None or params['wind_dir'] == wind_dir):
                recalculate = False

        if recalculate:
            if tqdm: print('parameters changed, or cached file not found: indexing dataset, please wait...')
            params = {'files': files,
                      'input_len': input_len, 'target_pos': target_pos, 'stride': stride,
                      'wind_dir': wind_dir}
            maxs = []
            Umean = Vmean = 0.
            Uvar = Vvar = 0.
            size = 0.
            has_wind = []
            for f in tqdm(files, unit=' files') if tqdm else files:
                maxs.append( load_map(f).max())
                if wind_dir:
                    Upath = join(wind_dir,'U',basename(f))
                    Vpath = join(wind_dir,'V',basename(f))
                    if isfile(Upath) and isfile(Vpath):
                        U = np.array(load_map(Upath), dtype=float)
                        Umean += U.sum()
                        Uvar += (U**2).sum()
                        V = load_map(Vpath)*1.
                        Vmean += V.sum()
                        Vvar += (V**2).sum()
                        size += U.size
                        has_wind.append(True)
                    else:
                        has_wind.append(False)
            if wind_dir:
                Umean /= size
                Vmean /= size
                params['U_moments'] = Umean, np.sqrt(Uvar/size - Umean**2)
                params['V_moments'] = Vmean, np.sqrt(Vvar/size - Vmean**2)    

            params['maxs'] = np.array(maxs)
            l = len(files)
            items = []

            dname = dirname(files[0])
            missing_dates = []
            for j in range(0, len(files), stride):
                item = [j]
                curr_date = basename( files[j])
                # check if the next input_len-1 files have correct dates
                j += 1
                num_obs = input_len - 1
                while num_obs and j<l:
                    next_available_date = basename(files[j])
                    curr_date = next_date(curr_date)
                    if curr_date == next_available_date:
                        item.append(j)
                        j += 1
                    else:
                        item.append(-1)
                        missing_dates.append(curr_date)
                    num_obs -= 1
                if j == l: break

                # get the target date
                pos = target_pos - input_len
                while pos:
                    curr_date = next_date(curr_date)
                    pos -= 1
                target_file = join(dname, curr_date)

                jend = min(j+target_pos, l)
                if target_file in files[j:jend]:
                    item.append( j+files[j:jend].index(target_file))
                else:
                    item.append(-1)
                    missing_dates.append(basename(target_file))
                items.append(item)

            params['items'] = np.array(items)
            params['has_wind'] = np.array(has_wind)
            params['missing_dates'] = missing_dates
            if cached:
                np.savez_compressed( cached, params)

        self.logging = logging
        self.tqdm = tqdm
        self.params = params
        self.norm_factors = [np.log(1+params['maxs'].max())]
        self.use_wind = False
                
        if wind_dir:
            for v in params['U_moments']: self.norm_factors.append(v)
            for v in params['V_moments']: self.norm_factors.append(v)
            self.use_wind = True

        self.do_not_read_map = False # for performances tests. could be removed
        
    def __len__(self):
        ## Question pour Anastase: on pourrait aussi avoir une fenêtre glissante sur les inputs_len ??
        ## le code de Vincent ne le prévoit pas: les inputs ne se superposent pas.
        return self.params['items'].shape[0]

    def read(self, idx):
        if self.do_not_read_map:
            return torch.zeros((1,1)), torch.zeros((1,1))
        ## Anastase: tensor() ou Tensor() ?
        rainmap = torch.Tensor(load_map(self.params['files'][idx]))
        return torch.log(rainmap.unsqueeze(0) + 1 + epsilon)/self.norm_factors[0], rainmap

    def read_wind(self, idx):
        if self.do_not_read_map:
            return torch.zeros((1,1)), torch.zeros((1,1))
        m,s = self.norm_factors[1], self.norm_factors[2]
        U = torch.Tensor(load_map( join(self.params['wind_dir'],'U',basename(self.params['files'][idx])))-m)/s
        m,s = self.norm_factors[3], self.norm_factors[4]        
        V = torch.Tensor(load_map( join(self.params['wind_dir'],'V',basename(self.params['files'][idx])))-m)/s
        return U.unsqueeze(0),V.unsqueeze(0)

    def __getitem__(self, i):
        item = self.params['items'][i]

        # Anastase: 
        # on peut changer ça, notamment uniquement si la target vaut -1.
        # rappel: un index à -1 indique une date manquante.
        # ces dates pourrait être remplacées par des images nulles,
        # avec un minimum de deux non nulles sans doute pour inférer la dynamique
        # if item[-1] == -1:
        if item.min() == -1:
            return None

        maps, _ = self.read(item[0])
        for idx in item[1:-1]:
            rmap, persistence = self.read(idx)
            maps = torch.cat((maps, rmap), dim=0)
        if self.use_wind:
            if not self.params['has_wind'][item[:-1]].all():
                return None
            Umaps, Vmaps = self.read_wind(item[0])
            for idx in item[1:-1]:                
                U,V = self.read_wind(idx)
                Umaps = torch.cat((Umaps, U), dim=0)
                Vmaps = torch.cat((Vmaps, V), dim=0)
            maps = torch.cat((maps, Umaps, Vmaps), dim=0)
        
        target_file = self.params['files'][item[-1]]
        
        return {
            'inputs': maps,
            'target': torch.Tensor(load_map(target_file)) if not self.do_not_read_map else torch.zeros(1),
            'target_name': target_file,
            'persistence': persistence
        }


###############################################################################
# Chunk loading
###############################################################################
def bouget21_chunked(samples_idx_s, data_type="train"):
    """Split in train/validation/test sets according to Section 4.1 from Bouget et al, 2021"""
    if data_type == "all":
        return samples_idx_s
    idx_s = []
    for idx in samples_idx_s:
        year, month, day, hour, mi, data_idx, channels = idx
        yday = datetime(year, month, day).timetuple().tm_yday - 1
        if data_type == "val" and year == 2018 and (yday // 7) % 2 == 0:
            idx_s.append(idx)
        elif (
            data_type == "test"
            and year == 2018
            and not (yday % 7 == 0 and hour == 0)
            and (yday // 7) % 2 != 0
        ):
            idx_s.append(idx)
        elif data_type == "train" and (year == 2016 or year == 2017):
            idx_s.append(idx)
    return np.array(idx_s)


class ChunksCache:
    def __init__(self, root_dir: str, max_size=100):
        self.root_dir = root_dir
        self.max_size = max_size
        self.loaded_chunks = {}
        self.n_loaded = 0
        self.n_missed = 0

    def add_to_cache(self, y: int, M: int, d: int, chunk: np.array):
        if len(self.loaded_chunks) >= self.max_size:
            self.loaded_chunks.popitem()
        self.loaded_chunks[(y, M, d)] = chunk

    def load_chunk(self, y: int, M: int, d: int):
        chunk_file = join(self.root_dir, "chunks", f"y{y:04d}", f"M{M:02d}", f"d{d:02d}", f"y{y:04d}-M{M:02d}-d{d:02d}.npy")
        try:
            return np.load(chunk_file, mmap_mode="r")
        except Exception:
            logging.error(f"Error loading {chunk_file}")
            logging.error(f"Loaded chunks count: {len(self.loaded_chunks)}")
            return None

    def get_chunk(self, y: int, M: int, d: int):
        self.n_loaded += 1
        if (y, M, d) in self.loaded_chunks:
            return self.loaded_chunks[(y, M, d)]
        self.n_missed += 1
        chunk = self.load_chunk(y, M, d)
        self.add_to_cache(y, M, d, chunk)
        return chunk

    def get_sample(self, sample_idx):
        chunk = self.get_chunk(*sample_idx[:3])
        return chunk[sample_idx[-2]:sample_idx[-2] + sample_idx[-1]]


class MeteonetDatasetChunked(Dataset):
    """
    A PyTorch Dataset class for loading and processing Meteonet data in chunks.

    This class handles the loading of Meteonet data, computes necessary statistics,
    and prepares the data for training and evaluation in a chunked manner.


        norm_factors (list): Normalization factors computed from the data.

    Methods:
        _compute_maps_moments():
            Computes the moments (mean, variance) and other statistics for the data maps.
        
        _compute_items():
            Computes the items (sequences of indices) for the dataset based on the stride and target position.
        
        _sample_name(sample_idx):
            Generates a sample name string from the sample index.
        
        __len__():
            Returns the number of items in the dataset.
        
        _get_inputs(item):
            Retrieves and processes the input maps for a given item.
        
        _get_targets(item):
            Retrieves and processes the target maps for a given item.
        
        __getitem__(i):
            Retrieves the inputs, targets, and other relevant data for a given index.
    """
    def __init__(self, root_dir: str, data_type: str, input_len=12, target_pos=18, stride=12, target_is_one_map=False):
        """
        Initializes the loader with the specified parameters.

        Args:
            root_dir (str): The root directory where the data is stored.
            data_type (str): The type of data to be loaded.
            input_len (int, optional): The length of the input sequence. Defaults to 12.
            target_pos (int, optional): The position of the target in the sequence. Defaults to 18.
            stride (int, optional): The stride length for sampling. Defaults to 12.
            target_is_one_map (bool, optional): Flag indicating if the target is a single map. Defaults to False.

        Attributes:
            data_type (str): The type of data to be loaded.
            root_dir (str): The root directory where the data is stored.
            target_is_one_map (bool): Flag indicating if the target is a single map.
            do_not_read_map (bool): Flag indicating if the map should not be read.
            samples (list): The list of samples obtained from the index file.
            loader (ChunksCache): The cache loader for chunks of data.
            params (dict): Dictionary containing various parameters for data loading and processing.
        """
        self.data_type = data_type
        self.root_dir = root_dir
        self.target_is_one_map = target_is_one_map
        self.do_not_read_map = False

        idx_s_all = np.load(join(root_dir, "chunks", "indexs.npy"), mmap_mode="r")
        self.samples = bouget21_chunked(idx_s_all, data_type)
        self.loader = ChunksCache(root_dir)

        self.params = {
            "root_dir": root_dir,
            "input_len": input_len,
            "target_pos": target_pos,
            "stride": stride,
            "maxs": None,
            "U_moments": None,
            "V_moments": None,
            "has_wind": None,
            "items": None,
            "missing_dates": None,
        }
        
        cache_file = join(root_dir, f"moments_{data_type}.npz")
        if isfile(cache_file):
            print(f"Loading cached moments for {data_type} data from {cache_file}")
            cached_params = np.load(cache_file, allow_pickle=True)
            self.params.update(cached_params['arr_0'].item())
            self.norm_factors = [np.log(1 + self.params["maxs"].max())] + list(self.params["U_moments"]) + list(self.params["V_moments"])
        else:
            self._compute_maps_moments()
            self._compute_items()
            # save only the computed parameters to a cache file
            to_save_keys = ["maxs", "U_moments", "V_moments", "has_wind", "items", "missing_dates"]
            to_save = {k: v for k, v in self.params.items() if k in to_save_keys}
            np.savez_compressed(cache_file, to_save)

    def _compute_maps_moments(self):
        maxs, Umean, Vmean, Uvar, Vvar, size = [], 0.0, 0.0, 0.0, 0.0, 0.0
        has_wind = []
        for sample_idx in tqdm(self.samples, unit=" samples", desc="Computing moments"):
            maps = self.loader.get_sample(sample_idx)
            maxs.append(maps[0].max())
            if maps.shape[0] == 3:
                U, V = maps[1].astype(float), maps[2].astype(float)
                Umean, Vmean = Umean + U.sum(), Vmean + V.sum()
                Uvar, Vvar = Uvar + (U**2).sum(), Vvar + (V**2).sum()
                size += U.size
                has_wind.append(True)
            else:
                has_wind.append(False)
        Umean, Vmean = Umean / size, Vmean / size
        self.params.update({
            "U_moments": (Umean, np.sqrt(Uvar / size - Umean**2)),
            "V_moments": (Vmean, np.sqrt(Vvar / size - Vmean**2)),
            "maxs": np.array(maxs),
            "has_wind": np.array(has_wind),
        })
        self.norm_factors = [np.log(1 + self.params["maxs"].max())] + list(self.params["U_moments"]) + list(self.params["V_moments"])

    def _compute_items(self):
        missing_dates, items = [], []
        l = len(self.samples)
        for j in tqdm(range(0, l, self.params["stride"]), unit=" samples", desc="Computing items"):
            item, curr_date = [j], self._sample_name(self.samples[j])
            k, num_obs = j + 1, self.params["target_pos"] - 1
            while num_obs and k < l:
                next_aavailable_date = self._sample_name(self.samples[k])
                curr_date = next_date(curr_date)
                if curr_date == next_aavailable_date:
                    item.append(k)
                    k += 1
                else:
                    item.append(-1)
                    missing_dates.append(curr_date)
                num_obs -= 1
            if k == l:
                break
            items.append(item)
        self.params.update({"items": np.array(items), "missing_dates": missing_dates})

    def _sample_name(self, sample_idx):
        year, month, day, hour, mi, _, _ = sample_idx
        return f"y{year}-M{month}-d{day}-h{hour}-m{mi}.npz"

    def __len__(self):
        return self.params["items"].shape[0]

    def _get_inputs(self, item: np.array):
        if self.do_not_read_map:
            return np.zeros((1, 1))
        input_maps = []
        for idx in item[:self.params["input_len"]]:
            cur_map = torch.Tensor(self.loader.get_sample(self.samples[idx]).copy())
            cur_map[0] = torch.log(cur_map[0] + 1 + epsilon) / self.norm_factors[0]
            cur_map[1] = (cur_map[1] - self.norm_factors[1]) / self.norm_factors[2]
            cur_map[2] = (cur_map[2] - self.norm_factors[3]) / self.norm_factors[4]
            input_maps.append(cur_map)
        return torch.cat(input_maps, dim=0)

    def _get_targets(self, item: np.array):
        if self.do_not_read_map:
            return np.zeros((1, 1))
        if self.target_is_one_map:
            return torch.Tensor(self.loader.get_sample(self.samples[item[-1]])[0].copy())
        targets = [torch.Tensor(self.loader.get_sample(self.samples[idx])[0].copy()) for idx in item[self.params["input_len"] + 1:]]
        return torch.cat(targets, dim=0)

    def __getitem__(self, i: int):
        item = self.params["items"][i]
        if item.min() == -1 or not self.params["has_wind"][item[:self.params["input_len"]]].all():
            return None
        return {
            "inputs": self._get_inputs(item),
            "target": self._get_targets(item),
            "persistance": torch.Tensor(self.loader.get_sample(self.samples[item[self.params["input_len"] - 1]])[0].copy()) if not self.do_not_read_map else np.zeros((1, 1)),
            "target_name": self._sample_name(self.samples[item[-1]]),
        }


###############################################################################
###############################################################################
class MeteonetTime(Dataset):
    """A class to check the time cost of loading 200000 files, obsolet"""
    def __init__(self, files, load=False):
        self.files = files
        self.load = load
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        file = self.files[i]
        if self.load:
            return torch.Tensor(load_map(file))
        return isfile(file)
    def timeit(self):
        tic = time.perf_counter()
        for a in self: pass
        tac =  time.perf_counter()
        print(f"{tac - tic:0.4f} seconds")
