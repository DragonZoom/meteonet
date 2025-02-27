import os
import numpy as np
from glob import glob
from os.path import join
import argparse
from tqdm import tqdm

from meteonet.utilities import split_date

def main(rainmaps_dir, dest_folder, compress, include_wind):
    ###############################################################################
    # collect all rainmaps
    all_files = glob(join(rainmaps_dir, '*.npz'))
    all_files = sorted(all_files, key=lambda f: split_date(f))
    print(f"Total rainmaps: {len(all_files)}")

    ###############################################################################
    # split to chunks by date
    chunks = {}
    for f in tqdm(all_files, unit="files", total=len(all_files), desc="Splitting files"):
        y, m, d, _, _ = split_date(f)
        if (y, m, d) not in chunks:
            chunks[(y, m, d)] = []
        chunks[(y, m, d)].append(f)
    print(f"Total chunks: {len(chunks)}")

    ###############################################################################
    # create chunks
    idx_s = []
    for (y, m, d), files in tqdm(chunks.items(), unit="chunks", desc="Creating chunks"):
        dst_folder = join(dest_folder, "chunks", f"y{y:04d}", f"M{m:02d}", f"d{d:02d}")
        os.makedirs(dst_folder, exist_ok=True)
        # merge files
        arr_s = []
        cur_idx = 0
        for f in files:
            y, M, d, h, mi = split_date(f)
            #
            rain_map = list(np.load(f, mmap_mode='r').values())[0]
            #
            if include_wind:
                wind_path_U = f.replace("rainmaps", os.path.join("windmaps", "U"))
                wind_path_V = f.replace("rainmaps", os.path.join("windmaps", "V"))
                #
                try:
                    wmap_U = list(np.load(wind_path_U, mmap_mode='r').values())[0]
                    wmap_V = list(np.load(wind_path_V, mmap_mode='r').values())[0]
                    cur_map = np.stack([rain_map, wmap_U, wmap_V], axis=0)
                except FileNotFoundError:
                    cur_map = rain_map
            else:
                cur_map = rain_map
            cur_map = cur_map.reshape(-1, 128, 128)
            #
            arr_s.append(cur_map)
            idx_s.append((y, M, d, h, mi, cur_idx, cur_map.shape[0]))
            #
            cur_idx += cur_map.shape[0]
        arr_s = np.concatenate(arr_s, axis=0)
        # save
        arr_file_name = f"y{y:04d}-M{m:02d}-d{d:02d}"
        if compress:
            arr_s_file = join(dst_folder, f"{arr_file_name}.npz")
            np.savez_compressed(arr_s_file, arr_s)
        else:
            arr_s_file = join(dst_folder, f"{arr_file_name}.npy")
            np.save(arr_s_file, arr_s)

    idx_s = np.array(idx_s, dtype=np.int16)
    np.save(join(dest_folder, "chunks", "indexs.npy"), idx_s)
    print(f'All chunks have been created and saved to {dest_folder}')
    print(f'Indexs shape: {idx_s.shape}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process rainmaps and create chunks.')
    parser.add_argument('--rainmaps_dir', type=str, default='data/rainmaps', help='Directory containing rainmaps')
    parser.add_argument('--dest_folder', type=str, default='data-chunked', help='Destination folder for chunks')
    parser.add_argument('--compress', type=bool, default=False, help='Compress chunks to .npz')
    parser.add_argument('--include_wind', type=bool, default=False, help='Include Wind maps')
    args = parser.parse_args()

    # print args
    print(f"rainmaps_dir: {args.rainmaps_dir}")
    print(f"dest_folder: {args.dest_folder}")
    print(f"compress: {args.compress}")
    print(f"include_wind: {args.include_wind}")

    main(args.rainmaps_dir, args.dest_folder, args.compress, args.include_wind)
