import torch
import numpy as np
from tqdm import tqdm
from meteonet.utilities import split_date
import os
import argparse
from models.FsrGAN import FirstStage
from meteonet.loader import MeteonetDatasetChunked
from meteonet.samplers import meteonet_sequential_sampler
from torch.utils.data import DataLoader

def evaluate_model(model, loader, dst_dir, device):
    os.makedirs(dst_dir, exist_ok=True)
    for data in tqdm(loader):
        x = data['inputs']
        x = x.to(device)
        with torch.no_grad():
            y_hat = model(x).cpu().numpy()
        for i in range(y_hat.shape[0]):
            target_name = data["target_name"][i].split(".")[0]
            filename = os.path.join(dst_dir, target_name + ".npy")
            np.save(filename, y_hat[i])

def main(args): 
    input_len = 12
    time_horizon = 6
    stride = 12
    batch_size = 32

    test_ds = MeteonetDatasetChunked(
        "./data-chunked/",
        "test",
        input_len,
        input_len + time_horizon,
        stride,
        target_is_one_map=True,
        use_wind=True,
        normalize_target=False
    )
    val_ds = MeteonetDatasetChunked(
        "./data-chunked/",
        "val",
        input_len,
        input_len + time_horizon,
        stride,
        target_is_one_map=True,
        use_wind=True,
        normalize_target=False
    )
    train_ds = MeteonetDatasetChunked(
        "./data-chunked/",
        "train",
        input_len,
        input_len + time_horizon,
        stride,
        target_is_one_map=True,
        use_wind=True,
        normalize_target=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size,
        sampler=meteonet_sequential_sampler(test_ds),
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size,
        sampler=meteonet_sequential_sampler(val_ds),
        num_workers=4,
        pin_memory=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size,
        sampler=meteonet_sequential_sampler(train_ds),
        num_workers=4,
        pin_memory=True,
    )

    model = FirstStage(input_len, time_horizon, size_factor=2)
    model.load_state_dict(torch.load(args.model_pt))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    evaluate_model(model, test_loader, os.path.join(args.dest_dir, 'test'), device)
    evaluate_model(model, val_loader, os.path.join(args.dest_dir, 'val'), device)
    evaluate_model(model, train_loader, os.path.join(args.dest_dir, 'train'), device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='First Stage Inference Script')
    parser.add_argument('--model_pt', type=str, default='best_runs/stage_1_size_2/model_s1_last_epoch.pt', help='Path to the model checkpoint')
    parser.add_argument('--dest_dir', type=str, default='cache/first_stage_predictions', help='Path to the directory where the predictions will be saved')
    args = parser.parse_args()
    main(args)
