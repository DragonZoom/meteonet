#
# Difference from train-reg is that stage1 predict sequence of radar images, 
# while train-reg predict a single radar image.
#

import argparse, torch, os
from platform import processor  # for M1/M2 support

parser = argparse.ArgumentParser(
    prog="train-reg", description="Traning a UNet for Meteonet nowforecast"
)

parser.add_argument(
    "-dd",
    "--data-dir",
    type=str,
    help="Directory containing data (should be chunked!)",
    dest="data_dir",
    default="data-chunked",
)
parser.add_argument(
    "-t",
    "--thresholds",
    type=float,
    nargs="+",
    help="Rainmap thresholds in mm/h (used by binary metrics)",
    dest="thresholds",
    default=[0.1, 1, 2.5],
)
parser.add_argument(
    "-Rd",
    "--run-dir",
    type=str,
    help="Directory to save logs and checkpoints",
    dest="run_dir",
    default="runs/stage1",
)
parser.add_argument(
    "-e", "--epochs", type=int, help="Number of epochs", dest="epochs", default=20
)
parser.add_argument(
    "-b", "--batch-size", type=int, help="Batch size", dest="batch_size", default=16
)
parser.add_argument(
    "-lrg1",
    "--learning-rate-g1",
    type=float,
    help="LR_WD format is: epoch:Learning rate, weight decay",
    dest="lr_g1",
    default=0.0005,
)
parser.add_argument(
    "-nw",
    "--num-workers",
    type=int,
    help="Numbers of workers for Cuda",
    dest="num_workers",
    default=8 if processor() != "arm" else 0,
)  # no multithreadings on M1/M2 :(
parser.add_argument(
    "-o",
    "--oversampling",
    type=float,
    help="Oversampling percentage of last class",
    dest="oversampling",
    default=0.9,
)
parser.add_argument(
    "-ss", "--snapshot-step", type=int, help="", dest="snapshot_step", default=5
)
parser.add_argument("-db", "--debug", type=bool, help="", dest="debug", default=False)
# models: FsrGAN, FsrGAN_no_wind
parser.add_argument("-m", "--model", type=str, help="", dest="model", default="FsrGAN")
args = parser.parse_args()

## user parameters
input_len = 12
time_horizon = 6
stride = input_len
clip_grad = 0.1

thresholds = [
    100 * k / 12 for k in args.thresholds
]  #  unit: CRF over 5 minutes in 1/100 of mm (as meteonet data)
model_size = 8  # to do
lr_wd_g1 = {0: (args.lr_g1, 1e-4)}

if torch.backends.cuda.is_built() and torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_built():
    device = "mps"
else:
    device = "cpu"

print(
    f"""
Data params:
   {input_len = } (history of {12*5} minutes)
   {time_horizon = } (nowcasting at {time_horizon*5} minutes)
   {stride = }
   model = FsrGAN regression
   model_size = ?
   {args.data_dir = }
   {len(thresholds)} classes ({thresholds=})
   
Train params:
   {args.epochs = } 
   {args.batch_size = }
   {lr_wd_g1 = }
   {clip_grad = }

Others params:
   {device = }
   {args.snapshot_step = }
   {args.num_workers = }
   {args.run_dir = }
   {args.debug = }
"""
)

device = torch.device(device)

# model
from models.FsrGAN import (
    FirstStage,
)
from trainers.gan import train_meteonet_gan
from datetime import datetime

use_window = True

if args.model == "FsrGAN_radar_only":
    from models.FsrGAN_no_wind import FirstStageRadarOnly
    model1_g = FirstStageRadarOnly(input_len, time_horizon)
    use_window = False

elif args.model == "FsrGAN":
    model1_g = FirstStage(input_len, time_horizon, 1)
else:
    raise ValueError(f"Unknown model {args.model}")

print(f'Use window: {use_window}')
# print num of parameters
print(f"Number of parameters: {sum(p.numel() for p in model1_g.parameters())}")

# data
from meteonet.loader import MeteonetDatasetChunked
from meteonet.samplers import meteonet_random_oversampler, meteonet_sequential_sampler
from torch.utils.data import DataLoader
from os.path import join


train_ds = MeteonetDatasetChunked(
    args.data_dir,
    "train" if not args.debug else "val_small",  # limit the number of files for testing
    input_len,
    input_len + time_horizon,
    stride,
    target_is_one_map=False,
    use_wind=use_window,
)
val_ds = MeteonetDatasetChunked(
    args.data_dir,
    "val" if not args.debug else "val_small",
    input_len,
    input_len + time_horizon,
    stride,
    target_is_one_map=False,
    use_wind=use_window,
)

val_ds.norm_factors = train_ds.norm_factors

# samplers for dataloaders
train_sampler = meteonet_random_oversampler(train_ds, thresholds[-1], args.oversampling)
val_sampler = meteonet_sequential_sampler(val_ds)

# dataloaders
train_loader = DataLoader(
    train_ds,
    args.batch_size,
    sampler=train_sampler,
    num_workers=args.num_workers,
    pin_memory=True,
)
val_loader = DataLoader(
    val_ds,
    args.batch_size,
    sampler=val_sampler,
    num_workers=args.num_workers,
    pin_memory=True,
)


print(
    f"""
size of train items/batch
      {len(train_ds)} {len(train_loader)}
size of val items/batch
      {len(val_ds)} {len(val_loader)}
"""
)


## Model & training procedure
from trainers.gan import train_meteonet_gan_stage1
from datetime import datetime

# try:

rundir = join(args.run_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
print(f"run files will be recorded in directory {rundir}")
os.system(f'mkdir -p "{rundir}"')
scores = train_meteonet_gan_stage1(
    train_loader,
    val_loader,
    model1_g,
    thresholds,
    args.epochs,
    lr_wd_g1,
    args.snapshot_step,
    rundir=rundir,
    device=device,
)

hyperparams = {
    "input_len": input_len,
    "time_horizon": time_horizon,
    "stride": stride,
    "thresholds": thresholds,
    "batch_size": args.batch_size,
    # 'clip_grad': clip_grad,
    "epochs": args.epochs,
    "lr_wd": lr_wd_g1,
    "oversampling": args.oversampling,
    # 'model_size': model_size,
    "data_dir": args.data_dir,
}

os.system(f'rm -f lastrun; ln -sf "{rundir}" lastrun')

import torch

torch.save({"hyperparams": hyperparams, "scores": scores}, join(rundir, "run_info.pt"))
