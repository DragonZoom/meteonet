#
#  An example of meteonet dataloader usage to train in a regresionn manner
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
    help="Directory containing data",
    dest="data_dir",
    default="data",
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
    default="runs",
)
parser.add_argument(
    "-e", "--epochs", type=int, help="Number of epochs", dest="epochs", default=20
)
parser.add_argument(
    "-b", "--batch-size", type=int, help="Batch size", dest="batch_size", default=128
)
parser.add_argument(
    "-lr",
    "--learning-rate",
    type=str,
    nargs="+",
    help="LR_WD format is: epoch:Learning rate, weight decay",
    dest="lr_wd",
    default=["0:8e-4,1e-5", "4:1e-4,5e-5"],
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
parser.add_argument("-lt", "--light", type=bool, help="", dest="light", default=False)

# parser.add_argument('-f', '--load', dest='load', type=str, default=False, help='Load model from a .pth file')
# parser.add_argument( '-gs', '--global-step-start', metavar='gstp', type=int, default=0,
#                     help='Number of the last global step of loaded model', dest='glb_step_start')
# parser.add_argument( '-es', '--epoch-start', metavar='es', type=int, default=0,
#                     help='Number of last epoch of the loaded model', dest='epoch_start')

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
lr_wd = dict()
for a in args.lr_wd:
    k, u = a.split(":")
    a, b = u.split(",")
    lr_wd[int(k)] = float(a), float(b)

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
   {lr_wd = }
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

from meteonet.loader import MeteonetDatasetChunked
from meteonet.samplers import meteonet_random_oversampler, meteonet_sequential_sampler
from torch.utils.data import DataLoader
from os.path import join


train_ds = MeteonetDatasetChunked(
    args.data_dir,
    'train' if not args.debug else 'test', # limit the number of files for testing
    input_len,
    input_len + time_horizon,
    stride,
    target_is_one_map=False,
)
val_ds = MeteonetDatasetChunked(
    args.data_dir,
    "val",
    input_len,
    input_len + time_horizon,
    stride,
    target_is_one_map=True,
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
from models.FsrGAN import (
    FirstStage,
    FsrSecondStageGenerator,
    FsrDiscriminator,
    FsrSecondStageGeneratorLight,
)
from trainers.gan import train_meteonet_gan
from datetime import datetime

model1_g = FirstStage(input_len, time_horizon)
if args.light:
    model2_g = FsrSecondStageGeneratorLight(input_len, time_horizon)
else:
    model2_g = FsrSecondStageGenerator(input_len, time_horizon)
model_d = FsrDiscriminator(time_horizon)

# try:

rundir = join(args.run_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
print(f"run files will be recorded in directory {rundir}")
os.system(f'mkdir -p "{rundir}"')
scores = train_meteonet_gan(
    train_loader,
    val_loader,
    model1_g,
    model2_g,
    model_d,
    thresholds,
    args.epochs,
    lr_wd,  # TODO: lr_wd_g1, lr_wd_g2, lr_wd_d
    lr_wd,
    lr_wd,
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
    "lr_wd": lr_wd,
    "oversampling": args.oversampling,
    # 'model_size': model_size,
    "data_dir": args.data_dir,
}

os.system(f'rm -f lastrun; ln -sf "{rundir}" lastrun')

import torch

torch.save({"hyperparams": hyperparams, "scores": scores}, join(rundir, "run_info.pt"))
