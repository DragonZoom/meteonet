#
# Difference from stage2 is that stage1 does not have a discriminator model.
# Stage2 is GAN training.
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
    default="data-chunked",
)
parser.add_argument(
    "-s1dd",
    "--s1-data-dir",
    type=str,
    help="Dirrectory with stage1 inference results",
    dest="s1_data_dir",
    default="cache/first_stage_predictions",
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
    default="runs/stage2",
)
parser.add_argument(
    "-e", "--epochs", type=int, help="Number of epochs", dest="epochs", default=20
)
parser.add_argument(
    "-b", "--batch-size", type=int, help="Batch size", dest="batch_size", default=128
)
parser.add_argument(
    "-lrg",
    "--learning-rate-g",
    type=float,
    help="Learning rate",
    dest="lr_g",
    default=0.0001,
)
parser.add_argument(
    "-lrd",
    "--learning-rate-d",
    type=float,
    help="Learning rate",
    dest="lr_d",
    default=0.00005,
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
lr_wd_g = {0: (args.lr_g, 1e-4)}
lr_wd_d = {0: (args.lr_d, 1e-4)}

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
   {lr_wd_g = }
   {lr_wd_d = }
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

if args.model == "FsrGAN":
    from models.FsrGAN import FsrDiscriminator, FsrSecondStageGenerator
    model_G = FsrSecondStageGenerator(input_len, time_horizon, size_factor=1)
    model_D = FsrDiscriminator(time_horizon, size_factor=64)
    use_wind = True
    target_is_one_map = False
elif args.model == "FsrGAN_radar_only":
    from models.FsrGAN import FsrDiscriminator
    from models.FsrGAN_no_wind import FsrSecondStageGeneratorRadarOnly
    model_G = FsrSecondStageGeneratorRadarOnly(input_len, time_horizon, size_factor=2, predict_sequence=True)
    T_D = time_horizon if model_G.predict_sequence else 1
    model_D = FsrDiscriminator(T_D, size_factor=16)
    target_is_one_map = not model_G.predict_sequence
    use_wind = False
else:
    raise ValueError(f"Unknown model {args.model}")

print(f"{target_is_one_map=}")
# print number of parameters for each model
print(f"Number of parameters for model_G: {sum(p.numel() for p in model_G.parameters())}")
print(f"Number of parameters for model_D: {sum(p.numel() for p in model_D.parameters())}")

# data
from meteonet.loader import MeteonetDatasetChunked, DatsetWrapperFsrSecondStage
from meteonet.samplers import meteonet_random_oversampler, meteonet_sequential_sampler
from torch.utils.data import DataLoader
from os.path import join


# Train
train_ds = MeteonetDatasetChunked(
    args.data_dir,
    "train" if not args.debug else "val_small",  # limit the number of files for testing
    input_len,
    input_len + time_horizon,
    stride,
    target_is_one_map=target_is_one_map,
    use_wind=use_wind,
    normalize_target=True,
    skip_withou_wind=True, # TODO: s1 was trained with wind
)
train_ds_wrapp = DatsetWrapperFsrSecondStage(
    os.path.join(args.s1_data_dir, "train" if not args.debug else "val"),
    train_ds,
)
# Validation
val_ds = MeteonetDatasetChunked(
    args.data_dir,
    "val" if not args.debug else "val_small",
    input_len,
    input_len + time_horizon,
    stride,
    target_is_one_map=target_is_one_map,
    use_wind=use_wind,
    normalize_target=False,
    skip_withou_wind=True, # TODO: s1 was trained with wind
)
val_ds.norm_factors = train_ds.norm_factors
val_ds_wrapp = DatsetWrapperFsrSecondStage(
    os.path.join(args.s1_data_dir, "val"), val_ds
)


# samplers for dataloaders
train_sampler = meteonet_random_oversampler(train_ds, thresholds[-1], args.oversampling)
val_sampler = meteonet_sequential_sampler(val_ds)

# dataloaders
train_loader = DataLoader(
    train_ds_wrapp,
    args.batch_size,
    sampler=train_sampler,
    num_workers=args.num_workers,
    pin_memory=True,
)
val_loader = DataLoader(
    val_ds_wrapp,
    args.batch_size,
    sampler=val_sampler,
    num_workers=args.num_workers,
    pin_memory=True,
)


print(
    f"""
size of train items/batch
      {len(train_ds_wrapp)} {len(train_loader)}
size of val items/batch
      {len(val_ds_wrapp)} {len(val_loader)}
"""
)


## Model & training procedure
from trainers.gan import train_meteonet_gan_stage2
from datetime import datetime

rundir = join(args.run_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
print(f"run files will be recorded in directory {rundir}")
os.system(f'mkdir -p "{rundir}"')
scores = train_meteonet_gan_stage2(
    train_loader,
    val_loader,
    model_G,
    model_D,
    thresholds,
    args.epochs,
    lr_wd_g,
    lr_wd_d,
    args.snapshot_step,
    rundir=rundir,
    device=device,
    target_is_one_map=target_is_one_map,
)

hyperparams = {
    "input_len": input_len,
    "time_horizon": time_horizon,
    "stride": stride,
    "thresholds": thresholds,
    "batch_size": args.batch_size,
    # 'clip_grad': clip_grad,
    "epochs": args.epochs,
    "lr_wd_g": lr_wd_g,
    "lr_wd_d": lr_wd_d,
    "oversampling": args.oversampling,
    # 'model_size': model_size,
    "data_dir": args.data_dir,
}

os.system(f'rm -f lastrun; ln -sf "{rundir}" lastrun')

import torch

torch.save({"hyperparams": hyperparams, "scores": scores}, join(rundir, "run_info.pt"))
