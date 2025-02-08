import argparse, torch, os
from platform import processor

import optuna
from optuna.trial import Trial
import numpy as np

########## (1) Your existing imports & argument parsing ##########
# We assume your main training script, data loading, and so on
# are accessible or included. The snippet below is a minimal working
# example focusing on the LR tuning with Optuna.

from trainers.gan import train_meteonet_gan
from meteonet.loader import MeteonetDatasetChunked
from meteonet.samplers import meteonet_random_oversampler, meteonet_sequential_sampler
from torch.utils.data import DataLoader

from models.FsrGAN import (
    FirstStage,
    FsrSecondStageGenerator,
    FsrDiscriminator,
)
# If you are testing a different model (FsrGAN_light or FsrGAN_no_wind),
# import accordingly.

def parse_args():
    parser = argparse.ArgumentParser(
        prog="lr_optuna",
        description="Hyperparameter optimization for FsrGAN using Optuna",
    )
    parser.add_argument(
        "-dd", "--data-dir", type=str, default="data-chunked",
        help="Directory containing data"
    )
    parser.add_argument(
        "-rn", "--run-dir", type=str, default="optuna_runs",
        help="Directory to save logs/checkpoints."
    )
    parser.add_argument(
        "-nt", "--n-trials", type=int, default=10,
        help="Number of Optuna trials"
    )
    parser.add_argument(
        "-ep", "--epochs", type=int, default=5,
        help="Number of epochs per trial (for faster search, use fewer epochs)."
    )
    parser.add_argument(
        "-bs", "--batch-size", type=int, default=64,
        help="Batch size"
    )
    parser.add_argument(
        "-m", "--model", type=str, default="FsrGAN",
        choices=["FsrGAN", "FsrGAN_light", "FsrGAN_no_wind"],
        help="Which model variant to use"
    )
    return parser.parse_args()

########## (2) Define the objective function for Optuna ##########
def objective(trial: Trial, args, train_loader, val_loader, thresholds, device):
    """
    1) Sample learning rates & WDs for generator1, generator2, and discriminator.
    2) Build the dictionaries: lr_wd_g1, lr_wd_g2, lr_wd_d.
    3) Construct & train the models with 'train_meteonet_gan'.
    4) Return validation metric as objective.
    """

    # --- 2.1) Sample LR & WD for each of the 3 modules --- #
    lr_g1 = trial.suggest_float("lr_g1", 1e-5, 5e-3, log=True)
    # wd_g1 = trial.suggest_float("wd_g1", 1e-7, 1e-3, log=True)

    lr_g2 = trial.suggest_float("lr_g2", 1e-5, 5e-3, log=True)
    # wd_g2 = trial.suggest_float("wd_g2", 1e-7, 1e-3, log=True)

    lr_d = trial.suggest_float("lr_d", 1e-5, 5e-3, log=True)
    # wd_d = trial.suggest_float("wd_d", 1e-7, 1e-3, log=True)

    # --- 2.2) Build the lr_wd dicts for the entire training schedule --- #
    # Here we define a single step at epoch=0. If you want multiple steps
    # (like epoch=0, 4, etc.), define more keys. Example:
    #   lr_wd_g1 = {0: (lr_g1, wd_g1), 4: (lr_g1*0.1, wd_g1*2)}
    #
    # For simplicity, we'll do a single-step at epoch=0:
    lr_wd_g1 = {0: (lr_g1, 1e-4)}
    lr_wd_g2 = {0: (lr_g2, 1e-4)}
    lr_wd_d  = {0: (lr_d,  1e-4)}

    # --- 2.3) Construct the model(s) based on args.model --- #
    input_len = 12
    time_horizon = 6

    if args.model == "FsrGAN":
        model1_g = FirstStage(input_len, time_horizon)
        model2_g = FsrSecondStageGenerator(input_len, time_horizon, size_factor=4)
        use_window = True
    elif args.model == "FsrGAN_light":
        from models.FsrGAN import FirstStage
        # from models.FsrGANLight import FsrSecondStageGeneratorLight
        #
        model1_g = FirstStage(input_len, time_horizon, size_factor=1)
        # model2_g = FsrSecondStageGeneratorLight(input_len, time_horizon)
        model2_g = FsrSecondStageGenerator(input_len, time_horizon, size_factor=8)
        use_window = True
    elif args.model == "FsrGAN_no_wind":
        from models.FsrGAN_no_wind import RadarSecondStageGenerator, RadarFirstStage
        model1_g = RadarFirstStage(input_len, time_horizon)
        model2_g = RadarSecondStageGenerator(input_len, time_horizon)
        use_window = False
    else:
        raise ValueError(f"Unknown model {args.model}")

    model_d = FsrDiscriminator(time_horizon)

    # --- 2.4) Run the training for 'args.epochs' --- #
    # train_meteonet_gan returns "scores" that you can examine to pick a metric
    # by which we do the hyperparameter optimization.
    epochs = args.epochs

    rundir_trial = os.path.join(args.run_dir, f"trial_{trial.number}")
    os.makedirs(rundir_trial, exist_ok=True)

    scores = train_meteonet_gan(
        train_loader,
        val_loader,
        model1_g,
        model2_g,
        model_d,
        thresholds,
        epochs,
        lr_wd_g1,
        lr_wd_g2,
        lr_wd_d,
        snapshot_step=9999,  # or something large to skip intermediate snapshots
        rundir=rundir_trial,
        device=device,
    )
    # 'scores' typically might include e.g. F1, BIAS, TS, RMSE, etc.

    # --- 2.5) Extract the final metric from 'scores' --- #
    # Suppose we want to maximize the final F1 on validation:
    # If 'scores' is a dict with something like {'val_f1': [...], 'val_ts': [...], ...}
    # you can pick the last value or the best. Adjust as needed.

    # if "f1" in scores:
    #     # Let's say 'scores["f1"]' is a list of epoch values
    #     final_f1 = scores["f1"][-1]  # last epoch
    # else:
    #     # fallback or if your scoring is different, adapt here
    #     final_f1 = 0.0

    # We want to maximize final_f1, but Optuna's default is to MINIMIZE
    # (unless we specify direction='maximize'). We can either:
    # A) Create the study with direction='maximize'
    # B) Return -final_f1 if direction='minimize'.

    # We'll do A) for clarity, so we return final_f1:
    return np.nansum(scores["val_f1_2"][-1])


def main():
    args = parse_args()

    ########## (3) Setup device ##########
    if torch.backends.cuda.is_built() and torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"
    device = torch.device(device)
    print(f"Using device = {device}")

    ########## (4) Build data / loaders ##########
    input_len = 12
    time_horizon = 6
    thresholds = [0.1, 1.0, 2.5]  # you can adapt
    # if needed: convert thresholds to 5-min increments (like in your code)
    thresholds = [100 * k / 12 for k in thresholds]

    print("Loading datasets... (this may take some time)")
    train_ds = MeteonetDatasetChunked(
        args.data_dir,
        "val_small",
        input_len,
        input_len + time_horizon,
        input_len,
        target_is_one_map=False,
        use_wind=(args.model != "FsrGAN_no_wind"),
        normalize_target=False,
    )
    val_ds = MeteonetDatasetChunked(
        args.data_dir,
        "val_small",
        input_len,
        input_len + time_horizon,
        input_len,
        target_is_one_map=True,
        use_wind=(args.model != "FsrGAN_no_wind"),
        normalize_target=False,
    )
    val_ds.norm_factors = train_ds.norm_factors

    # Samplers
    train_sampler = meteonet_random_oversampler(train_ds, thresholds[-1], 0.9)
    val_sampler = meteonet_sequential_sampler(val_ds)

    # Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0 if processor() == "arm" else 4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=0 if processor() == "arm" else 4,
        pin_memory=True,
    )

    print(f"Train size: {len(train_ds)}   Val size: {len(val_ds)}")

    ########## (5) Create and run the Optuna Study ##########
    # We want to MAXIMIZE the final F1 (or whatever metric)
    study = optuna.create_study(direction="maximize")

    # The objective function is partially applied with the data/loaders
    def wrapped_objective(trial):
        return objective(trial, args, train_loader, val_loader, thresholds, device)

    print(f"Starting Optuna with n_trials = {args.n_trials} ...")
    study.optimize(wrapped_objective, n_trials=args.n_trials)

    print("\n=== Optuna Search Complete ===")
    print(f"Best Value = {study.best_value}")
    print(f"Best Params= {study.best_params}")

    # Optionally, retrain with best_params or store them:
    os.makedirs(args.run_dir, exist_ok=True)
    best_params_path = os.path.join(args.run_dir, "best_optuna_params.txt")
    with open(best_params_path, "w") as f:
        f.write(str(study.best_params))
    print(f"Saved best params to: {best_params_path}")

    # Save the study for visualization
    import pickle
    study_path = os.path.join(args.run_dir, "optuna_study.pkl")
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    print(f"Saved Optuna study to: {study_path}")


if __name__ == "__main__":
    main()
