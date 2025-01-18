# A training procedure for Meteonet data

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from meteonet.utilities import calculate_CT, calculate_BS, map_to_classes

from tqdm import tqdm
from os.path import join


def separate_radar_wind(x):
    B, C, H, W = x.shape
    input_len = C // 3
    radars = x[:, :12].view(B, input_len, 1, H, W)
    u_maps = x[:, 12:24].view(B, input_len, 1, H, W)
    v_maps = x[:, 24:].view(B, input_len, 1, H, W)
    wind = torch.cat([u_maps, v_maps], dim=2)
    return radars, wind


def calculate_weights(radar):
    """
    Calculate the weights w(z) as defined in equation (2) based on radar echo value (1/100 mm).

    Args:
        radar (torch.Tensor): Radar echo values (tensor of shape [B, T, H, W]).

    Returns:
        torch.Tensor: Weight tensor with the same shape as radar.
    """
    weights = torch.ones_like(radar)
    weights = torch.where(radar >= 0.83, torch.tensor(2.0, device=radar.device), weights)
    weights = torch.where(radar >= 8.33, torch.tensor(6.0, device=radar.device), weights)
    weights = torch.where(radar >= 20.3, torch.tensor(10.0, device=radar.device), weights)
    weights = torch.where(radar >= 40, torch.tensor(20.0, device=radar.device), weights)
    weights = torch.where(radar >= 70, torch.tensor(60.0, device=radar.device), weights)
    return weights

def loss_stage1(predicted, ground_truth):
    """
    Compute the first stage loss as defined in equation (3).

    Args:
        predicted (torch.Tensor): Predicted radar echo sequence (tensor of shape [B, T, H, W]).
        ground_truth (torch.Tensor): Ground truth radar echo sequence (tensor of shape [B, T, H, W]).

    Returns:
        torch.Tensor: The scalar loss value.
    """
    _, T, _, _ = predicted.shape

    # Compute the weights based on ground truth radar echo values
    weights = calculate_weights(ground_truth)

    # Compute the squared difference and absolute difference
    squared_diff = (predicted - ground_truth) ** 2
    absolute_diff = torch.abs(predicted - ground_truth)

    # Compute the weighted loss for each pixel and each timestep
    weighted_loss = weights * (squared_diff + absolute_diff)

    # Average over spatial dimensions, time, and batch
    loss = (weighted_loss / T).sum()

    return loss


def train_meteonet_gan(
    train_loader,
    val_loader,
    model_generator1,
    model_generator2,
    model_discriminator,
    thresholds,
    epochs,
    lr_wd_g1,
    lr_wd_g2,
    lr_wd_d,
    snapshot_step=5,
    rundir="runs",
    clip_grad=0.1,
    tqdm=tqdm,
    device="cpu",
):

    print("Evaluation Persistence...")
    # on garde les métriques ensemblistes.
    # il faut ajouter la RMSE
    CT_pers = 0
    RMSE_pers = 0
    N = 0
    for batch in tqdm(val_loader):
        persistance = batch["persistence"]
        target = batch["target"][:, -1]
        
        CT_pers += calculate_CT(
            map_to_classes(persistance, thresholds), map_to_classes(target, thresholds)
        )
        f1_pers, bias_pers, ts_pers = calculate_BS(CT_pers, ["F1", "BIAS", "TS"])
        RMSE_pers += ((persistance - target) ** 2).mean()
        N += target.shape[0]

    RMSE_pers = (RMSE_pers / N) * 0.5

    writer = SummaryWriter(log_dir=rundir)

    loss = nn.MSELoss()
    loss.to(device)
    model_generator1.to(device)
    model_generator2.to(device)
    model_discriminator.to(device)

    print("Start training...")
    train_losses = []
    train_losses_g1 = []
    train_losses_g2 = []
    train_losses_d = []
    val_losses = []
    val_f1, val_bias, val_ts = [], [], []
    for epoch in range(epochs):
        if epoch in lr_wd_g1:
            lr, wd = lr_wd_g1[epoch]
            print(f"** scheduler: new Adam parameters at epoch {epoch}: {lr,wd}")
            optimizer_g1 = Adam(
                model_generator1.parameters(),
                lr=lr,
                weight_decay=wd,
                betas=(0.5, 0.999),
            )
        if epoch in lr_wd_g2:
            lr, wd = lr_wd_g2[epoch]
            print(f"** scheduler: new Adam parameters at epoch {epoch}: {lr,wd}")
            optimizer_g2 = Adam(
                model_generator2.parameters(),
                lr=lr,
                weight_decay=wd,
                betas=(0.5, 0.999),
            )
        if epoch in lr_wd_d:
            lr, wd = lr_wd_d[epoch]
            print(f"** scheduler: new Adam parameters at epoch {epoch}: {lr,wd}")
            optimizer_d = Adam(
                model_discriminator.parameters(),
                lr=lr,
                weight_decay=wd,
                betas=(0.5, 0.999),
            )

        model_generator1.train()
        model_generator2.train()
        model_discriminator.train()
        train_loss_g1 = 0
        train_loss_g2 = 0
        train_loss_d = 0
        N = 0
        for batch in tqdm(train_loader, unit=" batches"):
            x, y = batch["inputs"], batch["target"]
            x, y = x.to(device), y.to(device)

            # separate radar data from wind maps
            x_radar, x_wind = separate_radar_wind(x)
            B, T, H, W = y.shape
            
            # First stage generator
            fake_im_first_stage = model_generator1(x_radar, x_wind).view(B, T, H, W)
            # compute the loss for the first stage generator
            loss_g1 = loss_stage1(fake_im_first_stage, y)
            optimizer_g1.zero_grad()
            loss_g1.backward(retain_graph=True)
            optimizer_g1.step()
            
            l = loss_g1.item()
            train_losses_g1.append(l)
            train_loss_g1 += l
            
            
            # Second stage generator
            fake_im = model_generator2(x_radar, x_wind, fake_im_first_stage.detach().view(B, T, 1, H, W)).view(B, T, 1, H, W)
            
            # classify the real images
            yhat_real = model_discriminator(y.view(B, T, 1, H, W))

            # classify the fake images
            yhat_fake = model_discriminator(fake_im.detach())

            # compute the loss for the discriminator
            real_targets = torch.ones_like(yhat_real)
            fake_targets = torch.zeros_like(yhat_fake)
            loss_d = nn.BCELoss()(yhat_real, real_targets) + nn.BCELoss()(
                yhat_fake, fake_targets
            )
            optimizer_d.zero_grad()
            loss_d.backward(retain_graph=True)
            optimizer_d.step()
            l = loss_d.item()
            train_losses_d.append(l)
            train_loss_d += l

            # compute the loss for the generator
            W_LOSS_STAGE_2 = [1, 200, 200]
            yhat_fake_for_g = model_discriminator(fake_im)
            loss_g2 = W_LOSS_STAGE_2[0] * nn.BCELoss()(yhat_fake_for_g, torch.ones_like(yhat_fake_for_g))
            loss_g2 += W_LOSS_STAGE_2[1] * (fake_im.view(B, T, H, W) - y).pow(2).mean()
            loss_g2 += W_LOSS_STAGE_2[2] * (fake_im.view(B, T, H, W) - y).abs().mean()
            optimizer_g2.zero_grad()
            loss_g2.backward()
            optimizer_g2.step()
            l = loss_g2.item()
            train_losses_g2.append(l)
            train_loss_g2 += l

            N += x.shape[0]

        train_loss_g1 = train_loss_g1 / N
        train_loss_g2 = train_loss_g2 / N
        train_loss_d = train_loss_d / N
        train_losses.append(train_loss_g2)
        print(f"epoch {epoch+1} {train_loss_g2=}")

        model_generator1.eval()
        model_generator2.eval()
        val_loss = 0
        CT_pred = 0
        RMSE_pred = 0
        N = 0
        for batch in tqdm(val_loader, unit=" batches"):
            x, y = batch["inputs"], batch["target"]
            x, y = x.to(device), y.to(device)[:, -1]
            x_radar, x_wind = separate_radar_wind(x)
            with torch.no_grad():
                y_hat = model_generator1(x_radar, x_wind)
                y_hat = model_generator2(x_radar, x_wind, y_hat)[:, -1]
            l = loss(y_hat, y)
            val_loss += l.item()
            CT_pred += calculate_CT(
                map_to_classes(y, thresholds), map_to_classes(y_hat, thresholds)
            )
            RMSE_pred += ((y - y_hat) ** 2).mean()
            N += x.shape[0]

        f1_pred, bias, ts = calculate_BS(CT_pred, ["F1", "BIAS", "TS"])

        RMSE_pred = ((RMSE_pred) / N) * 0.5

        val_loss /= N
        val_losses.append(val_loss)

        val_f1.append(f1_pred)
        val_bias.append(bias)
        val_ts.append(ts)

        print(f"epoch {epoch+1} {val_loss=} {f1_pred=} {f1_pers=}")

        writer.add_scalar("train G1", train_loss_g1, epoch)
        writer.add_scalar("train G2", train_loss_g2, epoch)
        writer.add_scalar("train D", train_loss_d, epoch)
        writer.add_scalar("val", val_loss, epoch)
        for c in range(len(thresholds)):
            writer.add_scalar(f"F1_C{c+1}", f1_pred[c], epoch)
            writer.add_scalar(f"TS_C{c+1}", ts[c], epoch)
            writer.add_scalar(f"BIAS_C{c+1}", bias[c], epoch)
            writer.add_scalar("RMSE", RMSE_pred, epoch)

        if epoch % snapshot_step == snapshot_step - 1:
            torch.save(
                model_generator1.state_dict(), join(rundir, f"model_s1_epoch_{epoch}.pt")
            )
            torch.save(
                model_generator2.state_dict(), join(rundir, f"model_s2_epoch_{epoch}.pt")
            )
            torch.save(
                model_discriminator.state_dict(),
                join(rundir, f"model_discriminator_epoch_{epoch}.pt"),
            )

    print(f"Optimisation is over. Model weights had been saved in {rundir}")
    torch.save(model_generator1.state_dict(), join(rundir, "model_s1_last_epoch.pt"))
    torch.save(model_generator2.state_dict(), join(rundir, "model_s2_last_epoch.pt"))
    torch.save(
        model_discriminator.state_dict(),
        join(rundir, "model_discriminator_last_epoch.pt"),
    )

    return {
        "train_losses": train_losses,
        "train_losses_g1": train_losses_g1,
        "train_losses_g2": train_losses_g2,
        "train_losses_d": train_losses_d,
        "val_losses": val_losses,
        "val_f1": val_f1,
        "f1_pers": f1_pers,
        "val_bias": val_bias,
        "bias_pers": bias_pers,
        "val_ts": val_ts,
        "ts_pers": ts_pers,
        "val_rmse": RMSE_pred,
        "RMSE_pers": RMSE_pers,
    }
