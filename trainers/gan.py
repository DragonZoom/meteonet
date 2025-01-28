# A training procedure for Meteonet data

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from meteonet.utilities import calculate_CT, calculate_BS, map_to_classes

from tqdm import tqdm
from os.path import join
import time


def calculate_weights(radar, norm_factor):
    """
    Calculate the weights w(z) as defined in equation (2) based on radar echo value (1/100 mm).

    Args:
        radar (torch.Tensor): Radar echo values (tensor of shape [B, T, H, W]).

    Returns:
        torch.Tensor: Weight tensor with the same shape as radar.
    """
    weights = torch.ones_like(radar)
    weights = torch.where(radar >= np.log(0.833 + 1 + 1e-3) / norm_factor, torch.full_like(radar, 1.5), weights)
    weights = torch.where(radar >= np.log(8.333 + 1 + 1e-3) / norm_factor, torch.full_like(radar, 3.0), weights)
    weights = torch.where(radar >= np.log(20.83 + 1 + 1e-3) / norm_factor, torch.full_like(radar, 6.0), weights)
    # weights = torch.where(radar >= 40.00, torch.full_like(radar, 4.0), weights)
    # weights = torch.where(radar >= 70.00, torch.tensor(60.0, device=radar.device), weights)
    return weights

def loss_stage1(predicted, ground_truth, norm_factor):
    """
    Compute the first stage loss as defined in equation (3).

    Args:
        predicted (torch.Tensor): Predicted radar echo sequence (tensor of shape [B, T, H, W]).
        ground_truth (torch.Tensor): Ground truth radar echo sequence (tensor of shape [B, T, H, W]).

    Returns:
        torch.Tensor: The scalar loss value.
    """
    # Compute the weights based on ground truth radar echo values
    weights = calculate_weights(ground_truth, norm_factor)

    # Exponential time weights: Emphasize future time steps
    alpha = 1.6
    T = predicted.shape[1]  # Total number of timesteps
    time_weights = torch.tensor([alpha**t for t in range(T)], device=predicted.device)  # Shape: [T]
    time_weights = time_weights / time_weights.sum()  # Normalize time weights to sum to 1

    # Reshape time weights for broadcasting
    time_weights = time_weights.view(1, T, 1, 1)  # Shape: [1, T, 1, 1]

    # Compute the squared difference and absolute difference
    squared_diff = (predicted - ground_truth) ** 2
    absolute_diff = torch.abs(predicted - ground_truth)

    # Compute the weighted loss for each pixel and each timestep
    weighted_loss = weights * (squared_diff + absolute_diff)
    weighted_loss = weighted_loss * time_weights  # Apply time weights

    # Average over time, and batch
    loss = weighted_loss.mean()

    return loss


def train_meteonet_gan_stage1(
    train_loader,
    val_loader,
    model_generator,
    thresholds,
    epochs,
    lr_wd_g,
    snapshot_step=5,
    rundir="runs",
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
        target = val_loader.dataset.denormalize_rainmap(batch["target"][:,-1])
        
        CT_pers += calculate_CT(
            map_to_classes(persistance, thresholds), map_to_classes(target, thresholds)
        )
        f1_pers, bias_pers, ts_pers = calculate_BS(CT_pers, ["F1", "BIAS", "TS"])
        RMSE_pers += ((persistance - target) ** 2).mean()
        N += target.shape[0]
        # break

    RMSE_pers = RMSE_pers / N

    writer = SummaryWriter(log_dir=rundir)

    loss = nn.MSELoss()
    loss.to(device)
    model_generator.to(device)

    optimizer_g1 = Adam(
        model_generator.parameters(),
        lr=lr_wd_g[0][0],
        weight_decay=lr_wd_g[0][1],
        betas=(0.5, 0.999),
    )
    scheduler_g1 = torch.optim.lr_scheduler.StepLR(optimizer_g1, step_size=1, gamma=1.0)

    print("Start training...")
    train_losses = []
    train_losses_g1 = []
    val_losses1 = []
    val_f1_1, val_bias1, val_ts1 = [], [], []
    for epoch in range(epochs):
        # print learning rate
        print(f"** epoch {epoch+1} learning rate: {scheduler_g1.get_last_lr()}")
        gen1_inference_time = 0
        #

        model_generator.train()
        train_loss_g1 = 0
        N = 0
        for batch in tqdm(train_loader, unit=" batches"):
            x, y = batch["inputs"], batch["target"]
            x, y = x.to(device), y.to(device)

            # separate radar data from wind maps
            B, T, H, W = y.shape
            
            #
            # First stage generator
            start_time = time.time()
            #
            fake_im_first_stage = model_generator(x).view(B, T, H, W)
            # compute the loss for the first stage generator
            loss_g1 = loss_stage1(fake_im_first_stage, y, train_loader.dataset.norm_factors[0])
            optimizer_g1.zero_grad()
            loss_g1.backward()
            optimizer_g1.step()
            
            l = loss_g1.item()
            train_losses_g1.append(l)
            train_loss_g1 += l
            #
            gen1_inference_time += time.time() - start_time
            #
            N += x.shape[0]

        scheduler_g1.step()

        train_loss_g1 = train_loss_g1 / N
        train_losses.append(train_loss_g1)
        print(f"epoch {epoch+1} {train_loss_g1=}")
        print(f"Generator 1 inference time: {gen1_inference_time:.4f} seconds")

        model_generator.eval()
        val_loss1 = np.zeros(T)
        CT_pred1 = [0] * T
        RMSE_pred1 = np.zeros(T)
        N = 0
        for batch in tqdm(val_loader, unit=" batches"):
            x, y = batch["inputs"], batch["target"]
            x, y = x.to(device), val_loader.dataset.denormalize_rainmap(y.to(device))
            with torch.no_grad():
                y_hat1 = model_generator(x).squeeze(2)
                y_hat1 = val_loader.dataset.denormalize_rainmap(y_hat1)
                # y_hat1 = y_hat1[:, -1].squeeze(1)

            for i in range(y_hat1.shape[1]):
                val_loss1[i] += loss(y_hat1[:, i], y[:, i]).item()
                CT_pred1[i] += calculate_CT(
                    map_to_classes(y_hat1[:, i], thresholds), map_to_classes(y[:, i], thresholds)
                )
                RMSE_pred1[i] += ((y[:, i] - y_hat1[:, i]) ** 2).mean()

            # val_loss1 += loss(y_hat1, y).item()

            # CT_pred1 += calculate_CT(
            #     map_to_classes(y, thresholds), map_to_classes(y_hat1, thresholds)
            # )            

            # RMSE_pred1 += ((y - y_hat1) ** 2).mean()
            N += x.shape[0]

        print(f"epoch {epoch+1}")
        for i, ct in enumerate(CT_pred1):
            f1_pred1, bias1, ts1 = calculate_BS(ct, ["F1", "BIAS", "TS"])
            print(f"  T={i}: {f1_pred1=} {val_loss1[i]=} {RMSE_pred1[i]=}")

        RMSE_pred1 = RMSE_pred1[-1]
        val_loss1 = val_loss1[-1]
        CT_pred1 = CT_pred1[-1]



        RMSE_pred1 = (RMSE_pred1) / N

        val_loss1 /= N
        val_losses1.append(val_loss1)

        val_f1_1.append(f1_pred1)
        val_bias1.append(bias1)
        val_ts1.append(ts1)

        print(f"  {val_loss1=} {f1_pred1=}")
        print(f"  {f1_pers=}")

        writer.add_scalar("train G1", train_loss_g1, epoch)
        writer.add_scalar("val1", val_loss1, epoch)
        for c in range(len(thresholds)):
            writer.add_scalar(f"F1_C{c+1}_1", f1_pred1[c], epoch)
            writer.add_scalar(f"TS_C{c+1}_1", ts1[c], epoch)
            writer.add_scalar(f"BIAS_C{c+1}_1", bias1[c], epoch)
            writer.add_scalar("RMSE1", RMSE_pred1, epoch)

        if epoch % snapshot_step == snapshot_step - 1:
            torch.save(
                model_generator.state_dict(), join(rundir, f"model_s1_epoch_{epoch}.pt")
            )

    print(f"Optimisation is over. Model weights had been saved in {rundir}")
    torch.save(model_generator.state_dict(), join(rundir, "model_s1_last_epoch.pt"))

    return {
        "train_losses": train_losses,
        "train_losses_g1": train_losses_g1,
        "val_losses1": val_losses1,
        "val_f1_1": val_f1_1,
        "f1_pers": f1_pers,
        "val_bias1": val_bias1,
        "bias_pers": bias_pers,
        "val_ts1": val_ts1,
        "ts_pers": ts_pers,
        "val_rmse1": RMSE_pred1,
        "RMSE_pers": RMSE_pers,
    }


###############################################################################
#
###############################################################################



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
        target = batch["target"]
        
        CT_pers += calculate_CT(
            map_to_classes(persistance, thresholds), map_to_classes(target, thresholds)
        )
        f1_pers, bias_pers, ts_pers = calculate_BS(CT_pers, ["F1", "BIAS", "TS"])
        RMSE_pers += ((persistance - target) ** 2).mean()
        N += target.shape[0]
        # break

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
    val_losses1 = []
    val_losses2 = []
    val_f1_1, val_bias1, val_ts1 = [], [], []
    val_f1_2, val_bias2, val_ts2 = [], [], []
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

        #
        gen1_inference_time = 0
        gen2_inference_time = 0
        disc_real_inference_time = 0
        disc_fake_inference_time = 0
        gen2_loss_time = 0
        #

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
            B, T, H, W = y.shape
            
            # First stage generator
            fake_im_first_stage = model_generator1(x).view(B, T, H, W)
            # compute the loss for the first stage generator
            start_time = time.time()
            loss_g1 = loss_stage1(fake_im_first_stage, y, train_loader.dataset.norm_factors[0])
            optimizer_g1.zero_grad()
            loss_g1.backward()
            optimizer_g1.step()
            
            fake_im_first_stage = fake_im_first_stage.detach()
            l = loss_g1.item()
            train_losses_g1.append(l)
            train_loss_g1 += l
            gen1_inference_time += time.time() - start_time
            
            
            # Second stage generator
            start_time = time.time()
            fake_im = model_generator2(x, fake_im_first_stage.view(B, T, 1, H, W)).view(B, T, 1, H, W)
            gen2_inference_time += time.time() - start_time
            
            # classify the real images
            start_time = time.time()
            yhat_real = model_discriminator(y.view(B, T, 1, H, W))
            disc_real_inference_time += time.time() - start_time

            # classify the fake images
            start_time = time.time()
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
            disc_fake_inference_time += time.time() - start_time

            # compute the loss for the generator
            start_time = time.time()
            W_LOSS_STAGE_2 = [1, 1, 1]
            yhat_fake_for_g = model_discriminator(fake_im)
            loss_g2_d   = W_LOSS_STAGE_2[0] * nn.BCELoss()(yhat_fake_for_g, torch.ones_like(yhat_fake_for_g))
            loss_g2_p2  = W_LOSS_STAGE_2[1] * (fake_im.view(B, T, H, W) - y).pow(2).mean()
            loss_g2_abs = W_LOSS_STAGE_2[2] * (fake_im.view(B, T, H, W) - y).abs().mean()
            # print(f"loss_g2_d: {loss_g2_d.item()} loss_g2_p2: {loss_g2_p2.item()} loss_g2_abs: {loss_g2_abs.item()}")
            loss_g2 = loss_g2_d + loss_g2_p2 + loss_g2_abs
            optimizer_g2.zero_grad()
            loss_g2.backward()
            optimizer_g2.step()
            l = loss_g2.item()
            train_losses_g2.append(l)
            train_loss_g2 += l
            gen2_loss_time += time.time() - start_time

            N += x.shape[0]

        train_loss_g1 = train_loss_g1 / N
        train_loss_g2 = train_loss_g2 / N
        train_loss_d = train_loss_d / N
        train_losses.append(train_loss_g2)
        print(f"epoch {epoch+1} {train_loss_g1=} {train_loss_g2=} {train_loss_d=}")
        print(f"Generator 1 inference time: {gen1_inference_time:.4f} seconds")
        print(f"Generator 2 inference time: {gen2_inference_time:.4f} seconds")
        print(f"Generator 2 loss time: {gen2_loss_time:.4f} seconds")
        print(f"Discriminator real inference time: {disc_real_inference_time:.4f} seconds")
        print(f"Discriminator fake inference time: {disc_fake_inference_time:.4f} seconds")

        model_generator1.eval()
        model_generator2.eval()
        val_loss1 = 0
        CT_pred1 = 0
        RMSE_pred1 = 0
        val_loss2 = 0
        CT_pred2 = 0
        RMSE_pred2 = 0
        N = 0
        for batch in tqdm(val_loader, unit=" batches"):
            x, y = batch["inputs"], batch["target"]
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                y_hat1 = model_generator1(x)
                y_hat = model_generator2(x, y_hat1)[:, -1].squeeze(1)
                y_hat1 = y_hat1[:, -1].squeeze(1)

            val_loss1 += loss(y_hat1, y).item()
            val_loss2 += loss(y_hat, y).item()

            CT_pred1 += calculate_CT(
                map_to_classes(y, thresholds), map_to_classes(y_hat1, thresholds)
            )            
            CT_pred2 += calculate_CT(
                map_to_classes(y, thresholds), map_to_classes(y_hat, thresholds)
            )

            RMSE_pred1 += ((y - y_hat1) ** 2).mean()
            RMSE_pred2 += ((y - y_hat) ** 2).mean()
            N += x.shape[0]

        f1_pred1, bias1, ts1 = calculate_BS(CT_pred1, ["F1", "BIAS", "TS"])
        f1_pred2, bias2, ts2 = calculate_BS(CT_pred2, ["F1", "BIAS", "TS"])

        RMSE_pred1 = ((RMSE_pred1) / N) * 0.5
        RMSE_pred2 = ((RMSE_pred2) / N) * 0.5

        val_loss1 /= N
        val_loss2 /= N
        val_losses1.append(val_loss1)
        val_losses2.append(val_loss2)

        val_f1_1.append(f1_pred1)
        val_bias1.append(bias1)
        val_ts1.append(ts1)

        val_f1_2.append(f1_pred2)
        val_bias2.append(bias2)
        val_ts2.append(ts2)

        print(f"epoch {epoch+1}")
        print(f"  {val_loss1=} {f1_pred1=}")
        print(f"  {val_loss2=} {f1_pred2=}")
        print(f"  {f1_pers=}")

        writer.add_scalar("train G1", train_loss_g1, epoch)
        writer.add_scalar("train G2", train_loss_g2, epoch)
        writer.add_scalar("train D", train_loss_d, epoch)
        writer.add_scalar("val1", val_loss1, epoch)
        writer.add_scalar("val2", val_loss2, epoch)
        for c in range(len(thresholds)):
            writer.add_scalar(f"F1_C{c+1}_1", f1_pred1[c], epoch)
            writer.add_scalar(f"F1_C{c+1}_2", f1_pred2[c], epoch)
            writer.add_scalar(f"TS_C{c+1}_1", ts1[c], epoch)
            writer.add_scalar(f"TS_C{c+1}_2", ts2[c], epoch)
            writer.add_scalar(f"BIAS_C{c+1}_1", bias1[c], epoch)
            writer.add_scalar(f"BIAS_C{c+1}_2", bias2[c], epoch)
            writer.add_scalar("RMSE1", RMSE_pred1, epoch)
            writer.add_scalar("RMSE2", RMSE_pred2, epoch)

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
        "val_losses1": val_losses1,
        "val_losses2": val_losses2,
        "val_f1_1": val_f1_1,
        "val_f1_2": val_f1_2,
        "f1_pers": f1_pers,
        "val_bias1": val_bias1,
        "val_bias2": val_bias2,
        "bias_pers": bias_pers,
        "val_ts1": val_ts1,
        "val_ts2": val_ts2,
        "ts_pers": ts_pers,
        "val_rmse1": RMSE_pred1,
        "val_rmse2": RMSE_pred2,
        "RMSE_pers": RMSE_pers,
    }
