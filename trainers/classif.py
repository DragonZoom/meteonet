# A training procedure for Meteonet data

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from loader.utilities import calculate_CT, calculate_BS, map_to_classes

from tqdm import tqdm
from os.path import join

def train_meteonet_classif( train_loader, val_loader, model, thresholds, epochs, lr_wd,
                            snapshot_step = 5, rundir='runs', clip_grad=0.1, tqdm=tqdm, device='cpu'):

    print('Evaluation Persistence...')
    CT_pers = 0
    for batch in tqdm(val_loader):
        CT_pers += calculate_CT( map_to_classes(batch['persistence'], thresholds),
                                 map_to_classes(batch['target'], thresholds))
        f1_pers, bias_pers, ts_pers = calculate_BS( CT_pers, ['F1','BIAS','TS'])

    writer = SummaryWriter(log_dir=rundir)

    loss = nn.BCEWithLogitsLoss()
    loss.to(device)
    model.to(device)
    
    print('Start training...')
    train_losses = []
    val_losses = []
    val_f1, val_bias, val_ts = [], [], []
    for epoch in range(epochs):
        if epoch in lr_wd:
            lr, wd = lr_wd[epoch]
            print(f'** scheduler: new Adam parameters at epoch {epoch}: {lr,wd}')
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
        
        model.train()  
        train_loss = 0
        N = 0
        for batch in tqdm(train_loader):
            x,y = batch['inputs'], map_to_classes( batch['target'], thresholds)
            x,y = x.to(device), y.to(device)

            y_hat = model(x)
            l = loss(y_hat, y)
            train_loss += l.item()

            optimizer.zero_grad()
            l.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_grad)
            optimizer.step()

            N += x.shape[0]

        train_loss /= N
        train_losses.append(train_loss)
        print(f'epoch {epoch+1} {train_loss=}')

        model.eval()
        val_loss = 0
        CT_pred = 0
        N = 0
        for batch in tqdm(val_loader):
            x,y = batch['inputs'], map_to_classes( batch['target'], thresholds)
            x,y = x.to(device), y.to(device)
            with torch.no_grad():
                y_hat = model(x)
            l = loss(y_hat, y)
            val_loss += l.item()
            CT_pred += calculate_CT(torch.sigmoid(y_hat)>.5, y)
            N += x.shape[0]
    
        f1_pred, bias, ts =  calculate_BS( CT_pred, ['F1','BIAS','TS'])
            
        val_loss /= N
        val_losses.append(val_loss)
        
        val_f1.append(f1_pred)
        val_bias.append(bias)
        val_ts.append(ts)

        print(f'epoch {epoch+1} {val_loss=} {f1_pred=} {f1_pers=}')

        writer.add_scalar('train', train_loss, epoch)
        writer.add_scalar('val', val_loss, epoch)
        for c in range(len(thresholds)):
            writer.add_scalar(f'F1_C{c+1}', f1_pred[c], epoch)
            writer.add_scalar(f'TS_C{c+1}', ts[c], epoch)
            writer.add_scalar(f'BIAS_C{c+1}', bias[c], epoch)
    
        if epoch % snapshot_step == snapshot_step-1:
            torch.save(model.state_dict(), join(rundir, f'model_epoch_{epoch}.pt'))
    
    print( f'Optimisation is over. Model weights had been saved in {rundir}')
    torch.save(model.state_dict(), join(rundir, "model_last_epoch.pt"))

    return {'train_losses': train_losses, 'val_losses': val_losses,
            'val_f1': val_f1, 'f1_pers': f1_pers ,
            'val_bias': val_bias, 'bias_pers': bias_pers,
            'val_ts': val_ts, 'ts_pers': ts_pers
            }

