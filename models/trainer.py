# Supervised training
# inspired by https://github.com/arxyzan/vanilla-transformer/blob/main/train.py
# (c) bereziat 2023, version 0.3

import torch, os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from os.path import join
from torch.optim import Adam

class SchedulerAdam:
    def __init__(self, model, lr, wd):
        self.model = model
        self.lr = lr
        self.wd = dw
        self.optim = None
    def __call__(epoch):
        if epoch in self.lr:
            self.optim = Adam(self.model.parameter(), lr=lr, weight_decay=wd)
        return self.optim
        
class TrainerSupervised:
    def __init__(self, model, loss, scheduler, get_xy, scores_fun = None, rundir="runs", device = 'cpu', clip=False):
        """
        scheduler(epoch)->optimizer: a function geting an epoch number and returning an optimizer function
        get_xy(batch)->tuple[X,Y]: from a batch, returns a tuple (X,Y) such as loss(Y, model(X)) applies
        scores_func(pred, true) -> Tensor[float,...]: a function getting a prediction and a ground truth, returning a Tensor of scores
        
        """
        self.model = model
        self.loss = loss
        self.scheduler = scheduler
        self.optim = None
        self.scores_fun = scores_fun
        self.device = device
        self.clip = clip
        self.get_xy = get_xy

        self.rundir = join(rundir,f'{datetime.now()}')
        os.system(f'mkdir -p "{self.rundir}"')
        os.system(f'rm -f lastrun; ln -sf "{self.rundir}" lastrun')

        self.writer = SummaryWriter(log_dir=self.rundir)
        
    def train( self, train):
        self.model.train()
        training_loss = 0
        N = 0
        pbar = tqdm(train, unit='batch')
        for data in pbar:
            X,Y = self.get_xy(data)

            # Loading data
            if self.device != 'cpu':
                X = X.to(self.device)
                Y = Y.to(self.device)
            N += X.shape[0]

            # Generating output
            Y_hat = self.model(X)
          
            # Calculating loss
            loss = self.loss(Y_hat, Y)

            if self.clip: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            # Updating weights according to the calculated loss
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
          
            # Incrementing loss
            training_loss += loss.item()
    
        return training_loss / N

    def evaluate(self, valid):
        validation_loss = 0
        validation_scores = 0

        self.model.eval()
        N = 0
        validation_loss = 0
        pbar = tqdm(valid, unit='batch')
        for data in pbar:
            X,Y = self.get_xy(data)

            if self.device != 'cpu':
                X = X.to(self.device)
                Y = Y.to(self.device)
            N += X.shape[0]
            
            with torch.no_grad():
                Y_hat = self.model(X)
                
            validation_loss += self.loss(Y_hat,Y).item()
            if self.scores_fun: validation_scores += self.scores_fun(Y_hat, Y).cpu()
                 
        return validation_loss / N, validation_scores
        
    def fit(self, train, valid, epochs, snapshot_step = 5):
        train_loss = []
        valid_loss = []
        scores_list = []
        
        for epoch in range(epochs):
            self.optim = self.scheduler( epoch)

            # train pass
            print(f'\033[91mTraining epoch {epoch}\033[00m')
            train_loss.append(self.train(train))
            print( f'Loss: {train_loss[-1]}')

            # evaluation pass
            print(f'\033[92mEvaluation epoch {epoch}\033[00m')
            loss, scores = self.evaluate(valid)
            valid_loss.append( loss)
            scores_list.append(scores)
            print( f'Loss: {train_loss[-1]}')

            # save weights
            if epoch % snapshot_step == snapshot_step-1:
                torch.save(self.model, join(self.rundir,f'model_epoch{epoch}.tch'))

            # save losses and scores
            self.writer.add_scalar('train', train_loss[-1], epoch)
            self.writer.add_scalar('val', valid_loss[-1], epoch)
            if self.scores_fun:
                for i,sc in enumerate(list(scores)):
                    self.writer.add_scalar(f'score {i}', sc, epoch)
            # pbar.set_postfix({'train':train_loss[-1],'val':valid_loss[-1]})

        # save last epoch
        torch.save(self.model, join(self.rundir,f'model_epoch{epochs-1}.tch'))
        # save losses
        torch.save({'train':train_loss,'valid':valid_loss,'scores':scores_list},
                    join(self.traindir,'losses.tch'))
        return train_loss, valid_loss, scores_list
