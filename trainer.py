# Training
# inspired by https://github.com/arxyzan/vanilla-transformer/blob/main/train.py
import torch, os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, loss, optim, get_xy, logdir='runs', device = 'cpu', clip=False):
        """ get_xy: from a batch, returns a tuple (X,Y) such as loss(Y, model(X)) applies
        """
        self.model = model
        self.loss = loss
        self.optim = optim
        self.device = device
        self.clip = clip
        self.get_xy = get_xy
        # os.system('rm -rf ' + logdir)
        os.system('mkdir -p ' + logdir)
        self.traindir = os.path.join(logdir,"train")
        self.writer = SummaryWriter(log_dir=self.traindir)
        
    def train( self, trainset):
        self.model.train()
        training_loss = 0
        N = 0
        for data in trainset:
            X,Y = self.get_xy(data)

            # Loading data
            if self.device != 'cpu': X = X.to(self.device)
            N += X.shape[0]

            # Generating output
            Y_hat = self.model(X)
          
            # Calculating loss
            loss = self.loss(Y, Y_hat)
          
            if self.clip: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            # Updating weights according
            # to the calculated loss
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
          
            # Incrementing loss
            training_loss += loss.item()
    
        return training_loss / N

    def evaluate(self, valid):
        validation_loss = 0
        self.model.eval()
        N = 0
        validation_loss = 0
        for X,_,_ in valid:
            if self.device != 'cpu': X = X.to(self.device)
            N += X.shape[0]
            with torch.no_grad():
                X_hat = self.model(X)
            validation_loss += self.loss(X,X_hat).item()
        return validation_loss / N
        
    def fit(self, trainset, valid, epochs):
        train_loss = []
        valid_loss = []

        pbar = tqdm(range(epochs), unit='epoch')
        for epoch in pbar:
            train_loss.append(self.train(trainset))
            valid_loss.append(self.evaluate(valid))

            if epoch%5 == 0:
                torch.save(self.model,os.path.join(self.traindir,f'model_epoch{epoch}.tch'))
            self.writer.add_scalar('train', train_loss[-1], epoch)
            self.writer.add_scalar('val', valid_loss[-1], epoch)

            pbar.set_postfix({'train':train_loss[-1],'val':valid_loss[-1]})

        # save last epoch
        torch.save(self.model,os.path.join(self.traindir,f'model_epoch{epochs-1}.tch'))
        # save losses
        torch.save({'train':train_loss,'valid':valid_loss}, os.path.join(self.traindir,'losses.tch'))
        return train_loss, valid_loss
