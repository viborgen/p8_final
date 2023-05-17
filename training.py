#based on https://github.com/EmilienDupont/coin
import util
import torch
import tqdm
from collections import OrderedDict
import numpy as np
import wandb


class Trainer():
    def __init__(self, representation, lr=2e-4):    
        """
        Initialize the Trainer class.

        Args:
            representation: The model representation to train.
            lr (float): Learning rate for the optimizer.
        """
        self.representation = representation
        self.optimizer = torch.optim.Adam(self.representation.parameters(), lr=lr)
        self.steps = 0  # Number of steps taken in training
        self.loss_func = torch.nn.MSELoss()
        self.best_vals = {'psnr': 0.0, 'loss': 1e8, 'mse': 1e8, 'visqol': 0}
        self.logs = {'psnr': [], 'loss': [], 'mse': [], 'visqol': []}
        # Store parameters of best model (in terms of highest PSNR achieved)
        self.best_model = OrderedDict((k, v.detach().clone()) for k, v in self.representation.state_dict().items())

    def train(self, coordinates, features, num_iters,max_val,min_val,sr,sweeping,retrainMixed=False,mode="audio"):
        """
        Train the model.

        Args:
            coordinates: Tensor of input coordinates.
            features: Tensor of target features.
            num_iters (int): Number of training iterations.
            max_val (float): Maximum value for denormalization.
            min_val (float): Minimum value for denormalization.
            sr (int): Sample rate.
            sweeping (bool): True if conducting a sweep, False otherwise.
        """
        if(retrainMixed):
            scaler = torch.cuda.amp.GradScaler()
        with tqdm.trange(num_iters, ncols=100) as t:
            for i in t:
                # Update model
                self.optimizer.zero_grad()
                if(retrainMixed):                    
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        predicted = self.representation(coordinates)
                        loss = self.loss_func(predicted, features)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    predicted = self.representation(coordinates)
                    loss = self.loss_func(predicted, features)
                    loss.backward()
                    self.optimizer.step()

                # Calculate psnr
                psnr = util.get_clamped_psnr(predicted, features)
                if(i % 2000 == 0):
                    #peaq = get_clamped_peaq(predicted,features,sr) 
                    visqol = util.visqol(features,predicted,sr,mode)
                mse = util.get_clamped_mse(predicted,features)
                
                # Log results to W&B
                if(sweeping):
                    wandb.log({
                        'loss': loss.item(),
                        'best_loss': self.best_vals['loss'],
                        'psnr': psnr,
                        'best_psnr': self.best_vals['psnr'],
                        'mse': mse,
                        'best_mse': self.best_vals['mse'],
                        'visqol': visqol,
                        'best_visqol': self.best_vals['visqol']
                    }, step=i)

                # Update best values
                if loss.item() < self.best_vals['loss']:
                    self.best_vals['loss'] = loss.item()
                if mse < self.best_vals['mse']:
                    self.best_vals['mse'] = mse
                if visqol > self.best_vals['visqol']:
                    self.best_vals['visqol'] = visqol
                if psnr > self.best_vals['psnr']:
                    self.best_vals['psnr'] = psnr
                    # If model achieves best PSNR seen during training, update model
                    if i > int(num_iters / 2.):
                        for k, v in self.representation.state_dict().items():
                            self.best_model[k].copy_(v)

                #saves a soundclip to Wandb every 10000 iterations
                if(i % 10000 == 0):
                    if(i != 0):
                        if(sweeping): #if we are sweeping, log audio to wandb
                            with torch.no_grad():
                                stft_recon = self.representation(coordinates).to('cpu').numpy() #makes prediction
                                stft_recon.squeeze() #removes extra dimension
                                denorm_matrix = stft_recon * (max_val - min_val) + min_val #denormalizes
                                wandb.log({"audio_sample": wandb.Audio(denorm_matrix, sample_rate=sr)}) #logs audio to wandb