import models
import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from torchvision.utils import make_grid

from time import time
import os
import argparse

# for debugging
torch.autograd.set_detect_anomaly(True)

STR_TO_CLASS = {'Conv':models.ConvVAE, 'CLSTM':models.CLSTMVAE}

class GenerateCallback(pl.Callback):

    def __init__(self, batch_size=4, every_n_epochs=1, save_to_disk=False):
        """
        Inputs:
            batch_size - Number of images to generate
            every_n_epochs - Only save those images every N epochs (otherwise tensorboard gets quite large)
            save_to_disk - If True, the samples and image means should be saved to disk as well.
        """
        super().__init__()
        self.batch_size = batch_size
        self.every_n_epochs = every_n_epochs
        self.save_to_disk = save_to_disk

    def on_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        if (trainer.current_epoch+1) % self.every_n_epochs == 0:
            self.sample_and_save(trainer, pl_module, trainer.current_epoch+1)

    def sample_and_save(self, trainer, pl_module, epoch):
        """
        Function that generates and save samples from the VAE.
        The generated samples and mean images should be added to TensorBoard and,
        if self.save_to_disk is True, saved inside the logging directory.
        Inputs:
            trainer - The PyTorch Lightning "Trainer" object.
            pl_module - The VAE model that is currently being trained.
            epoch - The epoch number to use for TensorBoard logging and saving of the files.
        """
        # Hints:
        # - You can access the logging directory path via trainer.logger.log_dir, and
        # - You can access the tensorboard logger via trainer.logger.experiment
        # - Use the torchvision function "make_grid" to create a grid of multiple images
        # - Use the torchvision function "save_image" to save an image grid to disk 

        #x_samples, x_mean = pl_module.sample(self.batch_size)
        
        # log images to tensorboard
        trainer.logger.experiment.add_image('Sample - Epoch {}'.format(epoch),make_grid(x_samples, nrow=self.batch_size))
        trainer.logger.experiment.add_image('Mean - Epoch {}'.format(epoch),make_grid(x_mean, nrow=self.batch_size))


def train_VAE(env_name, data_dir, lr, val_perc, eval_freq, batch_size, epochs, lr_gamma, lr_decrease_freq, model_dir, log_dir, model_class, lr_step_mode):
    
    # determine run name
    run_name = env_name + '_' + model_class + 'VAE' + str(int(time()))
    print(f'\nName of this run: {run_name}')
    
    # make sure that relevant dirs exist
    os.makedirs(os.path.join(log_dir, run_name), exist_ok=True)
    os.makedirs(os.path.join(model_dir, run_name), exist_ok=True)
    
    kernel_size = 3
    num_frames = 4
    latent_dim = 64 # CHANGED
    num_channels = [32,64,128,256] # channels

    # generate lstm kwarg list
    lstm_kwarg_list = [{'out_channels':n} for n in num_channels]
    lstm_kwarg_list[0]['in_channels'] = 3
    lstm_kwarg_list[0]['img_shape'] = (64,64)
    lstm_kwarg_list[0]['kernel_size'] = kernel_size
    for i in range(1, len(lstm_kwarg_list)):
        lstm_kwarg_list[i]['in_channels'] = num_channels[i-1]
        lstm_kwarg_list[i]['kernel_size'] = kernel_size
    print('\nlstm_kwarg_list : ', lstm_kwarg_list)
    
    # set encoder and decoder kwargs
    encoder_kwargs = {
        'lstm_kwarg_list':lstm_kwarg_list,
        'num_frames':num_frames,
        'latent_dim':latent_dim
    }
    decoder_kwargs = {
        'latent_dim':latent_dim,
        'num_channels':num_channels, #TODO: MAKE IT WORK IN REVERSE FOR DECODER
        'kernel_size':kernel_size
    }
    
    # init model
    optim_kwargs = {'lr':lr}
    scheduler_kwargs = {'lr_gamma':lr_gamma, 'lr_decrease_freq':lr_decrease_freq, 'lr_step_mode':lr_step_mode}
    model = STR_TO_CLASS[model_class](encoder_kwargs, decoder_kwargs, optim_kwargs, scheduler_kwargs)

    model = models.MyDataParallel(model)
    
    ''' # not necessary in pytorch-lightning
    # get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # enable multiple GPU use
    if torch.cuda.device_count() > 1:
        print("\nUsing", torch.cuda.device_count(), "GPUs!")
        model = models.MyDataParallel(model)
    model = model.to(device)

    # init optimizer and lr scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    
    # init tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))
    '''
    
    # load data
    data = datasets.OfflineData(env_name, data_dir, num_frames)
    lengths = [len(data)-int(len(data)*val_perc), int(len(data)*val_perc)]
    train_data, val_data = random_split(data, lengths)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)

    num_batches = len(train_data) // batch_size
    if len(train_data) % batch_size != 0:
        num_batches += 1

    print(f'\nnum train samples = {len(train_data)} --> {num_batches} train batches')
    print(f'\nnum val samples = {len(val_data)}')

    trainer=pl.Trainer(
                    precision=16, 
                    gpus=4, 
                    max_epochs=epochs, 
                    weights_save_path=os.path.join(model_dir, run_name)
                )
    trainer.fit(model, train_loader, val_loader)

    ''' # not necessary in pytorch-lightning
    for epoch in range(epochs):
        print(f'\n\nEpoch {epoch+1}:')

        for i, (_, img, _, _, _, _) in enumerate(train_loader):
            optimizer.zero_grad()

            _, _, bpd = model(img.to(device))
            bpd = bpd.mean() # compute mean over all devices, necessary when using multiple GPUs
            bpd.backward()
            optimizer.step()
            
            writer.add_scalar('Training/batch_bpd', bpd.item(), global_step=epoch * num_batches + i + 1)
        
        # Validation
        model.eval()
        val_bpd = 0
        for i, (_, img, _, _, _, _) in enumerate(val_loader):
            with torch.no_grad():
                _, _, bpd = model(img.to(device))
                val_bpd += bpd.mean().item()
        val_bpd /= i+1
        writer.add_scalar('Validation/bpd',val_bpd,global_step=num_batches * (epoch + 1))

        # save model
        print('Saving model..')
        torch.save(model.state_dict(), os.path.join(model_dir, run_name, f'epoch_{epoch+1}.pt'))

        # every N epochs, reduce lr by multiplying it with lr_gamma
        if (epoch+1) % lr_decrease_freq == 0:
            print(f'Multiplying learning rate with factor {lr_gamma}')
            scheduler.step()

        # every N epochs, display result of reconstruction
        if (epoch+1) % eval_freq == 0:
            print('Reconstructing a random validation sample..')
            # sample a random val image
            id = np.random.choice(len(val_data))
            _, img, _, _, _, _ = val_data[id]
            with torch.no_grad():
                img = img.to(device)
                rec_img = model.reconstruct_only(img.reshape((1,*img.shape)))
                if len(rec_img.shape) < len(img.shape):
                    rec_img = rec_img.reshape((1, *rec_img.shape))
                img_stack = torch.cat([img, rec_img],dim=0)
            writer.add_images('Reconstruction', img_stack, global_step=num_batches * (epoch + 1))
        
        # re-enable train mode
        model.train()
    writer.close()
    '''

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir')
    parser.add_argument('--model_dir')
    parser.add_argument('--log_dir')
    parser.add_argument('--env_name')
    parser.add_argument('--model_class', choices=['Conv','CLSTM'])
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate')
    parser.add_argument('--lr_gamma', default=0.5, type=float, help='Learning rate adjustment factor')
    parser.add_argument('--lr_step_mode', default='epoch', choices=['epoch', 'step'], type=str, help='Learning rate adjustment interval')
    parser.add_argument('--lr_decrease_freq', default=1, type=int, help='Learning rate adjustment frequency')
    parser.add_argument('--val_perc', default=0.1, type=float, help='How much of the data should be used for validation')
    parser.add_argument('--eval_freq', default=1, type=int, help='How often to reconstruct a random val image for tensorboard')

    args = vars(parser.parse_args())

    train_VAE(**args)
