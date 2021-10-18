
"""
Using a reward model doesn't make sense if we are already monitoring the 1-step TD error
"""


# import torch
# from torch.optim import AdamW
# import torch.nn as nn
# from torch.utils.data import DataLoader

# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint

# import numpy as np
# import os
# import argparse

# import datasets
# from vecobs_vqvae import VecObsVQVAE


# class RewardMLP(pl.LightningModule):
#     def __init__(
#         self, 
#         hidden_dims, 
#         learning_rate, 
#         visual_model_cls,
#         visual_model_path,
#         dynamics_model_cls,
#         dynamics_model_path
#     ):
#         super().__init__()
#         self.save_hyperparameters()

#         # load VAE
#         self.visual_model = visual_model_by_str[visual_model_cls].load_from_checkpoint(visual_model_path)
#         self.visual_model.eval()

#         if self.hparams.visual_model_cls == 'vqvae':
#             if use_one_hot:
#                 print('\nUsing one-hot representation')
#                 self.latent_dim = self.visual_model.quantizer.num_variables * self.visual_model.quantizer.codebook_size
#             else:
#                 print('\nUsing learned embedding representation')
#                 self.latent_dim = self.visual_model.quantizer.num_variables * self.visual_model.quantizer.embedding_dim

#         elif self.hparams.visual_model_cls == 'vae':
#             self.latent_dim = self.visual_model.hparams.encoder_kwargs['latent_dim']
#         print(f'\nlatent_dim = {self.latent_dim}')

#         self.input_dim = self.latent_dim + 64

#         # create MLP
#         self.mlp = [nn.Sequential(nn.Linear(self.input_dim, hidden_dims[0]), nn.GELU())]
#         for i in range(len(hidden_dims)-1):
#             self.mlp.append(nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.GELU()))
#         self.mlp.append(nn.Linear(hidden_dims[-1], 1))
#         self.mlp = nn.Sequential(*self.mlp)
        
#         # set up loss function
#         self.loss_fn = nn.MSELoss(reduction='none')
        

#     def forward(self, vec_obs):
#         out = vec_obs
#         if self.use_quantizer:
#             out = self.quantizer.encode_only(out)[0]
#         return self.mlp(out)
    
#     def training_step(self, batch, batch_idx):
#         vec_obs, reward = batch
#         vec_obs = vec_obs[0]
#         reward = reward[0]
#         print(f'{vec_obs.shape = }')
#         print(f'{reward.shape = }')
#         if self.hparams.seq_len == 2:
#             vec_obs = torch.diff(vec_obs, n=1, dim=0)
#             reward = reward[1:]
#         reward = torch.log2(1+reward)
#         loss_scaling_factor = 2 ** reward.detach()
#         predicted_reward = self.forward(vec_obs)[:,0]
        
#         loss = torch.mean(self.loss_fn(predicted_reward, reward) * loss_scaling_factor)

#         self.log('Training/loss', loss, on_step=True)
#         return loss

#     def configure_optimizers(self):
#         # set up optimizer, only train mlp params
#         optimizer =  AdamW(self.mlp.parameters(), lr=self.hparams.learning_rate)
#         return optimizer
        
# def train(
#     env_name, 
#     data_dir, 
#     log_dir, 
#     epochs, 
#     save_freq,
#     lr,
#     model_class, 
#     load_from_checkpoint, 
#     version,
#     visual_model_cls,
#     visual_model_path,
#     dynamics_model_cls,
#     dynamics_model_path
# ):
#     pl.seed_everything(1337)

#     # make sure that relevant dirs exist
#     os.makedirs(log_dir, exist_ok=True)
#     print(f'Saving logs and model to {log_dir}')

#     # set hidden dims and input dim
#     # TODO parse these as args
#     hidden_dims = [1000,1000]
    
#     # instantiate model
#     model = RewardMLP(
#         hidden_dims, 
#         lr, 
#         visual_model_cls, 
#         visual_model_path, 
#         dynamics_model_cls, 
#         dynamics_model_path
#     )
        
#     # load data
#     data = datasets.TrajectoryData(env_name, data_dir)
#     dataloader = DataLoader(data, num_workers=1, pin_memory=True, batch_size=1)

#     # create callbacks to sample reconstructed images and for model checkpointing
#     checkpoint_callback = ModelCheckpoint(mode="min", monitor="Training/loss", save_last=True, every_n_train_steps=save_freq)
#     config = dict(
#         env_name=env_name,
#         visual_model_cls=visual_model_cls,
#         visual_model_path=visual_model_path,
#         dynamics_model_cls=dynamics_model_cls,
#         dynamics_model_path=dynamics_model_path,
#         use_one_hot=use_one_hot,
#     )
#     tags = [visual_model_cls, 'one_hot_'+str(use_one_hot), 'dyn_'+dynamics_model_cls]
#     wandb_logger = WandbLogger(project='RewardModel', config=config, tags=tags)
#     trainer=pl.Trainer(
#         logger=wandb_logger,
#         progress_bar_refresh_rate=1, #every N batches update progress bar
#         callbacks=[checkpoint_callback],
#         gpus=torch.cuda.device_count(),
#         default_root_dir=log_dir,
#         max_epochs=epochs,
#         log_every_n_steps=10,
#     )
                    
#     # fit model
#     trainer.fit(model, dataloader)
    

# if __name__=='__main__':
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--data_dir', default='/home/lieberummaas/datadisk/minerl/data')
#     parser.add_argument('--log_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
#     parser.add_argument('--env_name', default='MineRLObtainIronPickaxeVectorObf-v0')
    
#     parser.add_argument('--save_freq', default=100, type=int)
#     parser.add_argument('--epochs', default=1, type=int)
#     parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate')
#     parser.add_argument('--load_from_checkpoint', default=False, action='store_true')
#     parser.add_argument('--version', default=0, type=int, help='Version of model, if training is resumed from checkpoint')

#     parser.add_argument('--model_class', default='MLP', type=str)
#     parser.add_argument('--visual_model_cls', type=str, default='vae', choices=['vae', 'vqvae'])
#     parser.add_argument('--visual_model_path', type=str, default='none')
#     parser.add_argument('--dynamics_model_cls', type=str, default='none', choices=['mdn', 'none'])
#     parser.add_argument('--dynamics_model_path', type=str, default='none')

#     args = vars(parser.parse_args())

#     train(**args)
