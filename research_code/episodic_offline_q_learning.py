import argparse
import os
import einops
from vqvae import VQVAE
from action_vqvae import ActionVQVAE
from vecobs_vqvae import VecObsVQVAE

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np 

from datasets import TrajectoryData

class ConvFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=2), # 64 -> 32
            nn.GELU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2), # 32 -> 16
            nn.GELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2), # 16 -> 8
            nn.GELU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2), # 8 -> 4
            nn.GELU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2), # 4 -> 2
            nn.GELU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2), # 2 -> 1
            nn.GELU()
        )
    
    def forward(self, x):
        conv_out = self.conv(x)
        return einops.rearrange(conv_out, 'b c h w -> b (c h w)')
    
    @property
    def trainable_parameters(self):
        return self.parameters()

class VQVAEFeatureExtractor(nn.Module):
    def __init__(self, model_path, finetune_vqvae=False):
        super().__init__()
        self.vqvae = VQVAE.load_from_checkpoint(model_path)
        
        self.finetune_vqvae = finetune_vqvae

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.vqvae.hparams.args.embedding_dim, out_channels=64, kernel_size=3, padding=1, stride=2), # 16 -> 8
            nn.GELU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2), # 8 -> 4
            nn.GELU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2), # 4 -> 2
            nn.GELU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2), # 2 -> 1
            nn.GELU()
        )

        self.trainable_params = list(self.conv.parameters())
        if finetune_vqvae:
            self.trainable_params += list(self.vqvae.parameters())
        else:
            self.vqvae.eval()
            
    def forward(self, x):
        # center image
        x = self.vqvae.recon_loss.inmap(x)

        # quantize
        if self.finetune_vqvae:
            vqvae_latent = self.vqvae.encode_with_grad(x)[0]
        else:
            vqvae_latent, ind, *_ = self.vqvae.encode_only(x)
        
        # distill
        conv_out = self.conv(vqvae_latent)
        
        return einops.rearrange(conv_out, 'b c h w -> b (c h w)')

    @property
    def trainable_parameters(self):
        return self.trainable_params

class VectorQuantizer(nn.Module):
    def __init__(self, model_class, model_path=None, centroids=True):
        super().__init__()
        self.centroids = centroids
        if centroids:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = torch.from_numpy(np.load(model_path)).to(device)
        elif model_path is not None:
            self.model = model_class.load_from_checkpoint(model_path)
        else:
            self.model = None
            
        dummy_input = torch.zeros(1,64)
        self.output_dim = self.forward(dummy_input).shape[1]
    
    def forward(self, x):
        if self.centroids:
            return self._compute_closest_centroids(x)  
        elif self.model is None:
            return x
        else:
            return self.model.encode_only(x)[0]

    def _compute_closest_centroids(self, x):
        return torch.argmin((self.model[None,...] - x[:,None]).pow(2).sum(-1),-1)

    @property
    def trainable_params(self):
        return []

class ActionQuantizer(VectorQuantizer):
    def __init__(self, model_path=None, action_centroids=True):
        super().__init__(ActionVQVAE, model_path)
        
class VecobsQuantizer(VectorQuantizer):
    def __init__(self, model_path=None, vecobs_centroids=False):
        super().__init__(VecObsVQVAE, model_path)
    

class OfflineQLearner(pl.LightningModule):
    def __init__(
        self, 
        model_cls_name, 
        model_path, 
        discount_factor, 
        lr, 
        action_quantizer_path=None, 
        action_centroids=True,
        vecobs_quantizer_path=None,
        max_batch_size=1000,
        margin=0.8
    ):
        super().__init__()
        self.save_hyperparameters()

        self.action_quantizer = ActionQuantizer(action_quantizer_path, action_centroids)
        self.action_dim = self.action_quantizer.output_dim
        
        self.vecobs_quantizer = VecobsQuantizer(vecobs_quantizer_path)
        self.vecobs_dim = self.vecobs_quantizer.output_dim

        # init discount matrix
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        discount_array = torch.tensor([self.hparams.discount_factor ** i for i in range(30001)], device=device)
        self.discount_matrix = torch.zeros((30000,30000), device=device)
        for i in range(len(self.discount_matrix)):
            self.discount_matrix[i,i:] = discount_array[:-i-1]

        # init model
        # TODO make this more customizable
        if model_cls_name == 'vqvae':
            self.pov_feature_extractor = VQVAEFeatureExtractor(model_path)
        elif model_cls_name == 'conv':
            self.pov_feature_extractor = ConvFeatureExtractor()
        
        # compute q_net input dim
        dummy_pov = torch.zeros(1,3,64,64, dtype=torch.float32)
        pov_features = self.pov_feature_extractor(dummy_pov)
        assert len(pov_features.shape) == 2, f'{pov_features.shape = } but expected shape (T, D)'

        q_net_input_dim = pov_features.shape[-1] + self.action_dim + self.vecobs_dim
        self.q_net = nn.Sequential(
            nn.Linear(q_net_input_dim, 1),
            nn.GELU(),
        )

    def _sub_batch_processing(self, pov_obs, vec_obs, actions):
        # preprocess
        pov_features = self.pov_feature_extractor(pov_obs)
        vec_obs = self.vecobs_quantizer(vec_obs)
        actions = self.action_quantizer(actions)

        # stack inputs
        q_net_input = torch.cat([pov_features, vec_obs, actions], dim=1)

        # predict q values
        predicted_q_values = self.q_net(q_net_input)
        
        return predicted_q_values
    
    def forward(self, pov_obs, vec_obs, actions):
        predicted_q_values = []
        for i in range((len(pov_obs) - 1) // self.hparams.max_batch_size + 1):
            start = i * self.hparams.max_batch_size
            stop = (i + 1) * self.hparams.max_batch_size
            predicted_q_values.append(self._sub_batch_processing(
                pov_obs[start:stop],
                vec_obs[start:stop],
                actions[start:stop],
            ))
        return torch.cat(predicted_q_values, dim=0)
            
    def training_step(self, batch, batch_idx):
        # unpack batch
        pov_obs, vec_obs, actions, rew = batch
        
        targets = self.compute_q_values(rew)
        print(f'{targets = }')
        print(f'{rew = }')
        predicted_q_values = self.forward(pov_obs, vec_obs, actions).squeeze()
        
        loss = nn.MSELoss(reduction='mean')(predicted_q_values, targets)
        self.log('Training/loss', loss, on_step=True)

        self.logger.experiment.add_histogram('Training/Predicted_Q', predicted_q_values, self.global_step)
        self.logger.experiment.add_histogram('Training/True_Q', targets, self.global_step)

        """
        figure = plt.figure()
        plt.plot(np.arange(len(predicted_q_values)), predicted_q_values.detach().cpu().numpy())
        plt.xlabel('Timestep t')
        plt.ylabel('Q')
        self.logger.experiment.add_figure('Training/Predicted_Q',figure, self.global_step)

        figure = plt.figure()
        plt.plot(np.arange(len(targets)), targets.detach().cpu().numpy())
        plt.xlabel('Timestep t')
        plt.ylabel('Q')
        self.logger.experiment.add_figure('Training/True_Q',figure, self.global_step)
        """

        return loss
    
    def configure_optimizers(self):
        params = list(self.q_net.parameters())
        params += list(self.pov_feature_extractor.trainable_parameters)
        self.optimizer = torch.optim.AdamW(params, lr=self.hparams.lr)
        return self.optimizer
    
    def compute_q_values(self, rew):
        assert len(rew.shape) == 1, f"{rew.shape = }, but expected (T,)"
        # compute q values from rewards and discount matrix
        print(f'self.discount_matrix[:len(rew), :len(rew)] = \n {self.discount_matrix[:len(rew), :len(rew)]}')
        q_values = self.discount_matrix[:len(rew), :len(rew)] @ rew
        return q_values

    def _large_margin_classification_loss(self, q_values, expert_action):
        '''
        Computes the large margin classification loss J_E(Q) from the DQfD paper
        '''
        idcs = torch.arange(0,len(q_values),dtype=torch.long)
        q_values = q_values + self.hparams.margin
        q_values[idcs, expert_action] = q_values[idcs,expert_action] - self.hparams.margin
        return (torch.max(q_values, dim=1)[0] - q_values[idcs,expert_action]).mean()

def main(
    env_name,
    log_dir,
    data_dir,
    model_class_name,
    model_path,
    finetune_vqvae,
    num_workers,
    save_freq,
    lr,
    discount_factor,
    load_from_checkpoint,
    version,
    epochs,
    action_quantizer_version,
    vecobs_quantizer_version,
    margin,
    action_num_centroids
):
    pl.seed_everything(1337)

    #
    log_path = os.path.join(log_dir, 'EpisodicOfflineQLearner', env_name, model_class_name)
    
    # load data
    data = TrajectoryData(env_name, data_dir)
    data_loader = DataLoader(data, batch_size=None, num_workers=num_workers)

    # set up model    
    if action_quantizer_version is None:
        action_quantizer_path = os.path.join(data_dir, env_name+'_centroids_'+str(action_num_centroids)+'.npy')
        centroids = True
    else:
        action_quantizer_path = os.path.join(log_dir, 'ActionVQVAE', env_name, 'lightning_logs', 'version_'+str(action_quantizer_version), 'checkpoints', 'last.ckpt')
        centroids = False
    
    if vecobs_quantizer_version is None:
        vecobs_quantizer_path = None
    else:
        vecobs_quantizer_path = os.path.join(log_dir, 'VecObsVQVAE', env_name, 'lightning_logs', 'version_'+str(vecobs_quantizer_version), 'checkpoints', 'last.ckpt')

    model = OfflineQLearner(
        model_cls_name=model_class_name, 
        model_path=model_path, 
        discount_factor=discount_factor, 
        lr=lr,
        action_quantizer_path=action_quantizer_path,
        action_centroids=centroids,
        vecobs_quantizer_path=vecobs_quantizer_path,
        margin=margin
    )

    # define callbacks
    callbacks = [ModelCheckpoint(monitor='Training/loss', mode='min', every_n_train_steps=save_freq, save_last=True)]
    
    # set up trainer
    trainer = pl.Trainer(
        progress_bar_refresh_rate=1,
        log_every_n_steps=1,
        callbacks=callbacks,
        gpus=torch.cuda.device_count(),
        accelerator='dp',
        default_root_dir=log_path,
        max_epochs=epochs
    )
    
    # train model
    trainer.fit(model, data_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='MineRLTreechopVectorObf-v0')
    parser.add_argument('--data_dir', default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--log_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
    parser.add_argument('--model_class_name', default='conv', choices=['conv', 'vqvae'])
    parser.add_argument('--model_path')
    parser.add_argument('--finetune_vqvae', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--discount_factor', default=0.99, type=float)
    parser.add_argument('--margin', default=0.8, type=float)
    parser.add_argument('--load_from_checkpoint', action='store_true')
    parser.add_argument('--version', default=0, type=int, help='Version of model, if training is resumed from checkpoint')
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--action_quantizer_version', default=None, type=int, help='Version of action quantizer')
    parser.add_argument('--action_num_centroids', default=150, type=int, help='Number of clusters for actions, if using kmeans instead of vqvae')
    parser.add_argument('--vecobs_quantizer_version', default=None, type=int, help='Version of vecobs quantizer')
    
    args = parser.parse_args()
    
    main(**vars(args))