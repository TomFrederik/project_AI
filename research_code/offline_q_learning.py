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
    
    def forward(self, x):
        if self.hparams.finetune_vqvae:
            vqvae_latent = self.vqvae.encode_with_grad(x)[0]
        else:
            vqvae_latent = self.vqvae.encode_only(x)[0]
        conv_out = self.conv(vqvae_latent)
        return einops.rearrange(conv_out, 'b c h w -> b (c h w)')

    @property
    def trainable_parameters(self):
        return self.trainable_params

class VectorQuantizer(nn.Module):
    def __init__(self, model_class, model_path=None):
        super().__init__()
        if model_path is not None:
            self.model = model_class.load_from_checkpoint(model_path)
        else:
            self.model = None
            
        dummy_input = torch.zeros(1,64)
        self.output_dim = self.forward(dummy_input).shape[1]
        
    def forward(self, x):
        if self.model is None:
            return x
        else:
            return self.model.encode_only(x)[0]

    @property
    def trainable_params(self):
        return []

class ActionQuantizer(VectorQuantizer):
    def __init__(self, model_path=None):
        super().__init__(ActionVQVAE, model_path)
        
class VecobsQuantizer(VectorQuantizer):
    def __init__(self, model_path=None):
        super().__init__(VecObsVQVAE, model_path)
    

class OfflineQLearner(pl.LightningModule):
    def __init__(self, model_cls_name, model_path, discount_factor, lr, action_quantizer_path=None, vecobs_quantizer_path=None):
        super().__init__()
        self.save_hyperparameters()

        self.action_quantizer = ActionQuantizer(action_quantizer_path)
        self.action_dim = self.action_quantizer.output_dim
        
        self.vecobs_quantizer = VecobsQuantizer(vecobs_quantizer_path)
        self.vecobs_dim = self.vecobs_quantizer.output_dim
        
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
            nn.Linear(q_net_input_dim, 1000),
            nn.GELU(),
            nn.Linear(1000, 1000),
            nn.GELU(),
            nn.Linear(1000, 100),
            nn.GELU(),
            nn.Linear(100, 1),
            nn.GELU()
        )


    def forward(self, pov_obs, vec_obs, actions):
        pov_features = self.pov_feature_extractor(pov_obs)

        actions = self.action_quantizer(actions)
        vec_obs = self.vecobs_quantizer(vec_obs)

        q_net_input = torch.cat([pov_features, vec_obs, actions], dim=1)

        predicted_q_values = self.q_net(q_net_input)
        
        return predicted_q_values
        
    def training_step(self, batch, batch_idx):
        # unpack batch
        
        pov_obs, vec_obs, actions, rew = batch
        
        targets = self.compute_q_values(rew)

        predicted_q_values = self.forward(pov_obs, vec_obs, actions).squeeze()
                
        loss = nn.MSELoss(reduction='mean')(predicted_q_values, targets)
        self.logger.experiment.add_histogram('Predicted_Q', predicted_q_values, self.global_step)
        self.logger.experiment.add_histogram('True_Q', targets, self.global_step)
        self.log('Training/loss', loss, on_step=True)
        return loss
    
    def configure_optimizers(self):
        params = list(self.q_net.parameters())
        params += list(self.pov_feature_extractor.trainable_parameters)
        self.optimizer = torch.optim.AdamW(params, lr=self.hparams.lr)
        return self.optimizer
    
    def compute_q_values(self, rew):
        assert len(rew.shape) == 1, f"{rew.shape = }, but expected (T,)"
        # compute discount factors
        discount_matrix = torch.tensor([self.hparams.discount_factor ** i for i in range(len(rew))], device=self.device)
        discount_matrix = einops.repeat(discount_matrix, 'T -> repeat T', repeat=len(discount_matrix))
        discount_matrix = torch.triu(discount_matrix)
        # compute q values from rewards and discount matrix
        q_values = discount_matrix @ rew
        return q_values

def main(
    env_name,
    log_dir,
    data_dir,
    batch_size,
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
    vecobs_quantizer_version
):
    pl.seed_everything(1337)

    #
    log_path = os.path.join(log_dir, 'OfflineQLearner', env_name, model_class_name)
    
    # load data
    data = TrajectoryData(env_name, data_dir)
    data_loader = DataLoader(data, batch_size=None, num_workers=num_workers)

    # set up model    
    if action_quantizer_version is None:
        action_quantizer_path = None
    else:
        action_quantizer_path = os.path.join(log_dir, 'ActionVQVAE', env_name, 'lightning_logs', 'version_'+str(action_quantizer_version), 'checkpoints', 'last.ckpt')
    
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
        vecobs_quantizer_path=vecobs_quantizer_path
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
        max_epochs=epochs,
        track_grad_norm=2
    )
    
    # train model
    trainer.fit(model, data_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='MineRLTreechopVectorObf-v0')
    parser.add_argument('--data_dir', default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--log_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
    parser.add_argument('--model_class_name', default='conv')
    parser.add_argument('--model_path')
    parser.add_argument('--finetune_vqvae', action='store_true')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--discount_factor', default=0.99, type=float)
    parser.add_argument('--load_from_checkpoint', action='store_true')
    parser.add_argument('--version', default=0, type=int, help='Version of model, if training is resumed from checkpoint')
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--action_quantizer_version', default=None, type=int, help='Version of action quantizer')
    parser.add_argument('--vecobs_quantizer_version', default=None, type=int, help='Version of vecobs quantizer')
    
    args = parser.parse_args()
    
    main(**vars(args))