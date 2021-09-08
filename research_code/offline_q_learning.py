import argparse
import os
import einops
from vqvae import VQVAE

from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets import TrajectoryData

class VAEFeatureExtractor(nn.Module):
    def __init__(self, model_cls, model_path):
        self.model = model_cls.load_from_checkpoint(model_path)
    
    def forward(self, x):
        return self.model.encode_only(x)


class OfflineQLearner(pl.LightningModule):
    def __init__(self, model_cls, model_path, discount_factor, lr):
        super().__init__()
        self.save_hyperparameters()
        
        # init model
        # TODO make this more customizable
        self.pov_feature_extractor = VAEFeatureExtractor(model_cls, model_path)
        
        # compute q_net input dim
        dummy_pov = torch.zeros(1,3,64,64, dtype=torch.float32)
        pov_features = self.pov_feature_extractor(dummy_pov)
        assert len(pov_features.shape) == 2, f'{pov_features.shape = } but expected shape (T, D)'

        q_net_input_dim = pov_features.shape[-1] + 64 + 64
        self.q_net = nn.Sequential(
            nn.Linear(q_net_input_dim, 100),
            nn.GELU(),
            nn.Linear(100, 100),
            nn.GELU(),
            nn.Linear(100, 100),
            nn.GELU(),
            nn.Linear(100, 1)
        )


    def forward(self, pov_obs, vec_obs, actions):
        pov_features = self.pov_feature_extractor(pov_obs)
        
        q_net_input = torch.cat(pov_features, vec_obs, actions, dim=1)
        
        predicted_q_values = self.q_net(q_net_input)
        
        return predicted_q_values
        
    def training_step(self, batch, batch_idx):
        # unpack batch
        pov_obs, vec_obs, actions, rew = batch
        
        targets = self.compute_q_values(rew)

        predicted_q_values = self.forward(pov_obs, vec_obs, actions)
        print(f'{predicted_q_values.shape = }')
                
        loss = nn.MSELoss(predicted_q_values, targets)
        
        self.log('Training/loss', loss, on_step=True)
        return loss
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
    
    def compute_q_values(self, rew):
        assert len(rew.shape) == 1, f"{rew.shape = }, but expected (T,)"
        # compute discount factors
        print(f'{discount_matrix.shape = }')
        discount_matrix = torch.tensor([self.hparams.discount_factor ** i for i in range(len(rew))], device=self.device)
        print(f'{discount_matrix.shape = }')
        discount_matrix = einops.repeat(discount_matrix, 'T -> repeat T', repeat=len(discount_matrix))
        print(f'{discount_matrix.shape = }')
        discount_matrix = torch.triu(discount_matrix)
        print(f'{discount_matrix.shape = }')
        q_values = discount_matrix @ rew
        print(f'{q_values.shape = }')
        print(f'{q_values = }')
        return q_values

def main(
    env_name,
    log_dir,
    data_dir,
    batch_size,
    num_workers,
    save_freq,
    lr,
    discount_factor,
    load_from_checkpoint,
    version,
    epochs
):
    
    #
    log_path = os.path.join(log_dir, 'OfflineQLearner', env_name, 'VQVAE')
    
    # load data
    data = TrajectoryData(env_name, data_dir, num_workers)
    data_loader = DataLoader(data, batch_size=None)

    # init model    
    model = OfflineQLearner(
        model_cls=VQVAE, 
        model_path=model_path, 
        discount_factor=discount_factor, 
        lr=lr
    )

    callbacks = [ModelCheckpoint(monitor='Training/loss', mode='min', every_n_train_steps=save_freq, save_last=True)]
    
    trainer = pl.Trainer(
        progress_bar_refresh_rate=1,
        log_every_n_steps=1,
        callbacks=callbacks,
        gpus=torch.cuda.device_count(),
        accelerator='dp',
        default_root_dir=log_path,
        max_epochs=1
    )
    
    trainer.fit(model, data_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='MineRLTreechopVectorObf-v0')
    parser.add_argument('--data_dir', default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--log_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
    parser.add_argument('--model_class', default='vqvae')
    parser.add_argument('--model_path')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--discount_factor', default=0.99, type=float)
    parser.add_argument('--load_from_checkpoint', action='store_true')
    parser.add_argument('--version', default=0, type=int, help='Version of model, if training is resumed from checkpoint')
    parser.add_argument('--epochs', default=1, type=int)
    
    args = parser.parse_args()
    
    main(**vars(args))