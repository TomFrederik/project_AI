import argparse 
from BCModels import BCModel
from vqvae import VQVAE
from visual_models import ResnetVAE
from datasets import BufferedBatchDataset
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import torch

def train(
    env_name,
    data_dir,
    log_dir,
    feature_extractor_class_name,
    feature_extractor_version,
    lr,
    action_num_centroids, 
    action_vqvae_version, 
    vecobs_vqvae_version,
    hidden_dim,
    batch_size,
    num_epochs,
    save_freq
):
    # set seed
    pl.seed_everything(1337)

    # get feature extractor class
    feature_extractor_class = {
        'vqvae':VQVAE,
        'vae':ResnetVAE,
        'conv':None
    }[feature_extractor_class_name]
    
    # get paths
    if feature_extractor_class is not None:
        feature_extractor_path = os.path.join(log_dir, feature_extractor_class.__name__, env_name, 'lightning_logs', 'version_'+feature_extractor_version, 'checkpoints', 'last.ckpt')
    else:
        feature_extractor_path = None
    if action_num_centroids is not None:
        action_centroids_path = os.path.join(data_dir, env_name+'_'+str(action_num_centroids)+'_centroids.npy')
    if action_vqvae_version is not None:
        action_vqvae_path = os.path.join(log_dir, 'ActionVQVAE', env_name, 'lightning_logs', 'version_'+action_vqvae_version, 'checkpoints', 'last.ckpt')
    else:
        action_vqvae_path = None
    if vecobs_vqvae_version is not None:
        vecobs_vqvae_path = os.path.join(log_dir, 'VecObsVQVAE', env_name, 'lightning_logs', 'version_'+vecobs_vqvae_version, 'checkpoints', 'last.ckpt')
    else:
        vecobs_vqvae_path = None
    
    # set up BC model    
    model = BCModel(
        feature_extractor_path,
        feature_extractor_class,
        lr,
        action_centroids_path,
        action_vqvae_path,
        vecobs_vqvae_path,
        hidden_dim
    )
    
    # set up data
    data = BufferedBatchDataset(env_name, data_dir, batch_size, num_epochs)
    dataloader = DataLoader(data, num_workers=1)
    
    # set log dir for trainer
    default_log_dir = os.path.join(log_dir, 'BC', env_name, feature_extractor_class_name)
    if action_vqvae_path is None and vecobs_vqvae_path is None:
        default_log_dir = os.path.join(default_log_dir, 'action_centroids_only')
    if action_vqvae_path is not None and vecobs_vqvae_path is None:
        default_log_dir = os.path.join(default_log_dir, 'action_vqvae_only')
    if action_vqvae_path is None and vecobs_vqvae_path is not None:
        default_log_dir = os.path.join(default_log_dir, 'action_centroids_vecobs_vqvae')
    if action_vqvae_path is not None and vecobs_vqvae_path is not None:
        default_log_dir = os.path.join(default_log_dir, 'action_vqvae_vecobs_vqvae')
    
    # set up trainer
    callbacks = [ModelCheckpoint(monitor='Training/loss', mode='min', every_n_train_steps=save_freq, save_last=True)]
    trainer = pl.Trainer(
        progress_bar_refresh_rate=10,
        log_every_n_steps=10,
        callbacks=callbacks,
        gpus=torch.cuda.device_count(),
        accelerator='dp',
        default_root_dir=default_log_dir,
        max_epochs=num_epochs
    )
    
    # fit model
    trainer.fit(model, dataloader)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MineRLTreechopVectorObf-v0')
    parser.add_argument('--data_dir', type=str, default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--log_dir', type=str, default='/home/lieberummaas/datadisk/minerl/run_logs')
    parser.add_argument('--feature_extractor_class_name', type=str, default='vqvae', choices=['vqvae', 'vae', 'conv'])
    parser.add_argument('--feature_extractor_version', type=str, default='0')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--action_num_centroids', type=int, default=None)
    parser.add_argument('--action_vqvae_version', type=str, default=None)
    parser.add_argument('--vecobs_vqvae_version', type=str, default=None)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=100)
    
    args = parser.parse_args()
    
    train(**vars(args))