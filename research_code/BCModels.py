import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

from action_vqvae import ActionVQVAE
from vecobs_vqvae import VecObsVQVAE
from vqvae import VQVAE


class ActionDiscretizer(nn.Module):
    def __init__(self, centroid_path=None, vqvae_path=None, device=None):
        super().__init__()
        
        # set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        # set model
        if centroid_path is None and vqvae_path is None:
            raise ValueError('No ActionDiscretizer detected!')
        elif centroid_path is not None and vqvae_path is not None:
            raise ValueError('Detected both centroids and vqvae, please only supply one!')
        elif centroid_path is not None:
            print('Using centroids for actions..')
            self.use_centroids = True
            self.centroids = torch.from_numpy(np.load(centroid_path)).to(device)
        elif vqvae_path is not None:
            print('Using vqvae for actions..')
            self.use_centroids = False
            self.model = ActionVQVAE.load_from_checkpoint(vqvae_path)
    
    def forward(self, x):
        if self.use_centroids:
            action_idcs = torch.argmin((self.centroids[None,:,:] - x[:,None,:]).pow(2).sum(dim=2), dim=1)
            print(f"{action_idcs.shape = }")
            return action_idcs
        else:
            return self.model.encode_only(x)[1] # return only ind

class VecObsDiscretizer(nn.Module):
    def __init__(self, vqvae_path=None):
        super().__init__()
        
        # set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        # set model
        if vqvae_path is not None:
            print('Using vqvae for actions..')
            self.use_vqvae = True
            self.model = VecObsVQVAE.load_from_checkpoint(vqvae_path)
        else:
            print('Not using a vecobs discretizer..')
            self.use_vqvae = False
            
    def forward(self, x):
        if self.use_vqvae:
            return self.model.encode_only(x)[0] # return only z_q
        else:
            return x

class FeatureExtractor(nn.Module):
    def __init__(self, model_path, model_class):
        super().__init__()
        
        # set up first model
        self.first_model = model_class.load_from_checkpoint(model_path)

        if isinstance(self.first_model, VQVAE):
            self.need_conv = True
            self.second_model = nn.Sequential(
                nn.Conv2d(in_channels=self.first_model.hparams.embedding_dim, out_channels=256, kernel_size=3, padding=1, stride=2), # 16 -> 8
                nn.GELU(),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2), # 8 -> 4
                nn.GELU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=2), # 4 -> 2
                nn.GELU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=2), # 2 -> 1
                nn.GELU()
            )
        else:
            self.need_conv = False
            self.second_model = nn.Sequential(
                nn.Linear(self.model.hparams.encoder_kwargs['latent_dim'], 256),
                nn.GELU(),
                nn.Linear(256, 512),
                nn.GELU(),
                nn.Linear(512, 512),
                nn.GELU(),
                nn.Linear(512, 512),
                nn.GELU()
            )
    def forward(self, x):
        print(f'{x.shape = }')
        if self.need_conv:
            latent = self.first_model.encode_only(x)[0]
            print(f'{latent.shape = }')
            latent = self.second_model(latent)
            print(f'{latent.shape = }')
            latent = einops.rearrange(latent, 'b c h w -> b (c h w)')
            print(f'{latent.shape = }')
        else:
            latent = self.first_model.encode_only(x)[2]
            print(f'{latent.shape = }')
            latent = self.second_model(latent)
            print(f'{latent.shape = }')
        return latent
    
    @property
    def trainable_params(self):
        '''
        Only the parameters of the second model should be trained.
        '''
        return list(self.second_model.parameters())
    
    
class BCModel(pl.LightningModule):
    def __init__(
        self, 
        feature_extractor_path, 
        feature_extractor_class, 
        lr,
        action_centroids_path=None, 
        action_vqvae_path=None, 
        vecobs_vqvae_path=None,
        hidden_dim=1024
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.feature_extractor = FeatureExtractor(feature_extractor_path, feature_extractor_class)
        self.action_discretizer = ActionDiscretizer(action_centroids_path, action_vqvae_path)
        self.vecobs_discretizer = VecObsDiscretizer(vecobs_vqvae_path)
        
        # compute action dim
        dummy_action = torch.zeros(1,64).to(self.action_discretizer.device)
        discretized_dummy_action = self.action_discretizer(dummy_action)
        self.action_dim = discretized_dummy_action.shape[1]
        
        # compute feature dim
        dummy_povobs = torch.zeros(1,3,64,64).to(self.feature_extractor.device)
        dummy_features = self.feature_extractor(dummy_povobs)
        assert len(dummy_features.shape) == 2, f"Expected len(dummy_features.shape) = 2, but got {len(dummy_features.shape) = }"
        self.feature_dim = dummy_features.shape[1]
        
        # compute vecobs dim
        dummy_vecobs = torch.zeros(1,64).to(self.vecobs_discretizer.device)
        discretized_dummy_vecobs = self.vecobs_discretizer(dummy_vecobs)
        self.vecobs_dim = discretized_dummy_vecobs.shape[1]
        
        print(f'\n{self.action_dim = }')
        print(f'{self.feature_dim = }')
        print(f'{self.vecobs_dim = }\n')
        
        # set up action predictor
        self.input_dim = self.feature_dim + self.vecobs_dim
        self.output_dim = self.action_dim
        self.action_predictor = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.output_dim)
        )
        
        # set up loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, pov_obs, vec_obs):
        
        # extract features and discretize
        features = self.feature_extractor(pov_obs)
        discr_vecobs = self.vecobs_discretizer(vec_obs)
        
        # cat tensors before passing through action predictor
        predictions = self.action_predictor(torch.cat([features, discr_vecobs, discr_action], dim=1))
        
    
    def training_step(self, batch, batch_idx):
        # unpack batch and prepare for forward pass
        obs, actions, *_ = batch
        vec_obs = obs['vector'].float()[0]
        pov_obs = obs['pov'].float() / 255
        actions = actions['vector'].float()
        assert [*pov_obs.shape[1:]] == [3, 64, 64], f"pov_obs shape should end with [3, 64, 64] but is {pov_obs.shape}"
        
        # predict actions
        predicted_actions = self.forward(pov_obs, vec_obs)
        
        # discretize target actions
        targets = self.action_discretizer(actions)
        
        # compute loss
        loss = self.loss_fn(predicted_actions, targets)
        
        # log
        self.log('Training/loss', loss, on_step=True)
        
        return loss
    
    def configure_optimizers(self):
        params = list(self.action_predictor.parameters()) + self.feature_extractor.trainable_params
        self.optimizer = torch.optim.AdamW(params, lr=self.hparams.lr)
        return self.optimizer