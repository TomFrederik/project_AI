import torch
import torch.nn as nn
import pytorch_lightning as pl
import einops

from scipy.cluster.vq import kmeans2


class StateVQVAE(pl.LightningModule):
    def __init__(self, optim_kwargs, num_input_channels):
        super().__init__()
        self.save_hyperparameters()
        
        self.cnn_encoder = CNNEncoder(num_input_channels)
        self.cnn_decoder = CNNDecoder(num_input_channels)
        self.lstm_encoder = LSTMEncoder(input_size=2048 + 64, hidden_size=2048)
        self.lstm_decoder = LSTMDecoder(input_size=2048 + 2048 + 64, hidden_size=2048)
        self.linear = nn.Linear(2048, 2048)
        self.quantizer = StateQuantizer(codebook_size=512, embedding_dim=64, latent_size=32) #TODO
        self.model_list = [self.cnn_encoder, self.cnn_decoder, self.lstm_encoder, self.lstm_decoder, self.linear, self.quantizer]
        self.loss_fn = nn.MSELoss()
        
    def forward(self, pov_obs, vec_obs, actions):
        B, T, C, H, W = pov_obs.shape()
        print('B, T, C, H, W = ', B, T, C, H, W)
        
        # encode all images
        encoded_images = einops.rearrange(self.cnn_encoder(einops.rearrange(pov_obs, 'b t c h w -> (b t) c h w')), '(b t) c h w -> b t c h w')
        print(f'{encoded_images.shape = }')
        
        # create lstm input
        h_0 = torch.cat([encoded_images[0], vec_obs[0]], dim=1) # (B D)
        enc_lstm_input = torch.cat([encoded_images[1:], vec_obs[1:], actions], dim=1) # (B D+A)
        print(f'{enc_lstm_input.shape = }')
        
        # encode with lstm
        enc_hidden_state_seq = self.lstm_encoder(enc_lstm_input, h_0)
        print(f'{enc_hidden_state_seq.shape = }')
        
        # quantize
        quantizer_input = einops.rearrange(enc_hidden_state_seq, 'b t d -> (b t) d')
        quantizer_input = einops.rearrange(quantizer_input, 'bt (d1 d2) -> bt d1 d2', d1=32, d2=64)
        discrete_embeddings, latent_loss = self.quantizer(quantizer_input)
        discrete_embeddings = einops.rearrange(discrete_embeddings, '(b t) d1 d2 -> b t (d1 d2)')
        print(f'{discrete_embeddings.shape = }')
        
        # prepare decoder input
        dec_lstm_input = torch.cat([discrete_embeddings, encoded_images[:-1], actions], dim=1)
        print(f'{dec_lstm_input.shape = }')
        
        # decode with lstm
        dec_hidden_state_seq = self.lstm_decoder(dec_lstm_input)
        print(f'{dec_hidden_state_seq.shape = }')
        
        # apply linear and decode with cnn decoder
        predictions = einops.rearrange(self.cnn_decoder(self.linear(einops.rearrange(dec_hidden_state_seq, 'b t d -> (b t) d'))), '(b t) d -> b t d')
        print(f'{predictions.shape = }')

        return predictions, latent_loss
        
        
    def training_step(self, batch, batch_idx):
        # unpack batch
        pov_obs, vec_obs, actions = batch
        
        # make predictions
        predictions, latent_loss = self(pov_obs, vec_obs, actions)
        
        # compute loss
        reconstruction_loss = self.loss_fn(predictions, pov_obs)
        loss = reconstruction_loss + latent_loss
        
        # logging
        self.log('Training/loss', loss, on_step=True)
        self.log('Training/reconstruction_loss', reconstruction_loss, on_step=True)
        self.log('Training/latent_loss', latent_loss, on_step=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # unpack batch
        pov_obs, vec_obs, actions = batch
        
        # make predictions
        predictions, latent_loss = self(pov_obs, vec_obs, actions)
        
        # compute loss
        reconstruction_loss = self.loss_fn(predictions, pov_obs)
        loss = reconstruction_loss + latent_loss
        
        # logging
        self.log('Validation/loss', loss, on_epoch=True)
        self.log('Validation/reconstruction_loss', reconstruction_loss, on_epoch=True)
        self.log('Validation/latent_loss', latent_loss, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        torch.optim.AdamW([list(m.parameters()) for m in self.model_list], **self.hparams.optim_kwargs)
    

class CNNEncoder(nn.Module):
    def __init__(self, num_input_channels):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=num_input_channels, out_channels=256, kernel_size=3, padding=1, stride=1), # input shape is (16,16)
            nn.AdaptiveAvgPool2d(output_size=(8,8)),
            nn.GELU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.AdaptiveAvgPool2d(output_size=(4,4)),
            nn.GELU(),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=1),
            nn.AdaptiveAvgPool2d(output_size=(2,2)),
            nn.GELU(),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1, stride=1),
        )
    def forward(self, x):
        return self.conv_net(x)

class CNNDecoder(nn.Module):
    def __init__(self, num_output_channels):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.UpsamplingNearest2d((2,2)),
            nn.Conv2d(2048, 1024, 3, 1, 1),
            nn.GELU(),
            nn.UpsamplingNearest2d((4,4)),
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.GELU(),
            nn.UpsamplingNearest2d((8,8)),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.GELU(),
            nn.UpsamplingNearest2d((16,16)),
            nn.Conv2d(256, num_output_channels, 3, 1, 1)
        )
    def forward(self, x):
        return self.conv_net(x)

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(
            batch_first=True,
            input_size=input_size,
            hidden_size=hidden_size
        )
    def forward(self, x, h_0):
        c_0 = torch.zeros_like(h_0)
        output, _ = self.lstm(x, (h_0, c_0))
        return output

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(
            batch_first=True,
            input_size=input_size,
            hidden_size=hidden_size
        )
    def forward(self, x):
        output, _ = self.lstm(x)
        return output
    
class StateQuantizer(nn.Module):
    def __init__(self, codebook_size, latent_size, embedding_dim):
        super().__init__()
        
        # save params
        self.codebook_size = codebook_size
        self.latent_size = latent_size
        self.embedding_dim = embedding_dim
        
        # some hparams
        self.kld_scale = 10.0
        self.commitment_cost = 0.25
        
        # create a separate embedding for every position
        # but in a batched way
        # I think this is equivalent, modulo the initialization via kmeans which will work a bit different in this batched way
        self.embedding = nn.Embedding(codebook_size, embedding_dim*latent_size)
        
        self.register_buffer('data_initialized', torch.zeros(1))
        
    def forward(self, z):
        B, N, D = z.shape()

        # this is a bit unnecessary, could just skip the rearrange in the statevqvae
        # basically just included as legacy, to be more recognizable coming from the original paper
        z = einops.rearrange(z, 'b n d -> b (n d)')        
        
        # init embedding
        if self.training and self.data_initialized.item() == 0:
            print('Running kmeans to init embeddings')
            kd = kmeans2(z.data.cpu().numpy(), self.codebook_size, minit='points')
            self.embedding.data.copy_(torch.from_numpy(kd[0]))
            self.data_initialized.fill_(1)
        
        # compute closest embedding vector
        dist = self.get_dist(z)
        ind = torch.argmin(dist, dim=1)
        z_q = self.embedding(ind)
        
        # compute losses
        latent_loss = self.commitment_cost * (z_q.detach() - z).pow(2).mean() + (z_q - z.detach()).pow(2).mean()
        latent_loss *= self.kld_scale
        
        # straight through gradient
        z_q = z + (z_q - z).detach()
        
        return z_q, latent_loss
                    
    def get_dist(self, z):
        '''
        returns distance from z to each embedding vec
        '''
        dist = (
            z.pow(2).sum(1, keepdim=True)
            - 2 * z @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(1, keepdim=True).t()
        )
        return dist