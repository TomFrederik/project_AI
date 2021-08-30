import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
import einops
import json

from vqvae import VQVAE

from scipy.cluster.vq import kmeans2


class StateVQVAE(pl.LightningModule):
    def __init__(self, optim_kwargs, framevqvae):
        super().__init__()
        self.save_hyperparameters()
        
        # load vqvae model
        self.vqvae = VQVAE.load_from_checkpoint(framevqvae)
        self.vqvae.eval()
        
        self.encoding_batch_size = 300
        self.max_seq_len = 500

        num_input_channels = self.vqvae.hparams.args.embedding_dim        
        
        self.cnn_encoder = CNNEncoder(num_input_channels)
        self.cnn_decoder = CNNDecoder(num_input_channels)
        self.lstm_encoder = LSTMEncoder(input_size=2048 + 64, hidden_size=2048)
        self.lstm_decoder = LSTMDecoder(input_size=2048 + 2048, hidden_size=2048)
        self.quantizer = StateQuantizer(codebook_size=512, embedding_dim=128, latent_size=16) #TODO
        self.model_list = [self.cnn_encoder, self.cnn_decoder, self.lstm_encoder, self.lstm_decoder, self.quantizer]
    
    def loss_fn(self, predictions, targets):
        return ((predictions - targets) ** 2).mean() / (2 * self.data_var)
    
    def _apply_frame_encoding(self, pov_obs):
        # encode pov obs
        t = pov_obs.shape[1]
        pov_obs = einops.rearrange(pov_obs, 'b t c h w -> (b t) c h w')
        enc_pov_obs = []
        for i in range((len(pov_obs)-1) // self.encoding_batch_size + 1):
            enc_pov_obs.append(self.vqvae.encode_only(pov_obs[i * self.encoding_batch_size:(i+1) * self.encoding_batch_size])[0])
        pov_obs = torch.cat(enc_pov_obs, dim=0)
        return einops.rearrange(pov_obs, '(b t) c h w -> b t c h w', t=t)

    def _predict_subsequence(self, encoded_images, actions, enc_lstm_input, enc_first_hidden, dec_first_hidden=None, enc_first_cell=None, dec_first_cell=None):
        T = enc_lstm_input.shape[1]
        # encode with lstm
        enc_hidden_state_seq, (enc_last_hidden, enc_last_cell) = self.lstm_encoder(enc_lstm_input, enc_first_hidden, enc_first_cell)
        #print(f'{enc_hidden_state_seq.shape = }')
        
        # quantize
        '''
        quantizer_input = einops.rearrange(enc_hidden_state_seq, 'b t d -> (b t) d')
        quantizer_input = einops.rearrange(quantizer_input, 'bt (d1 d2) -> bt d1 d2', d1=32, d2=64)
        discrete_embeddings, enc_latent_loss = self.quantizer(quantizer_input)
        discrete_embeddings = einops.rearrange(discrete_embeddings, '(b t) d -> b t d', t=T)
        #print(f'{discrete_embeddings.shape = }')
        #print(f'{latent_loss.shape = }')
        '''
        enc_latent_loss = 0
        
        # prepare decoder input
        dec_lstm_input = torch.cat([enc_hidden_state_seq, encoded_images[:,:-1], actions[:,:-1]], dim=2)
        #print(f'{dec_lstm_input.shape = }')
        
        # decode with lstm
        dec_hidden_state_seq, (dec_last_hidden, dec_last_cell) = self.lstm_decoder(dec_lstm_input, dec_first_hidden, dec_first_cell)
        quantizer_input = einops.rearrange(dec_hidden_state_seq, 'b t d -> (b t) d')
        quantizer_input = einops.rearrange(quantizer_input, 'bt (d1 d2) -> bt d1 d2', d1=16, d2=128)
        discrete_embeddings, dec_latent_loss = self.quantizer(quantizer_input)
        #print(f'{dec_hidden_state_seq.shape = }')
        
        latent_loss = enc_latent_loss + dec_latent_loss
        
        # decode with cnn decoder
        predictions = einops.rearrange(
            self.cnn_decoder(
                einops.rearrange(discrete_embeddings, 'bt d -> bt d 1 1')
            ),
            '(b t) c h w -> b t c h w', 
            t=T
        )
        
        return predictions, latent_loss, (enc_last_hidden, enc_last_cell), (dec_last_hidden, dec_last_cell)

    def forward(self, pov_obs, vec_obs, actions, max_seq_len=None):
        if max_seq_len is None:
            max_seq_len = self.max_seq_len

        # apply frame vqvae
        pov_obs = self._apply_frame_encoding(pov_obs)
        
        # normalize and center
        pov_obs /= self.data_max
        pov_obs -= self.data_mean
        
        B, T, C, H, W = pov_obs.shape
        #print('B, T, C, H, W = ', B, T, C, H, W)
        
        # encode all images
        encoded_images = einops.rearrange(self.cnn_encoder(einops.rearrange(pov_obs, 'b t c h w -> (b t) c h w')), '(b t) c h w -> b t (c h w)', t=T)
        #print(f'{encoded_images.shape = }')
        
        # create lstm input
        h_0 = torch.cat([encoded_images[:,0], vec_obs[:,0]], dim=1)[:,None] # (B 1 D)
        if max_seq_len < pov_obs.shape[1]:
            # split into max_seq_len sized subsequences
            enc_lstm_input = torch.cat([encoded_images[:,1:max_seq_len+1], vec_obs[:,1:max_seq_len+1], actions[:,:max_seq_len]], dim=2) # (B T D+A)
            #print(f'{enc_lstm_input.shape = }')
            
            predictions, latent_loss, (enc_last_hidden, enc_last_cell), (dec_last_hidden, dec_last_cell) = self._predict_subsequence(encoded_images[:,:max_seq_len+1], actions[:,:max_seq_len+1], enc_lstm_input, h_0)
            #print(f'{predictions.shape=}')
            
            all_predictions = [predictions]
            all_latent_losses = [latent_loss]
            remaining_idx = (pov_obs.shape[1]-2) // max_seq_len * max_seq_len
            for i in range((pov_obs.shape[1]-2) // max_seq_len - 1):
                #print(f'{i = }')
                enc_lstm_input = torch.cat([encoded_images[:,1+(i+1)*max_seq_len:(i+2)*max_seq_len+1], vec_obs[:,1+(i+1)*max_seq_len:(i+2)*max_seq_len+1], actions[:,(i+1)*max_seq_len:(i+2)*max_seq_len]], dim=2) # (B T D+A)
                predictions, latent_loss, (enc_last_hidden, enc_last_cell), (dec_last_hidden, dec_last_cell) = self._predict_subsequence(encoded_images[:,max_seq_len*(i+1):max_seq_len*(i+2)+1], actions[:,max_seq_len*(i+1):max_seq_len*(i+2)+1], enc_lstm_input, enc_last_hidden.detach(), enc_last_cell.detach(), dec_last_hidden.detach(), dec_last_cell.detach())
                all_predictions.append(predictions)
                all_latent_losses.append(latent_loss)

            # predict remaining
            enc_lstm_input = torch.cat([encoded_images[:,1+remaining_idx:], vec_obs[:,1+remaining_idx:], actions[:,remaining_idx:-1]], dim=2) # (B T D+A)
            predictions, latent_loss, (enc_last_hidden, enc_last_cell), (dec_last_hidden, dec_last_cell) = self._predict_subsequence(encoded_images[:,remaining_idx:], actions[:,remaining_idx:], enc_lstm_input, enc_last_hidden.detach(), enc_last_cell.detach(), dec_last_hidden.detach(), dec_last_cell.detach())
            all_predictions.append(predictions)
            all_latent_losses.append(latent_loss)
            
            predictions = torch.cat(all_predictions, dim=1)
            #print(f'{predictions.shape=}')
            
            latent_loss = torch.sum(torch.tensor(all_latent_losses, device=self.device))
        else:
            enc_lstm_input = torch.cat([encoded_images[:,1:], vec_obs[:,1:], actions[:,:-1]], dim=2) # (B T D+A)
            predictions, latent_loss, *_ = self._predict_subsequence(enc_lstm_input, h_0)
        
        # skip connection
        #predictions = predictions + pov_obs[:,:-1]
        
        # uncenter and unnormalize
        predictions += self.data_mean
        predictions *= self.data_max

        return predictions, pov_obs[:,1:], latent_loss
        
        
    def training_step(self, batch, batch_idx):
        # unpack batch
        pov_obs, vec_obs, actions = batch
        
        # make predictions
        predictions, targets, latent_loss = self(pov_obs, vec_obs, actions)
        
        # compute loss
        reconstruction_loss = self.loss_fn(predictions, targets)
        loss = reconstruction_loss + latent_loss
        #print(f'attempting backward with sequence of length {pov_obs.shape[1]}...')

        # logging
        self.log('Training/loss', loss, on_step=True)
        self.log('Training/reconstruction_loss', reconstruction_loss, on_step=True)
        self.log('Training/latent_loss', latent_loss, on_step=True)
        
        return loss
    
    def configure_optimizers(self):
        params = []
        for m in self.model_list:
            params += list(m.parameters())
        optimizer = torch.optim.AdamW(params, **self.hparams.optim_kwargs)
        return optimizer
    
    def find_data_mean_var(self, dataloader=None, load_from=None, save_to="./stats.json"):
        if load_from is None:
            if dataloader is None:
                raise ValueError('Need to specify either dataloader or file to load stats from')
            encoded_povs = []
            for batch_id, batch in enumerate(dataloader):
                print(f'{batch_id = }')
                pov_obs = batch[0]
                encoded_povs.append(self._apply_frame_encoding(pov_obs.to(self.device)).to('cpu')[0])
                print(f'{encoded_povs[-1].shape = }')
            encoded_povs = einops.rearrange(torch.cat(encoded_povs, dim=0), 'b c h w -> (b c h w)')
            self.data_mean = encoded_povs.mean().item()
            self.data_var = encoded_povs.var().item()
            self.data_max = encoded_povs.max().item()

            with open(save_to, 'w') as f:
                json.dumps(dict(mean=self.data_mean, var=self.data_var, max=self.data_max))
        else:
            with open(load_from) as f:
                stats = json.load(f)
            self.data_mean = stats['mean']
            self.data_var = stats['var']
            self.data_max = stats['max']

        print(f'\n{self.data_mean = }')
        print(f'{self.data_var = }')
        print(f'{self.data_max = }\n')
        
class CNNEncoder(nn.Module):
    def __init__(self, num_input_channels):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=num_input_channels, out_channels=256, kernel_size=3, padding=1, stride=2), # input shape is (16,16)
            #nn.AdaptiveAvgPool2d(output_size=(8,8)),
            nn.GELU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2),
            #nn.AdaptiveAvgPool2d(output_size=(4,4)),
            nn.GELU(),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=2),
            #nn.AdaptiveAvgPool2d(output_size=(2,2)),
            nn.GELU(),
            nn.Conv2d(in_channels=1024, out_channels=1984, kernel_size=3, padding=1, stride=2),
            #nn.AdaptiveAvgPool2d(output_size=(1,1))
        )
    def forward(self, x):
        return self.conv_net(x)

class CNNDecoder(nn.Module):
    def __init__(self, num_output_channels):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=3, padding=1, stride=2, output_padding=1), # 1 -> 2
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1, stride=2, output_padding=1), # 2 -> 4
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=2, output_padding=1), # 4 -> 8
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=num_output_channels, kernel_size=3, padding=1, stride=2, output_padding=1) # 8 -> 16
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
    def forward(self, x, h_0, c_0=None):
        if c_0 is None:
            c_0 = torch.zeros_like(h_0)
        output, (last_hidden, last_cell) = self.lstm(x, (h_0, c_0))
        return output, (last_hidden, last_cell)

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(
            batch_first=True,
            input_size=input_size,
            hidden_size=hidden_size
        )
    def forward(self, x, h_0=None, c_0=None):
        if h_0 is None and c_0 is None:
            output, (last_hidden, last_cell) = self.lstm(x)
        else:
            output, (last_hidden, last_cell) = self.lstm(x, (h_0, c_0))
        return output, (last_hidden, last_cell)
    
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
        #self.embedding = nn.Embedding(codebook_size, embedding_dim*latent_size)
        # nevermind.. 
        self.embedding = nn.Embedding(codebook_size, embedding_dim)


        self.register_buffer('data_initialized', torch.zeros(1))
        
    def forward(self, z):
        B, N, D = z.shape

        # this is a bit unnecessary, could just skip the rearrange in the statevqvae
        # basically just included as legacy, to be more recognizable coming from the original paper
        z_e = einops.rearrange(z, 'b n d -> (b n) d')        
        
        # init embedding
        if self.training and self.data_initialized.item() == 0:
            print('Running kmeans to init embeddings')
            kd = kmeans2(z_e.data.cpu().numpy(), self.codebook_size, minit='points')
            self.embedding.weight.data.copy_(torch.from_numpy(kd[0]))
            self.data_initialized.fill_(1)
        
        # compute closest embedding vector
        dist = self.get_dist(z_e)
        ind = torch.argmin(dist, dim=1)
        z_q = self.embed_code(ind)
        
        # compute losses
        latent_loss = self.commitment_cost * (z_q.detach() - z_e).pow(2).mean() + (z_q - z_e.detach()).pow(2).mean()
        latent_loss *= self.kld_scale
        
        # straight through gradient
        z_q = z_e + (z_q - z_e).detach()
        z_q = einops.rearrange(z_q, '(b n) d -> b (n d)', n=N)
        return z_q, latent_loss

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embedding.weight)

    def get_dist(self, z):
        '''
        returns distance from z to each embedding vec
        '''
        dist = (
            z.pow(2).sum(1, keepdim=True)
            - 2 * z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1, keepdim=True).t()
        )
        return dist