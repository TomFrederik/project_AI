import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import einsum
import numpy as np
import pytorch_lightning as pl
import einops
import json

from vqvae import VQVAE
from vqvae_model.deepmind_enc_dec import DeepMindDecoder, DeepMindEncoder

from scipy.cluster.vq import kmeans2


class StateVQVAE(pl.LightningModule):
    def __init__(
        self, 
        optim_kwargs, 
        framevqvae, 
        embedding_dim, 
        codebook_size, 
        gumbel, 
        lstm_enc_hidden_size=1024, 
        lstm_enc_input_size=1024, 
        lstm_dec_hidden_size=1024, 
        latent_size=32,
        tau=1
    ):
        super().__init__()
        self.save_hyperparameters()
        

        # load vqvae model
        self.vqvae = VQVAE.load_from_checkpoint(framevqvae)
        self.vqvae.eval()
        
        self.encoding_batch_size = 300
        self.max_seq_len = 100

        num_input_channels = self.vqvae.hparams.args.embedding_dim        
        print(f'{num_input_channels = }')
        #self.cnn_encoder = CNNEncoder(num_input_channels)
        #self.cnn_decoder = CNNDecoder(num_input_channels)
        self.cnn_encoder = DeepMindEncoder(num_input_channels, n_hid=64)
        dummy = torch.zeros((2,num_input_channels,16,16), device=self.device)
        self.cnn_encoded = self.cnn_encoder(dummy)
        cnn_encoded_flat_dim = self.cnn_encoded.shape[1]*self.cnn_encoded.shape[2]*self.cnn_encoded.shape[3]

        self.first_projection = nn.Linear(cnn_encoded_flat_dim, lstm_enc_input_size-128)
        print(f'First: {cnn_encoded_flat_dim} -> {lstm_enc_input_size-128}')
        self.lstm_starting_state_projection = nn.Linear(lstm_enc_input_size-64, lstm_enc_hidden_size)
        print(f'lstm starting proj: {lstm_enc_input_size-128} -> {lstm_enc_hidden_size}')

        self.lstm_encoder = LSTMEncoder(input_size=self.hparams.lstm_enc_input_size, hidden_size=self.hparams.lstm_enc_hidden_size)
        
        if gumbel:
            self.quantizer = StateGumbelQuantizer(codebook_size=codebook_size, embedding_dim=embedding_dim)#, straight_through=True)
            self.quantizer_dim = codebook_size
        else:
            self.quantizer = StateQuantizer(codebook_size=codebook_size, embedding_dim=embedding_dim)
            self.quantizer_dim = embedding_dim
        
        self.second_projection = nn.Linear(lstm_enc_hidden_size, self.quantizer_dim*latent_size)
        print(f'Second: {lstm_enc_hidden_size} -> {self.quantizer_dim*latent_size}')
        
        self.lstm_decoder = LSTMDecoder(input_size=embedding_dim*latent_size + cnn_encoded_flat_dim + 128, hidden_size=lstm_dec_hidden_size)

        self.third_projection = nn.Linear(lstm_dec_hidden_size, cnn_encoded_flat_dim)
        print(f'Third: {lstm_dec_hidden_size} -> {cnn_encoded_flat_dim}')

        self.cnn_decoder = DeepMindDecoder(n_init=self.cnn_encoded.shape[1], n_hid=64, output_channels=self.vqvae.hparams.args.num_embeddings)#num_input_channels)
        
        self.model_list = [
            self.cnn_encoder, 
            self.cnn_decoder, 
            self.first_projection, 
            self.lstm_starting_state_projection, 
            self.second_projection, 
            self.third_projection, 
            self.lstm_encoder, 
            self.lstm_decoder, 
            self.quantizer
        ]

    def loss_fn(self, predictions, targets):
        return nn.CrossEntropyLoss()(predictions, targets)
        #return ((predictions - targets) ** 2).mean() / (2 * self.data_var)
    
    def _apply_frame_encoding(self, pov_obs):
        # encode pov obs
        t = pov_obs.shape[1]
        pov_obs = einops.rearrange(pov_obs, 'b t c h w -> (b t) c h w')
        enc_pov_obs = []
        frame_quantization_idcs = []
        all_neg_dist = []
        
        for i in range((len(pov_obs)-1) // self.encoding_batch_size + 1):
            z_q, ind, neg_dist = self.vqvae.encode_only(pov_obs[i * self.encoding_batch_size:(i+1) * self.encoding_batch_size])
            enc_pov_obs.append(z_q)
            frame_quantization_idcs.append(ind)
            all_neg_dist.append(neg_dist)
        
        pov_obs = torch.cat(enc_pov_obs, dim=0)
        frame_quantization_idcs = torch.cat(frame_quantization_idcs, dim=0)
        all_neg_dist = einops.rearrange(torch.cat(all_neg_dist, dim=0), '(b t) c h w -> b t c h w', t=t)[:,:-1]

        return einops.rearrange(pov_obs, '(b t) c h w -> b t c h w', t=t), frame_quantization_idcs, all_neg_dist

    def _predict_subsequence(self, enc_lstm_input, dec_lstm_input, enc_first_hidden, enc_first_cell=None, dec_first_hidden=None, dec_first_cell=None):
        # encode with lstm
        T = enc_lstm_input.shape[1]
        enc_hidden_state_seq, (enc_last_hidden, enc_last_cell) = self.lstm_encoder(enc_lstm_input, enc_first_hidden, enc_first_cell)

        #k = einops.rearrange(enc_hidden_state_seq, 'b t d -> (b t) d').shape
        #print(f'pre second proj: {k}')
        quantizer_input = self.second_projection(einops.rearrange(enc_hidden_state_seq, 'b t d -> (b t) d'))
        #print(f'post second proj: {quantizer_input.shape}')
        quantizer_input = einops.rearrange(quantizer_input, 'bt (latent_size quantizer_dim) -> bt latent_size quantizer_dim', latent_size=self.hparams.latent_size, quantizer_dim=self.quantizer_dim)
        discrete_embeddings, latent_loss = self.quantizer(quantizer_input)
        # prepare decoder input
        dec_lstm_input = torch.cat(
            [
                einops.rearrange(discrete_embeddings, '(b t) latent_size embed_dim -> b t (latent_size embed_dim)', t=T), 
                dec_lstm_input
            ], 
            dim=2
        )
        
        # decode with lstm
        dec_hidden_state_seq, (dec_last_hidden, dec_last_cell) = self.lstm_decoder(dec_lstm_input, dec_first_hidden, dec_first_cell)
        #quantizer_input = einops.rearrange(dec_hidden_state_seq, 'b t (c h w) -> (b t) c h w', h=self.enc_img_shape[-1], w=self.enc_img_shape[-1], c=self.enc_img_shape[0])
        #quantizer_input = self.second_projection(quantizer_input)
        #discrete_embeddings, latent_loss = quantizer_input, 0
        #discrete_embeddings, latent_loss = self.quantizer(quantizer_input)
        #k = einops.rearrange(dec_hidden_state_seq, 'b t d -> (b t) d').shape
        #print(f'pre third proj: {k}')
        cnn_decoder_input = self.third_projection(einops.rearrange(dec_hidden_state_seq, 'b t d -> (b t) d'))
        #print(f'post third proj: {cnn_decoder_input.shape}')
        cnn_decoder_input = einops.rearrange(cnn_decoder_input, 'bt (c h w) -> bt c h w', c=self.cnn_encoded.shape[1], h=self.cnn_encoded.shape[2], w=self.cnn_encoded.shape[3])
        #raise ValueError
        # decode with cnn decoder
        predictions = einops.rearrange(
            self.cnn_decoder(cnn_decoder_input),
            '(b t) c h w -> b t c h w', 
            t=T
        )
        
        return predictions, latent_loss, (enc_last_hidden, enc_last_cell), (dec_last_hidden, dec_last_cell)

    def forward(self, pov_obs, vec_obs, actions, max_seq_len=None):
        if max_seq_len is None:
            max_seq_len = self.max_seq_len
        # apply frame vqvae
        pov_obs, frame_quantization_idcs, neg_dist = self._apply_frame_encoding(pov_obs)
        assert pov_obs.shape[-2:] == torch.Size([16,16]), f"Expected pov_obs.shape to end with (16,16) but got {pov_obs.shape}"
        # pov_obs.shape = [B, T, 32,16,16]
        # normalize and center
        #pov_obs -= self.data_mean
        #pov_obs /= self.data_var ** 0.5
        
        B, T, C, H, W = pov_obs.shape
        #print('B, T, C, H, W = ', B, T, C, H, W)
        
        # encode all images
        encoded_images = einops.rearrange(self.cnn_encoder(einops.rearrange(pov_obs, 'b t c h w -> (b t) c h w')), 'bt c h w -> bt (c h w)')
        # encoded_imgs.shape = [B, T, 1024]
        # vec_obs.shape = [B, T, 64] # <- float vector
        # actions.shape = [B, T, 64]
        # dec_lstm_input = [B, T, 1152]
        dec_lstm_input = torch.cat([einops.rearrange(encoded_images, '(b t) d -> b t d', t=T)[:,:-1], vec_obs[:,:-1], actions[:,:-1]], dim=2)
        encoded_images = self.first_projection(encoded_images)
        encoded_images = einops.rearrange(encoded_images, '(b t) d -> b t d', b=B, t=T)
        #print(f'{encoded_images.shape = }')

        # create lstm input
        #print(f'pre lstm starting proj: {torch.cat([encoded_images[:,0], vec_obs[:,0]], dim=1).shape}')
        enc_first_hidden = self.lstm_starting_state_projection(torch.cat([encoded_images[:,0], vec_obs[:,0]], dim=1))[:,None]
        #print(f'post lstm starting proj: {enc_first_hidden.shape}')
        enc_lstm_input = torch.cat([encoded_images[:,1:], vec_obs[:,1:], actions[:,:-1]], dim=2)
        if max_seq_len < pov_obs.shape[1]-1:
            # split into max_seq_len sized subsequences
            
            # make predictions for first subsequence
            # save last states of lstms for subsequent subsequences
            predictions, latent_loss, (enc_last_hidden, enc_last_cell), (dec_last_hidden, dec_last_cell) = self._predict_subsequence( 
                enc_lstm_input[:,:max_seq_len],
                dec_lstm_input[:,:max_seq_len], 
                enc_first_hidden
            )
            #print(f'\n{predictions=}\n')
            #print(f'\n{encoded_images[:,:max_seq_len]=}\n')
            #print(f'{predictions.shape=}')
            
            all_predictions = [predictions]
            all_latent_losses = [latent_loss]
            for i in range((pov_obs.shape[1]-2) // max_seq_len - 1):
                #print(f'{i = }')
                start_idx = (i+1)*max_seq_len
                end_idx = (i+2)*max_seq_len

                predictions, latent_loss, (enc_last_hidden, enc_last_cell), (dec_last_hidden, dec_last_cell) = self._predict_subsequence(
                    enc_lstm_input[:,start_idx:end_idx], 
                    dec_lstm_input[:,start_idx:end_idx],
                    enc_last_hidden.detach(), 
                    enc_last_cell.detach(), 
                    dec_last_hidden.detach(), 
                    dec_last_cell.detach()
                )
                
                all_predictions.append(predictions)
                all_latent_losses.append(latent_loss)

            # predict remaining frames
            remaining_idx = (pov_obs.shape[1]-2) // max_seq_len * max_seq_len
            predictions, latent_loss, (enc_last_hidden, enc_last_cell), (dec_last_hidden, dec_last_cell) = self._predict_subsequence(
                enc_lstm_input[:,remaining_idx:], 
                dec_lstm_input[:,remaining_idx:],
                enc_last_hidden.detach(), 
                enc_last_cell.detach(), 
                dec_last_hidden.detach(), 
                dec_last_cell.detach()
            )
            all_predictions.append(predictions)
            all_latent_losses.append(latent_loss)
            
            predictions = torch.cat(all_predictions, dim=1)
            #print(f'{predictions.shape=}')
            
            latent_loss = torch.sum(torch.tensor(all_latent_losses, device=self.device))
        else:
            raise NotImplementedError
            enc_lstm_input = torch.cat([encoded_images[:,1:], vec_obs[:,1:], actions[:,:-1]], dim=2) # (B T D+A)
            predictions, latent_loss, *_ = self._predict_subsequence(enc_lstm_input, h_0)
        
        # predictions.shape = [B, T-1, 32, 16, 16]
        # compute log_priors
        #TODO: Deal with memory leak
        log_prior = self._compute_log_prior(neg_dist)
        predictions += log_prior

        #return predictions, pov_obs[:,1:], latent_loss
        return predictions, frame_quantization_idcs[1:], latent_loss
        
    def _compute_log_prior(self, neg_dist, batch_size=5000):
        B, T, C, H, W = neg_dist.shape
        neg_dist = einops.rearrange(neg_dist, 'b t c h w -> (b t h w) c')
        log_prior = []
        for i in range(len(neg_dist) // batch_size + 1):
            log_prior.append(nn.functional.log_softmax(neg_dist[i*batch_size:(i+1)*batch_size] / self.hparams.tau, dim=1))
        log_prior = einops.rearrange(torch.cat(log_prior, dim=0), '(b t h w) c -> b t c h w', b=B, t=T, c=C, h=H, w=W)
        return log_prior
        
    def training_step(self, batch, batch_idx):
        # unpack batch
        pov_obs, vec_obs, actions = batch
        
        # make predictions
        predictions, targets, latent_loss = self(pov_obs, vec_obs, actions)
        
        # compute loss
        reconstruction_loss = self.loss_fn(einops.rearrange(predictions, 'b t c h w -> (b t) c h w'), targets)
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
            nn.Conv2d(in_channels=1024, out_channels=1920, kernel_size=3, padding=1, stride=2),
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
    def __init__(self, codebook_size, embedding_dim):
        super().__init__()
        
        # save params
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        
        # some hparams
        self.kld_scale = 10.0
        self.commitment_cost = 0.1
        
        # create a separate embedding for every position
        # but in a batched way
        # I think this is equivalent, modulo the initialization via kmeans which will work a bit different in this batched way
        #self.embedding = nn.Embedding(codebook_size, embedding_dim*latent_size)
        # nevermind.. 
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.register_buffer('data_initialized', torch.zeros(1))

        
    
    def init_embedding(self, vectors):
        print('Running kmeans to init embeddings')
        kd = kmeans2(vectors.data.cpu().numpy(), self.codebook_size, minit='points')
        #print(kd[0])
        self.embedding.weight.data.copy_(torch.from_numpy(kd[0]))
        self.data_initialized.fill_(1)

    def forward(self, z):
        B, N, D = z.shape

        # this is a bit unnecessary, could just skip the rearrange in the statevqvae
        # basically just included as legacy, to be more recognizable coming from the original paper
        z_e = einops.rearrange(z, 'b n d -> (b n) d')        
        # init quantizer embedding
        if self.data_initialized.item() == 0:
            self.init_embedding(z_e)
        # compute closest embedding vector
        dist = self.get_dist(z_e)
        ind = torch.argmin(dist, dim=1)
        #print(np.unique(ind.cpu().numpy()))
        z_q = self.embed_code(ind)
        
        # compute embedding loss
        #print(f'{self.embedding.weight.grad = }')
        #print(f'{(z_q-z_e).pow(2).mean() = }')
        #latent_loss = self.commitment_cost * (z_q - z_e).pow(2).mean()
        latent_loss = self.commitment_cost * (z_q.detach() - z_e).pow(2).mean() + (z_q - z_e.detach()).pow(2).mean()
        latent_loss *= self.kld_scale
        #print(latent_loss)
        #latent_loss.backward()
        #print(f'{self.embedding.weight.grad[self.embedding.weight.grad != 0] = }')
        #print(f'{z_q.grad = }')
        #print(f'{z_e.grad = }')
        #out = self.embedding(torch.arange(self.codebook_size, dtype=torch.long, device=self.embedding.weight.device))
        #loss = out.pow(2).mean()
        #latent_loss.backward(retain_graph=True)
        #print(f'{self.embedding.weight = }')
        
        #raise ValueError
        
        
        # straight through gradient
        z_q = z_e + (z_q - z_e).detach()
        z_q = einops.rearrange(z_q, '(b n) d -> b n d', n=N)
        #print(self.embedding.weight)
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

class StateGumbelQuantizer(nn.Module):
    """
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(self, codebook_size, embedding_dim, straight_through=False):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size

        self.straight_through = straight_through
        self.temperature = 1.0
        self.kld_scale = 10

        self.embedding = nn.Embedding(codebook_size, embedding_dim)

    def forward(self, z):

        # force hard = True when we are in eval mode, as we must quantize
        hard = self.straight_through if self.training else True
        soft_one_hot = F.gumbel_softmax(z, tau=self.temperature, dim=2, hard=hard)
        #print(f'{z.shape = }')
        #print(f'{soft_one_hot.shape = }')
        #print(f'{self.embedding.weight.shape = }')
        z_q = soft_one_hot @ self.embedding.weight
        #z_q = einsum('b lat codebook, codebook embed -> b lat embed', soft_one_hot, self.embedding.weight)
        #print(f'{z_q.shape = }')

        # + kl divergence to the prior loss
        qy = F.softmax(z, dim=2)
        latent_loss = self.kld_scale * torch.sum(qy * torch.log(qy * self.codebook_size + 1e-10), dim=2).mean()
        ind = soft_one_hot.argmax(dim=2)
        return z_q, latent_loss#, ind