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
from action_vqvae import ActionVQVAE
from vecobs_vqvae import VecObsVQVAE
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
        tau=1,
        action_quantizer=None,
        vecobs_quantizer=None,
        discard_priors=False,
        perplexity_freq=1, 
        log_perplexity=True,
        max_seq_len=100
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.encoding_batch_size = 100

        # load vqvae model
        self.vqvae = VQVAE.load_from_checkpoint(framevqvae)
        self.vqvae.eval()

        # load action quantizer
        if action_quantizer is None:
            self.action_quantizer = None
            self.action_dim = 64
        else:
            self.action_quantizer = ActionVQVAE.load_from_checkpoint(action_quantizer)
            self.action_quantizer.eval()
            self.action_dim = self.action_quantizer.quantizer.embedding_dim * self.action_quantizer.quantizer.latent_size

        # load vec obs quantizer
        if vecobs_quantizer is None:
            self.vecobs_quantizer = None
            self.vecobs_dim = 64
        else:
            self.vecobs_quantizer = VecObsVQVAE.load_from_checkpoint(vecobs_quantizer)
            self.vecobs_quantizer.eval()
            self.vecobs_dim = self.vecobs_quantizer.quantizer.embedding_dim * self.vecobs_quantizer.quantizer.latent_size

        print(f'\n{lstm_enc_input_size = }')
        print(f'{self.vecobs_dim = }')
        print(f'{self.action_dim = }\n')
        if lstm_enc_input_size <= self.vecobs_dim + self.action_dim:
            raise ValueError(f"lstm_enc_input_size <= self.vecobs_dim + self.action_dim: {lstm_enc_input_size} <= {self.vecobs_dim} + {self.action_dim}")
        elif lstm_enc_input_size <= + self.action_dim:
            raise ValueError(f"lstm_enc_input_size <= self.action_dim: {lstm_enc_input_size} <= {self.action_dim}")

        num_input_channels = self.vqvae.hparams.args.embedding_dim        
        print(f'{num_input_channels = }')
        self.cnn_encoder = DeepMindEncoder(num_input_channels, n_hid=64)
        dummy = torch.zeros((2,num_input_channels,16,16), device=self.device)
        self.cnn_encoded = self.cnn_encoder(dummy)
        cnn_encoded_flat_dim = self.cnn_encoded.shape[1]*self.cnn_encoded.shape[2]*self.cnn_encoded.shape[3]

        self.first_projection = nn.Linear(cnn_encoded_flat_dim, lstm_enc_input_size-self.vecobs_dim-self.action_dim)
        print(f'First: {cnn_encoded_flat_dim} -> {lstm_enc_input_size-self.vecobs_dim-self.action_dim}')
        self.lstm_starting_state_projection = nn.Linear(lstm_enc_input_size-self.action_dim, lstm_enc_hidden_size)
        print(f'lstm starting proj: {lstm_enc_input_size-self.vecobs_dim-self.action_dim} -> {lstm_enc_hidden_size}')

        self.lstm_encoder = LSTMEncoder(input_size=self.hparams.lstm_enc_input_size, hidden_size=self.hparams.lstm_enc_hidden_size)
        
        if gumbel:
            self.quantizer = StateGumbelQuantizer(codebook_size=codebook_size, embedding_dim=embedding_dim)#, straight_through=True)
            self.quantizer_dim = codebook_size
        else:
            self.quantizer = StateQuantizer(codebook_size=codebook_size, embedding_dim=embedding_dim)
            self.quantizer_dim = embedding_dim
        
        self.second_projection = nn.Linear(lstm_enc_hidden_size, self.quantizer_dim*latent_size)
        print(f'Second: {lstm_enc_hidden_size} -> {self.quantizer_dim*latent_size}')
        
        self.lstm_decoder = LSTMDecoder(input_size=embedding_dim*latent_size + cnn_encoded_flat_dim + self.vecobs_dim + self.action_dim, hidden_size=lstm_dec_hidden_size)

        self.third_projection = nn.Linear(lstm_dec_hidden_size, cnn_encoded_flat_dim)
        print(f'Third: {lstm_dec_hidden_size} -> {cnn_encoded_flat_dim}\n')

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

        self.loss_fn = nn.CrossEntropyLoss()
    
    @torch.no_grad()
    def _compute_perplexity(self, ind):
        # debugging: cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
        encodings = einops.rearrange(F.one_hot(ind, self.quantizer.codebook_size).float(), 'b n c -> (b n) c')
        avg_probs = encodings.mean(0)
        perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
        cluster_use = torch.sum(avg_probs > 0)
        return perplexity, cluster_use

    def _apply_frame_encoding(self, pov_obs):
        # encode pov obs
        t = pov_obs.shape[1]
        pov_obs = einops.rearrange(pov_obs, 'b t c h w -> (b t) c h w')
        
        enc_pov_obs = torch.zeros(pov_obs.shape[0], self.vqvae.quantizer.embedding_dim, 16, 16).to(self.device)
        frame_quantization_idcs = []
        all_neg_dist = torch.zeros(pov_obs.shape[0], self.vqvae.quantizer.n_embed, 16, 16).to(self.device)
        
        for i in range((len(pov_obs)-1) // self.encoding_batch_size + 1):
            enc_pov_obs[i * self.encoding_batch_size:(i+1) * self.encoding_batch_size], ind, all_neg_dist[i * self.encoding_batch_size:(i+1) * self.encoding_batch_size] = self.vqvae.encode_only(pov_obs[i * self.encoding_batch_size:(i+1) * self.encoding_batch_size])
            #enc_pov_obs.append(z_q)
            frame_quantization_idcs.append(ind)
            #all_neg_dist.append(neg_dist)
        
        #pov_obs = torch.cat(enc_pov_obs, dim=0)
        enc_pov_obs = einops.rearrange(enc_pov_obs, '(b t) c h w -> b t c h w', t=t)
        frame_quantization_idcs = torch.cat(frame_quantization_idcs, dim=0)
        all_neg_dist = einops.rearrange(all_neg_dist, '(b t) c h w -> b t c h w', t=t)[:,:-1]
        
        return enc_pov_obs, frame_quantization_idcs, all_neg_dist

    def _compute_log_prior(self, neg_dist):
        B, T, C, H, W = neg_dist.shape
        log_prior = []
        neg_dist = einops.rearrange(neg_dist, 'b t c h w -> (b t h w) c')
        log_prior.append(nn.functional.log_softmax(neg_dist / self.hparams.tau, dim=1))

        log_prior = einops.rearrange(torch.cat(log_prior, dim=0), '(b t h w) c -> b t c h w', b=B, t=T, c=C, h=H, w=W)
        return log_prior
        
    def _predict_subsequence(self, enc_lstm_input, dec_lstm_input, enc_first_hidden, enc_first_cell=None, dec_first_hidden=None, dec_first_cell=None):

        # encode with lstm
        T = enc_lstm_input.shape[1]
        enc_hidden_state_seq, (enc_last_hidden, enc_last_cell) = self.lstm_encoder(enc_lstm_input, enc_first_hidden, enc_first_cell)

        # prepare quantizer input
        quantizer_input = self.second_projection(einops.rearrange(enc_hidden_state_seq, 'b t d -> (b t) d'))
        quantizer_input = einops.rearrange(quantizer_input, 'bt (latent_size quantizer_dim) -> bt latent_size quantizer_dim', latent_size=self.hparams.latent_size, quantizer_dim=self.quantizer_dim)
        
        # quantize
        discrete_embeddings, latent_loss, ind = self.quantizer(quantizer_input)
        
        # prepare lstm decoder input
        dec_lstm_input = torch.cat(
            [
                einops.rearrange(discrete_embeddings, '(b t) latent_size embed_dim -> b t (latent_size embed_dim)', t=T), 
                dec_lstm_input
            ], 
            dim=2
        )
        
        # decode with lstm
        dec_hidden_state_seq, (dec_last_hidden, dec_last_cell) = self.lstm_decoder(dec_lstm_input, dec_first_hidden, dec_first_cell)

        # prepare cnn decoder input
        cnn_decoder_input = self.third_projection(einops.rearrange(dec_hidden_state_seq, 'b t d -> (b t) d'))
        cnn_decoder_input = einops.rearrange(cnn_decoder_input, 'bt (c h w) -> bt c h w', c=self.cnn_encoded.shape[1], h=self.cnn_encoded.shape[2], w=self.cnn_encoded.shape[3])

        # decode with cnn decoder
        # decode in batches because otherwise out of memory
        predictions = []
        for i in range((len(cnn_decoder_input) - 1) // self.encoding_batch_size + 1):
            predictions.append(self.cnn_decoder(cnn_decoder_input[i*self.encoding_batch_size:(i+1)*self.encoding_batch_size]))
        predictions = einops.rearrange(torch.cat(predictions, dim=0), '(b t) c h w -> b t c h w', t=T)
        
        return predictions, latent_loss, ind, (enc_last_hidden, enc_last_cell), (dec_last_hidden, dec_last_cell)

    def _process_subsequence(
        self, 
        pov_obs, 
        vec_obs, 
        actions, 
        enc_lstm_input, 
        dec_lstm_input, 
        enc_first_hidden, 
        enc_first_cell=None, 
        dec_first_hidden=None, 
        dec_first_cell=None
    ):
        # apply frame vqvae
        pov_obs, frame_quantization_idcs, neg_dist = self._apply_frame_encoding(pov_obs)
        assert pov_obs.shape[-2:] == torch.Size([16,16]), f"Expected pov_obs.shape to end with (16,16) but got {pov_obs.shape}"

        # save for future reference  
        B, T, C, H, W = pov_obs.shape

        # apply quantizers on action and vec_obs
        if self.action_quantizer is not None:
            # apply action vqvae
            actions = einops.rearrange(self.action_quantizer.encode_only(einops.rearrange(actions, 'b t d -> (b t) d'))[0], '(b t) d -> b t d', b=B)

        if self.vecobs_quantizer is not None:
            # apply vec obs vqvae
            vec_obs = einops.rearrange(self.vecobs_quantizer.encode_only(einops.rearrange(vec_obs, 'b t d -> (b t) d'))[0], '(b t) d -> b t d', b=B)

        # encode all latent images further with cnn encoder
        encoded_images = einops.rearrange(self.cnn_encoder(einops.rearrange(pov_obs, 'b t c h w -> (b t) c h w')), 'bt c h w -> bt (c h w)')

        # prepare lstm input
        dec_lstm_input = torch.cat([einops.rearrange(encoded_images, '(b t) d -> b t d', t=T)[:,:-1], vec_obs[:,:-1], actions[:,:-1]], dim=2)
        encoded_images = self.first_projection(encoded_images)
        encoded_images = einops.rearrange(encoded_images, '(b t) d -> b t d', b=B, t=T)
        enc_first_hidden = self.lstm_starting_state_projection(torch.cat([encoded_images[:,0], vec_obs[:,0]], dim=1))[:,None]
        #print(f'post lstm starting proj: {enc_first_hidden.shape}')
        enc_lstm_input = torch.cat([encoded_images[:,1:], vec_obs[:,1:], actions[:,:-1]], dim=2)

        # make predictions
        predictions, latent_loss, ind, (enc_last_hidden, enc_last_cell), (dec_last_hidden, dec_last_cell) = self._predict_subsequence( 
            enc_lstm_input,
            dec_lstm_input, 
            enc_first_hidden,
            enc_first_cell,
            dec_first_hidden,
            dec_first_cell
        )

        # compute priors and update
        if not self.hparams.discard_priors:
            log_prior = self._compute_log_prior(neg_dist)
            predictions += log_prior
        else:
            pass
        
        # compute loss
        reconstruction_loss = self.loss_fn(predidtions, frame_quantization_idcs[1:])

        return reconstruction_loss, latent_loss, ind, (enc_last_hidden, enc_last_cell), (dec_last_hidden, dec_last_cell)

    def forward(self, pov_obs, vec_obs, actions, max_seq_len=None):
        if max_seq_len is None:
            max_seq_len = self.hparams.max_seq_len
        
        # init lstm states
        enc_last_hidden = None
        enc_last_cell = None
        dec_last_hidden = None
        dec_last_cell = None

        # init loss
        total_reconstruction_loss = 0
        total_latent_loss = 0
        
        # init index tensor
        all_ind = []

        for i in range((pov_obs.shape[1] - 1) // max_seq_len + 1):
            start_idx = i * max_seq_len
            stop_idx = (i + 1) * max_seq_len
            reconstruction_loss, latent_loss, ind, (enc_last_hidden, enc_last_cell), (dec_last_hidden, dec_last_cell) = self._process_subsequence(
                pov_obs[:,start_idx:stop_idx],
                vec_obs[:,start_idx:stop_idx],
                actions[:,start_idx:stop_idx],
                enc_lstm_input,#TODO
                dec_lstm_input,#TODO
                enc_last_hidden,
                enc_last_cell,
                dec_last_hidden,
                dec_last_cell
            )
            
            # save subsequence outputs
            total_latent_loss = total_latent_loss + latent_loss
            total_reconstruction_loss = total_reconstruction_loss + reconstruction_loss
            all_ind.append(ind)
        
        all_ind = torch.cat(all_ind, dim=0)

        return total_reconstruction_loss, total_latent_loss, all_ind

    def forward(self, pov_obs, vec_obs, actions, max_seq_len=None):
        if max_seq_len is None:
            max_seq_len = self.hparams.max_seq_len
        
        # apply frame vqvae
        pov_obs, frame_quantization_idcs, neg_dist = self._apply_frame_encoding(pov_obs)
        assert pov_obs.shape[-2:] == torch.Size([16,16]), f"Expected pov_obs.shape to end with (16,16) but got {pov_obs.shape}"
                
        B, T, C, H, W = pov_obs.shape
        #print('B, T, C, H, W = ', B, T, C, H, W)

        if self.action_quantizer is not None:
            # apply action vqvae
            actions = einops.rearrange(self.action_quantizer.encode_only(einops.rearrange(actions, 'b t d -> (b t) d'))[0], '(b t) d -> b t d', b=B)
        
        if self.vecobs_quantizer is not None:
            # apply vec obs vqvae. If it was None, then this is the identity mapping
            vec_obs = einops.rearrange(self.vecobs_quantizer.encode_only(einops.rearrange(vec_obs, 'b t d -> (b t) d'))[0], '(b t) d -> b t d', b=B)

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
            predictions, latent_loss, ind, (enc_last_hidden, enc_last_cell), (dec_last_hidden, dec_last_cell) = self._predict_subsequence( 
                enc_lstm_input[:,:max_seq_len],
                dec_lstm_input[:,:max_seq_len], 
                enc_first_hidden
            )
            #print(f'\n{predictions=}\n')
            #print(f'\n{encoded_images[:,:max_seq_len]=}\n')
            #print(f'{predictions.shape=}')
            
            all_predictions = [predictions]
            all_latent_losses = [latent_loss]
            all_ind = [ind]
            for i in range((pov_obs.shape[1]-2) // max_seq_len - 1):
                start_idx = (i+1)*max_seq_len
                end_idx = (i+2)*max_seq_len

                predictions, latent_loss, ind, (enc_last_hidden, enc_last_cell), (dec_last_hidden, dec_last_cell) = self._predict_subsequence(
                    enc_lstm_input[:,start_idx:end_idx], 
                    dec_lstm_input[:,start_idx:end_idx],
                    enc_last_hidden.detach(), 
                    enc_last_cell.detach(), 
                    dec_last_hidden.detach(), 
                    dec_last_cell.detach()
                )
                
                all_predictions.append(predictions)
                all_latent_losses.append(latent_loss)
                all_ind.append(ind)

            # predict remaining frames
            remaining_idx = (pov_obs.shape[1]-2) // max_seq_len * max_seq_len
            predictions, latent_loss, ind, (enc_last_hidden, enc_last_cell), (dec_last_hidden, dec_last_cell) = self._predict_subsequence(
                enc_lstm_input[:,remaining_idx:], 
                dec_lstm_input[:,remaining_idx:],
                enc_last_hidden.detach(), 
                enc_last_cell.detach(), 
                dec_last_hidden.detach(), 
                dec_last_cell.detach()
            )
            all_predictions.append(predictions)
            all_latent_losses.append(latent_loss)
            all_ind.append(ind)
            
            #all_predictions = torch.cat(all_predictions, dim=1)
            all_ind = torch.cat(all_ind, dim=0)
            #print(f'{predictions.shape=}')
            
            latent_loss = torch.sum(torch.tensor(all_latent_losses, device=self.device))
        else:
            raise NotImplementedError
            enc_lstm_input = torch.cat([encoded_images[:,1:], vec_obs[:,1:], actions[:,:-1]], dim=2) # (B T D+A)
            predictions, latent_loss, *_ = self._predict_subsequence(enc_lstm_input, h_0)
        
        # predictions.shape = [B, T-1, 32, 16, 16]
        # compute log_priors and update
        if not self.hparams.discard_priors:
            log_prior = self._compute_log_prior(neg_dist)
            for i in range(len(all_predictions)):
                all_predictions[i] += log_prior[:,i*self.hparams.max_seq_len:(i+1)*self.hparams.max_seq_len] 
        else:
            pass

        #return predictions, pov_obs[:,1:], latent_loss
        return all_predictions, frame_quantization_idcs[1:], latent_loss, all_ind
        
    def training_step(self, batch, batch_idx):
        # unpack batch
        pov_obs, vec_obs, actions = batch
        
        # make predictions
        predictions, targets, latent_loss, ind = self(pov_obs, vec_obs, actions)
        
        # compute loss
        reconstruction_loss = 0
        for i in range(len(predictions)):
            preds = predictions[i][0]
            targs = targets[i*self.hparams.max_seq_len:(i+1)*self.hparams.max_seq_len]
            print(f"{preds.shape = }")
            print(f"{targs.shape = }")
            reconstruction_loss = reconstruction_loss + self.loss_fn(preds, targs)
        loss = reconstruction_loss + latent_loss
        #print(f'attempting backward with sequence of length {pov_obs.shape[1]}...')

        # logging
        self.log('Training/loss', loss, on_step=True)
        self.log('Training/reconstruction_loss', reconstruction_loss, on_step=True)
        self.log('Training/latent_loss', latent_loss, on_step=True)
        if self.hparams.log_perplexity:
            if (self.global_step + 1) % self.hparams.perplexity_freq == 0:
                self.eval()
                perplexity, cluster_use = self._compute_perplexity(ind)
                self.train()
                self.log('Training/perplexity', perplexity, prog_bar=True)
                self.log('Training/cluster_use', cluster_use, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        params = []
        for m in self.model_list:
            params += list(m.parameters())
        self.optimizer = torch.optim.AdamW(params, **self.hparams.optim_kwargs)
        return self.optimizer

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
        return z_q, latent_loss, ind