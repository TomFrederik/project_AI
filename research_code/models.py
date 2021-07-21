import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import torchdiffeq as teq
import utils

from torchvision.transforms import ToPILImage # used to transform input to VAE to uint image


class MDNLSTMDynamicsModel(pl.LightningModule):
    def __init__(self, lstm_kwargs, latent_dim, VAE_path, optim_kwargs, scheduler_kwargs, seq_len):
        super().__init__()
        
        # save params
        self.save_hyperparameters()
    
        # load VAE
        self.VAE = ConvVAE.load_from_checkpoint(VAE_path)
        self.VAE.eval()

        # save some vars
        self.scheduler_kwargs = scheduler_kwargs
        self.optim_kwargs = optim_kwargs
        self.seq_len = seq_len
        self.mse_loss = nn.MSELoss()
        self.merge = MergeFramesWithBatch()
        self.split = SplitFramesFromBatch(self.seq_len)
        self.split_pred = SplitFramesFromBatch(self.seq_len-1)
        self.lstm = nn.LSTM(**lstm_kwargs, batch_first=True)
        # we KNOW that the latent space follows a multi-variate, diagonal covariance, Gaussian,
        # so we will only need a single (multi-dimensional) component to model the encoding
        # The obfuscated vector could be multimodal, but we assume for now that it isn't
        self.mdn_network = nn.Sequential(nn.Linear(lstm_kwargs['hidden_size'], 2*(latent_dim+64)))
        self.elu = nn.ELU()    
    
    def forward(self, model_input):
        lstm_out, _ = self.lstm(model_input)
        lstm_out = lstm_out[:,:-1] # skip last element, since we can't score it against a target
        mdn_in = self.merge(lstm_out) # merge frames with batch
        mean, preact_std = torch.chunk(self.mdn_network(mdn_in), chunks=2, dim=1)
        std = self.elu(preact_std) + 1 # make sure std is non-negative

        # compute log prob of last frames under computed Gaussian
        merged_input = self.merge(model_input[:,1:,:-64]) # don't want to predict actions, just observation
        log_prob = self._get_log_p(merged_input, mean, std)
        log_prob = self.split_pred(log_prob) # split again
        
        # sample from the multi-dim gaussian parameterized by the mdn outputs --> only used to compare with NeuralODE
        pred_z = mean + std * torch.normal(torch.zeros_like(mean), torch.ones_like(std))
        pred_z = self.split_pred(pred_z)
        return log_prob, pred_z

    def _get_log_p(self, x, mean, std):
        '''
        Computes log prob of a x under a diagonal multivariate gaussian
        Shapes:
        x - (B*T, D)
        mu - (B*T, D)
        std - (B*T, D)
        '''
        D = x.shape[1]
        return -0.5 * D * np.log(2*np.pi) - torch.sum(2 * torch.log(std) + (x - mean).abs().pow(2) / (2 * std.abs().pow(2)), dim=1)

    def _step(self, batch):
        '''
        Helper function which encodes the pov obs, cats them with vec obs and action to pass through self.forward
        returns prediction and target
        '''
        # get data
        pov, vec, actions = batch
        pov = self.merge(pov) # merge frames with batch for batch processing by VAE
        pov = self.VAE.encode_only(pov)
        pov = self.split(pov) # split frames from batch again
        obs = torch.cat([pov, vec], dim=2)
        target_obs = obs[:,1:,:]
        model_input = torch.cat([obs, actions], dim=2)
        # create predictions
        log_p, pred_z = self(model_input)
        return log_p, pred_z, target_obs
    
    def training_step(self, batch, batch_idx):
        log_p, pred_obs, target_obs = self._step(batch)
        # score and log predictions
        mse_loss = self.mse_loss(pred_obs, target_obs)
        nll_loss = -1 * log_p.mean()
        self.log('Training/nll_loss', nll_loss, on_step=True)
        self.log('Training/mse_loss', mse_loss, on_step=True)
        return nll_loss
        
    def validation_step(self, batch, batch_idx):
        log_p, pred_obs, target_obs = self._step(batch)
        # score and log predictions
        mse_loss = self.mse_loss(pred_obs, target_obs)
        nll_loss = -1 * log_p.mean()
        self.log('Validation/nll_loss', nll_loss, on_step=False, on_epoch=True)
        self.log('Validation/mse_loss', mse_loss, on_step=False, on_epoch=True)
        return nll_loss
        
    def configure_optimizers(self):
        # set up optimizer
        optimizer = torch.optim.Adam(self.parameters(), **self.optim_kwargs)
        # set up scheduler
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.scheduler_kwargs['lr_gamma'])
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': self.scheduler_kwargs['lr_step_mode'],
            'frequency': self.scheduler_kwargs['lr_decrease_freq'],
        }
        return {'optimizer':optimizer, 'lr_scheduler':lr_dict}

    @torch.no_grad()
    def predict_recursively(self, input):
        '''
        Auto-regressively applies dynamics model to extrapolate from image
        Input shape should be (B, D), i.e. have no time component yet
        '''
        #TODO make it so that it can take new actions intead of repeating action
        out = input[:,None,:]
        action = out[:,:,-64:]
        for t in range(self.seq_len):
            # predict new frame / latent space
            lstm_out, _ = self.lstm(out)
            mdn_in = lstm_out[:,-1,:] # only take last output
            mean, preact_std = torch.chunk(self.mdn_network(mdn_in), chunks=2, dim=1)
            std = self.elu(preact_std) + 1 # make sure std is non-negative

            # sample from the multi-dim gaussian parameterized by the mdn outputs --> only used to compare with NeuralODE
            pred_z = mean + std * torch.normal(torch.zeros_like(mean), torch.ones_like(std))
            pred_z = pred_z.reshape((pred_z.shape[0],1,pred_z.shape[1]))
            pred_z = torch.cat([pred_z, action], dim=2) # repeat action, #TODO: make it so that action can be given as input?
            
            out = torch.cat([out, pred_z], dim=1) # add new frame to sequence
        
        return out[:,:,:-128] # return generated sequence, but only z part, i.e. not vec obs and vec act.


class DynamicsBaseModel(nn.Module):

    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        hidden_dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], input_dim))

        self.net = nn.Sequential(*layers)
        
    def forward(self, t, model_input):
        '''
        t - time, needed for odeint, but not used in model
        input should be of shape (B, latent_dim + vec_obs_dim + action_dim), e.g. (B, 256)
        '''
        return self.net(model_input)

class NODEDynamicsModel(pl.LightningModule):
    def __init__(self, base_model_class, base_model_kwargs, VAE_path, optim_kwargs, scheduler_kwargs, seq_len):
        super().__init__()
        
        # save params
        self.save_hyperparameters()
    
        # load VAE
        self.VAE = ConvVAE.load_from_checkpoint(VAE_path)
        self.VAE.eval()

        # save some vars
        self.scheduler_kwargs = scheduler_kwargs
        self.optim_kwargs = optim_kwargs
        self.seq_len = seq_len
        self.base_model = base_model_class(**base_model_kwargs)
        self.criterion = nn.MSELoss()
        self.timesteps = None
        self.merge = MergeFramesWithBatch()
        self.split = SplitFramesFromBatch(self.seq_len)
        
    
    def forward(self, model_input):
        if self.timesteps is None:
            self.timesteps = torch.linspace(0,self.seq_len,self.seq_len, device=self.device)
        # pass through ode solver
        pred_y = teq.odeint_adjoint(self.base_model, model_input, self.timesteps, adjoint_options={"norm": "seminorm"})
        return pred_y

    def _step(self, batch):
        '''
        Helper function
        '''
        # get data
        pov, vec, actions = batch
        pov = self.merge(pov) # merge frames with batch for batch processing
        pov = self.VAE.encode_only(pov)
        pov = self.split(pov) # split frames from batch again
        obs = torch.cat([pov, vec], dim=2)
        input_obs, target_obs = obs[:,0,:], obs[:,1:,:] # split into input and target
        model_input = torch.cat([input_obs, actions[:,0,:]], dim=1)
        # create predictions
        pred_obs = self(model_input)[:,:,:obs.shape[2]] # throw away the predicted trajectories of actions
        pred_obs = pred_obs[1:,:,:].transpose(0,1) # flip to batch first, and throw away initial value, since it didn't change
        return pred_obs, target_obs
    
    def training_step(self, batch, batch_idx):
        pred_obs, target_obs = self._step(batch)
        # score and log predictions
        loss = self.criterion(pred_obs, target_obs)
        self.log('Training/loss', loss, on_step=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        pred_obs, target_obs = self._step(batch)
        # score and log predictions
        loss = self.criterion(pred_obs, target_obs)
        self.log('Validation/loss', loss, on_epoch=True, on_step=False)
        return loss
    
    def configure_optimizers(self):
        # set up optimizer
        optimizer = torch.optim.Adam(self.parameters(), **self.optim_kwargs)
        # set up scheduler
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.scheduler_kwargs['lr_gamma'])
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': self.scheduler_kwargs['lr_step_mode'],
            'frequency': self.scheduler_kwargs['lr_decrease_freq'],
        }
        return {'optimizer':optimizer, 'lr_scheduler':lr_dict}

class MergeFramesWithBatch(nn.Module):
    '''
    Transforms tensor of shape (N, T, ...) to tensor of shape (N * T, ...)
    '''
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input.reshape((input.shape[0] * input.shape[1], *input.shape[2:]))

class SplitFramesFromBatch(nn.Module):
    '''
    Transforms tensor of shape (N * T, ...) to tensor of shape (N, T, ...)
    '''
    def __init__(self, num_frames):
        super().__init__()
        self.num_frames = num_frames
    
    def forward(self, input):
        return input.reshape((-1, self.num_frames, *input.shape[1:]))

class SplitChannelsFromClasses(nn.Module):
    '''
    Reshapes a (N, num_classes * num_channels, H, W) tensor
    into a     (N, num_classes, num_channels, H, W) tensor
    '''

    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, input):
        return input.reshape((input.shape[0], -1, self.num_channels, *input.shape[2:]))

class Transpose(nn.Module):
    '''
    Transposes two dimensions of a tensor
    '''
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    
    def forward(self, input):
        return input.transpose(dim1, dim2)

class ConvLSTM(nn.Module):

    def __init__(self, lstm_kwarg_list, num_frames):
        super().__init__()
        model_list = []
        self.lstm_kwarg_list = lstm_kwarg_list
        for i, kwargs in enumerate(self.lstm_kwarg_list):
            # if img_shape is not provided, compute it automatically:
            if 'img_shape' not in kwargs:
                
                try:
                    last_img_shape = self.lstm_kwarg_list[i-1]['img_shape']
                except:
                    raise ValueError('No input img shape detected. Provide img shape for input to first layer')
                
                h_new = np.floor((last_img_shape[0] - self.lstm_kwarg_list[i-1]['kernel_size'])/2 + 1).astype(int)
                kwargs['img_shape'] = (h_new, h_new)
                print(f'No img_shape for layer {i+1} provided. Automatically computed img_shape: {(h_new, h_new)}')

            # append modules
            model_list.append(ConvLSTMCell(**kwargs, last_only=False)) # return all hidden states so that later layers can also view everything.
            model_list.append(MergeFramesWithBatch()) # merge frames with batch for BN
            model_list.append(nn.BatchNorm2d(num_features=kwargs['out_channels'])) # BatchNorm
            model_list.append(nn.MaxPool2d(kernel_size=kwargs['kernel_size'], stride=2)) # MaxPooling
            model_list.append(SplitFramesFromBatch(num_frames=num_frames)) # split frames from batch again for next iteration
        
        self.net = nn.Sequential(*model_list)

        h_out = np.floor((self.lstm_kwarg_list[-1]['img_shape'][0] - self.lstm_kwarg_list[-1]['kernel_size'])/2 + 1).astype(int)
        self.out_img_shape = (self.lstm_kwarg_list[-1]['out_channels'], h_out, h_out)

    def forward(self, input):
        return self.net(input)

class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, img_shape, last_only=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.last_only = last_only
        self.img_shape = img_shape
        
        # define layers
        padding = (kernel_size - 1) // 2
        
        self.conv_xi = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_hi = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.W_ci = nn.Parameter(data=torch.zeros(img_shape))

        self.conv_xf = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_hf = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.W_cf = nn.Parameter(data=torch.zeros(img_shape))


        self.conv_xc = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_hc = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        
        self.conv_xo = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_ho = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.W_co = nn.Parameter(data=torch.zeros(img_shape))
        

        # init weight vectors
        nn.init.kaiming_normal_(self.W_ci.data, nonlinearity='sigmoid')
        nn.init.kaiming_normal_(self.W_cf.data, nonlinearity='sigmoid')
        nn.init.kaiming_normal_(self.W_co.data, nonlinearity='sigmoid')
        
            
    def forward(self, input):
        '''
        Args:
            input   - should be of shape (B, T, C, H, W)
        Returns:
            h - hidden states of shape (B, T, C, H, W)
            or
            curr_h - last hidden state of shape (B, C, H, W)
        '''
        assert len(input.shape) == 5, f'input shape should have 5 entries, but is {input.shape}'
        
        # get sequence length T
        seq_len = input.shape[1]

        # init h and c
        h = torch.zeros((*input.shape[:2], self.out_channels, *input.shape[3:]), device=input.device)
        c = torch.zeros((*input.shape[:2], self.out_channels, *input.shape[3:]), device=input.device)

        curr_h = torch.zeros((input.shape[0], self.out_channels, *input.shape[3:]), device=input.device)
        curr_c = torch.zeros((input.shape[0], self.out_channels, *input.shape[3:]), device=input.device)

        list_of_h = []

        for t in range(seq_len):
            x_t = input[:,t]
            
            i_t = torch.sigmoid(self.conv_xi(x_t) + self.conv_hi(curr_h) + self.W_ci * (curr_c))
            f_t = torch.sigmoid(self.conv_xf(x_t) + self.conv_hf(curr_h) + self.W_cf * (curr_c))
            
            curr_c = f_t * curr_c + i_t * torch.tanh(self.conv_xc(x_t) + self.conv_hc(curr_c))
            
            o_t = torch.sigmoid(self.conv_xo(x_t) + self.conv_ho(curr_h) + self.W_co) * (curr_c)
            curr_h = o_t * torch.tanh(curr_c)

            list_of_h.append(curr_h)

        if self.last_only:
            return curr_h
        else:
            return torch.stack(list_of_h, dim=1)
            raise NotImplementedError("Currently only support returning last hidden state")

class CLSTMEncoder(nn.Module):

    def __init__(self, lstm_kwarg_list, num_frames, latent_dim):
        super().__init__()
        self.CLSTM = ConvLSTM(lstm_kwarg_list, num_frames)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        self.clstm_out_shape = self.CLSTM.out_img_shape

        self.linear = nn.Linear(np.prod(self.clstm_out_shape)*num_frames, 2*latent_dim)
    
    def forward(self, x):
        '''
        x - input, for mineRL (B, T, C, H, W) batch of stack of frames
        '''
        clstm_out = self.CLSTM(x)
        latent_out = self.linear(self.flatten(clstm_out))
        mean, log_std = torch.chunk(latent_out, chunks=2, dim=-1)

        return latent_out

class ConvEncoder(nn.Module):

    def __init__(self, latent_dim, num_channels, kernel_size, img_shape):
        '''
        Args:
            latent_dim - dimension of the latent space
            num_channels - list of ints describing the number of channels after each convolution. 
                           len(num_channels) is used to determine number of layers.
            kernel_size - int, size of the kernels used in the convolutions
        '''
        super().__init__()
        
        # compute padding for same-padding
        padding = (kernel_size - 1) // 2
        
        num_channels = [3] + num_channels
        model_list = []
        self.img_shape_list = [img_shape]
        for i in range(len(num_channels)-1):
            model_list.append(nn.Conv2d(in_channels=num_channels[i], out_channels=num_channels[i+1], kernel_size=kernel_size, stride=1, padding=padding))
            model_list.append(nn.BatchNorm2d(num_features=num_channels[i+1])) # BatchNorm
            model_list.append(nn.GELU()) # activation function
            model_list.append(nn.MaxPool2d(kernel_size=kernel_size, stride=2)) # MaxPooling
            h_new = np.floor((self.img_shape_list[-1][0] - kernel_size)/2 + 1).astype(int)
            self.img_shape_list.append((h_new, h_new))

        print('Encoder img_shapes:', self.img_shape_list)
        
        self.conv_net = nn.Sequential(*model_list)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear = nn.Linear(self.img_shape_list[-1][0] * self.img_shape_list[-1][1] * num_channels[-1], 2*latent_dim)

        self.conv_out_shape = (num_channels[-1], *self.img_shape_list[-1])

    def forward(self, x):
        '''
        x - input, for mineRL (B, C, H, W) batch of frames
        '''
        conv_out = self.conv_net(x)
        latent_out = self.linear(self.flatten(conv_out))

        mean, log_std = torch.chunk(latent_out, chunks=2, dim=-1)
        
        return mean, log_std

class ConvDecoder(nn.Module):

    def __init__(self, latent_dim, num_channels, kernel_size, conv_out_shape, img_shape_list):
        '''
        Args:
            latent_dim - dimension of the latent space
            num_channels - list of ints describing the number of channels after each convolution. 
                           len(num_channels) is used to determine number of layers.
            kernel_size - int, size of the kernels used in the convolutions
            conv_out_shape - shape of output of ConvLSTM or Conv_net of Encoder
            list of img_shapes - [..., (x,x), (64,64)], list of image shapes
        '''
        super().__init__()
        
        
        print(f'\nImage sizes for deconv: {img_shape_list}')

        self.conv_out_shape = conv_out_shape

        # compute padding for same-padding
        padding = (kernel_size - 1) // 2

        self.linear = nn.Sequential(nn.Linear(latent_dim, np.prod(conv_out_shape)), nn.GELU())
        
        num_channels = [conv_out_shape[0]] + num_channels

        layer_list = []
        for i in range(len(num_channels)-1):
            layer_list.append(nn.Upsample(img_shape_list[i+1]))
            layer_list.append(nn.Conv2d(num_channels[i], num_channels[i+1], kernel_size=kernel_size, padding=padding))
            layer_list.append(nn.GELU())

        # conv to 3 * 255 channels
        layer_list.append(nn.Conv2d(num_channels[-1], 768, kernel_size=kernel_size, padding=padding))
        layer_list.append(SplitChannelsFromClasses(num_channels=3))
        self.deconv = nn.Sequential(*layer_list)
    
    def forward(self, z):
        '''
        z - latent vector
        '''
        
        out = self.linear(z)
        out = out.reshape((z.shape[0], *self.conv_out_shape))
        return self.deconv(out)

class MyDataParallel(nn.DataParallel):
    '''
    Wrapper class for DataParallel which allows access to the methods of the base model
    '''
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class CLSTMVAE(pl.LightningModule):

    def __init__(self, encoder_kwargs, decoder_kwargs, optim_kwargs, scheduler_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.optim_kwargs = optim_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        
        self.encoder = CLSTMEncoder(**encoder_kwargs)
        img_shape_list = [l['img_shape'] for l in self.encoder.CLSTM.lstm_kwarg_list]
        img_shape_list.reverse()
        self.decoder = ConvDecoder(**decoder_kwargs, clstm_out_shape=self.encoder.clstm_out_shape, img_sizes=img_shape_list)
        self.CE_loss = nn.CrossEntropyLoss(reduction='none')

        self.transform = ToPILImage(mode='RGB')

    def sample(self, mean, log_std):
        '''
        Implements reparameterizatio trick to sample from the given normal distribution
        '''
        z = mean + torch.exp(log_std) * torch.normal(torch.zeros_like(mean), torch.ones_like(log_std))
        return z
    

    @torch.no_grad()
    def reconstruct_only(self, x):
        '''
        Encodes x, samples z and reconstructs x
        x - shape (B, C, H, W)
        '''
        # encode
        mean, log_std = self.encoder(x)

        # sample latent vector
        z = self.sample(mean, log_std)

        # decode
        probs = torch.nn.functional.softmax(self.decoder(z), dim=1)

        # sample from the 255-category distribution
        sampled_x = torch.multinomial(probs.transpose(0,1).flatten(start_dim=1).transpose(0,1), 1).squeeze().reshape(x.shape[2:])
        
        return sampled_x.float() / 255
        
    def forward(self, x):
        '''
        x - input, for mineRL (B, T, C, H, W) batch of stack of frames
        '''
        # encode
        mean, log_std = self.encoder(x)

        # compute KL difstance, i.e. regularization loss
        L_regul = (0.5 * (torch.exp(2 * log_std) + mean ** 2 - 1 - 2 * log_std)).sum(dim=-1)

        # sample latent vector
        z = self.sample(mean, log_std)

        # decode
        x_new = self.decoder(z)
        
        # convert x to classes
        x = (x[:,-1] * 255).type(torch.long)
        
        # compute reconstruction loss, sum over all dimension except batch
        L_reconstr = self.CE_loss(x_new, x).sum(dim=list(range(1,len(x.shape))))

        # compute elbo
        elbo =  L_reconstr + L_regul

        # convert into bits per dimension loss
        bpd = (elbo * np.log2(np.exp(1)) / np.prod(x.shape[1:])).mean()

        # take means
        L_reconstr = torch.mean(L_reconstr)
        L_regul = torch.mean(L_regul)
        
        return L_reconstr, L_regul, bpd

    def configure_optimizers(self):
        # set up optimizer
        optimizer = torch.optim.Adam(self.parameters(), **self.optim_kwargs)
        # set up 
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.scheduler_kwargs['lr_gamma'])
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': self.scheduler_kwargs['lr_step_mode'],
            'frequency': self.scheduler_kwargs['lr_decrease_freq'],
        }
        return {'optimizer':optimizer, 'lr_scheduler':lr_dict}
    
    def training_step(self, batch, batch_idx):
        img = batch[1]
        #print(img)
        #print(img.shape)
        _, _, bpd = self(img)
        self.log('Training/batch_bpd', bpd.mean().item(),on_step=True)

        return bpd
    
    def validation_step(self, batch, batch_idx):
        _, img, _, _, _, _ = batch
        _, _, bpd = self(img)
        self.log('Validation/bpd', bpd.mean().item(), on_epoch=True)
        
        return bpd

class VAE(pl.LightningModule):
    '''
    A base class for VAEs
    '''
    def __init__(self, encoder_class, decoder_class, encoder_kwargs, decoder_kwargs, learning_rate, scheduler_kwargs, batch_size):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.scheduler_kwargs = scheduler_kwargs

        self.encoder = encoder_class(**encoder_kwargs)
        img_shape_list=self.encoder.img_shape_list.copy()
        img_shape_list.reverse()
        self.decoder = decoder_class(**decoder_kwargs, conv_out_shape=self.encoder.conv_out_shape, img_shape_list=img_shape_list)
        self.CE_loss = nn.CrossEntropyLoss(reduction='none')

        self.transform = ToPILImage(mode='RGB')
    
    def sample(self, mean, log_std):
        '''
        Implements reparameterizatio trick to sample from the given normal distribution
        '''
        z = mean + torch.exp(log_std) * torch.normal(torch.zeros_like(mean), torch.ones_like(log_std))
        return z
    
    @torch.no_grad()
    def reconstruct_only(self, x):
        '''
        Encodes x, samples z and reconstructs x
        x - shape (B, C, H, W)
        '''
        # encode
        mean, log_std = self.encoder(x)

        # sample latent vector
        z = self.sample(mean, log_std)

        # decode
        probs = torch.nn.functional.softmax(self.decoder(z), dim=1)

        # sample from the 255-category distribution
        sampled_x = torch.multinomial(probs.transpose(0,1).flatten(start_dim=1).transpose(0,1), 1).squeeze().reshape(x.shape)
        
        return sampled_x.float() / 255
    
    @torch.no_grad()
    def encode_only(self, x):
        '''
        Encodes images into their latent space (via sampling).
        Does not track gradients. Use e.g. as preprocessing/embedding.
        Args:
            x - input, for mineRL (B, C, H, W) batch of frames
        Returns:
            latent - tensor of shape (B, L), where L is the latent dimension
        '''
        mean, log_std = self.encoder(x)
        latent = self.sample(mean, log_std)

        return latent

    @torch.no_grad()
    def decode_only(self, z):
        # decode
        probs = torch.nn.functional.softmax(self.decoder(z), dim=1)

        # sample from the 255-category distribution
        sampled_x = torch.multinomial(probs.transpose(0,1).flatten(start_dim=1).transpose(0,1), 1).squeeze().reshape((-1, 3, 64, 64))
        
        return sampled_x.float() / 255
    
    def forward(self, x):
        '''
        x - input, for mineRL (B, C, H, W) batch of frames
        '''
        # encode
        mean, log_std = self.encoder(x)

        # compute KL difstance, i.e. regularization loss
        L_regul = (0.5 * (torch.exp(2 * log_std) + mean ** 2 - 1 - 2 * log_std)).sum(dim=-1)

        # sample latent vector
        z = self.sample(mean, log_std)

        # decode
        x_new = self.decoder(z)
        
        # convert x to classes
        x = (x * 255).type(torch.long)
        
        # compute reconstruction loss, sum over all dimension except batch
        L_reconstr = self.CE_loss(x_new, x).sum(dim=list(range(1,len(x.shape))))

        # compute elbo
        elbo =  L_reconstr + L_regul

        # convert into bits per dimension loss
        bpd = (elbo * np.log2(np.exp(1)) / np.prod(x.shape[1:])).mean()

        # take means
        L_reconstr = torch.mean(L_reconstr)
        L_regul = torch.mean(L_regul)
        
        return L_reconstr, L_regul, bpd
    
    def configure_optimizers(self):
        # set up optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # set up 
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.scheduler_kwargs['lr_gamma'])
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': self.scheduler_kwargs['lr_step_mode'],
            'frequency': self.scheduler_kwargs['lr_decrease_freq'],
        }
        return {'optimizer':optimizer, 'lr_scheduler':lr_dict}
    
    def training_step(self, batch, batch_idx):
        img = batch[1]
        _, _, bpd = self(img)
        self.log('Training/batch_bpd', bpd.mean().item(),on_step=True)

        return bpd
    
    def validation_step(self, batch, batch_idx):
        _, img, _, _, _, _ = batch
        _, _, bpd = self(img)
        self.log('Validation/bpd', bpd.mean().item(), on_epoch=True)
        return bpd
    
class ConvVAE(VAE):

    def __init__(self, encoder_kwargs, decoder_kwargs, learning_rate, scheduler_kwargs, batch_size):
        kwargs = {
            'encoder_class':ConvEncoder,
            'decoder_class':ConvDecoder,
            'encoder_kwargs':encoder_kwargs,
            'decoder_kwargs':decoder_kwargs,
            'learning_rate':learning_rate,
            'scheduler_kwargs':scheduler_kwargs,
            'batch_size':batch_size
        }
        super().__init__(**kwargs)

class ResnetVAE(VAE):
    
    def __init__(self, encoder_kwargs, decoder_kwargs, learning_rate, scheduler_kwargs, batch_size):
        kwargs = {
            'encoder_class':ResnetEncoder,
            'decoder_class':ResnetDecoder,
            'encoder_kwargs':encoder_kwargs,
            'decoder_kwargs':decoder_kwargs,
            'learning_rate':learning_rate,
            'scheduler_kwargs':scheduler_kwargs,
            'batch_size':batch_size
        }
        super().__init__(**kwargs)


class ResnetEncoder(nn.Module):

    def __init__(self, ):
        pass

    def forward(self, input):
        pass

class ResnetDecoder(nn.Module):

    def __init__(self, ):
        pass

    def forward(self, input):
        pass

class DynamicsModel(pl.LightningModule):

    def __init__(self, input_size, num_layers, num_hidden, optim_kwargs, scheduler_kwargs):
        self.save_hyperparameters()

        self.optim_kwargs = optim_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=num_hidden, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(self.input_size, self.input_size-64) # want to predict latent + vec_obs, not action
        self.criterion = nn.MSELoss()

    def forward(self, input):
        '''
        input should be of shape (B, T, D), where D = L + 64 + 64 and L is the latent dimension of the encoding.
        '''
        print('LSTM input shape', input.shape)
        lstm_out = self.lstm(input)[0] # return last hidden state at every step
        pred = self.linear(lstm_out)
        return pred

    
    def configure_optimizers(self):
        # set up optimizer
        optimizer = torch.optim.Adam(self.parameters(), **self.optim_kwargs)
        # set up 
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.scheduler_kwargs['lr_gamma'])
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': self.scheduler_kwargs['lr_step_mode'],
            'frequency': self.scheduler_kwargs['lr_decrease_freq'],
        }
        return {'optimizer':optimizer, 'lr_scheduler':lr_dict}

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        print('pred shape', pred.shape)
        # pred should be of same shape as input, i.e. (B, L + 128)
        # pred is scored against original sequence
        loss = self.criterion(pred[:,:-1], batch[:,1:,:-64])
        self.log('Training/loss', loss.mean().item(), on_step=True)
    
    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        loss = self.criterion(pred[:,:-1], batch[:,1:,:-64])
        self.log('Validation/loss', loss.mean().item(), on_step=True)


class BCLinear(pl.LightningModule):

    def __init__(self, input_dim, hidden_dims, output_dim, learning_rate, scheduler_kwargs, centroids_path, VAE_path):
        super().__init__()
        self.save_hyperparameters()
        
        self.VAE = ConvVAE.load_from_checkpoint(VAE_path)
        self.VAE.eval()
        self.centroids = torch.from_numpy(np.load(centroids_path))
        self.learning_rate = learning_rate
        self.scheduler_kwargs = scheduler_kwargs
        self.loss_fct = nn.CrossEntropyLoss()

        hidden_dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.net = nn.Sequential(*layers)
        
    def forward(self, model_input):
        '''
        input should be of shape (B, latent_dim + vec_obs_dim), e.g. (B, 192)
        '''
        return self.net(model_input)

    def configure_optimizers(self):
        # set up optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # set up 
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.scheduler_kwargs['lr_gamma'])
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': self.scheduler_kwargs['lr_step_mode'],
            'frequency': self.scheduler_kwargs['lr_decrease_freq'],
        }
        return {'optimizer':optimizer, 'lr_scheduler':lr_dict}
    
    def training_step(self, batch, batch_idx):
        # get model input and target actions
        pov, vec, actions = batch
        model_input = torch.cat([self.VAE.encode_only(pov), vec], dim=1)
        
        # generate predictions
        pred = self(model_input)
        
        # map action to centroids
        actions = self.remap_actions(actions)
        
        # compute loss and log
        loss = self.loss_fct(pred, actions) 
        self.log('Training/loss', loss.mean().item(), on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # get model input and target actions
        pov, vec, actions = batch
        model_input = torch.cat([self.VAE.encode_only(pov), vec], dim=1)
        
        # generate predictions
        pred = self(model_input)
        
        # map action to centroids
        actions = self.remap_actions(actions)
        
        # compute loss and log
        loss = self.loss_fct(pred, actions) 
        self.log('Validation/loss', loss.mean().item())
        return loss

    @torch.no_grad()
    def remap_actions(self, actions):
        if self.device != self.centroids.device:
            self.centroids = self.centroids.to(self.device)
        # compute distances between action vectors and centroids
        distances = torch.sum((actions - self.centroids[:, None]) ** 2, dim=2)
        # Get the index of the closest centroid to each action.
        # This is an array of (batch_size,)
        actions = torch.argmin(distances, dim=0)
        return actions
    



