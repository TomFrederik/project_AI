import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from torch.cuda.amp import autocast 
from torch.optim import AdamW


import util_models
from torchvision.transforms import ToPILImage # used to transform input to VAE to uint image
from deepspeed.ops.adam import  DeepSpeedCPUAdam, FusedAdam
import deepspeed


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
        #padding = (kernel_size - 1) // 2
        
        num_channels = [3] + num_channels
        model_list = []
        self.img_shape_list = [img_shape]
        for i in range(len(num_channels)-1):
            model_list.append(nn.Conv2d(in_channels=num_channels[i], out_channels=num_channels[i+1], kernel_size=kernel_size, stride=2))#, padding=padding))
            model_list.append(nn.BatchNorm2d(num_features=num_channels[i+1])) # BatchNorm
            model_list.append(nn.GELU()) # activation function
            #model_list.append(nn.MaxPool2d(kernel_size=kernel_size, stride=2)) # MaxPooling
            h_new = np.floor((self.img_shape_list[-1][0] - kernel_size)/2 + 1).astype(int)
            self.img_shape_list.append((h_new, h_new))

        print('Encoder img_shapes:', self.img_shape_list)
        
        self.conv_net = nn.Sequential(*model_list)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear = nn.Linear(self.img_shape_list[-1][0] * self.img_shape_list[-1][1] * num_channels[-1], 2*latent_dim)

    def forward(self, x):
        '''
        x - input, for mineRL (B, C, H, W) batch of frames
        '''
        conv_out = self.conv_net(x)
        latent_out = self.linear(self.flatten(conv_out))

        mean, log_std = torch.chunk(latent_out, chunks=2, dim=-1)
        
        return mean, log_std

class ConvDecoder(nn.Module):

    def __init__(self, latent_dim, num_channels, kernel_size, img_shape_list):
        '''
        Args:
            latent_dim - dimension of the latent space
            num_channels - list of ints describing the number of channels after each convolution. 
                           len(num_channels) is used to determine number of layers.
            kernel_size - int, size of the kernels used in the convolutions
            list of img_shapes - [..., (x,x), (64,64)], list of image shapes
        '''
        super().__init__()
        
        
        print(f'\nImage sizes for deconv: {img_shape_list}')
        self.img_shape_list = img_shape_list

        # compute padding for same-padding
        padding = (kernel_size - 1) // 2

        self.linear = nn.Sequential(nn.Linear(latent_dim, np.prod(img_shape_list[0])*num_channels[0]), nn.GELU())
        
        layer_list = []
        for i in range(1,len(num_channels)):
            layer_list.append(nn.UpsamplingNearest2d(img_shape_list[i]))
            layer_list.append(nn.Conv2d(num_channels[i-1], num_channels[i], kernel_size=kernel_size, padding=padding))
            layer_list.append(nn.GELU())
        layer_list.append(nn.UpsamplingNearest2d(img_shape_list[-1]))

        # conv to 3 * 255 channels
        layer_list.append(nn.Conv2d(num_channels[-1], 768, kernel_size=kernel_size, padding=padding))
        layer_list.append(util_models.SplitChannelsFromClasses(num_channels=3))
        self.deconv = nn.Sequential(*layer_list)
    
    def forward(self, z):
        '''
        z - latent vector
        '''
        
        out = self.linear(z)
        out = out.reshape((z.shape[0], -1, *self.img_shape_list[0]))
        return self.deconv(out)


class VAE(pl.LightningModule):
    '''
    A base class for VAEs
    '''
    def __init__(self, encoder_class, decoder_class, encoder_kwargs, decoder_kwargs, learning_rate, scheduler_kwargs, batch_size, beta=1):
        '''
        beta - positive float which controls the strength of the regularization, as described in beta-VAE paper. 
        '''
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.scheduler_kwargs = scheduler_kwargs
        self.beta = beta

        self.encoder = encoder_class(**encoder_kwargs)
        if decoder_class == ResnetDecoder:
            self.decoder = decoder_class(**decoder_kwargs)
        else:
            img_shape_list=self.encoder.img_shape_list.copy()
            img_shape_list.reverse()
            self.decoder = decoder_class(**decoder_kwargs, img_shape_list=img_shape_list)

        self.CE_loss = nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = nn.MSELoss()

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
        #sampled_x = self.decoder(z)
        return sampled_x.float() / 255
        #return sampled_x

    @torch.no_grad()
    def encode_only(self, x):
        '''
        Encodes images into their latent space (via sampling).
        Does not track gradients. Use e.g. as preprocessing/embedding.
        Args:
            x - input, for mineRL (B, C, H, W) batch of frames
        Returns:
            mean
            std
            sample - tensor of shape (B, L), where L is the latent dimension
        '''
        mean, log_std = self.encoder(x)
        sample = self.sample(mean, log_std)

        return mean, torch.exp(log_std), sample

    @torch.no_grad()
    def decode_only(self, z):
        # decode
        probs = torch.nn.functional.softmax(self.decoder(z), dim=1)

        # sample from the 255-category distribution
        sampled_x = torch.multinomial(probs.transpose(0,1).flatten(start_dim=1).transpose(0,1), 1).squeeze().reshape((-1, 3, 64, 64))
        #sampled_x = self.decoder(z)

        return sampled_x.float() / 255

    def forward(self, x):
        '''
        x - input, for mineRL (B, C, H, W) batch of frames
        '''
        # encode
        x.requires_grad = True
        
        #mean, log_std = deepspeed.checkpointing.checkpoint(self.encoder, x)
        mean, log_std = self.encoder(x)
        
        # compute KL difstance, i.e. regularization loss
        L_regul = (0.5 * (torch.exp(2 * log_std) + mean ** 2 - 1 - 2 * log_std)).sum(dim=-1).mean()

        # sample latent vector
        z = self.sample(mean, log_std)
        # decode
        #x_new = deepspeed.checkpointing.checkpoint(self.decoder, z)
        x_new = self.decoder(z)
        
        # convert x to classes
        x = (x * 255).type(torch.long)

        # make sure that x_new is fp32, otherwise nan error in loss
        #x_new = x_new.float()

        # compute reconstruction loss, sum over all dimension except batch
        L_reconstr = self.CE_loss(x_new, x).sum(dim=list(range(1,len(x.shape)))).mean()
        #L_reconstr = self.mse_loss(x_new,x)
        print(f"L_reconstr = {L_reconstr.item()}")

        # compute elbo
        elbo =  L_reconstr + self.beta * L_regul

        # convert into bits per dimension loss
        bpd = (elbo * np.log2(np.exp(1)) / np.prod(x.shape[1:])).mean()

        return L_reconstr, L_regul, bpd
    
    def configure_optimizers(self):
        # set up optimizer
        optimizer =  AdamW(self.parameters(), lr=self.learning_rate)
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
        print(f'Step {self.global_step + 1}: bpd = {bpd.mean().item()}')
        
        self.log('Training/batch_bpd', bpd.mean().item(),on_step=True)

        return bpd
    
    def validation_step(self, batch, batch_idx):
        img = batch[1]
        _, _, bpd = self(img)
        self.log('Validation/bpd', bpd.mean().item(), on_epoch=True)
        return bpd
    
class ConvVAE(VAE):

    def __init__(self, encoder_kwargs, decoder_kwargs, learning_rate, scheduler_kwargs, batch_size, beta):
        kwargs = {
            'encoder_class':ConvEncoder,
            'decoder_class':ConvDecoder,
            'encoder_kwargs':encoder_kwargs,
            'decoder_kwargs':decoder_kwargs,
            'learning_rate':learning_rate,
            'scheduler_kwargs':scheduler_kwargs,
            'batch_size':batch_size,
            'beta':beta
        }
        super().__init__(**kwargs)

class ResnetVAE(VAE):
    
    def __init__(self, encoder_kwargs, decoder_kwargs, learning_rate, scheduler_kwargs, batch_size, beta):
        kwargs = {
            'encoder_class':ResnetEncoder,
            'decoder_class':ResnetDecoder,
            'encoder_kwargs':encoder_kwargs,
            'decoder_kwargs':decoder_kwargs,
            'learning_rate':learning_rate,
            'scheduler_kwargs':scheduler_kwargs,
            'batch_size':batch_size,
            'beta':beta
        }
        super().__init__(**kwargs)


class ResnetEncoder(nn.Module):
    '''
    Adapted from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html#ResNet
    '''
    def __init__(self, num_blocks, latent_dim, num_channels, kernel_size, img_shape):
        '''
        Args:
            num_blocks - list of ints describing the number of resnet blocks per group (group = work on img of same width and height)
            latent_dim - dimension of the latent space
            num_channels - list of ints describing the number of channels after each convolution. 
                           len(num_channels) is used to determine number of layers.
            kernel_size - int, size of the kernels used in the convolutions
        '''
        super().__init__()
        assert len(num_blocks) == len(num_channels), f"len(num_blocks) != len(num_channels): = {num_blocks}, {num_channels}"

        # A first convolution on the original image to scale up the channel size
        self.input_net = nn.Sequential(nn.Conv2d(3, num_channels[0], kernel_size=kernel_size, padding=1, bias=False))

        # Creating the ResNet blocks
        blocks = []
        for block_idx, block_count in enumerate(num_blocks):
            for bc in range(block_count):
                subsample = (bc == 0 and block_idx > 0) # Subsample the first block of each group, except the very first one.
                blocks.append(
                    PreActResNetBlock(c_in=num_channels[block_idx if not subsample else (block_idx-1)],
                                             subsample=subsample,
                                             c_out=num_channels[block_idx])
                )
        self.blocks = nn.Sequential(*blocks)

        # Mapping to classification output
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((2,2)),
            nn.Flatten(),
            nn.Linear(num_channels[-1] * 4, 2 * latent_dim)
        )

        '''
        # init parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        '''

    def forward(self, input):
        x = self.input_net(input)
        for b in self.blocks:
            x = b(x)
        x = self.output_net(x)
        mean, log_std = torch.chunk(x, chunks=2, dim=-1)        
        return mean, log_std

class ResnetDecoder(nn.Module):

    def __init__(self, num_blocks, latent_dim, num_channels, kernel_size):
        '''
        Args:
            num_blocks - list of ints describing the number of resnet blocks per group (group = work on img of same width and height)
            latent_dim - dimension of the latent space
            num_channels - list of ints describing the number of channels after each convolution. 
            kernel_size - int, size of the kernels used in the convolutions
        '''
        super().__init__()
        
        # compute padding for same-padding
        padding = (kernel_size - 1) // 2

        self.latent_to_2d = nn.Sequential(nn.Linear(latent_dim, 16*num_channels[0]), nn.GELU())
        
        # Creating the ResNet blocks
        layers = []
        for block_idx, block_count in enumerate(num_blocks):
            if block_idx > 0:
                layers.append(nn.Conv2d(in_channels=num_channels[block_idx-1], out_channels=num_channels[block_idx], kernel_size=1, stride=1, padding=0)) # change channel_dim
            for bc in range(block_count):
                # in decoder, we handle upsampling outside the blocks
                layers.append(PreActResNetBlock(c_in=num_channels[block_idx], c_out=num_channels[block_idx]))
            if block_idx < len(num_blocks)-1:
                layers.append(nn.UpsamplingNearest2d(scale_factor=2))
        layers.append(nn.GELU())


        # conv to 3 * 255 channels
        layers.append(nn.Conv2d(num_channels[-1], 768, kernel_size=kernel_size, padding=padding))
        layers.append(util_models.SplitChannelsFromClasses(num_channels=3))
        #layers.append(nn.Conv2d(num_channels[-1], 3, kernel_size=kernel_size, padding=padding))
        #layers.append(nn.Sigmoid())
        self.deconv = nn.Sequential(*layers)


        '''
        # init parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        '''
    def forward(self, z):
        '''
        z - latent vector
        '''
        
        out = self.latent_to_2d(z)
        out = out.reshape((z.shape[0], -1, 4, 4))
        return self.deconv(out)

class PreActResNetBlock(nn.Module):
    """
    Adapted from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html#ResNet
    """
    def __init__(self, c_in, subsample=False, c_out=-1):
        """
        Pre-activation resnet block
        Inputs:
            c_in - Number of input channels
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()

        if not subsample:
            c_out = c_in

        # Network representing F
        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            nn.GELU(),
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False)
        )
        # 1x1 convolution needs to apply non-linearity as well as not done on skip connection
        self.downsample = nn.Sequential(
            nn.BatchNorm2d(c_in),
            nn.GELU(),
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, bias=False)
        ) if subsample else None

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        return out
