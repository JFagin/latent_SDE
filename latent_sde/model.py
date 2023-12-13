# Joshua Fagin
# Ji Won Park: older vserion of code https://github.com/jiwoncpark/magnify 
# Also inspired by https://github.com/google-research/torchsde/blob/master/examples/latent_sde_lorenz.py

import torch
from torch import nn
from torch.distributions import Normal
import torchsde
from latent_sde.GRUD import GRUD
import torch.nn.functional as F
import logging
from math import log
import numpy as np


class Encoder(nn.Module):
    """
    Encode the input data X into a context vector
    """
    def __init__(self, input_size, hidden_size, output_size, device, dropout_rate=0.0,num_RNN_layers=1):
        """
        input_size: int, dimension of the input data
        hidden_size: int, dimension of the hidden state
        output_size: int, dimension of the context vector
        device: torch.device, device to run the model on (cpu or cuda)
        dropout_rate: float, dropout rate, should be between 0 and 1
        num_RNN_layers: int, number of GRU layers to use. Should be at least 1. The first RNN layer is always the GRU-D layer.
        """
        super(Encoder, self).__init__()

        self.grud = nn.ModuleList()
        self.grud.append(GRUD(input_size, hidden_size, device=device))
        self.grud.append(nn.LayerNorm(hidden_size))
        self.grud.append(nn.Dropout(dropout_rate))

        self.num_RNN_layers = num_RNN_layers
        if num_RNN_layers > 1:
            self.gru = nn.ModuleList()
            for i in range(num_RNN_layers-1):
                self.gru.append(nn.GRU(hidden_size, hidden_size, batch_first=True))
                self.gru.append(nn.LayerNorm(hidden_size))
                self.gru.append(nn.Dropout(dropout_rate))
        
        self.linear_layers = nn.ModuleList()
        self.linear_layers.append(nn.Linear(hidden_size, hidden_size))
        self.linear_layers.append(nn.LeakyReLU())
        self.linear_layers.append(nn.LayerNorm(hidden_size))
        self.linear_layers.append(nn.Dropout(dropout_rate))
        self.linear_layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, X):
        """
        X: torch.Tensor, input data, shape [B, T, input_size], where B is the batch size, T is the number of time steps, and input_size is the dimension of the input data (brightness and uncertainty)
        
        Here we include a residual connection from the input to the output of the GRU-D layer and additional hidden GRU layer.
        """
        X = X.permute(1,0,2) # [T, B, input_size]

        for layer in self.grud:
            if isinstance(layer, GRUD):
                X, _ = layer(X)
            else:
                X = layer(X)

        if self.num_RNN_layers > 1:
            for layer in self.gru:
                if isinstance(layer, nn.GRU):
                    skip, _ = layer(X)
                else:
                    skip = layer(skip)

        for i,layer in enumerate(self.linear_layers):
            if i == 0 and self.num_RNN_layers > 1:
                X = layer(X+skip)
            else:
                X = layer(X)

        return X.permute(1,0,2) # [T, B, output_size]


class ResMLP(nn.Module):
    """
    Residual MLP that predicts the target quantities
    """
    def __init__(self, dim_in, dim_out, dim_hidden=64, activation='leaky_relu', dropout_rate=0.0):
        super(ResMLP, self).__init__()
        """
        dim_in: int, dimension of input
        dim_out: int, dimension of output
        dim_hidden: int, dimension of hidden layers
        activation: str, type of activation, should be one of ['leaky_relu', 'softplut'] or defaults to Relu
        dropout_rate: float, dropout rate, should be between 0 and 1
        """
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.dropout_rate = dropout_rate
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        else:
            self.activation = nn.ReLU()

        self.pre_skip = nn.Sequential(nn.Linear(self.dim_in, self.dim_hidden),
                                      self.activation,
                                      nn.LayerNorm(self.dim_hidden),
                                      nn.Dropout(self.dropout_rate),
                                      nn.Linear(self.dim_hidden, self.dim_hidden),
                                      self.activation,
                                      nn.LayerNorm(self.dim_hidden),
                                      nn.Dropout(self.dropout_rate),
                                      nn.Linear(self.dim_hidden, self.dim_in),
                                      nn.LayerNorm(self.dim_in),
                                      )
                                      
        self.post_skip = nn.Sequential(nn.Linear(self.dim_in, self.dim_hidden),
                                       self.activation,
                                       nn.LayerNorm(self.dim_hidden),
                                       nn.Dropout(self.dropout_rate),
                                       nn.Linear(self.dim_hidden, self.dim_out)
                                       )

    def forward(self, z0):
        # z0 ~ [B, dim_in]
        out = self.pre_skip(z0)  # [B, dim_in]
        out = out + z0  # [B, dim_in], skip connection
        out = self.post_skip(out)  # projector

        return out

class Projector(nn.Module):
    def __init__(self, latent_dim, num_bands, device, uncertainty_output=True, dim_hidden=64, RNN_decoder=False, num_RNN_layers=1,dropout_rate=0.0,include_latent=True):
        """
        Projector that maps the output of the neural SDE to the target quantities

        latent_dim : int, dimension of the latent vector'
        num_bands : int, number of target quantities to predict 
        device : torch.device, device to run the model on
        uncertainty_output : bool, whether to output the uncertainty of the target quantities
        dim_hidden : int, dimension of the hidden layers
        RNN_decoder : bool, whether to use an RNN decoder
        num_RNN_layers : int, number of RNN layers. If RNN_decoder is False, this is ignored.
        dropout_rate : float, dropout rate. Should be between 0 and 1.
        include_latent : bool, whether to include the latent vector in the output of the projector.
        """
        super(Projector, self).__init__()

        dim_out = num_bands
        if uncertainty_output:
            dim_out += num_bands

        dim_in = latent_dim+num_bands
        if include_latent:
            dim_in += latent_dim
        self.include_latent = include_latent

        self.lin_skip = nn.ModuleList()
        self.lin_skip.append(nn.Linear(dim_in, dim_hidden))
        self.lin_skip.append(nn.LeakyReLU())
        self.lin_skip.append(nn.LayerNorm(dim_hidden))
        self.lin_skip.append(nn.Dropout(dropout_rate))
        self.lin_skip.append(nn.Linear(dim_hidden, dim_out))

        self.layers = nn.ModuleList()
        if RNN_decoder:
            for i in range(num_RNN_layers):
                if i ==0:
                    self.layers.append(GRUD(dim_in, dim_hidden, device=device))
                else:
                    self.layers.append(nn.GRU(dim_hidden, dim_hidden, batch_first=True))
                self.layers.append(nn.LayerNorm(dim_hidden))
                self.layers.append(nn.Dropout(dropout_rate))
        else:
            for i in range(num_RNN_layers):
                if i ==0:
                    self.layers.append(nn.Linear(dim_in, dim_hidden))
                else:
                    self.layers.append(nn.Linear(dim_hidden, dim_hidden))    
                self.layers.append(nn.LeakyReLU())
                self.layers.append(nn.LayerNorm(dim_hidden))
                self.layers.append(nn.Dropout(dropout_rate))
        self.layers.append(nn.Linear(dim_hidden, dim_out))

    def forward(self, X, z0, input_uncertainty):
        """
        X: light curve, has shape [T, B, dim_in] where dim_in is the latent dimension
        z0: latent vector, has shape [B, dim_in] where dim_in is the latent dimension. Only used if include_latent=True
        input_uncertainty: input_uncertainty for the light curve (photometric and systematic noise), has shape [T, B, num_bands]
        """

        if self.include_latent:
            X = torch.cat((X, z0.unsqueeze(0).repeat(X.shape[0],1,1)), dim=2) # [T, B, dim_in+dim_in]
        X = torch.cat((X, input_uncertainty), dim=2) # [T, B, num_bands+num_bands+latent_dim]
        X = X.permute(1,0,2) # [B, T, num_bands+num_bands+latent_dim]

        for i,layer in enumerate(self.lin_skip):
            if i==0:
                skip = layer(X) 
            else:
                skip = layer(skip)

        for layer in self.layers:
            if isinstance(layer, nn.GRU) or isinstance(layer, GRUD):
                X, _ = layer(X)
            else:
                X = layer(X)

        X = X+skip # [B, T, dim_out], skip connection

        return X.permute(1,0,2) # [T, B, output_size]

class LatentSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, data_size, latent_size, context_size, hidden_size,
                 n_params, device, include_prior_drift=True,dt=1.0e-3,anneal_dt=False,uncertainty_output=True,
                 dropout_rate=0.0, normalize=True,num_RNN_layers=1,
                 use_SDE=True,num_epoch=0,RNN_decoder=False,num_RNN_layers_decoder=1,give_redshift=False,self_supervised=False):
        super(LatentSDE, self).__init__()

        self.epoch = 0
        self.include_prior_drift = include_prior_drift
        self.dt = dt
        self.anneal_dt = anneal_dt
        self.give_redshift = give_redshift
        self.uncertainty_output = uncertainty_output

        self.num_epoch = num_epoch
        self.normalize = normalize
        self.dropout_rate = dropout_rate
        self.use_SDE = use_SDE
        self.RNN_decoder = RNN_decoder
        self.num_RNN_layers = num_RNN_layers
        self.num_RNN_layers_decoder = num_RNN_layers_decoder
        self.latent_size = latent_size
        self.self_supervised = self_supervised

        self.self_supervised_fraction_max = 0.4 # Max fraction of the LC observations that are used for supervision if self_supervised=True.

        # Encoder
        self.encoder = Encoder(input_size=data_size, hidden_size=hidden_size, output_size=context_size, device=device, 
                               dropout_rate=self.dropout_rate, num_RNN_layers=self.num_RNN_layers)

        # include the hidden state of the encoder as well as the mean and stdev of the LC and uncertainties
        if uncertainty_output:
            qz0_net_output_size = 4*latent_size
        else:
            qz0_net_output_size = 2*latent_size
        qz0_net_input_size = context_size+context_size+context_size + 2 # +2 for the mean and stdev of the LC
        if give_redshift:
            qz0_net_input_size += 1 # add one for the redshift

        self.qz0_net = nn.Sequential(
            nn.Linear(qz0_net_input_size,hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_size, qz0_net_output_size),
        )

        # Decoder.
        self.f_net = ResMLP(latent_size + context_size, latent_size, dim_hidden=hidden_size, dropout_rate=self.dropout_rate, activation='leaky_relu')
        self.h_net = ResMLP(latent_size, latent_size, dim_hidden=hidden_size, dropout_rate=self.dropout_rate, activation='leaky_relu')
 
        # This needs to be an element-wise function for the SDE to satisfy diagonal noise.
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.LeakyReLU(),
                    nn.LayerNorm(hidden_size),
                    nn.Dropout(self.dropout_rate),

                    nn.Linear(hidden_size, hidden_size),
                    nn.LeakyReLU(),
                    nn.LayerNorm(hidden_size),
                    nn.Dropout(self.dropout_rate),

                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid()
                )
                for _ in range(latent_size)
            ]
        )
        self.out_dim = data_size//2
        if self.use_SDE:
            self.projector = Projector(latent_size, self.out_dim, device, uncertainty_output, dim_hidden=hidden_size, RNN_decoder=self.RNN_decoder,
                                       num_RNN_layers=num_RNN_layers_decoder,dropout_rate=self.dropout_rate,include_latent=True)

        param_mlp_dim_in = latent_size
        if self.include_prior_drift:
            param_mlp_dim_in += 4*latent_size + context_size + context_size + context_size + 2 # +2 for the mean and stdev of the LC
        if uncertainty_output:
            param_mlp_dim_in += 2*latent_size
        if give_redshift:
            param_mlp_dim_in += 1 # add one for the redshift

        # mean vector for n_params parameters
        # correlation matrix of size n_params*n_params with n_params*(n_params-1)/2 free parameters
        num_output = n_params + n_params*(n_params+1)//2

        self.param_mlp = ResMLP(param_mlp_dim_in, num_output, dim_hidden=hidden_size, activation='leaky_relu', dropout_rate=self.dropout_rate)

        if uncertainty_output:
            self.pz0_mean = nn.Parameter(torch.zeros(1, 2*latent_size))
            self.pz0_logstd = nn.Parameter(torch.zeros(1, 2*latent_size))            
        else:
            self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
            self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        self._ctx = None

        self.epsilon = 0.001 # epsilon so our predicted uncertainty is never 0.

    def toggle_SDE(self, use_SDE):
        """
        Trun on/off the SDE. If the SDE is turned off, the model just predicts parameters (i.e. no LC reconstruction).
        """
        self.use_SDE = use_SDE

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_dt(self):
        """
        Anneal the time step for faster training. This is only done if the epoch or num_epoch is updated.
        """
        if self.anneal_dt:
            if self.epoch <= 0.25*self.num_epoch:
                return 2.5*self.dt
            elif self.epoch <= 0.5*self.num_epoch:
                return 2.0*self.dt
            elif self.epoch <= 0.75*self.num_epoch:
                return 1.5*self.dt
            else:
                return self.dt
        else:
            return self.dt

    def contextualize(self, ctx):
        """Set the context vector, which is an encoding of the observed
        sequence

        Parameters
        ----------
        ctx : tuple
            A tuple of tensors of sizes (T,), (T, batch_size, d)
        """
        self._ctx = ctx

    def f(self, t, y):
        """Network that decodes the latent and context
        (posterior drift function)
        Time-inhomogeneous (see paper sec 9.12)

        """
        ts, ctx = self._ctx
        # ts ~ [T]
        # ctx ~ [T, B, context_size]
        ts = ts.to(t.device)
        # searchsorted output: if t were inserted into ts, what would the
        # indices have to be to preserve ordering, assuming ts is sorted
        # training time: t is tensor with no size (scalar)
        # inference time: t ~ [num**2, 1] from the meshgrid

        i = min(torch.searchsorted(ts, t.min(), right=True), len(ts) - 1)

        # training time: y ~ [B, latent_dim]
        # inference time: y ~ [num**2, 1] from the meshgrid
        
        f_in = torch.cat((y, ctx[i]), dim=1) 
        
        # Training time (for each time step)
        # t ~ []
        # y ~ [B, latent_dim]
        # f_in ~ [B, latent_dim + context_dim]
        f_out = self.f_net(f_in)
        # f_out ~ [B, latent_dim]
        return f_out

    def h(self, t, y):
        """Network that decodes the latent
        (prior drift function)

        """
        return self.h_net(y)

    def g(self, t, y):
        """Network that decodes each time step of the latent
        (diagonal diffusion)

        """
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        g_out = torch.cat(out, dim=1)  # [B, latent_dim]
        return g_out

    def forward(self, xs, true_LC, ts, redshift=None, adjoint=False, method="euler"):
        """
        xs ~ [T, B, 2*num_bands], observed data with uncertainty and unobserved points set to zero.
        true_LC ~ [T, B, num_bands], True light curve with no uncertainty and no unobserved points.
        ts ~ [T] # Time steps across the batch. Here ts is constant and we use masking in xs instead.
        redshift ~ [B], redshift of each object in the batch. Default: None.
        adjoint ~ bool, whether to use adjoint method for neural SDE. Default: False, adjoint = True is much slower.
        method ~ str, which method to use for neural SDE. Options: "euler", "milstein", "srk". Default: "euler".
        
        # Example plotting for debugging. 
        # This plots the first first band with error bars and the true light curve:
        import matplotlib.pyplot as plt
        cadence = 3.0 # Time between observations in days
        plt.errorbar(cadence*np.arange(xs.shape[0]),xs[:,0,0].cpu().numpy(),yerr=xs[:,0,6].cpu().numpy(),fmt='o',label='observed')
        plt.plot(cadence*np.arange(xs.shape[0]),true_LC[:,0,0].cpu().numpy(),label='true')
        plt.legend()
        plt.xlabel('time [days]')
        plt.ylabel('magnitude')
        plt.show()
        """
        if self.give_redshift:
            assert redshift is not None, "redshift must be given if give_redshift is True"
            redshift = redshift.unsqueeze(-1)

        #gets the mean of the light curve for each band/uncertainty
        xs = torch.clone(xs)

        if self.self_supervised:
            # additional mask for part of xs to use portion of observed data for self-supervised learning

            # Define the probability of zeros
            p = self.self_supervised_fraction_max * torch.rand(xs.shape[1]).to(xs.device) # random probability for each object in the batch
            #p = 0.0 * torch.rand(xs.shape[1]).to(xs.device) # random probability for each object in the batch
            
            p = p.view(1, xs.shape[1], 1) 
            # Create a tensor with random values between 0 and 1 of the same shape as xs
            random_tensor = torch.rand(xs[:,:,:self.out_dim].shape).to(xs.device)

            # Threshold the tensor based on the desired probability of zeros
            mask_supervision = torch.where(random_tensor < p, torch.zeros_like(random_tensor).to(xs.device), torch.ones_like(random_tensor).to(xs.device)).type_as(xs)

            mask_supervision = mask_supervision.repeat(1,1,2) # repeat for each band
            # get masked out tensor

            xs_supervise_part = xs * (1.0-mask_supervision)

            # Apply the mask to the input tensor
            xs = xs * mask_supervision
            
            mask_supervision = (xs_supervise_part[:,:,:self.out_dim] != 0.0).type_as(xs) # shape [T, B, num_bands]

        mask = (xs[:,:,:self.out_dim] != 0.0).type_as(xs) # shape [T, B, num_bands]

        mean = torch.sum(xs[:,:,:self.out_dim],dim=0)/torch.sum(mask,dim=0)

        #now get the std masking all the zeros of xs
        mean_diff = mask*(xs[:,:,:self.out_dim] - mean)**2 # mask out the zeros again
        std = torch.sqrt(torch.sum(mean_diff,dim=0)/torch.sum(mask,dim=0))
        
        mean = mean.mean(dim=1).unsqueeze(-1) # shape [B, 1, 1], averge over the bands
        std = std.mean(dim=1).unsqueeze(-1) # shape [B, 1, 1], averge over the bands

        #now normalize the light curve so each band has mean 0 and std 1
        if self.normalize:
            
            xs[:,:,:self.out_dim] = mask*((xs[:,:,:self.out_dim] - mean)/std)
            xs[:,:,self.out_dim:] = mask*(xs[:,:,self.out_dim:]/std)
            
            
        # Encoder
        ctx = self.encoder(torch.flip(xs, dims=(0,)))  # [T, B, context_size] 
        
        num_nan = torch.isnan(ctx).sum()
        if num_nan > 0:
            logging.warning(f"{num_nan} nan detected in ctx")
        ctx = torch.nan_to_num(torch.flip(ctx, dims=(0,)))
        self.contextualize((ts, ctx))
        
        # Get the latent vector
        qz0_input = torch.cat((ctx[0],ctx.mean(dim=0),ctx.std(dim=0),mean,std),dim=1)
        if self.give_redshift:
            # include redshift in the input to the qz0_net
            qz0_input = torch.cat((qz0_input,redshift),dim=1)
        qz0_mean, qz0_logstd = self.qz0_net(qz0_input).chunk(chunks=2, dim=1) # ([B, latent_dim], [B, latent_dim])
        z0 = qz0_mean + qz0_logstd.exp()*torch.randn_like(qz0_mean) # [B, latent_dim]

        if self.uncertainty_output:
            z0_SDE = z0[:, :self.latent_size] # [B, latent_dim]
            z0_RNN = z0[:, self.latent_size:] # [B, latent_dim]
        else:
            z0_SDE = z0
            z0_RNN = z0

        # Decoder
        if self.use_SDE:
            # Integrate the SDE
            if adjoint:
                # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
                adjoint_params = (
                        (ctx,) +
                        tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(self.h_net.parameters())
                )
                zs, log_ratio = torchsde.sdeint_adjoint(
                    self, z0_SDE, ts, adjoint_params=adjoint_params, dt=self.get_dt(),
                    logqp=True, method=method)
                # zs ~ [T, B, latent_dim] 
                # log_ratio ~ [T-1, B]
            else:
                zs, log_ratio = torchsde.sdeint(self, z0_SDE, ts, dt=self.get_dt(), logqp=True, method=method)

            num_nan = torch.isnan(zs).sum()
            if num_nan > 0:
                logging.warning(f"{num_nan} nan detected in zs")

            #project to the predicted light curve. 
            _xs = self.projector(torch.nan_to_num(zs),z0_RNN,xs[:,:,self.out_dim:]) #[T,B,num_bands]
            _xs = torch.nan_to_num(_xs, nan=0.0, posinf=0.0, neginf=0.0) # nan_to_num is not really needed here, but just in case
            # if self.uncertainty_output: then _xs ~ [T, B, 2*num_bands], else _xs ~ [T, B, num_bands]
            # mean = _xs[:,:,:self.out_dim] and log_var = _xs[:,:,self.out_dim:]

            #now unnormalize the light curve
            if self.normalize:
                xs[:,:,:self.out_dim] = mask*(xs[:,:,:self.out_dim]*std + mean)
                xs[:,:,self.out_dim:] = mask*(xs[:,:,self.out_dim:]*std)


                if self.uncertainty_output:
                    # normalize the mean
                    _xs[:,:,:self.out_dim] = _xs[:,:,:self.out_dim]*std + mean
                    # normalize the log variance
                    _xs[:,:,self.out_dim:] = _xs[:,:,self.out_dim:]+2*torch.log(std) # log(a^2std^2) = log(a^2)+2*log(b) , b > 0
                else:
                    _xs = _xs*std + mean
            # _xs ~ [T, B, Y_out_dim] 
            #xs_dist = Normal(loc=_xs,scale=self.extra_logstd.exp()+xs[:,:,self.out_dim:]+self.epsilon) #Add epsilon to avoid zero scales from the mask

            if self.uncertainty_output:
                # Gaussian log-likelihood = 0.5*((x-mu)^2/sigma^2 + log(sigma^2) + log(2pi))
                if self.self_supervised:

                    log_pxs = 0.5*mask_supervision*(((xs_supervise_part[:,:,:self.out_dim]-_xs[:,:,:self.out_dim])**2)/(torch.exp(_xs[:,:,self.out_dim:])+xs_supervise_part[:,:,self.out_dim:]**2+1e-7)+torch.log(torch.exp(_xs[:,:,self.out_dim:])+xs_supervise_part[:,:,self.out_dim:]**2+1e-7))
                    log_pxs = log_pxs.sum()/mask_supervision.sum()+np.log(2*np.pi)
                    #print(log_pxs)
                else:
                    # use the true light curve
                    log_pxs = 0.5*(((true_LC-_xs[:,:,:self.out_dim])**2)/torch.exp(_xs[:,:,self.out_dim:])+_xs[:,:,self.out_dim:]+np.log(2*np.pi)).mean() 
                
                xs_dist = Normal(loc=_xs[:,:,:self.out_dim],scale=xs[:,:,self.out_dim:]+self.epsilon)
                log_pxs2 = -(mask*xs_dist.log_prob(xs[:,:,:self.out_dim])).sum()/mask.sum()

                log_pxs = log_pxs + log_pxs2

                #if you wanted to add the predicted uncertainty as Gaussian noise
                #_xs += torch.exp(log_var).sqrt()*torch.randn_like(_xs) #add the predicted uncertainty as Gaussian noise
                
            else:
                xs_dist = Normal(loc=_xs,scale=xs[:,:,self.out_dim:]+self.epsilon)
                log_pxs = -(mask*xs_dist.log_prob(xs[:,:,:self.out_dim])).sum()/mask.sum()  # scalar. Negative log-likelihood.

            # Sum across times and dimensions, mean across examples in batch
            qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
            pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())      
            logqp0 = torch.distributions.kl_divergence(qz0, pz0).mean()
            logqp_path = log_ratio.mean()
        else:
            _xs = 0 
            log_pxs = 0 
            logqp0 = 0 
            logqp_path = 0 

        # Parameter predictions
        if self.include_prior_drift:
            mlp_input = torch.cat(( #all of the things that the black hole parameters are predicted from:
                                                qz0_mean,
                                                qz0_logstd,
                                                mean, #mean of the light curve for each band/uncertainty
                                                std, #std of the light curve for each band/uncertainty
                                                ctx[0],  # last context point (last output of RNN in the encoder)
                                                ctx.mean(dim=0), # mean of the context points
                                                ctx.std(dim=0), # mean of the context points
                                                self.h(None, z0_SDE),
                                                self.f(ts, z0_SDE),
                                                self.g(None,z0_SDE),
                                                ),
                                                dim=-1)
            if self.give_redshift:
                mlp_input = torch.cat((mlp_input,redshift),dim=1)
        else:
            mlp_input = z0

        param_pred = self.param_mlp(mlp_input)

        # if self.uncertainty_output: then _xs has mean = _xs[:,:,:self.out_dim] and log_var = _xs[:,:,self.out_dim:]
        return log_pxs, logqp0 + logqp_path, param_pred, _xs

    @torch.no_grad()
    def sample_posterior(self, xs, ts, redshift=None, adjoint=False, method='euler'):
        '''
        Sample LC reconstruction predictions from the posterior distribution.

        This includes @torch.no_grad() decorator to avoid unnecessary memory usage and save time during inference.
        '''
        if self.give_redshift:
            assert redshift is not None, "redshift must be given if give_redshift is True"
            redshift = redshift.unsqueeze(-1)

        #gets the mean of the light curve for each band/uncertainty
        xs = torch.clone(xs)
        mask = (xs[:,:,:self.out_dim] != 0.0).type_as(xs) # shape [T, B, num_bands]

        mean = torch.sum(xs[:,:,:self.out_dim],dim=0)/torch.sum(mask,dim=0)

        #now get the std masking all the zeros of xs
        mean_diff = mask*(xs[:,:,:self.out_dim] - mean)**2 # mask out the zeros again
        std = torch.sqrt(torch.sum(mean_diff,dim=0)/torch.sum(mask,dim=0))
        
        mean = mean.mean(dim=1).unsqueeze(-1) # shape [B, num_bands,1], averge over the bands
        std = std.mean(dim=1).unsqueeze(-1) # shape [B, num_bands,1], averge over the bands

        #now normalize the light curve so each band has mean 0 and std 1
        if self.normalize:
            
            xs[:,:,:self.out_dim] = mask*((xs[:,:,:self.out_dim] - mean)/std)
            xs[:,:,self.out_dim:] = mask*(xs[:,:,self.out_dim:]/std)
            
            
        # Encoder
        ctx = self.encoder(torch.flip(xs, dims=(0,)))  # [T, B, context_size] 
        
        num_nan = torch.isnan(ctx).sum()
        if num_nan > 0:
            logging.warning(f"{num_nan} nan detected in ctx")
        ctx = torch.nan_to_num(torch.flip(ctx, dims=(0,)))
        self.contextualize((ts, ctx))
        
        # Get the latent vector
        qz0_input = torch.cat((ctx[0],ctx.mean(dim=0),ctx.std(dim=0),mean,std),dim=1)
        if self.give_redshift:
            # include redshift in the input to the qz0_net
            qz0_input = torch.cat((qz0_input,redshift),dim=1)
        qz0_mean, qz0_logstd = self.qz0_net(qz0_input).chunk(chunks=2, dim=1) # ([B, latent_dim], [B, latent_dim])
        z0 = qz0_mean + qz0_logstd.exp()*torch.randn_like(qz0_mean) # [B, latent_dim]

        if self.uncertainty_output:
            z0_SDE = z0[:, :self.latent_size] # [B, latent_dim]
            z0_RNN = z0[:, self.latent_size:] # [B, latent_dim]
        else:
            z0_SDE = z0
            z0_RNN = z0

        # Decoder
        if self.use_SDE:
            # Integrate the SDE
            if adjoint:
                # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
                adjoint_params = (
                        (ctx,) +
                        tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(self.h_net.parameters())
                )
                zs, log_ratio = torchsde.sdeint_adjoint(
                    self, z0_SDE, ts, adjoint_params=adjoint_params, dt=self.get_dt(),
                    logqp=True, method=method)
                # zs ~ [T, B, latent_dim] 
                # log_ratio ~ [T-1, B]
            else:
                zs, log_ratio = torchsde.sdeint(self, z0_SDE, ts, dt=self.get_dt(), logqp=True, method=method)

            num_nan = torch.isnan(zs).sum()
            if num_nan > 0:
                logging.warning(f"{num_nan} nan detected in zs")

            #project to the predicted light curve. 
            _xs = self.projector(torch.nan_to_num(zs),z0_RNN,xs[:,:,self.out_dim:]) #[T,B,num_bands]
            _xs = torch.nan_to_num(_xs, nan=0.0, posinf=0.0, neginf=0.0) # nan_to_num is not really needed here, but just in case

            #now unnormalize the light curve
            if self.normalize:
                if self.uncertainty_output:
                    # normalize the mean
                    _xs[:,:,:self.out_dim] = _xs[:,:,:self.out_dim]*std + mean
                    # normalize the log variance
                    _xs[:,:,self.out_dim:] = _xs[:,:,self.out_dim:]+2*torch.log(std) # log(a^2std^2) = log(a^2)+2*log(b) , b > 0
                else:
                    _xs = _xs*std + mean

        else:
            _xs = 0 #torch.zeros(1).to(xs.device)

        # if self.uncertainty_output: then _xs has mean = _xs[:,:,:self.out_dim] and log_var = _xs[:,:,self.out_dim:]
        return _xs

    @torch.no_grad()
    def sample_parameter_posterior(self, xs, ts, redshift=None):
        '''
        Sample parameter predictions from the posterior distribution.

        This includes @torch.no_grad() decorator to avoid unnecessary memory usage and save time during inference.
        '''
        if self.give_redshift:
            assert redshift is not None, "redshift must be given if give_redshift is True"
            redshift = redshift.unsqueeze(-1)

        #gets the mean of the light curve for each band/uncertainty
        xs = torch.clone(xs)
        mask = (xs[:,:,:self.out_dim] != 0.0).type_as(xs) # shape [T, B, num_bands]

        mean = torch.sum(xs[:,:,:self.out_dim],dim=0)/torch.sum(mask,dim=0)

        #now get the std masking all the zeros of xs
        mean_diff = mask*(xs[:,:,:self.out_dim] - mean)**2 # mask out the zeros again
        std = torch.sqrt(torch.sum(mean_diff,dim=0)/torch.sum(mask,dim=0))
        
        mean = mean.mean(dim=1).unsqueeze(-1) # shape [B, num_bands,1], averge over the bands
        std = std.mean(dim=1).unsqueeze(-1) # shape [B, num_bands,1], averge over the bands

        #now normalize the light curve so each band has mean 0 and std 1
        if self.normalize:
            
            xs[:,:,:self.out_dim] = mask*((xs[:,:,:self.out_dim] - mean)/std)
            xs[:,:,self.out_dim:] = mask*(xs[:,:,self.out_dim:]/std)
            
            
        # Encoder
        ctx = self.encoder(torch.flip(xs, dims=(0,)))  # [T, B, context_size] 
        
        num_nan = torch.isnan(ctx).sum()
        if num_nan > 0:
            logging.warning(f"{num_nan} nan detected in ctx")
        ctx = torch.nan_to_num(torch.flip(ctx, dims=(0,)))
        self.contextualize((ts, ctx))
        
        # Get the latent vector
        qz0_input = torch.cat((ctx[0],ctx.mean(dim=0),ctx.std(dim=0),mean,std),dim=1)
        if self.give_redshift:
            # include redshift in the input to the qz0_net
            qz0_input = torch.cat((qz0_input,redshift),dim=1)
        qz0_mean, qz0_logstd = self.qz0_net(qz0_input).chunk(chunks=2, dim=1) # ([B, latent_dim], [B, latent_dim])
        z0 = qz0_mean + qz0_logstd.exp()*torch.randn_like(qz0_mean) # [B, latent_dim]

        if self.uncertainty_output:
            z0_SDE = z0[:, :self.latent_size] # [B, latent_dim]
            z0_RNN = z0[:, self.latent_size:] # [B, latent_dim]
        else:
            z0_SDE = z0
            z0_RNN = z0  

        # Parameter predictions
        if self.include_prior_drift:
            mlp_input = torch.cat(( #all of the things that the black hole parameters are predicted from:
                                                qz0_mean,
                                                qz0_logstd,
                                                mean, #mean of the light curve for each band/uncertainty
                                                std, #std of the light curve for each band/uncertainty
                                                ctx[0],  # last context point (last output of RNN in the encoder)
                                                ctx.mean(dim=0), # mean of the context points
                                                ctx.std(dim=0), # mean of the context points
                                                self.h(None, z0_SDE),
                                                self.f(ts, z0_SDE),
                                                self.g(None,z0_SDE),
                                                ),
                                                dim=-1)
            if self.give_redshift:
                mlp_input = torch.cat((mlp_input,redshift),dim=-1)
        else:
            mlp_input = z0

        param_pred = self.param_mlp(mlp_input)

        return param_pred