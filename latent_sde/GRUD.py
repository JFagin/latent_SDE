#Joshua Fagin
#Adaptived from: https://github.com/zhiyongc/GRU-D/blob/master/GRUD.py

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math

class FilterLinear(nn.Module):
    def __init__(self, in_features, out_features, filter_square_matrix, bias=True):
        '''
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        '''
        super(FilterLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.filter_square_matrix = filter_square_matrix
        
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.filter_square_matrix.mul(self.weight), self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'
        
class GRUD(nn.Module):
    def __init__(self, input_size, hidden_size, device = torch.device('cpu')):
        super(GRUD, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.delta_size = input_size
        self.mask_size = input_size
        
        self.identity = torch.eye(input_size).to(self.device)
        self.zeros = torch.zeros(input_size).to(self.device)
        self.zeros_h = torch.zeros(hidden_size).to(self.device)
        
        self.zl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
        self.rl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
        self.hl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
        
        self.gamma_x_l = FilterLinear(self.delta_size, self.delta_size, self.identity)
        
        self.gamma_h_l = nn.Linear(self.delta_size, self.hidden_size)
        
    def step(self, x, x_last_obsv, x_mean, h, mask, delta):
        
        delta_x = torch.exp(-torch.max(self.zeros, self.gamma_x_l(delta)))
        delta_h = torch.exp(-torch.max(self.zeros_h, self.gamma_h_l(delta)))
        
        x = mask * x + (1 - mask) * (delta_x * x_last_obsv + (1 - delta_x) * x_mean)
        h = delta_h * h
        
        combined = torch.cat((x, h, mask), 1)
        z = torch.sigmoid(self.zl(combined))
        r = torch.sigmoid(self.rl(combined))
        combined_r = torch.cat((x, r * h, mask), 1)
        h_tilde = torch.tanh(self.hl(combined_r))
        h = (1 - z) * h + z * h_tilde
        
        return h
    
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        input_dim = x.size(2)

        # Get the mean of the observed values
        x_mean = torch.sum(x,dim=1)/torch.sum((x != 0.0).type_as(x),dim=1) 

        h = torch.zeros(batch_size, self.hidden_size).to(self.device)
        
        x_tm1 = torch.zeros(batch_size,input_dim).to(self.device)
        delta_t = torch.zeros(batch_size,input_dim).to(self.device)
        
        time_step = 1.0/seq_len
        obs_mask_t = 0.0
        outputs = None
        for t in range(seq_len):

            x_t = x[:,t] 
            
            if t > 0:
                delta_t = (1.0-obs_mask_t)*delta_t + time_step
            
            obs_mask_t = (x_t != 0.0).type_as(x_t) 

            x_tm1 = torch.where(obs_mask_t>0.0,x_t,x_tm1) 

            h = self.step(x_t, x_tm1, x_mean, h, obs_mask_t, delta_t)
                
            if outputs is None:
                outputs = h.unsqueeze(1)
            else:
                outputs = torch.cat((outputs, h.unsqueeze(1)), 1)

        return outputs,  h