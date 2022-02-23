import numpy as np
import matplotlib.pyplot as plt
import gc, time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


def read_dataset( args ):
    data = np.genfromtxt(args.dataset_file)
    data = ( data - np.mean(data, axis=0) ) / np.std( data, axis=0 )

    np.random.seed(args.seed)
    np.random.shuffle(data)
    
    if len( data.shape ) == 1:
        d = 1
        data = data.reshape((-1, 1))
    else:
        d = len( data[0] )
    
    train_data = data[int(0.1 * len(data)):, :]
    test_data = data[:int(0.1 * len(data)), :]
        
    train_data = torch.from_numpy( train_data )
    test_data = torch.from_numpy( test_data )

    return train_data, test_data, d


class StackedFlow(nn.Module):
    def __init__(self, num_flows, num_hidden_layers, num_hidden_nodes, std_a, initialization, args):
        super(StackedFlow, self).__init__()
        self.num_flows = num_flows

        self.flows_module = nn.ModuleList([])
        for i in range(self.num_flows):
            self.flows_module.append( Flow(  num_hidden_layers, num_hidden_nodes, std_a, initialization, args) )

    def forward(self, inp):
        log_det_jacob = 0
        for i, module_name in enumerate( self.flows_module ):
            inp, log_det_jacob_mod = self.flows_module[i]( inp )
            log_det_jacob += log_det_jacob_mod
        return inp, log_det_jacob



class Flow(nn.Module):
    def __init__(self, num_hidden_layers, num_hidden_nodes, std_a, initialization, args):
        super(Flow, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_nodes = num_hidden_nodes
        self.initialization = initialization
        self.d = args.d
        self.model_structure = args.model_structure
        self.parameterization = args.parameterization

        
        self.w_direction = nn.ParameterList( [None] * ( num_hidden_layers + 1 ) )
        self.w_direction[0] = nn.Parameter( self.lower_dimensional_weight( num_hidden_nodes[0] * self.d , self.d, self.give_std( initialization, num_hidden_nodes[0] ) ) ) #1.0 / np.sqrt( num_hidden_nodes[0] )
        for i in range( num_hidden_layers - 1 ):
            self.w_direction[ i + 1 ] = nn.Parameter( self.lower_dimensional_weight( num_hidden_nodes[i+1] * self.d , num_hidden_nodes[i] * self.d, self.give_std( initialization, num_hidden_nodes[i+1] ) ) ) # 1.0 / np.sqrt( num_hidden_nodes[i+1] )
        

        self.w_direction[ num_hidden_layers ] = nn.Parameter( self.lower_dimensional_weight( self.d, num_hidden_nodes[ self.num_hidden_layers - 1 ] * self.d, std_a ) )
        if args.train_last_layer == False:
            self.w_direction[num_hidden_layers].requires_grad = False
        
        
        self.mask = [None]*( num_hidden_layers + 1 )
        self.mask[0] = self.diagonal_mask( num_hidden_nodes[0] * self.d, self.d )
        for i in range( num_hidden_layers - 1 ):
            self.mask[ i + 1 ] = self.diagonal_mask( num_hidden_nodes[i+1] * self.d, num_hidden_nodes[i] * self.d )
        self.mask[ num_hidden_layers ] = self.diagonal_mask( self.d, num_hidden_nodes[num_hidden_layers - 1] * self.d )

        self.lower_mask =  [None]*( num_hidden_layers + 1 )
        self.lower_mask[0] = self.lower_mask_function( num_hidden_nodes[0] * self.d, self.d )
        for i in range( num_hidden_layers - 1 ):
            self.lower_mask[i+1] = self.lower_mask_function( num_hidden_nodes[i+1]*d, num_hidden_nodes[i] * self.d )
        self.lower_mask[ num_hidden_layers ] = self.lower_mask_function( self.d, num_hidden_nodes[num_hidden_layers-1] * self.d )

        self.b = nn.ParameterList( [None] * num_hidden_layers )
        for i in range( num_hidden_layers ):
            self.b[ i ] = nn.Parameter( torch.Tensor( num_hidden_nodes[ i ] * self.d ) )
            nn.init.normal_( self.b[i], std=self.give_std(initialization,num_hidden_nodes[i] ) )  #1/( np.sqrt( num_hidden_nodes[i] ) )

        if args.activation == "tanh":
            self.activation = nn.Tanh()

        self.gate = nn.Parameter( torch.Tensor(1) )
        nn.init.normal_(self.gate)

    def give_std(self, initialization, num_node):
        if initialization == "standard-normal":
            return 1.0
        elif initialization == "sqrt-(m)":
            return 1.0 / np.sqrt( num_node )
        else:
            raise Exception("Initialization is not from the choices")

    def give_parameterized_weight_mat(self, ind, parameterization):
        num_nodes = self.num_hidden_nodes[ ind ] if ind < self.num_hidden_layers else 1
        return nn.init.zeros_( torch.Tensor( num_nodes * self.d,1 ) )


    def lower_dimensional_weight(self, out, inp, std):
        weight_mat = torch.zeros(out, inp)

        ### All positive weights in clamp-zero and exp-weights

        if self.model_structure == "clamp-zero":
            for i in range(self.d):
                weight_mat[ (i * out) // self.d : ((i + 1) * out) // self.d,  
                    (i * inp) // self.d : ((i + 1) * inp) // self.d ]  = torch.nn.init.normal_( 
                        torch.Tensor( out // self.d, inp // self.d ), std=std ).abs().clamp(1e-10)
                weight_mat[ (i * out) // self.d : ((i + 1) * out) // self.d,  
                    0 : (i * inp) // self.d ]  = torch.nn.init.normal_( 
                        torch.Tensor( out // self.d, (i * inp) // self.d ), std=std )

        return weight_mat


    def lower_mask_function(self, out, inp):
        mask_mat = torch.zeros(out, inp)

        for i in range(self.d):
            mask_mat[ (i * out) // self.d : ((i + 1) * out) // self.d, 
            0 : ((i + 1) * inp) // self.d ] = 1

        return mask_mat


    def diagonal_mask(self, out, inp):
        mask_mat = torch.zeros(out, inp)

        for i in range(self.d):
            mask_mat[ (i * out) // self.d : ((i + 1) * out) // self.d, 
            (i * inp) // self.d : ((i + 1) * inp) // self.d ] = 1.0

        return mask_mat

    def give_weight_mat_log_grad(self, ind, inp_dim, out_dim):
        weight_mat = self.w_direction[ind] * self.lower_mask[ind]
        tmp = self.w_direction[ind][ self.mask[ind].bool() ].log().view( self.d * out_dim, inp_dim )
        return weight_mat, tmp


    def get_weight_mat(self, ind):
        if self.model_structure == "clamp-zero":
            if ind == 0:
                inp_dim = 1
            else:
                inp_dim = self.num_hidden_nodes[ ind-1 ]
            if ind == self.num_hidden_layers:
                out_dim = 1
            else:
                out_dim = self.num_hidden_nodes[ ind ]

            weight_mat, tmp = self.give_weight_mat_log_grad( ind, inp_dim, out_dim )
            log_grad_T = tmp.view( self.d, out_dim, inp_dim).transpose(-2, -1)

        return weight_mat, log_grad_T

    def get_normalizing_const(self):
        if self.initialization == "standard-normal":
            return 1.0 / np.sqrt( self.num_hidden_nodes[0] )
        elif self.initialization == "sqrt-(m)":
            return 1.0

    def forward(self, inp):
        out = inp
        grad = None

        normalizing_const = self.get_normalizing_const()

        for i in range( self.num_hidden_layers ):
            weight_mat, log_grad_T = self.get_weight_mat(i)
            log_grad = log_grad_T.transpose(-2, -1).unsqueeze(0).repeat( inp.shape[0], 1, 1, 1 )

            if grad is None:
                grad = log_grad
            else:
                grad = ( log_grad.unsqueeze(-2) + grad.transpose(-2, -1).unsqueeze(-3) ).logsumexp(-1)
            
            # The shape of log_grad should be (inp.shape[0], d, output_dimension, d(dimension of data) )
            out = F.linear( out, weight_mat, self.b[i] )
            log_grad_activation = 2 * ( math.log(2) - out - F.softplus( -2 * out ) )

            # The shape of log_grad_activation should be (inp.shape[0], d*output_dimension)
            out = normalizing_const * self.activation( out )
            grad = grad + log_grad_activation.view(grad.shape) + torch.log( torch.Tensor([normalizing_const]) )
        
        weight_mat, log_grad_T = self.get_weight_mat( self.num_hidden_layers )
        log_grad = log_grad_T.transpose(-2, -1).unsqueeze(0).repeat( inp.shape[0], 1, 1, 1 )
        grad = ( log_grad.unsqueeze(-2) + grad.transpose(-2, -1).unsqueeze(-3) ).logsumexp(-1)
        out = F.linear( out, weight_mat , torch.zeros(self.d) )

        return out, grad.squeeze().sum(-1)

