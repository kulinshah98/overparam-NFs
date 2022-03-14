import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats
import matplotlib.pyplot as plt



def read_dataset( dataset_file, results_file, args ):
    data = np.genfromtxt( dataset_file)

    d = 1 if ( len( data.shape ) == 1 ) else len( data[0] )

    if d==1:
        data = np.reshape(data, (-1, 1))

    data = data / (2.0 * np.amax( np.linalg.norm(data, axis=1) ) )

    minimum_point = np.amin(data, axis=0)
    maximum_point = np.amax(data, axis=0)

    return torch.from_numpy(data), d, minimum_point, maximum_point


class ELUPlus(nn.Module):
    def __init__(self):
        super(ELUPlus, self).__init__()
        self.elu = nn.ELU()
    def forward(self, x):
        return self.elu(x) + 1.


def give_std(num_nodes, i, args):
    if i==len(num_nodes)-2:
        return args.std_a
    if args.initialization == "gaussian-std":
        return 1.0
    if args.initialization == "gaussian-sqrt(m)":
        return 1.0 / np.sqrt( num_nodes[i + 1] )
    else:
        return None

def give_norm_factor(num_nodes, args):
    if args.initialization == "gaussian-sqrt(m)":
        return 1.0
    elif args.initialization == "gaussian-std":
        return 1.0 / np.sqrt( num_nodes[-2] )
    else:
        return None


class MLP(nn.Module):
    def __init__(self, inp_dim, num_hidden_layers, num_hidden_nodes, args):
        super(MLP, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_nodes = num_hidden_nodes
        self.d = args.d
        self.train_last_layer = args.train_last_layer

        self.num_nodes = [inp_dim] + self.num_hidden_nodes + [1]
        self.w = nn.ParameterList( [None] * (num_hidden_layers) )
        for i in range(len( self.num_nodes ) - 2):
            std = give_std(self.num_nodes, i, args)
            self.w[i] = nn.Parameter( torch.Tensor( self.num_nodes[i+1], self.num_nodes[i] ) )
            nn.init.normal_(self.w[i], std=std)

        std = give_std(self.num_nodes, len( self.num_nodes ) - 2, args)
        if self.train_last_layer == True:
            self.w_final = nn.Parameter( torch.Tensor( self.num_nodes[ len( self.num_nodes ) - 1 ], self.num_nodes[ len( self.num_nodes ) - 2 ] ) )
        else:
            self.w_final = torch.Tensor( self.num_nodes[ len( self.num_nodes ) - 1 ], self.num_nodes[ len( self.num_nodes ) - 2 ] )
        nn.init.normal_(self.w_final, std=std)

        self.b = nn.ParameterList([None] * num_hidden_layers )
        for i in range(len( self.num_nodes ) - 2):
            std = give_std(self.num_nodes, i, args)
            self.b[i] = nn.Parameter(torch.Tensor( self.num_nodes[i+1] ))
            nn.init.normal_(self.b[i], std=std)

        self.normalization_factor = give_norm_factor(self.num_nodes, args)
        self.phi = ELUPlus()

    def set_embedding(self, h):
        self.h = h

    def forward(self, inp):
        out = inp.view(-1, 1)
        out = torch.cat((self.h, out), 1)

        for i in range( len(self.num_nodes) - 2 ):
            out = F.linear( out, self.w[i], self.b[i] )
            out = F.relu( out )

        out = F.linear( out, self.w_final, torch.zeros( self.num_nodes[-1] ) )

        return self.phi( out * self.normalization_factor ).squeeze(dim=1)



def give_module_list(d, num_hidden_layers, num_hidden_nodes, args, device):
    module_list = []
    for i in range(d):
        module_list.append( MLP(i+1, num_hidden_layers, num_hidden_nodes, args) )
        module_list[i].to(device)
    return module_list


def give_function(d, module_list, inp, dtype, args, minimum_point):
    deriv_func = torch.zeros_like( inp )
    func = torch.zeros_like( inp )

    # Returns derivative of function (d dimensional vector), function values (d dimensional vector), determinant of Jacobian (scalar)
    for i in range( d ):
        module_list[i].set_embedding( inp[:, :i].type(dtype) )
        deriv_func[:, i] = module_list[i]( inp[:, i].type(dtype) )
        func[:, i] = integrate( module_list[i], inp[:, i].type(dtype), inp[:, :i].type(dtype), args, minimum_point[i] )

    jacob = torch.prod(deriv_func, 1)
    return func, deriv_func, jacob


def give_initial_weights(module_list, args):
    initial_w = [None] * len(module_list)
    initial_b = [None] * len(module_list)
    initial_w_final = [None] * len(module_list)
    with torch.no_grad():

        for k in range( len(module_list) ):
            initial_w[k] = np.zeros( ( module_list[k].w[0].data.shape[0], module_list[k].w[0].data.shape[1] ) )
            for i in range( module_list[k].w[0].data.shape[0] ):
                for j in range( module_list[k].w[0].data.shape[1] ):
                    initial_w[k][i][j] = module_list[k].w[0].data[i][j]

            initial_b[k] = np.zeros( module_list[k].b[0].data.shape[0] )
            for i in range( module_list[k].b[0].data.shape[0] ):
                initial_b[k][i] = module_list[k].b[0].data[i]

            initial_w_final[k] = np.zeros( ( module_list[k].w_final.data.shape[0], module_list[k].w_final.data.shape[1] ) )
            for i in range( module_list[k].w_final.data.shape[0] ):
                for j in range( module_list[k].w_final.data.shape[1] ):
                    initial_w_final[k][i][j] = module_list[k].w_final.data[i, j]

    return initial_w, initial_b, initial_w_final


def find_complete_training_loss(d, module_list, train_loader, dtype, base_dist, args, minimum_point):
    epoch_loss = 0.0
    for batch_idx, train_data in enumerate( train_loader ):
        len_train_data = len(train_data)

        func, deriv_func, jacob = give_function( d, module_list, train_data, dtype, args, minimum_point )

        data_log_prob = base_dist.log_prob( func.type(dtype) )
        prob_loss = data_log_prob.sum()
        jacob_loss = jacob.log().sum()
        epoch_loss += ( -prob_loss.item() - jacob_loss.item()  )

    return epoch_loss


class exponential:
    def __init__(self, d, lam):
        self.d = d
        self.lam = lam

    def log_prob(self, inp):
        sum = torch.sum(inp, 1)
        p = torch.exp( -sum )
        log_p = torch.log( p )
        return log_p


def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def compute_sr_weights(n_steps):
    sr_weights = np.zeros(n_steps)
    sr_weights[:] = 2.0 / n_steps
    steps = np.arange( n_steps ) + 1
    steps = steps * 2 / n_steps
    steps = steps - 1
    sr_weights = torch.Tensor(sr_weights).float()
    steps = torch.Tensor(steps).float()
    return sr_weights, steps


def sr_integrate(xT, h, n_steps, model, starting_point, compute_grad=False, prev_grad=None):
    sr_weights, steps = compute_sr_weights( n_steps )
    device = xT.get_device() if xT.is_cuda else "cpu"
    sr_weights, steps = sr_weights.to(device), steps.to(device)

    if compute_grad:
        g_param = 0.
    else:
        z = 0.

    x0 = torch.ones_like(xT) * starting_point

    for i in range(n_steps):
        x = (x0 + (xT - x0)*(steps[i] + 1)/2)
        if compute_grad:
            dg_param = computeIntegrand(x, h, model, prev_grad*(xT - x0)/2)
            g_param += sr_weights[i]*dg_param
        else:
            model.set_embedding(h)
            dz = model( x )
            z = z + sr_weights[i]*dz

    if compute_grad:
        return g_param

    return z*(xT - x0)/2


def computeIntegrand(x, h, model, grad_mult_vec):
    with torch.enable_grad():
        model.set_embedding( h )
        f = model.forward( x )
        g_param = _flatten(torch.autograd.grad(f, model.parameters(), grad_mult_vec))

    return g_param


class sr_integral(torch.autograd.Function):

    @staticmethod
    def forward(ctx, model, model_params, x, h, n_steps, starting_point):
        with torch.no_grad():
            vals = sr_integrate(x, h, n_steps, model, starting_point, compute_grad=False)
            ctx.func_model = model
            ctx.int_steps = n_steps
            ctx.starting_point = starting_point
            ctx.save_for_backward(x.clone(), h.clone())

        return vals

    @staticmethod
    def backward(ctx, grad_output):
        x, h = ctx.saved_tensors
        model = ctx.func_model
        n_steps = ctx.int_steps
        starting_point = ctx.starting_point
        param_grad = sr_integrate(x, h, n_steps, model, starting_point, True, grad_output)
        model.set_embedding( h )
        x_grad = model( x )
        return None, param_grad, x_grad * grad_output, None, None, None



def compute_cc_weights(n_steps):
    lam = np.arange(0, n_steps + 1, 1).reshape(-1, 1)
    lam = np.cos((lam @ lam.T) * math.pi / n_steps)
    lam[:, 0] = .5
    lam[:, -1] = .5 * lam[:, -1]
    lam = lam * 2 / n_steps
    W = np.arange(0, n_steps + 1, 1).reshape(-1, 1)
    W[np.arange(1, n_steps + 1, 2)] = 0
    W = 2 / (1 - W ** 2)
    W[0] = 1
    W[np.arange(1, n_steps + 1, 2)] = 0
    cc_weights = torch.tensor(lam.T @ W).float()
    steps = torch.tensor(np.cos(np.arange(0, n_steps + 1, 1).reshape(-1, 1) * math.pi / n_steps)).float()

    return cc_weights, steps


def cc_integrate(xT, h, n_steps, model, starting_point, compute_grad=False, prev_grad=None):
    cc_weights, steps = compute_cc_weights( n_steps )

    device = xT.get_device() if xT.is_cuda else "cpu"
    cc_weights, steps = cc_weights.to(device), steps.to(device)

    if compute_grad:
        g_param = 0.
    else:
        z = 0.

    x0 = torch.ones_like(xT) * starting_point

    for i in range(n_steps + 1):
        x = (x0 + (xT - x0)*(steps[i] + 1)/2)
        if compute_grad:
            dg_param = computeIntegrand(x, h, model, prev_grad*(xT - x0)/2)
            g_param += cc_weights[i]*dg_param
        else:
            model.set_embedding(h)
            dz = model(x)
            z = z + cc_weights[i]*dz

    if compute_grad:
        return g_param

    return z*(xT - x0)/2



class cc_integral(torch.autograd.Function):

    @staticmethod
    def forward(ctx, model, model_params, x, h, n_steps, start_point):
        with torch.no_grad():
            vals = cc_integrate(x, h, n_steps, model, start_point, compute_grad=False)
            ctx.func_model = model
            ctx.int_steps = n_steps
            ctx.start_point = start_point
            ctx.save_for_backward(x.clone(), h.clone())

        return vals

    @staticmethod
    def backward(ctx, grad_output):
        x, h = ctx.saved_tensors
        model = ctx.func_model
        n_steps = ctx.int_steps
        start_point = ctx.start_point
        param_grad = cc_integrate(x, h, n_steps, model, start_point, True, grad_output)
        model.set_embedding( h )
        x_grad = model(x)
        return None, param_grad, x_grad * grad_output, None, None, None



def integrate( model, x, h, args, starting_point ):
    if args.integrate_method == "right-sum":
        values = sr_integral.apply( model, _flatten( model.parameters() ), x, h, args.int_steps, starting_point )
    elif args.integrate_method == "clenshaw-curtis":
        values = cc_integral.apply( model, _flatten( model.parameters() ), x, h, args.int_steps, starting_point )

    return values
