from utils import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time, gc
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_layers', type=int, default=1)
parser.add_argument('--hidden_nodes', type=list, default=[6400])
parser.add_argument('--total_epochs', type=int, default=100)
parser.add_argument('--std_a', type=float, default=0.25 )
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--initialization', type=str, choices=["gaussian-sqrt(m)", "gaussian-std"], default="gaussian-sqrt(m)")
parser.add_argument('--base_distribution', choices=["exponential"], type=str, default="exponential")
parser.add_argument('--dataset_file', type=str, default="../datasets/grid-dataset.txt")
parser.add_argument('--results_file', type=str, default="results/")
parser.add_argument('--integrate_method', choices=["right-sum", "clenshaw-curtis"], type=str, default="right-sum")
parser.add_argument('--train_last_layer', type=bool, default=False)
parser.add_argument('--plot_change_in_w_b', type=bool, default=True)
parser.add_argument('--int_steps', type=int, default=50)
parser.add_argument('--learning_rate', type=float, default=0.005 )
parser.add_argument("--lr_mult_fact", type=float, default=1.0 )
parser.add_argument("--lr_epochs", type=int, default=20)
parser.add_argument('--seed', type=int, default=9)
args = parser.parse_args()
print(args)




torch.manual_seed(args.seed)

writer = SummaryWriter(args.results_file)
train_dataset, d, minimum_point, maximum_point = read_dataset( args.dataset_file, args.results_file, args )
args.d = d

print(train_dataset.shape)

logging.basicConfig(filename=args.results_file + "/logger.log", format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.info(args)

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info("%s", device)

epoch = -1

if use_cuda:
    train_dataset = train_dataset.cuda()
    torch.set_default_tensor_type(dtype)

module_list = give_module_list(d, args.hidden_layers, args.hidden_nodes, args, device)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
train_len = len(train_dataset)

params_list = list()
for module in module_list:
    params_list = params_list + list( module.parameters() )

optimizer = torch.optim.SGD( params_list, lr=args.learning_rate )
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_epochs, gamma=args.lr_mult_fact)


if args.base_distribution == "exponential":
    base_dist = exponential( d, [1.0]*d )


if args.plot_change_in_w_b:
    initial_w, initial_b, initial_w_final = give_initial_weights(module_list, args)
    for i in range(d):
        writer.add_scalar("Change_in_w_" + str(i), 0.0, epoch )
        writer.add_scalar("Change_in_b_" + str(i), 0.0, epoch )
        writer.add_scalar("Change_in_w_final_" + str(i), 0.0, epoch )

epoch = -1

for epoch in range( args.total_epochs ):
    print(epoch)
    epoch_loss = 0.0
    epoch_prob_loss = 0.0
    epoch_jacob_loss = 0.0

    for batch_idx, train_data in enumerate(train_loader):
        len_train_data = len(train_data)
        func, deriv_func, jacob = give_function( d, module_list, train_data, dtype, args, minimum_point)

        data_log_prob = base_dist.log_prob( func.type(dtype) )

        prob_loss = data_log_prob.sum()
        epoch_prob_loss -= prob_loss.item()

        jacob_loss = jacob.log().sum()
        epoch_jacob_loss -= jacob_loss.item()

        loss = ( - prob_loss - jacob_loss ) * 1.0 / len_train_data
        epoch_loss += ( -prob_loss.item() - jacob_loss.item() )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    if epoch%10 == 0 or epoch == args.total_epochs - 1:
        epoch_loss = find_complete_training_loss(d, module_list, train_loader, dtype, base_dist, args, minimum_point) * 1.0 / len( train_dataset )
        print("Epoch:", epoch, "Epoch loss:", epoch_loss)
        writer.add_scalar("Epoch_training_loss", epoch_loss, epoch)
        for i in range(d):
            writer.add_scalar("Change_in_w_" + str(i), np.linalg.norm(module_list[i].w[0].data.cpu().numpy() - initial_w[i]), epoch )
            writer.add_scalar("Change_in_b_" + str(i), np.linalg.norm(module_list[i].b[0].data.cpu().numpy() - initial_b[i]), epoch )
            writer.add_scalar("Change_in_w_final_" + str(i), np.linalg.norm(module_list[i].w_final.data.cpu().numpy() - initial_w_final[i]), epoch )
        logger.info("Epoch number: %f  Epoch loss: %f\n", epoch, epoch_loss )
