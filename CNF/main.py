from utils import *

import numpy as np
import matplotlib.pyplot as plt
import gc, math
import argparse
import logging
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.autograd import Variable 
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--num_flows', type=int, default=1)
parser.add_argument('--hidden_layers', type=int, default=1)
parser.add_argument('--hidden_nodes', nargs="+", type=int, default=[100])
parser.add_argument('--total_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--dataset_file', type=str, default="../datasets/grid-dataset.txt") #
parser.add_argument('--results_file', type=str, default="results/check")
parser.add_argument('--activation', type=str, default="tanh")
parser.add_argument('--model_structure', type=str, default="clamp-zero")
parser.add_argument('--parameterization', choices=["normal"], type=str, default="normal")
parser.add_argument('--train_last_layer', type=bool, default=True)
parser.add_argument('--plot_change_in_w_b', type=bool, default=True)
parser.add_argument('--clamping_epsilon', type=float, default=1e-6)
parser.add_argument('--std_a', type=float, default=0.005 )
parser.add_argument('--initialization', choices=["standard-normal", "sqrt-(m)"], type=str, default="standard-normal")
parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--optimizer', choices=["sgd"], type=str, default="sgd")
parser.add_argument("--lr_mult_fact", type=float, default=1.0)
parser.add_argument("--lr_epochs", type=int, default=10)
parser.add_argument('--seed', type=int, default=18)

args = parser.parse_args()
print(args)


def find_complete_training_loss():
    train_loss = 0.0
    train_prob_loss = 0.0
    train_jacob_loss = 0.0

    for batch_idx, train_data in enumerate(train_loader):
        len_train_data = len(train_data)
        output_data, log_det_jacob = model( train_data.type(dtype) )
        data_log_prob = base_dist.log_prob(output_data)

        prob_loss = data_log_prob.sum()
        jacob_loss = log_det_jacob.type(dtype).sum()

        train_prob_loss += (-prob_loss.item())
        train_jacob_loss += ( -jacob_loss.item() )
        train_loss += ( -prob_loss.item() - jacob_loss.item()  )

    return train_loss * 1.0 / len(train_dataset), train_prob_loss* 1.0 / len(train_dataset), train_jacob_loss * 1.0 / len(train_dataset)


def find_complete_test_loss():
    test_loss = 0.0
    test_prob_loss = 0.0
    test_jacob_loss = 0.0

    for batch_idx, test_data in enumerate(test_loader):
        len_test_data = len(test_data)
        output_data, log_det_jacob = model( test_data.type(dtype) )
        data_log_prob = base_dist.log_prob(output_data)

        prob_loss = data_log_prob.sum()
        jacob_loss = log_det_jacob.type(dtype).sum()

        test_prob_loss += (-prob_loss.item())
        test_jacob_loss += ( -jacob_loss.item() )
        test_loss += ( -prob_loss.item() - jacob_loss.item()  )

    return test_loss * 1.0 / len(test_dataset), test_prob_loss * 1.0 / len(test_dataset), test_jacob_loss * 1.0 / len(test_dataset)



torch.manual_seed( args.seed )
writer = SummaryWriter( args.results_file )
train_dataset, test_dataset, d = read_dataset( args )

logging.basicConfig(filename=args.results_file + "/logger.log", format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.info(args)

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info("%s", device)

if use_cuda:
    train_dataset = train_dataset.cuda()
    torch.set_default_tensor_type(dtype)

args.d = d
model = StackedFlow(args.num_flows, args.hidden_layers, args.hidden_nodes,  args.std_a, args.initialization, args)

model.to(device)

epoch = -1

if args.plot_change_in_w_b:
    initial_weight = [ [] for i in range(args.num_flows) ]
    initial_bias = [ [] for i in range(args.num_flows) ]
    
    with torch.no_grad():
        
        size = [d] + args.hidden_nodes + [d]
        for i in range( args.num_flows ):
            for j in range( args.hidden_layers + 1 ):
                initial_weight[i].append( model.flows_module[i].w_direction[j].data.clone() )
        
        for i in range( args.num_flows ):
            for j in range( args.hidden_layers ):
                initial_bias[i].append( model.flows_module[i].b[j].data.clone() )

        for i in range( args.num_flows ):
            for j in range( args.hidden_layers ):
                writer.add_scalar("Change_in_w_" + str(i) + "_" + str(j), np.linalg.norm( (model.flows_module[i].w_direction[j].cpu().numpy() - initial_weight[i][j].cpu().numpy() ) ), epoch )
                writer.add_scalar("Change_in_b_" + str(i) + "_" + str(j), np.linalg.norm( (model.flows_module[i].b[j].cpu().numpy() - initial_bias[i][j].cpu().numpy() ) ), epoch )
            writer.add_scalar("Change_in_w_" + str(i) + "_" + str(args.hidden_layers), np.linalg.norm( (model.flows_module[i].w_direction[ args.hidden_layers].cpu().numpy() - initial_weight[i][args.hidden_layers].cpu().numpy() ) ), epoch )

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

base_dist = D.MultivariateNormal(torch.zeros(d).to(device),  torch.eye(d).to(device) )
        
if args.optimizer == "sgd":
    optimizer = torch.optim.SGD( model.parameters(), lr=args.learning_rate )

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_epochs, gamma=args.lr_mult_fact)

prev_running_avg_loss = float('inf')
running_avg_loss = 0.0


epoch = -1

train_loss, train_prob_loss, train_jacob_loss = find_complete_training_loss()
test_loss, test_prob_loss, test_jacob_loss = find_complete_test_loss()

logger.info("Epoch number: %f\n Train loss: %f\n Train probability loss: %f\n Train Jacobian loss: %f\n Running average loss: %f\n\n", epoch, train_loss, train_prob_loss, train_jacob_loss, running_avg_loss)
print("Epoch number:", epoch, "Train loss:", train_loss, "Train probability loss:", train_prob_loss, "Train Jacobian loss:", train_jacob_loss, "Running average loss:", running_avg_loss)
print("Test loss:", test_loss, "Test probability loss:", test_prob_loss, "Test Jacobian loss:", test_jacob_loss)


for epoch in range(args.total_epochs):
    epoch_loss = 0.0
    epoch_prob_loss = 0.0
    epoch_jacob_loss = 0.0
    epoch_reg_loss = 0.0
    print("Epoch:", epoch)
    
    model.train()
    for batch_idx, train_data in enumerate(train_loader):
        # print(batch_idx)

        len_train_data = len(train_data)
        output_data, log_det_jacob = model( train_data.type(dtype) )
        data_log_prob = base_dist.log_prob(output_data)

        prob_loss = data_log_prob.sum()
        epoch_prob_loss += (-prob_loss.item())
        jacob_loss = log_det_jacob.type(dtype).sum()
        epoch_jacob_loss += ( -jacob_loss.item() )

        loss = ( - prob_loss - jacob_loss ) * 1.0 / len_train_data
        epoch_loss += ( -prob_loss.item() - jacob_loss.item()  ) 
        # print(epoch, batch_idx, loss, prob_loss, jacob_loss)
        # if batch_idx > 100: 
        #     exit(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        if args.model_structure == "clamp-zero":
            with torch.no_grad():
                for i, module_name in enumerate( model.flows_module ):
                    for j in range( model.flows_module[i].num_hidden_layers + 1 ):
                        model.flows_module[i].w_direction[j].data = model.flows_module[i].w_direction[j].data *(model.flows_module[i].lower_mask[j].data - model.flows_module[i].mask[j].data) + ((model.flows_module[i].w_direction[j].data.clamp( args.clamping_epsilon )) * model.flows_module[i].mask[j].data)



    epoch_loss = epoch_loss * 1.0 / len(train_dataset)
    running_avg_loss = (running_avg_loss * epoch + epoch_loss) / (epoch + 1)

    scheduler.step()

    train_loss, train_prob_loss, train_jacob_loss = find_complete_training_loss()
    test_loss, test_prob_loss, test_jacob_loss = find_complete_test_loss()

    logger.info("Epoch number: %f\n Train loss: %f\n Train probability loss: %f\n Train Jacobian loss: %f\n Running average loss: %f\n Learning rate: %f \n\n", epoch, train_loss, train_prob_loss, train_jacob_loss, running_avg_loss, optimizer.param_groups[0]['lr'])
    if epoch % 10 == 0 or epoch == (args.total_epochs - 1):
        print("Epoch number:", epoch, "Train loss:", train_loss, "Train probability loss:", train_prob_loss, "Train Jacobian loss:", train_jacob_loss, "Running average loss:", running_avg_loss)
        print("Test loss:", test_loss, "Test probability loss:", test_prob_loss, "Test Jacobian loss:", test_jacob_loss)

    writer.add_scalar("Running_Average_Training_Loss", running_avg_loss, epoch)
    writer.add_scalar("Train_training_loss", train_loss, epoch)
    writer.add_scalar("Train_probability_loss:", train_prob_loss, epoch )
    writer.add_scalar("Train_jacobian_loss:", train_jacob_loss, epoch)
    writer.add_scalar("Test_training_loss", test_loss, epoch)
    writer.add_scalar("Test_probability_loss:", test_prob_loss, epoch )
    writer.add_scalar("Test_jacobian_loss:", test_jacob_loss, epoch)

    for i in range( args.num_flows ):
            for j in range( args.hidden_layers ):
                writer.add_scalar("Change_in_w_" + str(i) + "_" + str(j), np.linalg.norm( (model.flows_module[i].w_direction[j].data.cpu().numpy() - initial_weight[i][j].cpu().numpy() ) ), epoch )
                writer.add_scalar("Change_in_b_" + str(i) + "_" + str(j), np.linalg.norm( (model.flows_module[i].b[j].data.cpu().numpy() - initial_bias[i][j].cpu().numpy() ) ), epoch )
            writer.add_scalar("Change_in_w_" + str(i) + "_" + str(args.hidden_layers), np.linalg.norm( (model.flows_module[i].w_direction[args.hidden_layers].data.cpu().numpy() - initial_weight[i][args.hidden_layers].cpu().numpy() ) ), epoch )



writer.close()

