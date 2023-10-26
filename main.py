import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import time
import os
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils import split_data, load_data_geometric, plot_accu_graph, plot_loss_graph, write_10_fold_result, print_args
from models.graphegin import GraphEGIN
import random

criterion = nn.CrossEntropyLoss()


def train(model, device, train_graphs, optimizer):
    model.train()

    loss_accum = 0
    for batch_graph in train_graphs:

        output = model(batch_graph)
        labels = batch_graph.y.to(device)

        # compute loss
        loss = criterion(output, labels)

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

    average_loss = loss_accum / len(train_graphs)

    return average_loss


def val(model, train_graphs, val_graphs):
    model.eval()

    output = []
    labels = []
    for batch in train_graphs:
        output.append(model(batch))
        labels.append(batch.y)

    output = torch.cat(output, 0)
    labels = torch.cat(labels, 0)
    num_train_graph = len(labels)
    pred = output.max(1, keepdim=True)[1]

    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(num_train_graph)

    output = []
    labels = []
    for batch in val_graphs:
        output.append(model(batch))
        labels.append(batch.y)

    output = torch.cat(output, 0)
    labels = torch.cat(labels, 0)
    num_val_graph = len(labels)
    val_loss = criterion(output, labels).detach().cpu().numpy()

    pred = output.max(1, keepdim=True)[1]
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_val = correct / float(num_val_graph)

    return acc_train, acc_val, val_loss


def main(args):
    seed = args.seed

    random.seed(seed)  # set random seed to reproduce result
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # Recommendation device: cpu if no eps / gpu if learn eps.
    if not args.learn_eps:
        device = 'cpu'

    graphs, num_classes = load_data_geometric(args.dataset)
    train_idx, val_idx = split_data(graphs, args.dataset, args.fold_index)

    train_loader = DataLoader(graphs[train_idx.tolist()], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(graphs[val_idx.tolist()], batch_size=args.batch_size, shuffle=True)

    model = GraphEGIN(args.num_layers, args.num_mlp_layers, graphs.num_edge_features, graphs.num_node_features,
                      args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.dot_update, device).to(device)

    print_args(args)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    pbar = tqdm(range(args.epochs), unit='epochs', file=sys.stdout)
    date = str(time.strftime("%m_%d_%H_%M", time.localtime()))

    if args.learn_eps:
        if not args.dot_update:
            path = 'result/{}/eps/'.format(args.dataset)
        else:
            path = 'result/{}/eps/dot/'.format(args.dataset)
    else:
        if not args.dot_update:
            path = 'result/{}/no_eps/'.format(args.dataset)
        else:
            path = 'result/{}/no_eps/dot/'.format(args.dataset)

    if args.output_file:
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
        with open(path + date + '_acc_results.txt', 'a+') as f:
            for element in args.__dict__:
                f.write(str(element) + " = " + str(args.__dict__[element]) + '\n')
            f.write("==============================================================================================\n")

    best_val_acc = 0
    all_train_loss = []
    all_val_loss = []
    all_train_acc = []
    all_val_acc = []

    for epoch in pbar:
        avg_loss = train(model, device, train_loader, optimizer)
        scheduler.step()
        acc_train, acc_val, val_loss = val(model, train_loader, val_loader)
        if acc_val > best_val_acc:
            best_val_acc = acc_val
        if epoch % 5 == 0 and args.print_console:
            tqdm.write("epoch: %d, train loss = %.6f, val loss = %.6f, train acc = %.2f%% , val acc = %.2f%%" % (
        epoch, avg_loss, val_loss, acc_train * 100, acc_val * 100))
        if args.output_file:
            with open(path + date + '_acc_results.txt', 'a+') as f:
                f.write("epoch: %d, train loss = %.6f, val loss = %.6f, train acc = %.2f%% , val acc = %.2f%%\n" % (
                    epoch, avg_loss, val_loss, acc_train * 100, acc_val * 100))

        all_train_loss.append(avg_loss)
        all_val_loss.append(val_loss)
        all_train_acc.append(acc_train * 100)
        all_val_acc.append(acc_val * 100)

    if args.plot_curve:
        plot_loss_graph(all_train_loss[1:], all_val_loss[1:], date, path)
        plot_accu_graph(all_train_acc, all_val_acc, date, path)

    print("best val acc: {} ".format(best_val_acc))

    return best_val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch EGIN implementation for graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--seed', type=int, default=-1,
                        help='random seed for training, -1: random')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--fold_index', type=int, default=0,
                        help='the index (0-9) of fold in 10-fold validation. 0: create indexes if does not exist / 10: run all 10-fold')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--dot_update', default=False, help='apply dot update function')
    parser.add_argument('--learn_eps', action="store_true",
                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--print_console', default=False, help='print result of each epoch on console')
    parser.add_argument('--output_file', default=False, help='save results as a output file')
    parser.add_argument('--plot_curve', default=False, help='plot loss and accuracy curves')
    args = parser.parse_args()

    # =======================================================================================================
    # specify parameters here manually without using command line.
    # 'Cuneiform','ER_MD','AIDS'
    args.dataset = 'MUTAG'
    args.epochs = 200
    args.batch_size = 64
    args.lr = 0.005
    args.fold_index = 10
    args.learn_eps = False
    args.output_file = True
    args.plot_curve = True
    args.print_console = True
    args.num_layers = 5
    args.hidden_dim = 32
    args.dot_update = False
    # =======================================================================================================

    if args.seed == -1:
        args.seed = random.randint(0, 100000)

    best_acc = []
    if args.fold_index == 10:
        for i in range(10):
            args.fold_index = i
            best_acc.append(main(args))
        print("10-fold mean:{}, std_dev:{}".format(np.mean(best_acc),np.std(best_acc)))

        if args.output_file:
            write_10_fold_result(args, best_acc)

    else:
        main(args)
