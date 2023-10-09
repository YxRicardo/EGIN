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
from utils import split_data, load_data_geometric, plot_accu_graph, plot_loss_graph
from models.graphcnn import GraphCNN

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


def test(model, train_graphs, test_graphs):
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
    for batch in test_graphs:
        output.append(model(batch))
        labels.append(batch.y)

    output = torch.cat(output, 0)
    labels = torch.cat(labels, 0)
    num_test_graph = len(labels)
    test_loss = criterion(output, labels).detach().cpu().numpy()

    pred = output.max(1, keepdim=True)[1]
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(num_test_graph)

    return acc_train, acc_test, test_loss


def main(args):
    # set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # recommend to use cpu if no eps and gpu if learn eps.
    if not args.learn_eps:
        device = 'cpu'

    graphs, num_classes = load_data_geometric(args.dataset)
    train_idx, test_idx = split_data(graphs, args.dataset, args.fold_index)

    train_loader = DataLoader(graphs[train_idx.tolist()], batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(graphs[test_idx.tolist()], batch_size=args.batch_size, shuffle=True)

    model = GraphCNN(args.num_layers, args.num_mlp_layers, graphs.num_edge_features, graphs.num_node_features,
                     args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, device).to(device)
    print(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    pbar = tqdm(range(args.epochs), unit='epochs', file=sys.stdout)
    date = str(time.strftime("%m_%d_%H_%M", time.localtime()))
    if args.learn_eps:
        path = 'result/{}/eps/'.format(args.dataset)
    else:
        path = 'result/{}/no_eps/'.format(args.dataset)

    if args.output_file:
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
        with open(path + date + '_acc_results.txt', 'a+') as f:
            for element in args.__dict__:
                f.write(str(element) + " = " + str(args.__dict__[element]) + '\n')
            f.write("==============================================================================================\n")

    best_test_acc = 0
    all_train_loss = []
    all_test_loss = []
    all_train_acc = []
    all_test_acc = []

    for epoch in pbar:
        avg_loss = train(model, device, train_loader, optimizer)
        scheduler.step()
        acc_train, acc_test, test_loss = test(model, train_loader, test_loader)
        if acc_test > best_test_acc:
            best_test_acc = acc_test
        tqdm.write("epoch: %d, train loss = %.6f, test loss = %.6f, train acc = %.2f%% , test acc = %.2f%%" % (
        epoch, avg_loss, test_loss, acc_train * 100, acc_test * 100))
        if args.output_file:
            with open(path + date + '_acc_results.txt', 'a+') as f:
                f.write("epoch: %d, train loss = %.4f, test loss = %.4f, train acc = %.2f%% , test acc = %.2f%%\n" % (
                    epoch, avg_loss, test_loss, acc_train * 100, acc_test * 100))

        all_train_loss.append(avg_loss)
        all_test_loss.append(test_loss)
        all_train_acc.append(acc_train * 100)
        all_test_acc.append(acc_test * 100)

    if args.plot_curve:
        plot_loss_graph(all_train_loss, all_test_loss, date, path)
        plot_accu_graph(all_train_acc, all_test_acc, date, path)

    return best_test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=150,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_index', type=int, default=0,
                        help='the index (0-9) of fold in 10-fold validation. -1:create indexes. 10: run all 10-fold')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--learn_eps', action="store_true",
                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--print_console', default=False, help='print result of each epoch on console')
    parser.add_argument('--output_file', default=False, help='save results as a output file')
    parser.add_argument('--plot_curve', default=False, help='plot loss and accuracy curves')
    args = parser.parse_args()

    # =======================================================================================================
    # specify parameters for testing purpose
    args.dataset = 'MUTAG'
    args.epochs = 300
    args.batch_size = 16
    args.lr = 0.001
    args.fold_index = 10
    args.learn_eps = False
    args.output_file = True
    args.plot_curve = True
    # =======================================================================================================

    best_acc = []
    if args.fold_index == 10:
        for i in range(10):
            args.fold_index = i
            best_acc.append(main(args))
        mean = np.mean(best_acc)
        std_dev = np.std(best_acc)
        print("10-fold mean:{}, std_dev:{}".format(mean,std_dev))
    else:
        main(args)
