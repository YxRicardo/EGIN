import random
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import KFold
import pickle
import os
import matplotlib.pyplot as plt
import time
import numpy as np

def load_data_geometric(dataset):
    root_dir = 'geometric_data/' + dataset
    dataset = TUDataset(root=root_dir, name=dataset, use_node_attr=True, use_edge_attr=True)
    num_class = dataset.num_classes
    return dataset, num_class


def split_data(graph_list, data, fold_idx):
    assert -1 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9, or -1 for overwriting."

    addr = 'pickle_data/' + data + '/10-fold_index.pkl'

    if fold_idx == 0 and not os.path.isfile(addr): # create 10-fold index and train by fold index 0
        graph_index = [i for i in range(len(graph_list))]
        num_folds = 10
        kf = KFold(n_splits=num_folds, shuffle=True)
        fold_indices = []
        for train_idx, test_idx in kf.split(graph_index):
            fold_indices.append((train_idx, test_idx))

        train_idx, test_idx = fold_indices[0]
        isExists = os.path.exists('pickle_data/' + data)
        if not isExists:
            os.makedirs('pickle_data/' + data)
        with open(addr, 'wb') as f:
            pickle.dump(fold_indices, f)

        random.shuffle(train_idx)
        random.shuffle(test_idx)
        return train_idx, test_idx

    else:
        if not os.path.isfile(addr):
            raise Exception ("No pickle data for 10-fold, please execute fold_index = 0 to create indices first")
        else:
            with open(addr, 'rb') as f:
                fold_indices = pickle.load(f)

        train_idx, test_idx = fold_indices[fold_idx]
        random.shuffle(train_idx)
        random.shuffle(test_idx)

        return train_idx, test_idx

def get_average(lst):
    if not lst:
        return 0
    total = 0
    for num in lst:
        total += num
    return total / len(lst)

def plot_loss_graph(train_loss, test_loss, date, addr):
    plt.plot(train_loss, color='r', label='train_loss')
    plt.plot(test_loss, color='b', label='val_loss')
    plt.title("Red:Train Loss  Blue:Validation Loss", fontsize=20)
    plt.xlabel("epoch", fontsize=12)
    plt.ylabel("loss", fontsize=12)
    plt.grid()
    plt.savefig(addr + date + '_loss_graph' + '.png')
    plt.clf()

def plot_accu_graph(train_acc, test_accu, date, addr):
    plt.plot(train_acc, color='r', label='train_accu')
    plt.plot(test_accu, color='b', label='val_accu')
    plt.title("Red:Train Accuracy  Blue:Validation Accuracy", fontsize=20)
    plt.xlabel("epoch", fontsize=12)
    plt.ylabel("accuracy", fontsize=12)
    plt.grid()
    plt.savefig(addr + date + '_accu_graph' + '.png')
    plt.clf()


def write_10_fold_result(args,test_acc,last_acc,avg_acc):
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

    date = str(time.strftime("%m_%d_%H_%M", time.localtime()))
    with open(path + date + '_10-fold_results.txt', 'a+') as f:
        for element in args.__dict__:
            f.write(str(element) + " = " + str(args.__dict__[element]) + '\n')
        f.write("==============================================================================================\n")
        for i in range(10):
            f.write("iteration: {}, best val acc = {:.6f}\n".format(i, test_acc[i]))
            f.write("iteration: {}, last val acc = {:.6f}\n".format(i, last_acc[i]))
            f.write("iteration: {}, last 20 epochs avg val acc = {:.6f}\n".format(i, avg_acc[i]))
            f.write("==============================================================================================\n")
        f.write("best val acc, 10-fold mean:{:.6f}, std_dev:{:.6f}\n".format(np.mean(test_acc), np.std(test_acc)))
        f.write("last val acc, 10-fold mean:{:.6f}, std_dev:{:.6f}\n".format(np.mean(last_acc), np.std(last_acc)))
        f.write("last 20 epochs avg val acc, 10-fold mean:{:.6f}, std_dev:{:.6f}".format(np.mean(avg_acc), np.std(avg_acc)))


def print_args(args):
    print("==============================================================================================")
    print("Arguments:")
    for index, element in enumerate(args.__dict__):
        if index % 3 == 2:
            print("{} = {},".format(element, args.__dict__[element]))
        else:
            print("{} = {}, ".format(element, args.__dict__[element]), end='')