import random
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import KFold
import pickle
import os
import matplotlib.pyplot as plt

def load_data_geometric(dataset):
    root_dir = 'geometric_data/' + dataset
    dataset = TUDataset(root=root_dir, name=dataset, use_node_attr=True, use_edge_attr=True)
    num_class = dataset.num_classes
    return dataset, num_class


def split_data(graph_list, data, fold_idx):
    assert -1 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9, or -1 for overwriting."

    if fold_idx == -1: # overwrite 10-fold index and train by fold index 0
        graph_index = [i for i in range(len(graph_list))]
        num_folds = 10
        kf = KFold(n_splits=num_folds, shuffle=True)
        fold_indices = []
        for train_idx, test_idx in kf.split(graph_index):
            fold_indices.append((train_idx, test_idx))

        train_idx, test_idx = fold_indices[0]

        addr = 'pickle_data/' + data + '/10-fold_index.pkl'
        isExists = os.path.exists('pickle_data/' + data)
        if not isExists:
            os.makedirs('pickle_data/' + data)
        with open(addr, 'wb') as f:
            pickle.dump(fold_indices, f)

        random.shuffle(train_idx)
        random.shuffle(test_idx)
        return train_idx, test_idx

    else:
        addr = 'pickle_data/' + data + '/10-fold_index.pkl'
        if not os.path.isfile(addr):
            raise Exception ("No pickle data for 10-fold")
        else:
            with open(addr, 'rb') as f:
                fold_indices = pickle.load(f)

        train_idx, test_idx = fold_indices[fold_idx]
        random.shuffle(train_idx)
        random.shuffle(test_idx)

        return train_idx, test_idx

def plot_loss_graph(train_loss, test_loss, date, addr):
    plt.plot(train_loss, color='r', label='train_loss')
    plt.plot(test_loss, color='b', label='test_loss')
    plt.title("Red:Train Loss  Bule:Test Loss", fontsize=20)
    plt.xlabel("epchos", fontsize=12)
    plt.ylabel("loss", fontsize=12)
    plt.grid()
    plt.savefig(addr + date + '_loss_graph' + '.png')
    plt.clf()

def plot_accu_graph(train_accu, test_accu, date, addr):
    plt.plot(train_accu, color='r', label='train_accu')
    plt.plot(test_accu, color='b', label='test_accu')
    plt.title("Red:Train Accuracy  Bule:Test Accuracy", fontsize=20)
    plt.xlabel("epchos", fontsize=12)
    plt.ylabel("accuracy", fontsize=12)
    plt.grid()
    plt.savefig(addr + date + '_accu_graph' + '.png')
    plt.clf()
