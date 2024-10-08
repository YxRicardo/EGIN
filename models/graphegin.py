import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.pool import global_add_pool
import sys
sys.path.append("models/")
from mlp import MLP
import copy

class GraphEGIN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, num_edge_feat, input_dim, hidden_dim, output_dim, final_dropout, learn_eps, dot_update, edge_mlp, edge_hidden_dim, device):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether. 
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''

        super(GraphEGIN, self).__init__()
        self.dot_update = dot_update
        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.learn_eps = learn_eps
        self.num_edge_feat = num_edge_feat
        self.eps = nn.Parameter(torch.zeros(self.num_layers-1))
        ###List of MLPs
        self.mlps = torch.nn.ModuleList()
        self.adj = None
        self.edge_rep = None
        self.edge_unit = None
        self.edge_mlp = edge_mlp
        self.edge_hidden_dim = edge_hidden_dim
        if self.edge_mlp:
            self.edge_mlps = torch.nn.ModuleList()
            for layer in range(self.num_layers-1):
                self.edge_mlps.append(torch.nn.Linear(num_edge_feat,self.edge_hidden_dim))

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)

        self.batch_norms = torch.nn.ModuleList()

        if self.dot_update:
            for layer in range(self.num_layers-1):
                if layer == 0:
                    self.mlps.append(MLP(num_mlp_layers, input_dim * num_edge_feat, hidden_dim, hidden_dim))
                else:
                    self.mlps.append(MLP(num_mlp_layers, hidden_dim * num_edge_feat, hidden_dim, hidden_dim))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        elif not self.edge_mlp:
            for layer in range(self.num_layers - 1):
                if layer == 0:
                    self.mlps.append(MLP(num_mlp_layers, input_dim + num_edge_feat, hidden_dim, hidden_dim))
                else:
                    self.mlps.append(MLP(num_mlp_layers, hidden_dim + num_edge_feat, hidden_dim, hidden_dim))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        else:
            for layer in range(self.num_layers - 1):
                if layer == 0:
                    self.mlps.append(MLP(num_mlp_layers, input_dim + self.edge_hidden_dim, hidden_dim, hidden_dim))
                else:
                    self.mlps.append(MLP(num_mlp_layers, hidden_dim + self.edge_hidden_dim, hidden_dim, hidden_dim))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))


        #Linear function that maps the hidden representation at dofferemt layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))



    def __batch_to_embedadj(self,batch):
        if batch.num_edge_features == 0:
            raise Exception("No edge feature in this dataset. Add (-egin false) to execute original GIN model")

        if batch.num_edge_features == 1:
            adj = torch.eye(batch.num_nodes, batch.num_nodes)
        else:
            adj = torch.zeros(batch.num_nodes, batch.num_nodes, batch.num_edge_features)
            for i in range(batch.num_nodes):
                adj[i][i] = torch.ones(batch.num_edge_features)
        for i in range(batch.num_edges):
            adj[batch.edge_index[0][i]][batch.edge_index[1][i]] = batch.edge_attr[i]

        if not self.dot_update:
            self.edge_rep = torch.sum(adj,dim=1).to(self.device)

            adj = torch.eye(batch.num_nodes, batch.num_nodes)
            for i in range(batch.num_edges):
                adj[batch.edge_index[0][i]][batch.edge_index[1][i]] = 1

            if self.learn_eps:
                self.edge_unit = torch.ones(batch.num_nodes,batch.num_edge_features).to(self.device)

        self.adj = adj.to(self.device)

    def embedadj_mm(self,embedadj,h):
        if len(embedadj.size()) == 2:
            h = torch.matmul(embedadj,h)
        else:
            h = torch.matmul(embedadj.permute(2, 0, 1), h)
            h = h.permute(1, 2, 0)
            h = torch.flatten(h, start_dim=1, end_dim=2)
        return h



    def egin_next_layer(self, h, layer):
        if self.dot_update:
        ###pooling neighboring nodes and center nodes altogether
            pooled = self.embedadj_mm(self.adj, h)
        elif not self.edge_mlp:
            pooled = torch.cat((torch.spmm(self.adj, h), self.edge_rep),dim=1)
        else:
            pooled = torch.cat((torch.spmm(self.adj, h), self.edge_mlps[layer](self.edge_rep)), dim=1)

        # representation of neighboring and center nodes
        pooled_rep = self.mlps[layer](pooled)

        h = self.batch_norms[layer](pooled_rep)


        # non-linearity
        h = F.relu(h)
        return h


    def egin_next_layer_eps(self, h, layer, batch):
        ###pooling neighboring nodes and center nodes altogether
        if self.dot_update:
            adj = copy.deepcopy(self.adj)
            if batch.num_edge_features == 1:
                adj = adj + torch.eye(batch.num_nodes, batch.num_nodes).to(self.device) * self.eps[layer]
            else:
                for i in range(batch.num_nodes):
                    adj[i][i] = adj[i][i] + torch.ones(batch.num_edge_features).to(self.device) * self.eps[layer]

            pooled = self.embedadj_mm(adj, h)

        elif not self.edge_mlp:
            pooled = torch.cat((torch.spmm(self.adj, h), self.edge_rep), dim=1) + torch.cat((h,self.edge_unit),dim=1) * self.eps[layer]

        else:
            pooled = torch.cat((torch.spmm(self.adj, h), self.edge_mlps[layer](self.edge_rep)), dim=1) + torch.cat((h, self.edge_mlps[layer](self.edge_unit)),
                                                                                            dim=1) * self.eps[layer]

        pooled_rep = self.mlps[layer](pooled)


        h = self.batch_norms[layer](pooled_rep)


        # non-linearity
        h = F.relu(h)
        return h


    def egin_forward(self,batch_graph_input):
        batch_graph = batch_graph_input.to(self.device)
        self.__batch_to_embedadj(batch_graph)
        X_concat = batch_graph.x

        # graph_pool = self.__preprocess_graphpool(batch_graph)
        # list of hidden representation at each layer (including input)
        hidden_rep = [X_concat]
        h = X_concat

        for layer in range(self.num_layers - 1):
            if self.learn_eps:
                h = self.egin_next_layer_eps(h, layer, batch_graph)
            else:
                h = self.egin_next_layer(h, layer)

            hidden_rep.append(h)

        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for layer, h in enumerate(hidden_rep):
            pooled_h = global_add_pool(h, batch_graph.batch)
            score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout,
                                          training=self.training)

        return score_over_layer


    def forward(self, batch_graph):
        return self.egin_forward(batch_graph)


