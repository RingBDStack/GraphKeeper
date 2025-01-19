import torch
import sys, os
import dgl
import copy
import numpy as np
from os import path
from tqdm import tqdm
from typing import Any, Dict, Optional, Sequence
from utils import set_weight_decay, validate
from .ours_utils import *
from Backbones.gnns import *
from collections import defaultdict
from sklearn.cluster import DBSCAN
from collections import Counter
import torch.nn.functional as F


def drop_feature(x, drop_prob):
    drop_mask = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

def mask_edge(graph, drop_prob):
    graph = copy.deepcopy(graph)
    num_edges = graph.number_of_edges()
    edge_delete = np.random.choice(num_edges, int(drop_prob*num_edges), replace=False)
    src, dst = graph.edges()
    not_equal = src[edge_delete].cpu() != dst[edge_delete].cpu()
    edge_delete = edge_delete[not_equal]
    graph.remove_edges(edge_delete)
    return graph

def addedges(subgraph):
    subgraph = copy.deepcopy(subgraph)
    nodedegree = subgraph.in_degrees().cpu()
    isolated_nodes = torch.where(nodedegree==1)[0]
    connected_nodes = torch.where(nodedegree!=1)[0]
    isolated_nodes = isolated_nodes.numpy()
    connected_nodes = connected_nodes.numpy()
    randomnode = np.random.choice(connected_nodes, isolated_nodes.shape[0])
    srcs = np.concatenate([isolated_nodes, randomnode])
    dsts = np.concatenate([randomnode, isolated_nodes])
    subgraph.add_edges(srcs, dsts)
    return subgraph


class Learner():
    def __init__(
        self,
        args: Dict[str, Any],
        backbone_output: int,
        device=None
    ) -> None:
        self.backbone_output = backbone_output
        self.device = device
        self.buffer_size: int = args["buffer_size"]
        self.gamma: float = 0.1
        self.model = Regression(
            self.backbone_output,
            self.buffer_size,
            device=self.device,
            dtype=torch.double
        )

    @torch.no_grad()
    def learn(self, X, labels) -> None:
        self.model.eval()
        X = X.to(self.device, non_blocking=True)
        y: torch.Tensor = labels
        y = y.to(self.device, non_blocking=True)
        self.model.fit(X, y)

    def before_validation(self) -> None:
        self.model.update()

    def inference(self, X):
        return self.model.inference(X)

class NET(torch.nn.Module):

    """
    Bare model baseline for NCGL tasks

    :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
    :param task_manager: Mainly serves to store the indices of the output dimensions corresponding to each task
    :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.

    """

    def __init__(self,
                 model,
                 task_manager,
                 args):
        """
        The initialization of the baseline

        :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
        :param task_manager: Mainly serves to store the indices of the output dimensions corresponding to each task
        :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.
        """
        super(NET, self).__init__()
        args.ours_args['latdim'] = int(args.ours_args['latdim'])
        args.ours_args['rank'] = int(args.ours_args['rank'])
        args.ours_args['buffer_size'] = int(args.ours_args['buffer_size'])
        
        self.args = args
        self.task_manager = task_manager

        self.GNN_pretrain = model
        self.current_expert = None
        self.Experts = []
        self.prototype = []

        self.opt = None

        self.expert_prototype = []

        self.GNN_proj = GCN_proj(args).cuda(args.gpu)
        for param in self.GNN_proj.parameters():
            param.requires_grad = False

        self.learner = Learner(
            args.ours_args, args.GCN_args['h_dims'][0], torch.device('cuda:0')
        )

    def sample_negative_edges(self, g, num_samples):
        all_edges = set(g.edges())
        neg_edges = []
        while len(neg_edges) < num_samples:
            u = np.random.choice(g.nodes().cpu().numpy())
            v = np.random.choice(g.nodes().cpu().numpy())
            if (u, v) not in all_edges and (v, u) not in all_edges and u != v:
                neg_edges.append((u, v))
        return torch.tensor(neg_edges).T
         

    def base_training(self, args, g, features, dataloader = None):
        opt = torch.optim.Adam(self.GNN_pretrain.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.GNN_pretrain.train()

        for epoch in range(2):
            opt.zero_grad()
            h = self.GNN_pretrain(g, features)
            pos_edges = g.edges()
            neg_edges = self.sample_negative_edges(g, len(pos_edges))
            pos_scores = h[pos_edges[0]] * h[pos_edges[1]]
            neg_scores = h[neg_edges[0]] * h[neg_edges[1]] 
            labels = torch.cat([torch.ones(pos_scores.size()), torch.zeros(neg_scores.size())])
            scores = torch.cat([pos_scores, neg_scores])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(scores.cpu(), labels.cpu())
            loss.backward()
            opt.step()


    @torch.no_grad()
    def forward(self, g, features, test_ids, task_max):
        g_ = addedges(g)
        testprototypes = torch.mean(self.GNN_proj(g_, features)[test_ids], dim=0)
        distances = []
        for expert_prototypes_tensor in self.expert_prototype[0:task_max]:
            dist = torch.norm(expert_prototypes_tensor - testprototypes)
            distances.append(dist)
        distances = torch.tensor(distances)
        prob = torch.exp(-distances)
        _, expert_index_proto = torch.max(prob,dim=0)
        expert_index = expert_index_proto.item()
        expert = self.Experts[expert_index]
        expert.eval()
        X = expert(g, features)
        output = self.learner.inference(X)
        torch.cuda.empty_cache()
        return output

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def expert_training(self, args, expert, opt, g, features, labels, train_ids, epochs, t, datas, dataloader = None):
        expert.train()
        for epoch in range(epochs):
            opt.zero_grad()
            X = expert(g, features)
            g_aug = mask_edge(g, drop_prob=0.2)
            features_aug = drop_feature(features, drop_prob=0.3)
            X_aug = expert(g_aug, features_aug)
            similarity_matrix = self.sim(X[train_ids],X_aug[train_ids])
            labels_matrix = labels[train_ids].unsqueeze(1) == labels[train_ids].unsqueeze(0)
            pos_mask = labels_matrix.fill_diagonal_(0)
            pos_sim = similarity_matrix * pos_mask.float()
            neg_mask = ~labels_matrix
            neg_sim = similarity_matrix * neg_mask.float()
            pos_exp = torch.exp(pos_sim).sum(dim=1)
            neg_exp = torch.exp(neg_sim).sum(dim=1)
            supcon_loss = -torch.log(pos_exp / (pos_exp + neg_exp)).mean()
            supcon_loss = supcon_loss * args.ours_args['sup_coef']

            if len(self.prototype) != 0:
                prototype_dis = torch.norm(X[train_ids].unsqueeze(1) - torch.stack(self.prototype), dim=2, p=2)
                prototype_loss = torch.mean(torch.reciprocal(torch.min(prototype_dis, dim=1)[0] + 1e-8)) 
                prototype_loss = prototype_loss * args.ours_args['proto_coef']
            else:
                prototype_loss = 0

            if prototype_loss != 0:
                loss = supcon_loss + prototype_loss
            else:
                loss = supcon_loss
            loss.backward()
            opt.step()
            
    def observe(self, args, g, features, labels, t, train_ids, ids_per_cls, dataset, epochs, datas):
        if t == 0:
            self.base_training(args, g, features)

        self.current_expert = GCN_LoRA(args, self.GNN_pretrain).to(device='cuda:0')
        self.opt = torch.optim.Adam(self.current_expert.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.expert_training(args, self.current_expert, self.opt, g, features, labels, train_ids, epochs, t, datas)

        for param in self.current_expert.parameters():
            param.requires_grad = False
        self.current_expert.eval()
        X = self.current_expert(g, features)
        self.learner.learn(X[train_ids], labels[train_ids])
        self.learner.before_validation()

        dbscan = DBSCAN(eps=0.1, min_samples=5)
        dbscan.fit(X[train_ids].detach().cpu().numpy())
        labels = dbscan.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
            cluster_embeddings = X[train_ids][cluster_indices]
            cluster_center = torch.mean(cluster_embeddings, dim=0)
            self.prototype.append(cluster_center)

        g_ = addedges(g)
        self.expert_prototype.append(torch.mean(self.GNN_proj(g_, features)[train_ids], dim=0))
        self.Experts.append(self.current_expert)
        torch.cuda.empty_cache()
