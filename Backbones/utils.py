import random
import pickle
import numpy as np
import torch
from torch import Tensor, device, dtype
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
# from dgl.data import CoraGraphDataset, CoraFullDataset, register_data_args, RedditDataset
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl, Evaluator
import copy
from sklearn.metrics import roc_auc_score, average_precision_score
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, RedditDataset, PubmedGraphDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset, CoauthorCSDataset, CoauthorPhysicsDataset, MUTAGDataset, AMDataset
import os
from dgl import save_graphs, load_graphs
from dgl.data import DGLDataset

class Linear_IL(nn.Linear):
    def forward(self, input: Tensor, n_cls=10000, normalize = True) -> Tensor:
        if normalize:
            return F.linear(F.normalize(input,dim=-1), F.normalize(self.weight[0:n_cls],dim=-1), bias=None)
        else:
            return F.linear(input, self.weight[0:n_cls], bias=None)

def accuracy(logits, labels, cls_balance=True, ids_per_cls=None):
    if cls_balance:
        logi = logits.cpu().numpy()
        _, indices = torch.max(logits, dim=1)
        ids = _.cpu().numpy()
        acc_per_cls = [torch.sum((indices == labels)[ids])/len(ids) for ids in ids_per_cls]
        return sum(acc_per_cls).item()/len(acc_per_cls)
    else:
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def mean_AP(args,logits, labels, cls_balance=True, ids_per_cls=None):
    eval_ogb = Evaluator(args.dataset)
    pos = (F.sigmoid(logits)>0.5)
    APs = 0
    if cls_balance:
        _, indices = torch.max(logits, dim=1)
        ids = _.cpu().numpy()
        acc_per_cls = [torch.sum((indices == labels)[ids])/len(ids) for ids in ids_per_cls]
        return sum(acc_per_cls).item()/len(acc_per_cls)
    else:
        input_dict = {"y_true": labels, "y_pred": logits}

        eval_result_ogb = eval_ogb.eval(input_dict)
        for c,ids in enumerate(ids_per_cls):
            TP_ = (pos[ids,c]*labels[ids,c]).sum()
            FP_ = (pos[ids,c]*(labels[ids, c]==False)).sum()
            med0 = TP_ + FP_ + 0.0001
            med1 = TP_ / med0
            APs += med1
        med2 = APs/labels.shape[1]

            #mAP_per_cls.append((TP / (TP+FP)).mean().item())
        #return (TP / (TP+FP)).mean().item()

        return med2.item()

def evaluate_batch(args,model, g, features, labels, mask, label_offset1, label_offset2, cls_balance=True, ids_per_cls=None):
    model.eval()
    with torch.no_grad():
        dataloader = dgl.dataloading.NodeDataLoader(g.cpu(), list(range(labels.shape[0])), args.nb_sampler, batch_size=args.batch_size, shuffle=False, drop_last=False)
        if args.method == "ours":
            output, output_l = model.forward_batch(dataloader)
        else:
            output = torch.tensor([]).to(device='cuda:0')
            output_l = torch.tensor([]).to(device='cuda:0')
            for input_nodes, output_nodes, blocks in dataloader:
                blocks = [b.to(device='cuda:0') for b in blocks]
                input_features = blocks[0].srcdata['feat']
                output_labels = blocks[-1].dstdata['label'].squeeze()
                output_predictions, _ = model.forward_batch(blocks, input_features)
                output = torch.cat((output,output_predictions),dim=0)
                output_l = torch.cat((output_l, output_labels), dim=0)

        #output, _ = model(g, features)
        #judget = (labels==output_l).sum()
        logits = output[:, label_offset1:label_offset2]
        if cls_balance:
            return accuracy(logits, labels.to(device='cuda:0'), cls_balance=cls_balance, ids_per_cls=ids_per_cls)
        else:
            return accuracy(logits[mask], labels[mask].to(device='cuda:0'), cls_balance=cls_balance, ids_per_cls=ids_per_cls)

def evaluate(args, model, g, features, labels, mask, label_offset1, label_offset2, cls_balance=True, ids_per_cls=None, save_logits_name=None):
    model.eval()
    with torch.no_grad():
        output, _ = model(g, features)
        logits = output[:, label_offset1:label_offset2]
        if cls_balance:
            return accuracy(logits, labels, cls_balance=cls_balance, ids_per_cls=ids_per_cls)
        else:
            return accuracy(logits[mask], labels[mask], cls_balance=cls_balance, ids_per_cls=ids_per_cls)

def evaluate_ours(args,model, g, features, labels, mask, label_offset1, label_offset2, cls_balance=True, ids_per_cls=None, task_max=0):
    model.eval()
    with torch.no_grad():
        output = model.forward(g, features, mask, task_max)
        logits = output[:, label_offset1:label_offset2]
        if cls_balance:
            return accuracy(logits, labels, cls_balance=cls_balance, ids_per_cls=ids_per_cls)
        else:
            return accuracy(logits[mask], labels[mask], cls_balance=cls_balance, ids_per_cls=ids_per_cls)

def evaluatewp(output, labels, mask, cls_balance=True, ids_per_cls=None):
    logits = output.detach()
    if cls_balance:
        return accuracy(logits, labels, cls_balance=cls_balance, ids_per_cls=ids_per_cls)
    else:
        return accuracy(logits[mask], labels[mask], cls_balance=cls_balance, ids_per_cls=ids_per_cls)


class incremental_graph_trans_(nn.Module):
    def __init__(self, dataset, n_cls):
        super().__init__()
        # transductive setting
        self.graph, self.labels = dataset[0]
        #self.graph = dgl.add_reverse_edges(self.graph)
        #self.graph = dgl.add_self_loop(self.graph)
        self.graph.ndata['label'] = self.labels
        self.d_data = self.graph.ndata['feat'].shape[1]
        self.n_cls = n_cls
        self.d_data = self.graph.ndata['feat'].shape[1]
        self.n_nodes = self.labels.shape[0]
        self.tr_va_te_split = dataset[1]

    def get_graph(self, tasks_to_retain=[], node_ids = None, remove_edges=True):
        # get the partial graph
        # tasks-to-retain: classes retained in the partial graph
        # tasks-to-infer: classes to predict on the partial graph
        node_ids_ = copy.deepcopy(node_ids)
        node_ids_retained = []
        ids_train_old, ids_valid_old, ids_test_old = [], [], []
        if len(tasks_to_retain) > 0:
            # retain nodes according to classes
            for t in tasks_to_retain:
                ids_train_old.extend(self.tr_va_te_split[t][0])
                ids_valid_old.extend(self.tr_va_te_split[t][1])
                ids_test_old.extend(self.tr_va_te_split[t][2])
                node_ids_retained.extend(self.tr_va_te_split[t][0] + self.tr_va_te_split[t][1] + self.tr_va_te_split[t][2])
            subgraph_0 = dgl.node_subgraph(self.graph, node_ids_retained, store_ids=True)
            if node_ids_ is None:
                subgraph = subgraph_0
        if node_ids_ is not None:
            # retrain the given nodes
            if not isinstance(node_ids_[0],list):
                # if nodes are not divided into different tasks
                subgraph_1 = dgl.node_subgraph(self.graph, node_ids_, store_ids=True)
                if remove_edges:
                    # to facilitate the methods like ER-GNN to only retrieve nodes
                    n_edges = subgraph_1.edges()[0].shape[0]
                    subgraph_1.remove_edges(list(range(n_edges)))
            elif isinstance(node_ids_[0],list):
                # if nodes are diveded into different tasks
                subgraph_1 = dgl.node_subgraph(self.graph, node_ids_[0], store_ids=True) # load the subgraph containing nodes of the first task
                node_ids_.pop(0)
                for ids in node_ids_:
                    # merge the remaining nodes
                    subgraph_1 = dgl.batch([subgraph_1,dgl.node_subgraph(self.graph, ids, store_ids=True)])

            if len(tasks_to_retain)==0:
                subgraph = subgraph_1

        if len(tasks_to_retain)>0 and node_ids is not None:
            subgraph = dgl.batch([subgraph_0,subgraph_1])

        old_ids = subgraph.ndata['_ID'].cpu()
        # ids_train = [(old_ids == i).nonzero()[0][0].item() for i in ids_train_old]
        # ids_val = [(old_ids == i).nonzero()[0][0].item() for i in ids_valid_old]
        # ids_test = [(old_ids == i).nonzero()[0][0].item() for i in ids_test_old]
        ids_train = [(old_ids == i).nonzero(as_tuple=False)[0][0].item() for i in ids_train_old]
        ids_val = [(old_ids == i).nonzero(as_tuple=False)[0][0].item() for i in ids_valid_old]
        ids_test = [(old_ids == i).nonzero(as_tuple=False)[0][0].item() for i in ids_test_old]
        node_ids_per_task_reordered = []
        for c in tasks_to_retain:
            # ids = (subgraph.ndata['label'] == c).nonzero()[:, 0].view(-1).tolist()
            ids = (subgraph.ndata['label'] == c).nonzero(as_tuple=False)[:, 0].view(-1).tolist()
            node_ids_per_task_reordered.append(ids)
        subgraph = dgl.add_self_loop(subgraph)

        return subgraph, node_ids_per_task_reordered, [ids_train, ids_val, ids_test]

def train_valid_test_split(ids,ratio_valid_test):
    va_te_ratio = sum(ratio_valid_test)
    train_ids, va_te_ids = train_test_split(ids, test_size=va_te_ratio)
    return [train_ids] + train_test_split(va_te_ids, test_size=ratio_valid_test[1]/va_te_ratio)

def load(name):
    graph_path = os.path.join(f'datas/{name}.bin')
    data = load_graphs(graph_path)
    return data

def get_dataset(name,args):
    if name[0:4] == 'ogbn':
        data = DglNodePropPredDataset(name, root=f'{args.ori_data_path}/ogb_downloaded')
        graph, label = data[0]
    elif name == 'Cora':
        data = CoraGraphDataset()
        graph, label = data[0], data[0].dstdata['label'].view(-1, 1)
    elif name == 'Citeseer':
        data = CiteseerGraphDataset()
        graph, label = data[0], data[0].dstdata['label'].view(-1, 1)
    elif name == 'Pubmed':
        data = PubmedGraphDataset()
        graph, label = data[0], data[0].dstdata['label'].view(-1, 1)
    elif name == 'Computer':
        data = AmazonCoBuyComputerDataset()
        graph, label = data[0], data[0].dstdata['label'].view(-1, 1)
    elif name == 'Photo':
        data = AmazonCoBuyPhotoDataset()
        graph, label = data[0], data[0].dstdata['label'].view(-1, 1)
    elif name == 'CoauthorCS':
        data = CoauthorCSDataset()
        graph, label = data[0], data[0].dstdata['label'].view(-1, 1)
    elif name == 'CoauthorPhysics':
        data = CoauthorPhysicsDataset()
        graph, label = data[0], data[0].dstdata['label'].view(-1, 1)
    elif name == 'WikiCS':
        data = load('WikiCS')
        graph, label = data[0][0], data[1]['labels']
    elif name == 'Chameleon':
        data = load('Chameleon')
        graph, label = data[0][0], data[1]['labels']
    elif name == 'Squirrel':
        data = load('Squirrel')
        graph, label = data[0][0], data[1]['labels']
    elif name == 'DBLP':
        data = load('DBLP')
        graph, label = data[0][0], data[1]['labels']
    elif name == 'Facebook':
        data = load('Facebook')
        graph, label = data[0][0], data[1]['labels']
    elif name == 'GitHub':
        data = load('GitHub')
        graph, label = data[0][0], data[1]['labels']
    elif name == 'LastFMAsia':
        data = load('LastFMAsia')
        graph, label = data[0][0], data[1]['labels']
    elif name == 'DeezerEurope':
        data = load('DeezerEurope')
        graph, label = data[0][0], data[1]['labels']
    elif name == 'Airport':
        data = load('Airport')
        graph, label = data[0][0], data[1]['labels']
    else:
        print('invalid data name')
        return None,None
    return data, graph, label


class NodeLevelDataset(incremental_graph_trans_):
    def __init__(self,name,default_split=False,ratio_valid_test=None,args=None):

        args.task_seq = []
        self.graphs = []
        self.labels = []
        self.tr_va_te_split = {}
        label_offset = 0
        node_offset = 0

        for name in args.multi_datasets:
            data, graph, label = get_dataset(name,args)
            
            feats = graph.ndata['feat']
            if args.latdim > feats.shape[0] or args.latdim > feats.shape[1]:
                dim = min(feats.shape[0], feats.shape[1])
                decom_feats, s, decom_featdim = torch.svd_lowrank(feats, q=dim, niter=args.niter)
                decom_feats = torch.cat([decom_feats, torch.zeros([decom_feats.shape[0], args.latdim-dim])], dim=1)
                s = torch.cat([s, torch.zeros(args.latdim - dim)])
            else:
                decom_feats, s, decom_featdim = torch.svd_lowrank(feats, q=args.latdim, niter=args.niter)
            decom_feats = decom_feats @ torch.diag(torch.sqrt(s))

            graph.ndata['feat'] = decom_feats

            n_cls = len(torch.unique(label.view(-1)))
            cls = [i for i in range(n_cls)]
            cls_id_map = {i: list((label.squeeze() == i).nonzero(as_tuple=False).squeeze().view(-1, ).numpy()) for i in cls}
            cls_sizes = {c: len(cls_id_map[c]) for c in cls_id_map}
            for c in cls_sizes:
                if cls_sizes[c] < 2:
                    cls.remove(c) # remove classes with less than 2 examples, which cannot be split into train, val, test sets
            cls_id_map = {i: list((label.squeeze() == i).nonzero(as_tuple=False).squeeze().view(-1, ).numpy()) for i in cls}
            n_cls = len(cls)

            args.task_seq.append(list(range(label_offset,label_offset+n_cls)))
            label += label_offset
            self.graphs.append(graph)
            self.labels.append(label)
            
            split_name = f'{args.data_path}/tr{round(1-ratio_valid_test[0]-ratio_valid_test[1],2)}_va{ratio_valid_test[0]}_te{ratio_valid_test[1]}_split_{name}.pkl'
            try:
                tr_va_te_split = pickle.load(open(split_name, 'rb')) # could use same split across different experiments for consistency
            except:
                if ratio_valid_test[1] > 0:
                    tr_va_te_split = {c: train_valid_test_split(cls_id_map[c], ratio_valid_test=ratio_valid_test)
                                    for c in
                                    cls}
                    print(f'splitting is {ratio_valid_test}')
                elif ratio_valid_test[1] == 0:
                    tr_va_te_split = {c: [cls_id_map[c], [], []] for c in
                                    cls}
                with open(split_name, 'wb') as f:
                    pickle.dump(tr_va_te_split, f)

            for key, value in tr_va_te_split.items():
                new_key = key + label_offset
                if new_key in self.tr_va_te_split:
                    raise ValueError(f"Key {new_key} already exists in the total dictionary.")
                new_value = [[node + node_offset for node in lst] for lst in value]
                self.tr_va_te_split[new_key] = new_value

            label_offset += n_cls  
            node_offset += graph.num_nodes()

        for g in self.graphs:
            for key in list(g.nodes['_N'].data.keys()):
                if key not in ['feat']:
                    del g.nodes['_N'].data[key]
            if '__orig__' in g.edata:
                del g.edata['__orig__']

        self.graph = dgl.batch(self.graphs)
        self.label = torch.cat(self.labels)

        super().__init__([[self.graph, self.label], self.tr_va_te_split], label_offset)