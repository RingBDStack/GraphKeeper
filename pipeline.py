import os
import pickle
import numpy as np
import torch
from Backbones.model_factory import get_model
from Backbones.utils import evaluate, NodeLevelDataset, evaluate_batch, evaluatewp, evaluate_ours
from training.utils import mkdir_if_missing
from dataset.utils import semi_task_manager
import importlib
import copy
import dgl


joint_alias = ['joint', 'Joint', 'joint_replay_all', 'jointtrain']
def get_pipeline(args):
    # choose the pipeline for the chosen setting
    if args.method in joint_alias:
        return pipeline_domain_IL_no_inter_edge_joint
    else:
        return pipeline_domain_IL_no_inter_edge


def data_prepare_DIL(args):
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    args.n_tasks = len(args.task_seq)
    n_cls_so_far = 0
    # check whether the preprocessed data exist and can be loaded
    str_int_tsk = 'no_inter_tsk_edge'
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        try:
            if args.load_check:
                subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(
                    f'{args.data_path}/{str_int_tsk}/{args.dataset}_{task_cls[0]}_{task_cls[-1]}.pkl', 'rb'))
            else:
                if f'{args.dataset}_{task_cls[0]}_{task_cls[-1]}.pkl' not in os.listdir(f'{args.data_path}/{str_int_tsk}'):
                    subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(
                        f'{args.data_path}/{str_int_tsk}/{args.dataset}_{task_cls[0]}_{task_cls[-1]}.pkl', 'rb'))
        except:
            # if not exist or cannot be loaded correctly, create new processed data
            print(f'preparing data for task {task}')
            mkdir_if_missing(f'{args.data_path}/inter_tsk_edge')
            mkdir_if_missing(f'{args.data_path}/no_inter_tsk_edge')
            if args.inter_task_edges:
                cls_retain = []
                for clss in args.task_seq[0:task + 1]:
                    cls_retain.extend(clss)
                subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = dataset.get_graph(
                    tasks_to_retain=cls_retain)
                with open(f'{args.data_path}/inter_tsk_edge/{args.dataset}_{task_cls[0]}_{task_cls[-1]}.pkl', 'wb') as f:
                    pickle.dump([subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids]], f)
            else:
                subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = dataset.get_graph(tasks_to_retain=task_cls)
                with open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls[0]}_{task_cls[-1]}.pkl', 'wb') as f:
                    pickle.dump([subgraph, ids_per_cls, [train_ids, valid_ids, test_ids]], f)



def pipeline_domain_IL_no_inter_edge(args, valid=False):
    data_prepare_DIL(args)
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    n_cls_so_far = 0
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls[0]}_{task_cls[-1]}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        if args.method == 'tpp':
            save_proto_name = save_model_name + '_prototypes'
            save_proto_path = f'{args.result_path}/{subfolder_c}val_models/{save_proto_name}.pkl'
        n_cls_so_far+=len(task_cls)
        subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(
            open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls[0]}_{task_cls[-1]}.pkl', 'rb'))
        subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)
        label_offset1, label_offset2 = task_manager.get_label_offset(task)

        for epoch in range(epochs):
            if args.method == 'ours':
                life_model_ins.observe(args, subgraph, features, labels, task, train_ids, ids_per_cls, dataset, epochs, args.multi_datasets)
                break
            else:
                life_model_ins.observe(args, subgraph, features, labels, task, train_ids, ids_per_cls, dataset)
            torch.cuda.empty_cache()


        if not valid:
            try:
                if args.method == "ours":
                    life_model_ins = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
                else:
                    model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
            except:
                if args.method == "ours":
                    life_model_ins.load_state_dict(torch.load(save_model_path.replace('.pkl','.pt')))
                else:
                    model.load_state_dict(torch.load(save_model_path.replace('.pkl','.pt')))
        acc_mean = []

        for t in range(task+1):
            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(open(
                f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t][0]}_{args.task_seq[t][-1]}.pkl', 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            test_ids = valid_ids_ if valid else test_ids_
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            if args.classifier_increase:
                if args.method == "ours":
                    acc = evaluate_ours(args,life_model_ins, subgraph, features, labels, test_ids, label_offset1, label_offset2, cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test, task_max=task+1)
                else:
                    acc = evaluate(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2, cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)

            acc_matrix[task][t] = round(acc*100,2)
            acc_mean.append(acc)

        accs = acc_mean[:task+1]
        meana = round(np.mean(accs)*100,2)
        meanas.append(meana)
        acc_mean = round(np.mean(acc_mean)*100,2)

        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            try:
                with open(save_model_path, 'wb') as f:
                    if args.method == "ours":
                        pickle.dump(life_model_ins, f)
                    else:
                        pickle.dump(model, f) # save the best model for each hyperparameter composition
            except:
                if args.method == "ours":
                    torch.save(life_model_ins.state_dict(), save_model_path.replace('.pkl','.pt'))
                else:
                    torch.save(model.state_dict(), save_model_path.replace('.pkl','.pt'))
        if args.method == 'lwf':
            prev_model = copy.deepcopy(model).cuda()

    # print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks-1):
        b = acc_matrix[args.n_tasks-1][t]-acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward),2)
    # print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix



def pipeline_domain_IL_no_inter_edge_joint(args, valid=False):
    args.method = 'joint_replay_all'
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    data_prepare_DIL(args)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    print(args.task_seq)
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    n_cls_so_far = 0
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls[0]}_{task_cls[-1]}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        n_cls_so_far += len(task_cls)
        task_manager.add_task(task, n_cls_so_far)
        subgraphs, featuress, labelss, train_idss, ids_per_clss = [], [], [], [], []
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(open(
                f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t][0]}_{args.task_seq[t][-1]}.pkl', 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            subgraphs.append(subgraph)
            featuress.append(features)
            labelss.append(labels)
            train_idss.append(train_ids)
            ids_per_clss.append(ids_per_cls)

        for epoch in range(epochs):
            life_model_ins.observe(args, subgraphs, featuress, labelss, task, train_idss, ids_per_clss, dataset)

        label_offset1, label_offset2 = task_manager.get_label_offset(task)
        if not valid:
            try:
                model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
            except:
                model.load_state_dict(torch.load(save_model_path.replace('.pkl','.pt')))
        acc_mean = []
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(open(
                f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t][0]}_{args.task_seq[t][-1]}.pkl', 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            test_ids = valid_ids_ if valid else test_ids_
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            if args.classifier_increase:
                acc = evaluate(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            try:
                with open(save_model_path, 'wb') as f:
                    pickle.dump(model, f) # save the best model for each hyperparameter composition
            except:
                torch.save(model.state_dict(), save_model_path.replace('.pkl','.pt'))

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix
