import torch.nn.functional as F
from .gnns import *

def get_model(dataset, args):
    # n_classes = args.n_cls_per_task
    if args.backbone == 'GAT':
        heads = ([args.GAT_args['heads']] * args.GAT_args['num_layers']) + [args.GAT_args['out_heads']]
        model = GAT(args, heads, F.elu)
    elif args.backbone == 'GCN':
        if args.method == "ours":
            model = GCN_base(args)
        else:
            model = GCN(args)
    elif args.backbone in ['CustomDecoupledSGC', 'CustomDecoupledS2GC', 'CustomDecoupledAPPNP', 'CustomFDGNN']:
        PDGNN = {'CustomDecoupledSGC':CustomDecoupledSGC, 'CustomDecoupledS2GC':CustomDecoupledS2GC, 'CustomDecoupledAPPNP':CustomDecoupledAPPNP, 'CustomFDGNN':CustomFDGNN}
        model = PDGNN[args.backbone](args)
    elif args.backbone == 'SGC':
        model = SGC(args)
    return model
