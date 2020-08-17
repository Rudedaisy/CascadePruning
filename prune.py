import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn


def prune(net, method='std', q=5.0, s=0.25):
    # Before the training started, generate the mask
    assert isinstance(net, nn.Module)
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv) or isinstance(m, PruneLinear):
            if method == 'percentage':
                m.prune_by_percentage(q)
            elif method == 'std':
                m.prune_by_std(s)

