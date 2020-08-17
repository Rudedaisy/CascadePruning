import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def quantize_whole_model(net, bits=8):
    """
    Quantize the whole model.
    :param net: (object) network model.
    :return: centroids of each weight layer, used in the quantization codebook.
    """
    cluster_centers = []
    assert isinstance(net, nn.Module)
    layer_ind = 0
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            """
            Apply quantization for the PrunedConv layer.
            --------------Your Code---------------------
            """
            mask = m.conv.weight.data != 0
            # linearly initialize centroids
            max = torch.max(m.conv.weight.data).cpu().numpy()
            min = torch.min(m.conv.weight.data).cpu().numpy()
            step = (max - min) / (2.0**bits)
            cluster_centers_l = np.array(list(np.arange(min+step,max+step,step))).reshape(-1,1)
            # enforce proper shape
            cluster_centers_l = cluster_centers_l[:(2**bits)]
            # flatten weight vals; we only consider nonzero weights
            weights = m.conv.weight.data[mask].view(-1,1).cpu().numpy()
            # apply kmeans to obtain centroids and labels
            cluster = KMeans(n_clusters=2**bits,init=cluster_centers_l,n_init=1).fit(weights) 
            # approximate weights using corresponding centroids
            cluster_centers_l = cluster.cluster_centers_
            cluster_centers.append(cluster_centers_l)
            approx_weights = cluster_centers_l[cluster.predict(m.conv.weight.data.view(-1,1).cpu().numpy())]
            # update weight information
            m.conv.weight.data[mask] = (torch.from_numpy(approx_weights).float().to(device=device).view((m.conv.weight.data.size())))[mask]
            
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
        elif isinstance(m, PruneLinear):
            """
            Apply quantization for the PrunedLinear layer.
            --------------Your Code---------------------
            """
            mask = m.linear.weight.data != 0
            # linearly initialize centroids
            max = torch.max(m.linear.weight.data).cpu().numpy()
            min = torch.min(m.linear.weight.data).cpu().numpy()
            step = (max - min) / (2.0**bits)
            cluster_centers_l = np.array(list(np.arange(min+step,max+step,step))).reshape(-1,1)
            # enforce proper shape
            cluster_centers_l = cluster_centers_l[:(2**bits)]
            # flatten weight vals
            weights = m.linear.weight.data[mask].view(-1,1).cpu().numpy()
            # apply kmeans to obtain centroids and labels
            cluster = KMeans(n_clusters=2**bits,init=cluster_centers_l,n_init=1).fit(weights) 
            # approximate weights using corresponding centroids
            cluster_centers_l = cluster.cluster_centers_
            cluster_centers.append(cluster_centers_l)
            approx_weights = cluster_centers_l[cluster.predict(m.linear.weight.data.view(-1,1).cpu().numpy())]
            # update weight information
            m.linear.weight.data[mask] = (torch.from_numpy(approx_weights).float().to(device=device).view((m.linear.weight.data.size())))[mask]
            
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
    #print("ALL CENTERS: ",np.array(cluster_centers))
    return np.array(cluster_centers).reshape((layer_ind,-1))

