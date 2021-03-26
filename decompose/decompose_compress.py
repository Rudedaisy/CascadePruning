###################################################################
#   Provide essential tools to compress the decomposed model using
#   Prunning and Clustering Method
#
#   UNFINISHED RESEARCH CODE
#   DO NOT DISTRIBUTE
#
#   Author: Shiyu Li
#   Date:   12/02/2019
##################################################################

import torch
import torch.nn as nn
import numpy as np
import copy
from sklearn.cluster import KMeans

def prune_by_std(model, s=0.75, device='cuda'):
    new_model = copy.deepcopy(model)
    params = new_model.state_dict()
    thres= []
    for key, item in params.items():
        if 'coefs' in  key:
            weights = item.cpu().numpy()
            threshold = np.std(weights.flatten()) * s
            weights[weights<=threshold] = 0
            sparsity = np.sum(weights==0) / len(weights.flatten())
            print(key, " sparsity:", sparsity, ", Threshold:", threshold)
            params[key] = torch.tensor(weights).to(device)
            thres.append(threshold)

    new_model.load_state_dict(params)

    return new_model, thres

def quant(model, bits=8, device='cuda'):
    new_model = copy.deepcopy(model)
    params = new_model.state_dict()
    for key, item in  params.items():
        if 'coefs' in key:
            # Eliminate Prunned Values
            ori_weight = item.cpu().numpy()
            weight = ori_weight[ori_weight != 0].reshape(-1, 1)

            # Deal With Exception
            # No need to quantize
            if weight.shape[0] <= (2 ** bits):
                print("Parameter Size %d less than the encoding size %d, skip." % (weight.shape[0], 2 ** bits))
            else:
                # Initilize centroids with linear method
                _min = np.min(weight)
                _max = np.max(weight)

                cur_centeroids = np.linspace(_min, _max, num=2 ** bits).reshape(-1, 1)

                kmeans = KMeans(n_clusters=len(cur_centeroids), init=cur_centeroids, algorithm='full')
                kmeans.fit(weight)

                new_weight = np.zeros_like(ori_weight)
                new_weight[ori_weight != 0] = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)

                new_params[key] = torch.tensor(new_weight).to(device)

    model.load_state_dict(new_params)
    return model
