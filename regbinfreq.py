import math
import numpy as np

import export

path = '/root/hostCurUser/root/CascadePruning/ckpt/ResNet03270957/retrain_weight_73_0.94.pt'
modelName = 'resnet50_c10'
chunkSize = 32

layers = export.loadCoeffIdx(path, modelName)
layers = layers

sq_ptrs = []
full_ptrs = []
ratios = []
total_ptrs = []
first_row_IFM_accesses = 0
for idx, layer in enumerate(layers):
    sq_ptrs.append([])
    full_ptrs.append([])
    ratios.append([])
    num_fold = math.ceil(len(layer) / chunkSize)
    layer = layer > 0

    for chunk_idx in range(num_fold):
        
        sq_ptrs[idx].append(0)
        full_ptrs[idx].append(len(layer[0]))
        if (chunk_idx+1) * chunkSize >= len(layer):
            maxL = len(layer)
        else:
            maxL = (chunk_idx+1) * chunkSize

        chunk = layer[chunk_idx * chunkSize:(chunk_idx * chunkSize + maxL)]
        sq_ptrs[idx][chunk_idx] = len(np.where(chunk.any(axis=0))[0])

        if chunk_idx == 0:
            first_row_IFM_accesses += len(np.where(chunk.any(axis=0))[0])

        if chunk_idx >= len(total_ptrs):
            total_ptrs.append(0)
        total_ptrs[chunk_idx] += len(np.where(chunk.any(axis=0))[0])

        if chunk_idx == 0:
            ratios[idx].append(sq_ptrs[idx][chunk_idx] / len(layer[0]))
        elif sq_ptrs[idx][chunk_idx-1] != 0:
            ratios[idx].append(sq_ptrs[idx][chunk_idx] / sq_ptrs[idx][chunk_idx-1])
        else:
            ratios[idx].append(0)

accelerator_ranges = [0, 2, 6, 14, 30, 62, 126, 254, 510]

weightedMean = 0
accelerator_ptrs = []
for idx in range(len(total_ptrs)):
    weightedMean += total_ptrs[idx] * (idx+1)

    acc_idx = -1
    for i in range(len(accelerator_ranges)-1):
        if idx >= accelerator_ranges[i] and idx < accelerator_ranges[i+1]:
            acc_idx = i
            break
    if acc_idx == -1:
        print("ERR: idx {} not supported in accelerator_range".format(idx))
        exit(1)
    if acc_idx >= len(accelerator_ptrs):
        accelerator_ptrs.append(0)
    accelerator_ptrs[acc_idx] += float(total_ptrs[idx])
m = float(max(accelerator_ptrs))
alpha = 0
numBins = 3
for idx in range(len(accelerator_ptrs)):
    print(str(accelerator_ptrs[idx]) + "/" + str(m))
    accelerator_ptrs[idx] = float(accelerator_ptrs[idx]) / m
    alpha += (accelerator_ptrs[idx] * ((accelerator_ranges[idx+1] - accelerator_ranges[idx]) / accelerator_ranges[numBins]))
for idx in range(len(total_ptrs)-1):
    total_ptrs[idx] -= total_ptrs[idx+1]

print(total_ptrs)
print(accelerator_ptrs)
