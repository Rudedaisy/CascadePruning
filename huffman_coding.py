import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# find the 2 smallest values in 'counts'
# idx1 is smallest, idx2 is 2nd smallest
def find_min_two(counts):
    min1 = min(counts)
    idx1 = np.argmin(counts)
    min2 = max(counts) + 1
    for i in range(len(counts)):
        if (counts[i] < min2) and i != idx1:
            min2 = counts[i]
            idx2 = i
    return idx1, idx2

def _huffman_coding_per_layer(weight, centers):
    """
    Huffman coding for each layer
    :param weight: weight parameter of the current layer.
    :param centers: KMeans centroids in the quantization codebook of the current weight layer.
    :return: 
            'encodings': Encoding map mapping each weight parameter to its Huffman coding.
            'frequency': Frequency map mapping each weight parameter to the total number of its appearance.
            'encodings' should be in this format:
            {"0.24315": '0', "-0.2145": "100", "1.1234e-5": "101", ...
            }
            'frequency' should be in this format:
            {"0.25235": 100, "-0.2145": 42, "1.1234e-5": 36, ...
            }
            'encodings' and 'frequency' does not need to be ordered in any way.
    """
    """
    Generate Huffman Coding and Frequency Map according to incoming weights and centers (KMeans centriods).
    --------------Your Code---------------------
    """
    # gather frequencies of each centroid
    frequency = {} # initialize dict
    counts = [0]*len(centers)
    for i in range(len(centers)):
        for val in weight.reshape(-1):
            if(val == centers[i]):
                counts[i] += 1
        frequency[str(centers[i])] = counts[i]
    
    # generate encodings
    encodings = {}
    encodes = [""]*len(counts)
    clumps = []
    for i in range(len(counts)):
        clumps.append([i])
    while(len(clumps) > 1):
        idx1, idx2 = find_min_two(counts)
        # append to encodings of each instance in the clumps
        for idx in clumps[idx1]:
            encodes[idx] = "1" + encodes[idx]
        for idx in clumps[idx2]:
            encodes[idx] = "0" + encodes[idx]
        # merge clumps
        counts[idx1] += counts[idx2]
        counts.pop(idx2)
        clumps[idx1] += clumps[idx2]
        clumps.pop(idx2)
    for i in range(len(encodes)):
        encodings[str(centers[i])] = encodes[i]
    #print("ENCODINGS: ",encodings)
    #print("FREQUENCIES: ",frequency)
    
    
    return encodings, frequency


def compute_average_bits(encodings, frequency):
    """
    Compute the average storage bits of the current layer after Huffman Coding.
    :param 'encodings': Encoding map mapping each weight parameter to its Huffman coding.
    :param 'frequency': Frequency map mapping each weight parameter to the total number of its appearance.
            'encodings' should be in this format:
            {"0.24315": '0', "-0.2145": "100", "1.1234e-5": "101", ...
            }
            'frequency' should be in this format:
            {"0.25235": 100, "-0.2145": 42, "1.1234e-5": 36, ...
            }
            'encodings' and 'frequency' does not need to be ordered in any way.
    :return (float) a floating value represents the average bits.
    """
    total = 0
    total_bits = 0
    for key in frequency.keys():
        total += frequency[key]
        total_bits += frequency[key] * len(encodings[key])
    return total_bits / total

def huffman_coding(net, centers):
    """
    Apply huffman coding on a 'quantized' model to save further computation cost.
    :param net: a 'nn.Module' network object.
    :param centers: KMeans centroids in the quantization codebook for Huffman coding.
    :return: frequency map and encoding map of the whole 'net' object.
    """
    assert isinstance(net, nn.Module)
    layer_ind = 0
    freq_map = []
    encodings_map = []
    all_avg_bits = []
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            weight = m.conv.weight.data.cpu().numpy()
            center = centers[layer_ind]
            orginal_avg_bits = round(np.log2(len(center)))
            print("Original storage for each parameter: %.4f bits" %orginal_avg_bits)
            encodings, frequency = _huffman_coding_per_layer(weight, center)
            freq_map.append(frequency)
            encodings_map.append(encodings)
            huffman_avg_bits = compute_average_bits(encodings, frequency)
            all_avg_bits.append(huffman_avg_bits)
            print("Average storage for each parameter after Huffman Coding: %.4f bits" %huffman_avg_bits)
            layer_ind += 1
            print("Complete %d layers for Huffman Coding..." %layer_ind)
        elif isinstance(m, PruneLinear):
            weight = m.linear.weight.data.cpu().numpy()
            center = centers[layer_ind]
            orginal_avg_bits = round(np.log2(len(center)))
            print("Original storage for each parameter: %.4f bits" %orginal_avg_bits)
            encodings, frequency = _huffman_coding_per_layer(weight, center)
            freq_map.append(frequency)
            encodings_map.append(encodings)
            huffman_avg_bits = compute_average_bits(encodings, frequency)
            all_avg_bits.append(huffman_avg_bits)
            print("Average storage for each parameter after Huffman Coding: %.4f bits" %huffman_avg_bits)
            layer_ind += 1
            print("Complete %d layers for Huffman Coding..." %layer_ind)

    return freq_map, encodings_map, all_avg_bits