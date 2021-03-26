###################################################################
#   Matrix Permutation Algorithm
#
#   UNFINISHED RESEARCH CODE
#   DO NOT DISTRIBUTE
#
#   Author: Shiyu Li
#   Date:   03/09/2020
##################################################################
import numpy as np
from itertools import combinations
import copy

def score(matrix, chunk_size=32, verbose=True):
    '''
    :param matrix: Coefficient Matrix, should be in size of out_channels x (in_channels x basis)
    :param chunk_size: number of chunks
    :return:
    '''

    score  = 0
    n_chunk = matrix.shape[0] // chunk_size + \
              (1 if matrix.shape[0] % chunk_size !=0 else 0)

    for chunk_id in range(n_chunk):
        chunk = matrix[chunk_id * chunk_size : (chunk_id+1) * chunk_size, :] \
                if chunk_id * chunk_size < matrix.shape[0] else \
                matrix[chunk_id * chunk_size: , :]
        score += np.sum(np.sum(chunk, axis=0)==0)

        if verbose:
            # For debug purpose only
            print("Chunk ID:%d, local score:%.2f"%(chunk_id, \
                    np.sum(np.sum(chunk, axis=0)==0) / matrix.shape[1]))

    score = score / (n_chunk * matrix.shape[1])

    return score

def greedy_permute(matrix, chunksize, step=512):
    pre_score = score(matrix, chunksize)

    permuted_mat = copy.deepcopy(matrix)

    for s in range(step):
        indices = [c for c in combinations(range(matrix.shape[1]), 2)]
        for perm in indices:
            cur_ind = list(range(matrix.shape[1]))
            cur_ind[perm[0]] = perm[1]
            cur_ind[perm[1]] = perm[0]
            #print(cur_ind)

            new_matrix = permuted_mat[:, cur_ind]
            new_score = score(new_matrix, chunksize, verbose=False)
            if new_score>pre_score:
                print("Step %d: %.4f" %(s, new_score))
                permuted_mat = new_matrix
                pre_score = new_score
                break
    return permuted_mat

def brute_permute(matrix, chunksize):
    pre_score = score(matrix, chunksize)
    new_score = 1
    permuted_mat = copy.deepcopy(matrix)

    steps = 0

    while new_score > pre_score:
        pre_score = new_score
        best_score = 0
        best_mat = []
        indices = [c for c in combinations(range(matrix.shape[1]), 2)]
        for perm in indices:
            new_matrix = permuted_mat[:, perm]
            new_s = score(new_matrix, chunksize, verbose=False)
            if new_score > best_score:
                best_score = new_s
                best_mat = new_matrix
        new_score = best_score
        permuted_mat = best_mat
        steps += 1

    print("Algorithm successfully finished!\n%d Steps, Score %.4f->%.4f"\
          %(steps,  score(matrix, chunksize), new_score))
