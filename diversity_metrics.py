import torch
import numpy as np
from scipy.stats import entropy
import sys


def diversity_calculator(rank_list, item_cate_matrix, size):
    rank_list = rank_list.long()
    ILD_perList = []
    IE_perList = []
    ds_perList = []
    num_item = item_cate_matrix.shape[0]
    freq_item = np.full(num_item, 0.0, dtype=float)

    for b in range(rank_list.size(0)): #for each user, size of the first dimension
        ILD = []
        cate_dis = torch.Tensor([])
        # sess_len = rank_list.size(1)
    
        for i in range(size): #for each item
            freq_item[rank_list[b, i].item()] += 1.0
            item_i_cate = item_cate_matrix[rank_list[b, i].item()] #rank_list[b, i].item() is item_id, which is also item_index

            cate_dis = torch.cat((cate_dis, item_i_cate), 0) #add a layer of item_i_cate

            for j in range(i + 1, size):
                item_j_cate = item_cate_matrix[rank_list[b, j].item()]

                distance = np.linalg.norm(np.array(item_i_cate) - np.array(item_j_cate)) #Euclidean distance, 0 or sqrt(2)
    
                ILD.append(distance)
                
        ILD_perList.append(np.mean(ILD)) 
        
        cate_dis_sum = cate_dis.reshape(-1, item_cate_matrix.shape[1]).sum(axis=0) #how much items for each category
        IE_perList.append(entropy(np.array(cate_dis_sum/sum(cate_dis_sum)), base=2))
        ds_perList.append(len(cate_dis_sum.nonzero())/size)

    ILD_batch = torch.FloatTensor(ILD_perList)
    ILD_batch[ILD_batch!=ILD_batch] = 0 #replace NaN with 0

    entropy_batch = torch.FloatTensor(IE_perList)
    entropy_batch[entropy_batch!=entropy_batch] = 0 #replace NaN with 0

    freq_item = freq_item / np.sum(freq_item)
    freq_item = np.sort(freq_item)
    entropy_aggregate = entropy(freq_item, base=2)
    gini_vector = [(2*i-num_item+1)*freq_item[i] for i in range(num_item)]
    gini = np.sum(gini_vector) / (num_item - 1)
    
    metrics = {
        'ild': ILD_batch.mean().item(),
        'entropy': entropy_batch.mean().item(),
        'diversity_score': torch.FloatTensor(ds_perList).mean().item(),
        'entropy_aggregate': entropy_aggregate,
        'gini': gini
    }
    return metrics


        

