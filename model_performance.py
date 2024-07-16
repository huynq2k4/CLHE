from accuracy_metrics import *
import pandas as pd
import json
import argparse
import os
from fair_metrics.Run_metrics_RecSys import metric_analysis as ma
from metric_utils.groupinfo import GroupInfo
import metric_utils.position as pos
from diversity_metrics import *

def convert_to_item_cate_matrix(group_item):
    # Get unique item IDs and category IDs
    item_ids = list(set(item_id for item_ids in group_item.values() for item_id in item_ids))
    category_ids = list(group_item.keys())

    # Create an item-category matrix with zeros
    item_cate_matrix = torch.zeros((len(item_ids), len(category_ids)), dtype=torch.float32)

    # Fill the matrix with ones where items belong to categories
    for item_id in item_ids:
        
        for category_id in category_ids:
       
            if item_id in group_item[category_id]:
                item_cate_matrix[int(item_id), int(category_id)] = 1.0

    return item_cate_matrix

def get_eval_repeat(dataset, size, file):
    dir = os.path.dirname(__file__)

    truth_file = f'{dir}/datasets/{dataset}/{dataset}_future.json'

    with open(truth_file, 'r') as f:
        data_truth = json.load(f)

    a_ndcg = []
    a_recall = []

    
    pred_file = f'{dir}/datasets/{dataset}/{dataset}_pred.json'

    with open(pred_file, 'r') as f:
        data_pred = json.load(f)
    
    ndcg = []
    recall = []

    for user in data_truth:
        if len(data_truth[user]) != 0:
            pred = data_pred[user]
            truth = data_truth[user]
            u_ndcg = get_NDCG(truth, pred, size)
            ndcg.append(u_ndcg)
            u_recall = get_Recall(truth, pred, size)
            recall.append(u_recall)


    
    a_ndcg.append(np.mean(ndcg))
    a_recall.append(np.mean(recall))

   


    file.write('recall: '+ str([round(num, 4) for num in a_recall]) +' '+ str(round(np.mean(a_recall), 4)) +' '+ str(round(np.std(a_recall) / np.sqrt(len(a_recall)), 4)) +'\n')
    file.write('ndcg: '+ str([round(num, 4) for num in a_ndcg]) +' '+ str(round(np.mean(a_ndcg), 4)) +' '+ str(round(np.std(a_ndcg) / np.sqrt(len(a_ndcg)), 4)) +'\n')



def get_eval_fairness(dataset, size, file, pweight): 
    dir = os.path.dirname(__file__)
    group_file = f'{dir}/datasets/{dataset}/bundle_popularity.json'
    with open(group_file, 'r') as f:
        group_item = json.load(f)
    group_dict = dict()
    for name, item in group_item.items():
        group_dict[name] = len(item) #the number of each group
    group = GroupInfo(pd.Series(group_dict), 'unpop', 'pop', 'popularity')

         
    EEL = []            
    EED = []             
    EER = []             
    DP = []           
    EUR = []          
    RUR = []       

    
    pred_file = f'{dir}/datasets/{dataset}/{dataset}_pred.json'
    

    with open(pred_file, 'r') as f:
        data_pred = json.load(f)

    

    truth_file = f'{dir}/datasets/{dataset}/{dataset}_future.json' # all users
    with open(truth_file, 'r') as f:
        data_truth = json.load(f)

    rows = []
    for user_id, items in data_truth.items():
        for i, item_id in enumerate(items):
            if item_id in group_item['pop']:
                rows.append((user_id, item_id, 1, 'pop', 1, 0))
            else:
                rows.append((user_id, item_id, 1, 'unpop', 0, 1))
    test_rates = pd.DataFrame(rows, columns=['user', 'item', 'rating', 'popularity', 'pop', 'unpop']) 
    
    row = [] #relev 
    ros = [] #recs
    for user_id, items in data_pred.items():
        
            for i, item_id in enumerate(items):
                if item_id in group_item['pop']:
                    row.append((user_id, item_id, 'pop', i+1))
                    if item_id in data_truth[user_id]:
                        ros.append((user_id, item_id, i+1, 'pop', 1, 1, 0))
                    else:
                        ros.append((user_id, item_id, i+1, 'pop', 0, 1, 0))
                else:
                    row.append((user_id, item_id, 'unpop', i+1))
                    if item_id in data_truth[user_id]:
                        ros.append((user_id, item_id, i+1, 'unpop', 1, 0, 1))
                    else:
                        ros.append((user_id, item_id, i+1, 'unpop', 0, 0, 1))
    recs = pd.DataFrame(ros, columns=['user', 'item', 'rank', 'popularity', 'rating', 'pop', 'unpop']) 
    relev = pd.DataFrame(row, columns=['user', 'item', 'popularity', 'rank']) #in line with recs

    MA = ma(recs, test_rates, group, original_relev=relev)
    default_results = MA.run_default_setting(listsize=size, pweight=pweight)

            
    EEL.append(default_results['EEL'])     
    EED.append(default_results['EED'])       
    EER.append(default_results['EER'])       
    DP.append(default_results['logDP'])          
    EUR.append(default_results['logEUR'])          
    RUR.append(default_results['logRUR'])      

    file.write('EEL: ' + str([round(num, 4) for num in EEL]) +' '+ str(round(np.mean(EEL), 4)) +' '+ str(round(np.std(EEL) / np.sqrt(len(EEL)), 4)) +'\n')
    file.write('EED: ' + str([round(num, 4) for num in EED]) +' '+ str(round(np.mean(EED), 4)) +' '+ str(round(np.std(EED) / np.sqrt(len(EED)), 4)) +'\n')
    file.write('EER: ' + str([round(num, 4) for num in EER]) +' '+ str(round(np.mean(EER), 4)) +' '+ str(round(np.std(EER) / np.sqrt(len(EER)), 4)) +'\n')
    file.write('DP: ' + str([round(num, 4) for num in DP]) +' '+ str(round(np.mean(DP), 4)) +' '+ str(round(np.std(DP) / np.sqrt(len(DP)), 4)) +'\n')
    file.write('EUR: ' + str([round(num, 4) for num in EUR]) +' '+ str(round(np.mean(EUR), 4)) +' '+ str(round(np.std(EUR) / np.sqrt(len(EUR)), 4)) +'\n')
    file.write('RUR: ' + str([round(num, 4) for num in RUR]) +' '+ str(round(np.mean(RUR), 4)) +' '+ str(round(np.std(RUR) / np.sqrt(len(RUR)), 4)) +'\n')

 

def get_eval_diversity(dataset, size, file): #evaluate diversity
    dir = os.path.dirname(__file__)
    group_file = f'{dir}/datasets/{dataset}/{dataset}_group_purchase_category.json'
    with open(group_file, 'r') as f:
        group_item = json.load(f)
    
    ILD = []
    ETP = []
    DS = []
    ETP_AGG = []
    GINI = []



    pred_file = f'{dir}/datasets/{dataset}/{dataset}_pred.json'
    

    with open(pred_file, 'r') as f:
        data_pred = json.load(f)

    
    test_dict = {user: data_pred[user][:size] + [0] * (size - len(data_pred[user][:size])) for user in data_pred}

    rank_list = torch.tensor(list(test_dict.values())) #torch.Size([user_num, size])

    item_cate_matrix = convert_to_item_cate_matrix(group_item)
    diversity = diversity_calculator(rank_list, item_cate_matrix)
    
    ILD.append(diversity['ild'])
    ETP.append(diversity['entropy'])
    DS.append(diversity['diversity_score'])
    ETP_AGG.append(diversity['entropy_aggregate'])
    GINI.append(diversity['gini'])


           
    #file.write('basket size: ' + str(size) + '\n')

    file.write('ILD: ' + str([round(num, 4) for num in ILD]) +' '+ str(round(np.mean(ILD), 4)) +' '+ str(round(np.std(ILD) / np.sqrt(len(ILD)), 4)) +'\n')
    file.write('ETP: ' + str([round(num, 4) for num in ETP]) +' '+ str(round(np.mean(ETP), 4)) +' '+ str(round(np.std(ETP) / np.sqrt(len(ETP)), 4)) +'\n')
    file.write('DS: ' + str([round(num, 4) for num in DS]) +' '+ str(round(np.mean(DS), 4)) +' '+ str(round(np.std(DS) / np.sqrt(len(DS)), 4)) +'\n')
    file.write('ETP_AGG: ' + str([round(num, 4) for num in ETP_AGG]) +' '+ str(round(np.mean(ETP_AGG), 4)) +' '+ str(round(np.std(ETP_AGG) / np.sqrt(len(ETP_AGG)), 4)) +'\n')
    file.write('GINI: ' + str([round(num, 4) for num in GINI]) +' '+ str(round(np.mean(GINI), 4)) +' '+ str(round(np.std(GINI) / np.sqrt(len(GINI)), 4)) +'\n')

 

def beyond_acc(dataset, topk, method_name):

    dir = os.path.dirname(__file__)
    eval_file = f'{dir}/datasets/{dataset}/eval_{method_name}.txt'
    f = open(eval_file, 'w')
    

    f.write('-------------'+dataset+'-------------- \n')
    for k in topk:
        f.write('list size: ' + str(k) + '\n')
        get_eval_repeat(dataset, k, f)
        get_eval_fairness(dataset, k, f, pweight='default')
        get_eval_diversity(dataset, k, f)
        f.write('\n')


if __name__ == '__main__':
    beyond_acc('Youshu', 20, 'CrossCBR')




