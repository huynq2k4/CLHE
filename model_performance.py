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

def get_eval_accuracy(dataset, data_pred, data_truth, topk):
    # dir = os.path.dirname(__file__)

    # truth_file = f'{dir}/datasets/{dataset}/{dataset}_future.json'

    # with open(truth_file, 'r') as f:
    #     data_truth = json.load(f)

    a_ndcg = []
    a_recall = []

    
    # pred_file = f'{dir}/datasets/{dataset}/{dataset}_pred.json'

    # with open(pred_file, 'r') as f:
    #     data_pred = json.load(f)
    for size in topk:
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
    print("Done accuracy!")
    return a_recall, a_ndcg



def get_eval_fairness(dataset, data_pred, data_truth, topk, pop_rate, pweight, pop_type='B-I'): 
    dir = os.path.dirname(__file__)
    group_file = f'{dir}/datasets/{dataset}/{dataset}_group_purchase_popularity.json'
    with open(group_file, 'r') as f:
        group_item = json.load(f)

    popularity = group_item[pop_type]
    pop_threshold = int(len(popularity) * pop_rate)
    group_item['pop'] = popularity[:pop_threshold]
    group_item['unpop'] = popularity[pop_threshold:]
    
    group_dict = dict()
    for name, item in group_item.items():
        group_dict[name] = len(item) #the number of each group
    group = GroupInfo(pd.Series(group_dict), 'unpop', 'pop', 'popularity')
    
    fair_res = {'EEL': [], 'EED': [], 'EER': [], 'DP': [], 'EUR': [], 'RUR': []}
    
    # pred_file = f'{dir}/datasets/{dataset}/{dataset}_pred.json'
    

    # with open(pred_file, 'r') as f:
    #     data_pred = json.load(f)

    

    # truth_file = f'{dir}/datasets/{dataset}/{dataset}_future.json' # all users
    # with open(truth_file, 'r') as f:
    #     data_truth = json.load(f)
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
    for size in topk:
        default_results = MA.run_default_setting(listsize=size, pweight=pweight)

                
        fair_res['EEL'].append(default_results['EEL'])     
        fair_res['EED'].append(default_results['EED'])       
        fair_res['EER'].append(default_results['EER'])       
        fair_res['DP'].append(default_results['logDP'])          
        fair_res['EUR'].append(default_results['logEUR'])          
        fair_res['RUR'].append(default_results['logRUR']) 

    print("Done fairness!")
    return fair_res     

 

def get_eval_diversity(dataset, data_pred, topk): #evaluate diversity
    dir = os.path.dirname(__file__)
    group_file = f'{dir}/datasets/{dataset}/{dataset}_group_purchase_category.json'
    with open(group_file, 'r') as f:
        group_item = json.load(f)
    
    div_res = {'ILD': [], 'ETP': [], 'DS': [], 'ETP_AGG': [], 'GINI': []}



    # pred_file = f'{dir}/datasets/{dataset}/{dataset}_pred.json'
    

    # with open(pred_file, 'r') as f:
    #     data_pred = json.load(f)

    item_cate_matrix = convert_to_item_cate_matrix(group_item)
    rank_list = torch.tensor(list(data_pred.values()))
    
    for size in topk:
        # test_dict = {user: data_pred[user][:size] + [0] * (size - len(data_pred[user][:size])) for user in data_pred}
        # rank_list = torch.tensor(list(test_dict.values())) #torch.Size([user_num, size])

        diversity = diversity_calculator(rank_list, item_cate_matrix, size)
    
        div_res['ILD'].append(diversity['ild'])
        div_res['ETP'].append(diversity['entropy'])
        div_res['DS'].append(diversity['diversity_score'])
        div_res['ETP_AGG'].append(diversity['entropy_aggregate'])
        div_res['GINI'].append(diversity['gini'])

    print("Done diversity!")
    return div_res

 

def beyond_acc(dataset, data_pred, data_truth, topk, method_name, pop_rate):

    if type(topk) != list:
        topk = [topk]

    recall, ndcg = get_eval_accuracy(dataset, data_pred, data_truth, topk)
    fairness_UI = get_eval_fairness(dataset, data_pred, data_truth, topk, pop_rate, pweight='default', pop_type='U-I')
    fairness_BI = get_eval_fairness(dataset, data_pred, data_truth, topk, pop_rate, pweight='default', pop_type='B-I')
    
    diversity = get_eval_diversity(dataset, data_pred, topk)

    dir = os.path.dirname(__file__)
    eval_file = f'{dir}/datasets/{dataset}/eval_{method_name}.txt'
    f = open(eval_file, 'w')
    

    f.write('-------------'+dataset+'-------------- \n')
    

    for i, k in enumerate(topk):
        f.write('LIST SIZE = ' + str(k) + '\n')
        f.write('Accuracy:\n')
        f.write('recall: '+ str(round(recall[i], 4)) + '\n')
        f.write('ndcg: '+ str(round(ndcg[i], 4)) + '\n')
        f.write('Fairness UI:\n')
        f.write('EEL: '+ str(round(fairness_UI['EEL'][i], 4)) + '\n')
        f.write('EED: '+ str(round(fairness_UI['EED'][i], 4)) + '\n')
        f.write('EER: '+ str(round(fairness_UI['EER'][i], 4)) + '\n')
        f.write('DP: '+ str(round(fairness_UI['DP'][i], 4)) + '\n')
        f.write('EUR: '+ str(round(fairness_UI['EUR'][i], 4)) + '\n')
        f.write('RUR: '+ str(round(fairness_UI['RUR'][i], 4)) + '\n')
        f.write('Fairness BI:\n')
        f.write('EEL: '+ str(round(fairness_BI['EEL'][i], 4)) + '\n')
        f.write('EED: '+ str(round(fairness_BI['EED'][i], 4)) + '\n')
        f.write('EER: '+ str(round(fairness_BI['EER'][i], 4)) + '\n')
        f.write('DP: '+ str(round(fairness_BI['DP'][i], 4)) + '\n')
        f.write('EUR: '+ str(round(fairness_BI['EUR'][i], 4)) + '\n')
        f.write('RUR: '+ str(round(fairness_BI['RUR'][i], 4)) + '\n')
        f.write('Diversity:\n')
        f.write('ILD: '+ str(round(diversity['ILD'][i], 4)) + '\n')
        f.write('ETP: '+ str(round(diversity['ETP'][i], 4)) + '\n')
        f.write('DS: '+ str(round(diversity['DS'][i], 4)) + '\n')
        f.write('ETP_AGG: '+ str(round(diversity['ETP_AGG'][i], 4)) + '\n')
        f.write('GINI: '+ str(round(diversity['GINI'][i], 4)) + '\n')
        f.write('\n')


if __name__ == '__main__':
    dir = os.path.dirname(__file__)
    dataset = 'pog'
    pred_file = f'{dir}/datasets/{dataset}/{dataset}_pred.json'
    
    with open(pred_file, 'r') as f:
        data_pred = json.load(f)

    truth_file = f'{dir}/datasets/{dataset}/{dataset}_future.json' # all users
    with open(truth_file, 'r') as f:
        data_truth = json.load(f)

    beyond_acc(dataset, data_pred, data_truth, [5, 10, 20], 'CLHE', 0.2)


# -------------pog-------------- 
# list size: 5
# recall: 0.0163
# ndcg: 0.0148
# EEL: 1.2975
# EED: 2.2427
# EER: 1.8951
# DP: 2.3245
# EUR: 1.9347
# RUR: 1.3892

# list size: 10
# recall: 0.0208
# ndcg: 0.0166
# EEL: 1.3773
# EED: 2.3822
# EER: 1.9548
# DP: 2.3233
# EUR: 1.9238
# RUR: 1.4839

# list size: 20
# recall: 0.0257
# ndcg: 0.0181
# EEL: 1.38
# EED: 2.3868
# EER: 1.9567
# DP: 2.3352
# EUR: 1.9291
# RUR: 1.5664

# list size: 40
# recall: 0.0257
# ndcg: 0.0181
# EEL: 1.38
# EED: 2.3868
# EER: 1.9567
# DP: 2.3352
# EUR: 1.9291
# RUR: 1.5664

# list size: 80
# recall: 0.0257
# ndcg: 0.0181
# EEL: 1.38
# EED: 2.3868
# EER: 1.9567
# DP: 2.3352
# EUR: 1.9291
# RUR: 1.5664