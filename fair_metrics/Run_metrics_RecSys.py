import pandas as pd
import numpy as np

import fair_metrics.singh_joachims as sj
import fair_metrics.biega as bg
import fair_metrics.diaz as dz

import metric_utils.groupinfo as gi
import metric_utils.position as pos

"""
    Run the fair ranking metrics in default settings or controlled settings.

    Args:
        ranked_lists (pandas.DataFrame): complete set of rankings to measure. top-k item id 
        test_rates (panda.DataFrame): test set. ground truth
        group (GroupInfo Object): contains group information from ranked lists. extend individual to group fairness
        original_relev(pandas.DataFrame): system given relevance score. predicted relevance
        IAA(boolean): measure fairness using IAA metric?
        EE(boolean): measure fairness using EE family metrics?
        AWRF(boolean): measure fairness using AWRF metric?
        DRR(boolean): measure fairness using DRR family metrics?
        FAIR(boolean): measure fairness using FAIR metric?

    """

class metric_analysis:
    
    ranked_lists = None
    test_rates = None
    group = None
    arg = None      #arg(string): metric parameter
    arg_val = None  #arg_val(float): metric parameter value
    
    def __init__(self, ranked_lists, test_rates, group, original_relev=None, IAA=True, EE=True, DRR=True):
        
        self.ranked_lists = ranked_lists
        self.test_rates = test_rates
        self.original_relev = original_relev
        self.group = group
        self.IAA = IAA
        self.EE= EE
        self.DRR = DRR
        
    def run_IAA(self, ranked_list, ranked_rel, pweight=pos.geometric()): #default weighting 

        """
        Measure fairness using IAA metric.
        Args:
            ranked_list(panda.DataFrame): truncated recommended ranked lists for user
            pweight(position object): user browsing model to measure position weight. Default: geometric
        Return:
            pandas.Series: fairness score of ranked lists for each algorithms
        """
        
        if 'cascade' in str(pweight) or 'equality' in str(pweight):
            weight_per_user = ranked_list.groupby('user').apply(lambda x: pweight(x))
            ind = [i[1] for i in weight_per_user.index]
            weight_vector = pd.Series(data = weight_per_user.values, index = ind)

        else:
            
            weight_vector = pweight(ranked_list)
            
        #algo = ranked_list['Algorithm'].unique()
        #orb = self.original_relev.loc[self.original_relev['Algorithm']==algo[0]]
        return pd.Series({'IAA': bg.unfairness(ranked_list, ranked_rel, self.group, weight_vector)})
                                   
    def run_EE(self, ranked_list, pweight=pos.cascade()): #default weighting 
        
        """
        Measure fairness using expected-exposure metric.
        Args:
            ranked_list(panda.DataFrame): truncated recommended ranked lists for user
            pweight(position object): user browsing model to measure position weight. Default: cascade
        Return:
            pandas.Series: fairness score of ranked lists for each algorithms
        """
        
        return dz.ee_for_ds(ranked_list, self.test_rates, self.group, pweight)
 
    def run_dp_eur_rur(self, ranked_list, pweight=pos.logarithmic()): #default weighting 
        
        """
        Measure fairness using logDP, logEUR, and logRUR metrics.
        Args:
            ranked_list(panda.DataFrame): truncated recommended ranked lists for user
            pweight(position object): user browsing model to measure position weight. Default: logarithmic
        Return:
            pandas.Series: fairness score of ranked lists for each algorithms
        """
        
        if 'cascade' in str(pweight) or 'equality' in str(pweight):
            weight_per_user = ranked_list.groupby('user').apply(lambda x: pweight(x))
            ind = [i[1] for i in weight_per_user.index]
            weight_vector = pd.Series(data = weight_per_user.values, index = ind)
            
        else:
            weight_vector = pweight(ranked_list) 
           
          
            
        return {
            'logDP': sj.demographic_parity(ranked_list, self.group, weight_vector),
            'logEUR': sj.exposed_utility_ratio(ranked_list, self.test_rates, self.group, weight_vector),
            'logRUR': sj.realized_utility_ratio(ranked_list, self.test_rates, self.group, weight_vector),
        }

    def run_stochastic_metric(self, ranked_list, ranked_rel, pweight):

        """
        Measure fairness using single ranking metrics.
        Args:
            ranked_list(panda.DataFrame): truncated recommended ranked lists for user
            pweight(position object): user browsing model to measure position weight.
        Return:
            pandas.Series: fairness scores of ranked lists for each algorithms!!
        """
        
        result = {}
        if pweight == 'default':
            #if self.IAA == True:
                #result = self.run_IAA(ranked_list, ranked_rel)
            if self.EE == True:
                result.update(self.run_EE(ranked_list))          
            if self.DRR == True:
                result.update(self.run_dp_eur_rur(ranked_list))
            return result

        #if self.IAA == True:
            #result = self.run_IAA(ranked_list, ranked_rel, pweight)
        if self.EE == True:
            result.update(self.run_EE(ranked_list, pweight))
        if self.DRR == True:
            result.update(self.run_dp_eur_rur(ranked_list, pweight))
        return result
    
    def run_default_setting(self, listsize=100, pweight='default'):
        
        """
        Measure fairness using single ranking metrics.
        Args:
            listsize(int): size of the ranked list. default:100
        Return:
            pandas.Series: fairness scores of ranked lists for each algorithms
        """
        
        truncated = self.ranked_lists[self.ranked_lists['rank']<=listsize] #truncate top-k  
        truncated_rel = self.original_relev[self.original_relev['rank']<=listsize]
       
        final_metrics = self.run_stochastic_metric(truncated, truncated_rel, pweight) 
        
        return final_metrics