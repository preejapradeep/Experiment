import copy
import csv
import json
import os
import sys
import time
import numpy as np
from edit_operation import choose_random_operation
# from insert_operation import choose_random_operation
# from deletion_operation import choose_random_operation
import pandas as pd
import matplotlib.pyplot as plt
import edist.tree_utils as tree_utils
import edist.ted as ted
import edist.sed as sed
import edist.aed as aed
import edist.dtw as dtw
import edist.seted as seted
import edist.tree_edits as tree_edits
from multiprocessing import Pool

sys.setrecursionlimit(8000)  # Set the recursion limit to 8000
distance_factor=3
with open("costs.json") as f:
    costs = json.load(f)

def getSimilarityTable():
    return explainer_dataframe

# delta: custom node distance function
def semantic_delta(x, y):
    # Read the costs of the edit operations from a file
    # with open("costs.json") as f:
    #     costs = json.load(f)
    dfd = getSimilarityTable()
    if(x==y):
         ret = 0.
    elif(x!=None and y==None): #insertion
         ret = costs['insertion']
        #  print(f"insertion: x: {x}, y: {y}, ret: {ret}")
    elif(x==None and y!=None): #deletion 
        ret = costs['deletion']
        # print(f"deletion: x: {x}, y: {y}, ret: {ret}")
    elif(x=='r'or y=='r'):  #we assign an infinite cost when comparing a root node
        ret = np.inf
    elif(x in ['s','p'] and y in['s','p']): #if both nodes are either sequence or priority, assign null cost
        ret = 0.
    # elif x == "s" and y == "s":
    #     ret = 0
    # elif x == "p" and y == "p":
    #     ret = 0
    elif(x in ['s','p'] or y in ['s','p']): #if one of the nodes is a sequence or priority, the other won't because of the previous rule
        ret = np.inf
        # ret = 1e4 # Replace np.inf with a large constant value
    elif x in dfd.columns and y in dfd.columns: #Both explainers are in similarity table, DF MUST BE LOADED BEFOREHAND
        # if (x!=y):
            ret = 1-dfd.loc[x][y]
    else:
        ret = costs['default_cost']
        # print("default_cost", ret)
    return ret

# Returns a list of explainers in the same order specified by the tree
def explainer_sequence(bt,node,adj,seq):
    seq.append(node)
    if adj: 
        for child in adj:
            explainer_sequence(bt, bt["nodes"][child],bt["adj"][child],seq)

def explainer_sequence_SequenceMatch(tree,node,adj_node,seq):
    if not adj_node: #leaf
        if node not in ["f","t","r"]: #explainer
            seq.append(node)
        return
    for child in adj_node:
        explainer_sequence_SequenceMatch(tree, tree["nodes"][child],tree["adj"][child],seq)

# TED computation
def ted_similarity(q,c,delta):
    s1=[]
    explainer_sequence(q,q["nodes"][0],q["adj"][0],s1)
    s2=[]
    explainer_sequence(c,c["nodes"][0],c["adj"][0],s2)
    dist = ted.ted(q["nodes"], q["adj"], c["nodes"], c["adj"],delta)
    # return dist
    maxdist = max(len(s1),len(s2))
    return  (1-(dist/(maxdist*costs['insertion'])))

# # TED computation
# def editDistFunc(x_nodes, x_adj, y_nodes, y_adj,delta):
#     # delta=custom node distance function
#     if delta is None:
#         return ted.standard_ted(x_nodes, x_adj, y_nodes, y_adj)
#     else:
#         return ted.ted(x_nodes, x_adj, y_nodes, y_adj,delta)

        
# sequence edit distance of Levenshtein (1965)
def levenshtein_similarity(q,c,delta):
    s1=[]
    explainer_sequence(q,q["nodes"][0],q["adj"][0],s1)
    s2=[]
    explainer_sequence(c,c["nodes"][0],c["adj"][0],s2)
    dist = sed.sed(s1,s2, delta)
    # return dist
    maxdist = max(len(s1),len(s2))
    return  (1-(dist/(maxdist*costs['insertion'])))

# sequence edit distance with affine gap costs using algebraic dynamic programming (ADP; Giegerich, Meyer, and Steffen, 2004),
# as applied by Paaßen, Mokbel, and Hammer (2016)
def aed_similarity(q,c,delta):
    s1=[]
    explainer_sequence(q,q["nodes"][0],q["adj"][0],s1)
    s2=[]
    explainer_sequence(c,c["nodes"][0],c["adj"][0],s2)
    dist = aed.aed(s1,s2,delta)
    # return dist
    maxdist = max(len(s1),len(s2))
    return  (1-(dist/(maxdist*costs['insertion'])))

def default_dtw_distance(x,y):
    if x==y:
        return 0
    else: 
        return costs['insertion']

# dynamic time warping distance of Vintsyuk (1968)   
def dtw_similarity(q,c,delta):
    s1=[]
    explainer_sequence(q,q["nodes"][0],q["adj"][0],s1)
    s2=[]
    explainer_sequence(c,c["nodes"][0],c["adj"][0],s2)
    # if delta==None:
    #     delta = default_dtw_distance
    dist = dtw.dtw(s1,s2,delta)
    # return dist
    maxdist = max(len(s1),len(s2))
    return  (1-(dist/(maxdist*costs['insertion'])))

# Hungarian algorithm of Kuhn, 1955 
def set_similarity(q,c,delta):
    s1=[]
    explainer_sequence(q,q["nodes"][0],q["adj"][0],s1)
    s2=[]
    explainer_sequence(c,c["nodes"][0],c["adj"][0],s2)
    dist = seted.seted(s1,s2,delta)
    # return dist
    maxdist = max(len(s1),len(s2))
    return  (1-(dist/(maxdist*costs['insertion'])))

# simple sequence match from the beginning of both sequences
def sequence_match_similarity(q,c,delta=None):
    sq=[]
    explainer_sequence_SequenceMatch(q,q["nodes"][0],q["adj"][0],sq)
    sc=[]
    explainer_sequence_SequenceMatch(c,c["nodes"][0],c["adj"][0],sc)
    min_size=min(len(sq),len(sc))
    maxdist = max(len(sq),len(sc))
    sim=0
    for i in range(min_size):
        x = sq[i]
        y = sc[i]
        if delta == None:
            if sq[i]==sc[i]:
                sim += 1
        else:
            sim += distance_factor * (1-delta(x,y))
    return sim/(maxdist*costs['insertion'])
    # return dist

# The overlap coefficient, or Szymkiewicz–Simpson coefficient
def overlap_similarity(q,c,delta:None): #Shared explainers
    mq= [node for node in q["nodes"] if node[0]=='/']
    mc= [node for node in c["nodes"] if node[0]=='/']
    if min(len(mq),len(mc)) == 0:
        return 0
    else:
        return len(set(mq)&set(mc))/min(len(mq),len(mc)) 
    # * distance_factor


def compute_edit_distance(random_index, random_bt, random_bt_dict, case_base, case_base_dict, structural_similarity, similarity_metrics):    
    global explainer_dataframe
    random_bt = case_base[random_index]
    random_bt_dict[random_index]=random_bt
    random_bt_p = copy.deepcopy(random_bt)
    results = pd.DataFrame(columns=['Random BT', 'Edits', 'Operation', 'Modified BT', 'Case', 'Structural Similarity', 'Metrics', 'Edit Distance'])
    # Loop to perform operations on the BT - maximum operation is the length of BT
    for edits in range(1, 11):
        # Choose a random edit operation and get the new BT
        random_bt_prime, edits, operation = choose_random_operation(random_bt_p, edits)
        # For each structural_similarity, compute the edit distance using each similarity_metrics
        for algorithm in structural_similarity:
            # Iterate over the list of files and read each one
            for metric in similarity_metrics:
                # # Get the filename without the extension
                metric_name = os.path.splitext(os.path.basename(metric))[0]
                # Compute the edit distance between the modified BT and all the BTs in the case base
                for bt_name, bt in case_base_dict.items():
                    explainer_dataframe = pd.read_csv(metric,index_col=0)
                    if algorithm == "Tree Edit Distance":
                        score = ted_similarity(bt,random_bt_prime, delta=semantic_delta)
                    elif algorithm == "Levenshtein Distance":
                        score = levenshtein_similarity(bt,random_bt_prime, delta=semantic_delta)
                    elif algorithm == "Affine Edit Distance":
                        score = aed_similarity(bt,random_bt_prime, delta=semantic_delta)
                    elif algorithm == "Dynamic Time Warping":
                        score = dtw_similarity(bt,random_bt_prime, delta=semantic_delta)
                    elif algorithm == "Set Edit Distance":
                        score = set_similarity(bt,random_bt_prime, delta=semantic_delta)
                    elif algorithm == "Overlap Coefficient":
                        score = overlap_similarity(bt,random_bt_prime, delta=semantic_delta)
                    elif algorithm == "Sequence Match":
                        score = sequence_match_similarity(bt,random_bt_prime, delta=semantic_delta)
                    # if (score < 0) or (1-score) < 0:
                    #     print("********************")
                    #     print("Similarity: " + str(score))
                    #     print("Edit distance: " + str(1-score))
                    #     print(algorithm)
                    #     print(metric)
                    #     print(bt)
                    #     print(random_bt_prime)
                    #     print("********************")
                    
                    results.loc[len(results.index)] = [random_bt, edits, operation, random_bt_prime, bt, algorithm, metric_name, 1-score]

        # Assign random_bt_prime to random_bt_p
        random_bt_p = copy.deepcopy(random_bt_prime)

    return results, random_bt_dict


if __name__ == '__main__': 

    start_time = time.time()

    folder_path = '/Users/ppradeep/Documents/BT Algorithm/randomgenerator/BT_Random/ModifiedBTExp_June/cb300cost3/'
    structural_similarity = ["Tree Edit Distance", "Levenshtein Distance", "Affine Edit Distance", "Dynamic Time Warping", "Set Edit Distance", "Sequence Match", "Overlap Coefficient"]
    similarity_metrics=["unweighted/common attributes.csv", "unweighted/cosine.csv", "unweighted/depth.csv","unweighted/detail.csv"]

    # Load case base from json file
    # casebase300
    with open("casebase300.json", "r") as f:
        case_base = json.load(f)

    # Create a folder to store the random cases
    folder_name = "BT_Random" 
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    random_bt_output = os.path.join(folder_name, "RandomBT.csv")

    case_base_dict, edit_distances,random_bt_dict,sorted_scores_dict = {},{},{},{}
    for i, bt in enumerate(case_base):
        case_base_dict[f'bt_{i}'] = bt

    dataframes = {}
    for metric in similarity_metrics:
        dataframes[metric] = pd.read_csv(metric)

    # open the CSV file in append mode
    with open(random_bt_output, 'w', newline='') as file:
        print('cb length:',len(case_base))
        writer = csv.writer(file)
        writer.writerow(['Random BT', 'Modified BT', 'Edits'])
        results = pd.DataFrame(columns=['Random BT', 'Edits', 'Operation', 'Modified BT', 'Case', 'Structural Similarity', 'Metrics', 'Edit Distance'])
        ranking_results = pd.DataFrame(columns=['Random BT', 'Edits', 'Operation', 'Modified BT', 'Case', 'Structural Similarity', 'Metrics', 'Edit Distance', 'Rank'])
        # Loop over the random BTs
        random_indices = []

        # create a sublist of the first 100 random BTs
        sublist = case_base[:100]

        # create args_list with the random BTs in the sublist
        args_list = [(i, bt, random_bt_dict, case_base, case_base_dict, structural_similarity, similarity_metrics) for i, bt in enumerate(sublist)]

        # Create a pool of worker processes to run the function in parallel
        with Pool() as pool:
            print('Loading pool....')
        # Map the function over the list of arguments and get the results
            all_results = pool.starmap(compute_edit_distance, args_list)
        
        for i in range(len(all_results)):
            result = all_results[i][0]  # get the result at index i
            # convert the result list to a pandas dataframe and add it to the new_results dataframe
            random_bt_results = pd.DataFrame(result, columns=['Random BT', 'Edits', 'Operation', 'Modified BT', 'Case', 'Structural Similarity', 'Metrics', 'Edit Distance'])
            results = pd.concat([results, random_bt_results])        
        file.close()

        # create a dictionary to store the data frames for each sheet
        sheet_dict = {}

        fig_num = 1
        for i in range(len(all_results)):
            random_bt_dict = all_results[i][1]  # get the result_dict at index i
            
            for key, bt in random_bt_dict.items():
            # Rank the casebase based on edit distance for each metric
                # filter results by the current random_bt
                random_bt_results = results.loc[results['Random BT'] == random_bt_dict[key]]
                # # Group the results by edits and metric
                groups = random_bt_results.groupby(['Edits', 'Metrics', 'Structural Similarity'])
                ranks = groups['Edit Distance'].rank(method='dense')
                # ranks = groups['Edit Distance'].rank(method='dense', ascending=False)
                # # Add the ranks to the results DataFrame
                random_bt_results['Rank'] = ranks
                # Add the filtered results to the sheet dictionary
                sheet_dict[key] = random_bt_results
                # save filtered results to a csv file
                random_bt_results.to_csv(folder_path + f'random_bt_{key}_results.csv', index=False)
                # plt.show()
        
    end_time = time.time()
    computation_time = end_time - start_time
    print("\nFinal Computation time:", computation_time, "seconds")
