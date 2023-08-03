import ast
import copy
import csv
import json
import random
import os
import time

import numpy as np
from astar import graph_edit_distance
# from exp_test import choose_random_operation
# from op import choose_random_operation
from edit_operation import choose_random_operation
# from deletion_operation import choose_random_operation
from mAstar import modified_graph_edit_distance
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns
from json2graph import plot_mAstar, plot_tree
import edist.tree_utils as tree_utils
import edist.ted as ted
import edist.sed as sed
import edist.aed as aed
import edist.dtw as dtw
import edist.seted as seted
import edist.tree_edits as tree_edits
import matplotlib.colors as mcolors
from multiprocessing import Pool
import sys

sys.setrecursionlimit(8000)  # Set the recursion limit to 5000



leave_change = 1
sq=[]
default_cost = 1

start_time = time.time()

# Load case base from json file
with open("casebase.json", "r") as f:
    case_base = json.load(f)


def ged(bt,bt_name, random_bt_prime, edits,explainer_costs):
	# print('explainer_costs', explainer_costs)
	# plot tree
	g1, g2 = plot_tree(bt,random_bt_prime)
	# print('g1 & g2:', g1.nodes(), g2.nodes())
	# print('g1 & g2:', g2.edges(), g2.edges())
	ed = graph_edit_distance(g1, g2, explainer_costs)
	# print('ed:', ed)
	if ed[1] is not None:  # Check for None values
		score = ed[1] # ged
		edit_distances[bt_name] = score	
		avg_editDistances.append(sum([score])/edits)
		# min_editDistances.append(min(score))
		# maxvalue = max(score)
	return score

def modifiedged(bt, bt_name, random_bt_prime, edits, explainer_costs):
	# plot tree
	g1, g2 = plot_mAstar(bt,random_bt_prime)
	# print('g1 & g2:', g1.nodes(), g2.nodes())
	# print('g1 & g2:', g1.edges(), g2.edges(), '\n')
	med = modified_graph_edit_distance(g1, g2, explainer_costs)
	if med[1] is not None:  # Check for None values
		score = med[1] # ged
		medit_distances[bt_name] = score	
		avg_editDistances.append(sum([score])/edits)
		# min_editDistances.append(min(score))
		# maxvalue = max(score)
	return score

def getSimilarityTable():
    return explainer_dataframe

# TED computation
def editDistFunc(x_nodes, x_adj, y_nodes, y_adj,delta):
    # delta=custom node distance function
    if delta is None:
        return ted.standard_ted(x_nodes, x_adj, y_nodes, y_adj)
    else:
        return ted.ted(x_nodes, x_adj, y_nodes, y_adj,delta)

# TED - delta: custom node distance function
def semantic_delta(x, y):
    dfd = getSimilarityTable()
    if(x==y):
         ret = 0.
    elif(x!=None and y==None): #insertion
         ret = costs['insertion']
    elif(x==None and y!=None): #deletion 
        ret = costs['deletion']
    elif(x=='r'or y=='r'):  #we assign an infinite cost when comparing a root node
        ret = np.inf
    elif(x in ['f','t'] and y in ['f','t']): #if both nodes are either failer or succeeder, assign null cost
        ret = 0.
    elif(x in ['s','p'] and y in['s','p']): #if both nodes are either sequence or priority, assign null cost
        ret = 0.
    elif(x in ['s','p'] or y in ['s','p']): #if one of the nodes is a sequence or priority, the other won't because of the previous rule
        ret = np.inf
        # print(f"ctrl Inf: x: {x}, y: {y}, ret: {ret}")
    elif(x in ['f','t'] and y[0]=='/'):
        ret = leave_change
    elif(x[0]=='/' and y in ['f','t']):
        ret = leave_change
    elif x in dfd.columns and y in dfd.columns: #Both explainers are in similarity table, DF MUST BE LOADED BEFOREHAND
        # if (x!=y):
        #     ret = 1-dfd.loc[x][y]
        if(dfd.loc[x][y]>.5):
            ret = 0
        else:
            ret = leave_change
            # print('ret:', ret)
    else:
        ret = default_cost
    return ret


# Returns a list of explainers in the same order specified by the tree
def explainer_sequence(bt,node,adj,seq):
    seq.append(node)
    if adj: 
        for child in adj:
            explainer_sequence(bt, bt["nodes"][child],bt["adj"][child],seq)

# sequence edit distance of Levenshtein (1965)
def levenshtein_similarity(q,c, delta):
    s1=[]
    explainer_sequence(q,q["nodes"][0],q["adj"][0],s1)
    s2=[]
    explainer_sequence(c,c["nodes"][0],c["adj"][0],s2)

    dist = sed.sed(s1,s2, delta)
    return dist

# simple sequence match from the begining of both sequences
def sequence_match_similarity(q,c,delta):
    
    sq=[]
    explainer_sequence(q,q["nodes"][0],q["adj"][0],sq)
    sc=[]
    explainer_sequence(c,c["nodes"][0],c["adj"][0],sc)
    
    min_size=min(len(sq),len(sc))
    sim=0
    for i in range(min_size):
        x = sq[i]
        y = sc[i]
        if delta == None:
            if sq[i]==sc[i]:
                sim += 1
        else:
            sim += delta(x,y)

    return sim/min_size

# sequence edit distance with affine gap costs using algebraic dynamic programming (ADP; Giegerich, Meyer, and Steffen, 2004),
# as applied by Paaßen, Mokbel, and Hammer (2016)
def aed_similarity(q,c, delta):
    s1=[]
    explainer_sequence(q,q["nodes"][0],q["adj"][0],s1)
    s2=[]
    explainer_sequence(c,c["nodes"][0],c["adj"][0],s2)
    dist = aed.aed(s1,s2,delta)
    maxdist = max(len(s1),len(s2))
    return (dist/maxdist)

def default_dtw_distance(x,y):
    if x==y:
        return 0

# dynamic time warping distance of Vintsyuk (1968)   
def dtw_similarity(q,c, delta):
    s1=[]
    explainer_sequence(q,q["nodes"][0],q["adj"][0],s1)
    s2=[]
    explainer_sequence(c,c["nodes"][0],c["adj"][0],s2)
    print('s1 & s2:', s1, s2)
    if delta==None:
        delta = default_dtw_distance
    print('delta:', delta)
    dist = dtw.dtw(s1,s2,delta)
    print('dist', dist)
    maxdist = max(len(s1),len(s2))
    print('maxdist', maxdist)
    print('dist/maxdist', dist/maxdist)
    return dist/maxdist

# Hungarian algorithm of Kuhn, 1955 
def set_similarity(q,c, delta):
    s1=[]
    explainer_sequence(q,q["nodes"][0],q["adj"][0],s1)
    s2=[]
    explainer_sequence(c,c["nodes"][0],c["adj"][0],s2)
    dist = seted.seted(s1,s2,delta)
    maxdist = max(len(s1),len(s2))
    return dist/maxdist

# The overlap coefficient, or Szymkiewicz–Simpson coefficient
def overlap_similarity(q,c,delta:None): #Shared explainers
    
    mq= [node for node in q["nodes"] if node[0]=='/']
    mc= [node for node in c["nodes"] if node[0]=='/']
    print('mq & mc', mq,mc)

    # Check if both mq and mc are empty
    if len(mq) == 0 and len(mc) == 0:
        return 1  # Return 1 if both sets are empty (they are equal)
    
    return len(set(mq)&set(mc))/min(len(mq),len(mc))

def compute_edit_distance(random_index, random_bt, case_base_dict, edit_distance_functions, similarity_metrics):    
    global explainer_dataframe
    random_bt = case_base[random_index]
    random_bt_dict[random_index]=random_bt
    random_bt_p = copy.deepcopy(random_bt)
    results = pd.DataFrame(columns=['Random BT', 'Edits', 'Modified BT', 'Case', 'Edit Distance Function', 'Metrics', 'Edit Distance'])
    # Loop to perform operations on the BT - maximum operation is the length of BT
    for edits in range(1, 11):
        print("************************************************")
        print('\n index:',random_index,'random_bt:', random_bt)
        # Choose a random edit operation and get the new BT
        random_bt_prime, edits = choose_random_operation(random_bt_p, edits)
        # For each edit_distance_functions, compute the edit distance using each similarity_metrics
        for algorithm in edit_distance_functions:
            # Iterate over the list of files and read each one
            for metric in similarity_metrics:
                # Get the filename without the extension
                metric_name = os.path.splitext(os.path.basename(metric))[0]
                # print('metric',metric_name)
                # Compute the edit distance between the modified BT and all the BTs in the case base
                for bt_name, bt in case_base_dict.items():
                    explainer_dataframe = pd.read_csv(metric,index_col=0)
                    # df = dataframes[metric]
                    # filtered_df = df[df["explainer"].str.startswith("/")]
                    # explainer_costs = filtered_df.to_dict()
                    # if algorithm == "Graph Edit Distance":
                    #     score = ged(bt, bt_name, random_bt_prime, edits, explainer_costs)
                    #     print('ged')
                    # elif algorithm == "Modified Graph Edit Distance":
                    #     score = modifiedged(bt, bt_name, random_bt_prime, edits, explainer_costs)
                    #     print('mged')
                
                    if algorithm == "Tree Edit Distance":
                        # print('\nTED: ', bt["nodes"],bt["adj"],' and ',random_bt_prime["nodes"],random_bt_prime["adj"])
                        score = editDistFunc(bt["nodes"],bt["adj"],random_bt_prime["nodes"],random_bt_prime["adj"], delta=semantic_delta)
                        # print('semantic_delta')
                    elif algorithm == "Levenshtein Distance":
                        score = levenshtein_similarity(bt,random_bt_prime, delta=semantic_delta)
                    elif algorithm == "Affine Edit Distance":
                        score = aed_similarity(bt,random_bt_prime, delta=semantic_delta)
                    elif algorithm == "Dynamic Time Warping":
                        print('\nDTW: ', bt["nodes"],bt["adj"],' and ',random_bt_prime["nodes"],random_bt_prime["adj"])
                        print('semantic_delta:', semantic_delta)
                        score = dtw_similarity(bt,random_bt_prime, delta=semantic_delta)
                        print('DTW:',score)
                    elif algorithm == "Set Edit Distance":
                        # print('\nSED: ', bt["nodes"],bt["adj"],' and ',random_bt_prime["nodes"],random_bt_prime["adj"])
                        score = set_similarity(bt,random_bt_prime, delta=semantic_delta)
                    elif algorithm == "Sequence Match":
                        score = sequence_match_similarity(bt,random_bt_prime, delta=semantic_delta)
                        print('SM:',score)
                    elif algorithm == "Overlap Similarity":
                        # print('\nOverlap: ', bt["nodes"],bt["adj"],' and ',random_bt_prime["nodes"],random_bt_prime["adj"])
                        score = overlap_similarity(bt,random_bt_prime, delta=semantic_delta)
                        # print(score)
                    results.loc[len(results.index)] = [random_bt, edits, random_bt_prime, bt, algorithm, metric_name, score]

        # Assign random_bt_prime to random_bt_p
        random_bt_p = copy.deepcopy(random_bt_prime)

    return results, random_bt_dict

# edit_distance_functions = ["Dynamic Time Warping"]
edit_distance_functions = ["Tree Edit Distance", "Levenshtein Distance", "Affine Edit Distance", "Dynamic Time Warping", "Set Edit Distance", "Sequence Match", "Overlap Similarity"]

# edit_distance_functions = ["Graph Edit Distance", "Modified Graph Edit Distance", "Tree Edit Distance", "Levenshtein Distance", "Affine Edit Distance", "Dynamic Time Warping", "Set Edit Distance"]
similarity_metrics=["weighted/ca.csv", "weighted/cosine.csv", "weighted/depth.csv","weighted/detail.csv"]
# similarity_metrics=["unweighted/ca.csv","unweighted/cosine.csv", "unweighted/depth.csv","unweighted/detail.csv","weighted/ca.csv", "weighted/cosine.csv", "weighted/depth.csv","weighted/detail.csv"]


# Read the costs of the edit operations from a file
with open("costs.json") as f:
    costs = json.load(f)

# Create a folder to store the random cases
folder_name = "BT_Random" 
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

random_bt_output = os.path.join(folder_name, "RandomBT.csv")

# Initialize a variable to store the smallest edit distance
smallest_edit_distance = float("inf")

case_base_dict = {}
for i, bt in enumerate(case_base):
    case_base_dict[f'bt_{i}'] = bt
# print('case_base_dict',case_base_dict, '\n')

edit_distances = {}
medit_distances = {}
random_bt_dict = {}


# folder_path = '/Users/ppradeep/Documents/BT Algorithm/randomgenerator/BT_Random/NewBTUseCase/ALL/'
folder_path = '/Users/ppradeep/Documents/BT Algorithm/randomgenerator/BT_Random/NewBTUseCase/New/UCM/'

avg_editDistances=[]
min_editDistances=[]
max_editDistances=[]

prime_dict, prime_edit_dict, sorted_scores_dict = {},{},{}

dataframes = {}
for metric in similarity_metrics:
    dataframes[metric] = pd.read_csv(metric)

if __name__ == '__main__':  
    # open the CSV file in append mode
    with open(random_bt_output, 'w', newline='') as file:
        print('cb length:',len(case_base))
        writer = csv.writer(file)
        writer.writerow(['Random BT', 'Modified BT', 'Edits'])
        results = pd.DataFrame(columns=['Random BT', 'Edits', 'Modified BT', 'Case', 'Edit Distance Function', 'Metrics', 'Edit Distance'])
        ranking_results = pd.DataFrame(columns=['Random BT', 'Edits', 'Modified BT', 'Case', 'Edit Distance Function', 'Metrics', 'Edit Distance', 'Rank'])
        # Loop over the random BTs
        random_indices = []

        # create a sublist of the first 100 random BTs
        sublist = case_base[:100]
        # print('sublist',sublist)

        # create args_list with the random BTs in the sublist
        args_list = [(i, bt, case_base_dict, edit_distance_functions, similarity_metrics) for i, bt in enumerate(sublist)]

        # # Create a list of arguments to pass to the compute_edit_distance function
        # args_list = [(random_index, random_bt, case_base_dict, edit_distance_functions, similarity_metrics) for random_index, random_bt in enumerate(case_base)]
        # print('args_list:', args_list)

        # Create a pool of worker processes to run the function in parallel
        with Pool() as pool:
            print('Loading pool....')
        # Map the function over the list of arguments and get the results
            all_results = pool.starmap(compute_edit_distance, args_list)
        
        print('len(all_results):',len(all_results))
        for i in range(len(all_results)):
            result = all_results[i][0]  # get the result at index i
            # random_bt_dict = all_results[i][1]  # get the result_dict at index i
            # print('random_bt_dict:',random_bt_dict)
            # convert the result list to a pandas dataframe and add it to the new_results dataframe
            random_bt_results = pd.DataFrame(result, columns=['Random BT', 'Edits', 'Modified BT', 'Case', 'Edit Distance Function', 'Metrics', 'Edit Distance'])
            results = pd.concat([results, random_bt_results])
            # print('random_bt_results:', random_bt_results)

        # # Sort edit_distances by value
        # sorted_scores = sorted(edit_distances.items(), key=lambda x: x[1])
        # sorted_scores_dict[str(random_bt)] = sorted_scores	

        # results.to_csv(folder_path + 'new_results.csv')
        
        file.close()


        # create a dictionary to store the data frames for each sheet
        sheet_dict = {}

        fig_num = 1
        for i in range(len(all_results)):
            random_bt_dict = all_results[i][1]  # get the result_dict at index i
            # print('random_bt_dict:',random_bt_dict)
            
            for key, bt in random_bt_dict.items():
            # Rank the casebase based on edit distance for each metric
                # filter results by the current random_bt
                random_bt_results = results.loc[results['Random BT'] == random_bt_dict[key]]
                # # Group the results by edits and metric
                groups = random_bt_results.groupby(['Edits', 'Metrics', 'Edit Distance Function'])
                # # Rank the scores for each group
                ranks = groups['Edit Distance'].rank(method='dense')
                # # Add the ranks to the results DataFrame
                random_bt_results['Rank'] = ranks
                # add the filtered results to the sheet dictionary
                sheet_dict[key] = random_bt_results
                # save filtered results to a csv file
                random_bt_results.to_csv(folder_path + f'random_bt_{key}_results.csv', index=False)

                # # Load the data from the CSV file
                # random_bt_results = pd.read_csv(folder_path + f'random_bt_{key}_results.csv')
                # edge_str = bt['edge']

                # # # Plot the edit distance vs edits for all the edit distance function in each metrics
                # grid = sns.FacetGrid(data=random_bt_results, col="Edit Distance Function", hue="Metrics", col_wrap=2, height=4, aspect=2, legend_out=True)
                # grid.map(sns.lineplot, "Edits", "Edit Distance")
                # grid.set_axis_labels("Edits", "Edit Distance")
                # grid.set_titles("{col_name}", size=6)
                # grid.fig.suptitle(f'Edit distance vs. Edits for Random BT : {edge_str}', y=0.99, size=8)
                # grid.fig.subplots_adjust(top=0.95)
                # grid.add_legend()

                # # Plot the edit distance vs edits for all the  metrics in each edit distance function

            # Record the position of random_BT in the ranked casebase.
                # # Filter the rows where 'Random BT' equals 'Case'
                filtered_results = random_bt_results.loc[random_bt_results['Random BT'] == random_bt_results['Case']]
                # Group the filtered results by 'Edits', 'Metrics', and 'Edit Distance Function'
                groups = filtered_results.groupby(['Edits', 'Metrics', 'Edit Distance Function'])
                # Get the rank of the 'Edit Distance' column within each group
                ranks = groups['Edit Distance'].rank(method='dense')
                # Add the 'Rank' column to the filtered results dataframe
                filtered_results['Rank'] = ranks
                # filtered_results.to_csv(folder_path + f'random_bt_{key}_ranking.csv', index=False)

                # Sort the filtered results based on rank for each Edit Distance Function, metric, and edit
                # filtered_results = filtered_results.sort_values(['Edit Distance Function', 'Metrics', 'Edits', 'Rank'])
                # filtered_results.to_csv(folder_path + f'random_bt_{key}_sortedranking.csv', index=False)

                # # Load the data from the CSV file
                # filtered_results = pd.read_csv(folder_path + f'random_bt_{key}_ranking.csv')

                # random_bt_results['Rank'] = random_bt_results['Rank'].astype(int)
                
                # # Plot the Rank vs Edits for all metrics in each Edit Distance Function
                # grid = sns.FacetGrid(random_bt_results, col="Edit Distance Function", hue="Metrics", col_wrap=2, height=4, aspect=2, legend_out=True)
                # grid.map_dataframe(sns.lineplot, x="Edits", y="Rank")
                # grid.set_axis_labels("Edits", "Rank")
                # grid.set_titles("{col_name}", size=6)
                # grid.fig.suptitle(f'Rank vs. Edits for Random BT : {edge_str}', y=0.99, size=8)
                # grid.fig.subplots_adjust(top=0.95)
                # grid.add_legend()   

                # # Calculate the maximum number of edits and maximum rank for any case in the dataset
                # max_edits = random_bt_results['Edits'].max()
                # max_rank = random_bt_results['Rank'].max()
            
            # # Plot the Rank vs Edits for all Edit Distance Function in each metrics
                # grid1 = sns.FacetGrid(data=filtered_results, col="Metrics", row="Edit Distance Function", hue="Edit Distance Function", height=4, aspect=2, legend_out=True)
                # grid1.map(sns.scatterplot, "Edits", "Rank")
                # grid.set(xlim=(0, max_edits), ylim=(0, max_rank))
                # grid1.set_axis_labels("Edits", "Rank")
                # grid1.set_titles("{row_name}, {col_name}", size=6)
                # grid1.fig.suptitle(f'Rank vs. Edits for Random BT : {edge_str}', y=0.99, size=8)
                # grid1.fig.subplots_adjust(top=0.95)
                # grid1.add_legend()


            # Compute the mean Edit Distance for each group
                avg_edits = groups['Edit Distance'].mean()
                # # Reset the index of the resulting dataframe
                # avg_edits = avg_edits.reset_index()


                # # Plot the average edit distance vs number of edits of all metrics for each edit distance function
                # g1 = sns.relplot(x='Edits', y='Edit Distance', hue='Metrics', col='Edit Distance Function', col_wrap=2, kind='line', height=4, aspect= 1.5, data=avg_edits)
                # g1.set_axis_labels('Edits', 'Average Edit Distance')
                # g1.fig.suptitle(f'Average Edit Distance vs Edits for Random BT: {edge_str}', fontsize=9)
                # g1.fig.subplots_adjust(top=0.9)
                # # # plt.savefig(folder_path + f'{key}_fig_{fig_num}.png')
                # # # fig_num += 1

                # f = plt.figure()
                # g1.on(f)
                # f.savefig(folder_path + f'{key}_fig_{fig_num}.png')

            # # Plot the average edit distance vs number of edits for all edit distance function for each metric
            #     gd = sns.FacetGrid(avg_edits, col='Metrics', hue="Edit Distance Function", col_wrap=2, height=4, aspect=1.5)
            #     gd.map_dataframe(sns.lineplot, x="Edits", y="Edit Distance")
            #     gd.set_axis_labels("Edits", "Average Edit Distance")
            #     gd.fig.suptitle(f'Random BT: {edge_str}', y=0.99, size=8)
            #     gd.set_titles("{col_name}", size=8)	
            #     gd.fig.subplots_adjust(top=0.95)
            #     gd.add_legend()

        

            
            


                plt.show()



        
    end_time = time.time()
    computation_time = end_time - start_time
    print("\nFinal Computation time:", computation_time, "seconds")
