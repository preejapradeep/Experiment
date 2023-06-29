# from collections import OrderedDict
from itertools import combinations
import json
import os
import random
# from matplotlib import pyplot as plt
from astar import graph_edit_distance
from mAstar import modified_graph_edit_distance
# from modifiedAstar_modified import modified_graph_edit_distance
# import networkx as nx
# import numpy as np
# import pandas as pd
# import igraph as ig
# from treelib import Node, Tree


# Display in topological order (stratifies a DAG into generations) - tree structure
def topo_pos(T):
    """Display in topological order, with simple offsetting for legibility"""
    pos_dict = {}
    for i, node_list in enumerate(nx.topological_generations(T)):
        x_offset = len(node_list) / 2
        y_offset = 0.1
        for j, name in enumerate(node_list):
            pos_dict[name] = (j - x_offset, -i + j * y_offset)
        # print('node_list',node_list)
    # print('pos_dict:',pos_dict)
    return pos_dict

# JSON data to graph structure 
def json2graph(data):
    g=nx.DiGraph()
    instanceBT,rootBT,firstChildInsBT,nextChildBT,nextChildInsBT_list = [],[],[],[],[]
    instanceBT.append(data.get('Instance'))
    root_dictBT = data.get('root')
    nodes_dict_BT = data.get('nodes')
    # print('nodes_dict_BT:',nodes_dict_BT)
    nodesBTArr={}
    for keyBT in nodes_dict_BT:
        nodesBTArr[nodes_dict_BT[keyBT]['id']]=nodes_dict_BT[keyBT]['Concept']+':'+nodes_dict_BT[keyBT]['Instance'] 
        # nodesBTArr[nodes_dict_BT[keyBT]['id']]=nodes_dict_BT[keyBT]['Instance'] 

    for keyBT in nodesBTArr:
        # print(keyBT,nodesBTArr,'\n')
        # print('keyBT',keyBT,'nodesBTArr:',nodesBTArr,'\n') 
        if keyBT == root_dictBT:
            rootBT.append(nodesBTArr[keyBT])
            # print('root->nodesBT2Arr[keyBT]',rootBT2,'\n')
            rootnodes_dict=nodes_dict_BT[keyBT]['firstChild']
            # print('nodesBTArr:',nodesBTArr,'\n') 
            # print('rootnodes_dict:',rootnodes_dict,'\n')
            firstChildIdBT=rootnodes_dict['Id']

            if firstChildIdBT in nodesBTArr:
                firstChildInsBT.append(nodesBTArr[firstChildIdBT])
                g.add_edges_from(zip(rootBT, firstChildInsBT))
               
                # print('g.edges:',rootBT,firstChildInsBT)
                # print('rootnodes_dict before pop:',rootnodes_dict,'\n')
                rootnodes_dict.pop('Id')
                # print('rootnodes_dict after pop:',rootnodes_dict,'\n')
                
                while rootnodes_dict!= None: 
                    # print('Next',rootnodes_dict['Next'])
                    if rootnodes_dict['Next']!=None:
                        rootnodes_dict=rootnodes_dict['Next']
                        # print('rootnodes_dict:',rootnodes_dict,'\n')
                        nextChildBT=rootnodes_dict['Id']
                        nextChildInsBT = (nodesBTArr[nextChildBT])
                        nextChildInsBT_list.append(nextChildInsBT)
                        # print('nextChildInsBT_list:',nextChildInsBT_list)
                        # print('nextChildBT:',nextChildBT,'nextChildInsBT:',nextChildInsBT,'\n')
                        for nextChildInsBT in nextChildInsBT_list:
                            # print('next:', nextChildInsBT)
                            g.add_edges_from(zip(rootBT,nextChildInsBT_list))
                            # print('g.edges1:',rootBT,nextChildInsBT)
                            # print(nextChildInsBT)
                            # if nextChildInsBT==('Priority', 'Priority'):
                            #     print('type of nextChildInsBT',nextChildInsBT)
                            #     # check for children - firstChild, Next
                            # elif nextChildInsBT==('Sequence', 'Sequence'):
                            #     print('type of nextChildInsBT',nextChildInsBT)
                            #     # check for children - firstChild, Next

                            nextChildInsBT_list.remove(nextChildInsBT)
                        rootnodes_dict.pop('Id')
                        # print('rootnodes_dict:',rootnodes_dict,'\n')   
                    else:
                        break
            else:
                continue
    return g.nodes(),g.edges()

# Ordering the edge list to tackle the problem that the NetworkX Graph class doesn't retain the order of nodes in an edge
def edgelist_order(G, edgelist):
    # keep ordering of edgelist
    # edgelist = [sorted(edge) for edge in edgelist]
    edgelist = [edge for edge in edgelist]
    # print('edgelist:',edgelist)
    mapping = {tuple(edge): index for (edge, index) in zip(edgelist, range(len(edgelist)))}
    # print('edge mapping:',mapping)
    edge_dict = {index: tuple(edge) for (edge, index) in zip(edgelist, range(len(edgelist)))}
    # print('edge_dict:',edge_dict)
    G.add_edges_from(edgelist)
    G_edges=sorted(G.edges(), key=lambda edge: mapping[tuple(list(edge))])
    # print(G_edges)
    # G_edges=sorted(G.edges(), key=lambda edge: mapping[tuple(sorted(list(edge)))])
    
    return G_edges

# Convert edge list to adjacency list
def ELtoAL(edges,nodes):       #converting edge list to adjacency list
    node_index,adj_dict = {},{}
    # adj_dict=OrderedDict()
    adj_list=[]
    for index, value in enumerate(nodes):  
        # print('\nnode index:',index, 'node value:', value,'\n')
        node_index[value]=index
    # print('\n node_index:',node_index)
    for edge in edges:  
        u,v = edge
        # print(edge[0],edge[1])
        u=node_index[edge[0]]
        v=node_index[edge[1]]
        # print('edge:',edge,'edge index:',u,v,'\n')
        if u not in adj_dict:
            adj_dict[u] = []
        if v not in adj_dict:
            adj_dict[v] = []
        adj_dict[u].append(v)
        # print('graph[u]',graph[u],'\n')
    #     # graph[v].append(u)
        # print('graph[v]',graph[v],'\n')
    #     # for adj in graph:
    #     #     print('adj:',adj,'\n')
    # print('adj_dict',adj_dict,'\n')

    # ## ordering the dictionary by sorting the keys if edges are not in order
    # cnt={}
    # for k,v in adj_dict.items():
    #     # print(k,v)
    #     cnt[k]=v
    #     # print('cnt[k]:',cnt[k])
    # sorted_adj_dict=dict(sorted([(k,v) for k,v in cnt.items()], key=lambda t:int(t[0])))
    # print('sorted_adj_dict:',sorted_adj_dict)

    for adj in list(adj_dict):
        # print('\adj:',AdjLis[adj],'\n')
        adj_list.append(adj_dict[adj])
    # print('G_adj:',adj_list)

    return adj_list

# Convert adjacency list to edge list
def ALtoEL(nodes,adj): #converting adjacency list to edge list
    # print('*********ALtoEL(adj,nodes)*********')
    # print('nodes:',nodes,'adj:',adj)
    edgelist =[]
    node_ind,adj_index={},{}
    # graph = {}
    for index, value in enumerate(nodes):      
        for ind, val in enumerate(adj):   
            # print('node index:',index, 'node value:', value,'\n')
            node_ind[index]=value
            # print('adj ind:',ind, 'adj val:', val,'\n')
            adj_index[ind]=val
    # print('node_index:',node_ind, 'adj_index:',adj_index)

    for i in adj_index:
        # print('i:',i,'adj_index[i]:',adj_index[i]) 
        for ad in adj_index[i]:
            # print('ad:',ad)
            if ad is not None: 
                # if ad==node_ind[]:
                u=node_ind[i]
                # if node_ind[ad] is not None:
                v=node_ind[ad]
                edge=(u,v)
                edgelist.append(edge)
                # print('u:',u,'v:',v, 'edge:',edge)
                # print('edgelist',edgelist,'\n')
            else:
                continue
    return edgelist

# To create a dictionary for nodes and edges
def dictionaryBT(T,E):
    bt_dict,edge_dict,control_dict,explainer_dict={},{},{},{}
    for index, value in enumerate(T): 
      bt_dict[index]=value
    # print('bt_dict:',bt_dict)

    # for index, value in enumerate(zip(list(T))): 
    #   print('\n','index:',index,'value:',value)

    for key in bt_dict:
      # print(key, '->', bt1_dict[key])
      if not bt_dict[key].startswith('/'):
        #   allroots[key]=bt_dict[key]
          if bt_dict[key].startswith('s') or bt_dict[key].startswith('p'): # omit 'r', 't' and 'f'- root, succeder, failure
            control_dict[key]=bt_dict[key]
      else:
        explainer_dict[key]=(bt_dict[key])

    
    # NetworkX Graph doesn't retain the order of nodes in an edge -> never use g.edges
    for index, value in enumerate(E):  
      edge_dict[index]=value
    # print('edge_dict:',edge_dict,'\n') 

    # predecessor(T,bt_dict)

    return bt_dict, edge_dict

# To create a parent_children dictionary
def parent_child(E):
    child,parent,sibling=[],[],[]
    # parents_children=OrderedDict()
    
    ## get all the parent and children  - split child & parent from G.edges()
    for edge in E:
        # print('edge:',edge)
        u, v = edge
        # print('child:',v, 'parent:',u)
        child.append(v)
        parent.append(u)


    dat = {'child':child, 'parent':parent} 
    # dat = {'child':['B','C','D','E','F'], 'parent':['A','A','C','D','D']}
    # print('dat:',dat,'\n')

    # Create DataFrame 
    df = pd.DataFrame(dat) 

    # # df.to_dict('dict')
    # # df.to_dict('list')
    # # print(df.set_index('ID').T.to_dict('list'))

    # print(df.to_dict('records')) # {child,parent} record
    # print(df.to_dict('index')) # similar to edge dict
    # print(dict(df.values),'\n')# child:parent dictionary

    # # print(dict(list(df.iterrows())))
    
    # # ## create parent-child dictionary
    # # parents_children = {parent: {child for child in dat['child']
    # #     if (parent,child) in list(zip(dat['parent'],dat['child']))} 
    # #         for parent in dat['parent']}
    # # print('parents_children:',parents_children)

    # # for children in parents_children.values():
    # #     # print(children)
    # #     for children_couple in combinations(children,2):
    # #         # print(children_couple)
    # #         T.add_edge(*children_couple)
    # #         sibling.append(children_couple)
    # # print('sibling:',sibling,'\n')
    return df
    
def predecessor(T,bt_dict):
    predecessor_dict={}
    child_dict,parent_dict={},{}
    # for key,child in bt_dict.items():
    #     # print('******bt_dict******')
    #     # print(key,' : ',child)
    #     # parent = list(T1.predecessors(child))
    #     pre = [pred for pred in T1.predecessors(child)]
        
    #     # how to fetch the index or key of parent?
    #     # print('parent:',parent, key, 'child:',[child])
    #     # parent_dict[parent]=child
    #     # pre=(parent,[key,child])
    #     # predecessor.append(pre)
    #     # print('child_dict[key]=child',parent)
    #     # print('parent',parent)
    predecessor={}
    # for i, p in control_dict.items():
    #     # print(i,' : ',p)
    #     parent_dict[p]=i
    # print('parent_dict:',parent_dict)
    # for i, p in explainer_dict.items():
    #     # print(i,' : ',p)
    #     child_dict[p]=i
    # print('child_dict:',child_dict)

    for key,child in bt_dict.items():
        parent = [pred for pred in T.predecessors(child)]
        # pre=(parent,[key,child])
        predecessor_dict[key]=parent
    # print('predecessor:',predecessor_dict)


    # for key,child in bt_dict2.items():
    #     # print('******bt_dict2******')
    #     # print(key,' : ',child)
    #     parent = list(T2.predecessors(child))
    #     # how to fetch the index or key of parent?
    #     # print('parent:',parent, key, 'child:',[child])
    #     pre1=(parent,[key,' : ',child])
    #     predecessor1.append(pre1)
    #     print('predecessor1:',predecessor1)

    return predecessor

# To find the successor
def successor(T,E,bt_dict):
    successor=[]
    for key,child in bt_dict.items():
        # print('******bt_dict******')
        # print(key,' : ',child)
        neighbour=list(T.successors(child))
        # print('neighbour:',neighbour)
        # how to fetch the index or key of parent?
        # print('child:',[child],'neighbour:',neighbour)
        suc=(child,neighbour)
        successor.append(suc)
    # print('successor:',successor,'\n')

    for children in successor:
        # print(children)
        for item in children:
            # print(item))
            for children_couple in combinations(item,2):
                # print(children_couple, item)
                if list(children_couple)==item:
                    # T.add_edge(*children_couple) # Add the edge to G.edge
                    E.append(children_couple)  # Add the edge to G_edge
                    # tuple(item)
    # print('E:',E,'\n')

    #         sibling.append(children_couple)
    # print('sibling:',sibling,'\n')

    return successor

# Writes a tree in node list/adjacency list format to a JSON file.
def to_json(filename,nodes,adj):
    """ Writes a tree in node list/adjacency list format to a JSON file.

    Parameters
    ----------
    filename: str
        The filename for the resulting JSON file.
    nodes: list
        The node list of the tree.
    adj: list
        The adjacency list of the tree.

    Raises
    ------
    Exception
        if the file is not accessible or the JSON writeout fails.

    """

    # dictionary = {'nodes' : nodes, 'adj' : adj}
    # print('dictionary',dictionary)
    # json_object = json.dumps(dictionary, indent=4)
    # print(json_object)
    with open(filename, 'w') as json_file:
        json.dump({'nodes' : nodes, 'adj' : adj}, json_file, indent='\t')
        # json_file.write(json_object)

# Writes a graph in node list/edge list format to a JSON file.
def to_json_edge(filename,nodes,edges):
    with open(filename, 'w') as json_file:
        json.dump({'nodes' : nodes, 'edges' : edges}, json_file, indent='\t')
#     return {'nodes' : nodes, 'adj' : adj}

# ## get a matplotlib visualization of a tree
def plot(H,D,E,options):
    # options = {
    # 'node_size': 500,
    # 'node_color' : "pink", "edge_color" : "tab:pink",
    # "width": 2,
    # "with_labels": True,
    # }
    
    # Y -> 'node_color' : "#89CFF0",  "edge_color" : "tab:blue",
    # X -> 'node_color' : "pink", "edge_color" : "tab:pink",
    #A0CBE2, X -> baby blue #89CFF0 Y -> cornflower blue #6495ED

    # pos=nx.spring_layout(G)
    # pos = topo_pos(G) # to get a tree structure (won't work for a graph with self-loops)

    # fig = plt.figure()

    # ax = plt.gca()
    # color_list=['green']
    # ax.set_prop_cycle('color', color_list)

    # fig, all_axes = plt.subplots(2, 2)
    # ax = all_axes.flat  
    # Set titles for the figure and the subplot respectively
    # fig.suptitle(G.nodes(), fontsize=10, fontweight='bold')
    # ax.set_title(G.nodes(),fontsize=8, fontweight='bold')
    # ax.set_ylabel('cost',style='italic',color='blue')
    # if edit is not None:
    #     ax.set_xlabel(edit,style='italic',color='blue')
    # ax.plot([.1], [.95], '-->')
    # ax.margins(0.11)
    # plt.tight_layout()
    # plt.axis("off")

    # ax.plot([2], [1], 'o')
    # ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),
    #         arrowprops=dict(facecolor='black', shrink=0.05))
    # ax.text(0.95, 0.01, 'colored text in axes coords',
        # verticalalignment='bottom', horizontalalignment='right',
        # transform=ax.transAxes,
        # color='green', fontsize=15)
    
    # name2num = {name: num + 1 for num, name in enumerate(list(G.nodes))}

    # name2num = {name: f'{name}{[num]}' for num, name in enumerate(list(G.nodes), 0)}
    # fig, (ax, ax1) = plt.subplots(1, 2)
    # ax.set_title((G.nodes()),fontsize=8, fontweight='bold')
    # ax.margins(0.11)
    # plt.tight_layout()
    # nx.draw_networkx(G, pos, ax=ax, **options)


    # H = nx.relabel_nodes(G, mapping=name2num, copy=True)
    # pos1 = topo_pos(H)
    # nx.draw_networkx(H, pos1, ax=ax1,  **options)

    # legend_text = "\n".join(f"{v} - {k}" for k, v in name2num.items())
    # props = dict(boxstyle="round", facecolor="w", alpha=0.5)
    # ax1.text(
    #     1.15,
    #     0.95,
    #     legend_text,
    #     transform=ax1.transAxes,
    #     fontsize=14,
    #     verticalalignment="top",
    #     bbox=props,
    # )

    # H = nx.DiGraph()
    ax1= plt.gca()
    pos1 = topo_pos(H)  # To draw the graph as tree. When we use topo_pos, the position in the edge list changes. To correct the edge list, we use nx.edge_dfs() function
    # pos1 = nx.spring_layout(H)

    # edgelist= [("p", "s"), ("s","DicePrivate"), ("s","KernelSHAPLocal")] 
    cost= 'cost = 1'
    # x= ["DicePublic"]
    # op = 'insert '
    edit= 'replace (ALE, KernelSHAPLocal), ' + cost
    # replace (A, 0), (C, 1), (D, 2), (E, 3); delete B; insert '4' as 1st child of 'P'
    # H_edges=edgelist_order(H,edgelist)
    # print('H.nodes:',H.nodes())
    # print('G.edges',H_edges,'\n')
    # bt_dict,edge_dict = dictionaryBT(H,H_edges)
    # adj_list = ELtoAL(H_edges,H.nodes)
    # print('H_adj:',adj_list,'\n')

    
    # adjacency=  
    # edges=  

    fig.suptitle(H.nodes(), fontsize=10, fontweight='bold')
    ax1.set_title(D,fontsize=8, fontweight='bold')
    # ax1.set_ylabel('delete (B, C), cost = 1',style='italic',color='black')
    # ax1.set_ylabel(A,style='italic',color='black')
    ax1.set_xlabel(E,style='italic',color='black')
    # ax1.set_xlabel(edit,style='italic',color='black')
    ax1.margins(0.11)
    nx.draw_networkx(H, pos1, ax=ax1, **options)

    # # nx.draw_networkx_labels(H, pos1, ax=ax1,
    # #                     labels=name2num,
    # #                     font_size=12)
    # # alpha=0.5 ->transparent
    plt.show() # display
    # plt.savefig('g1_1.png')

    return None

# to transform graph g1 into g2 using the edit operations from the output of the GED
def transform_graph(T1, T2, edit_operations):
    # Apply the edit operations to T1 to create the new graph T3
    T3 = nx.Graph(T1)

    # Node substitution
    for node, replacement in edit_operations['node_substitution'].items():
        if node != replacement:
            T3.add_node(replacement)
            T3.remove_node(node)
            # if replacement is not None:
            #     T1.nodes[node]['label'] = T2.nodes[replacement]['label']

    # Node deletion
    for node in edit_operations['node_deletion']:
        T1.remove_node(node)

    # Node insertion
    for node in edit_operations['node_insertion']:
        T1.add_node(node, label=T2.nodes[node]['label'])

    # Edge substitution
    for edge, replacement in edit_operations['edge_substitution'].items():
        T3.remove_edge(edge[0], edge[1])
        T3.add_edge(replacement[0], replacement[1])

    # Edge deletion
    for edge in edit_operations['edge_deletion']:
        T1.remove_edge(edge[0], edge[1])

    # Edge insertion
    for edge in edit_operations['edge_insertion']:
        T1.add_edge(edge[0], edge[1])

## get a matplotlib visualization of a tree
def plot_edits(H,operation,edit,cost):
    options = {
    'node_size': 500,
    'node_color' : "pink", "edge_color" : "tab:pink",
    "width": 2,
    "with_labels": True,
    }
    # Y -> 'node_color' : "#89CFF0",  "edge_color" : "tab:blue",
    # X -> 'node_color' : "pink", "edge_color" : "tab:pink",
    #A0CBE2, X -> baby blue #89CFF0 Y -> cornflower blue #6495ED

    # # Remove self-loops
    # H.remove_edges_from(nx.selfloop_edges(H))

    # # Set the node labels
    # for node in H.nodes():
    #     H.nodes[node]['label'] = node

    fig = plt.figure()
    ax= plt.gca()
    if operation:
        fig.suptitle(operation, fontsize=10, fontweight='bold')
        # fig.suptitle("Edit operation: " + operation, fontsize=10, fontweight='bold')
    # fig.suptitle(H.nodes(), fontsize=10, fontweight='bold')
    ax.set_title(list(H.nodes()),fontsize=8, fontweight='bold')
    # ax.set_title(operation,fontsize=8, fontweight='bold')
    ax.set_xlabel(str(edit),style='italic',color='black')
    ax.set_ylabel("Cost: " + str(cost),style='italic',color='black')
    ax.margins(0.11)
    if nx.recursive_simple_cycles(H)!=None:
        # print("Graph contains a cycle. Handle it or raise an error")
        pos1 = nx.spring_layout(H)
        nx.draw_networkx(H, pos1, ax, **options)
    else:
        pos = topo_pos(H)# To draw the graph as tree. When we use topo_pos, the position in the edge list changes. To correct the edge list, we use nx.edge_dfs() function
        nx.draw_networkx(H, pos, ax, **options) 
    plt.show() # display
    # plt.savefig('g1_1.png')

    return None

def compare_and_plot_graphs(T,T2,ED):
    options = {
    'node_size': 500,
    'node_color' : "pink", "edge_color" : "tab:pink",
    "width": 2,
    "with_labels": True,
    }

    options1 = {
    'node_size': 500,
    'node_color' : "#89CFF0",  "edge_color" : "tab:blue",
    "width": 2,
    "with_labels": True,
    }
    # # check T & G2 are identical or not
    # title=identicalBT(T,T2)

    # Convert the graphs to tree representations
    T = nx.dfs_tree(T)
    T2 = nx.dfs_tree(T2)
    # print(T.nodes())

    T_edges=list(nx.edge_dfs(nx.Graph(T.edges), T.nodes()))
    T2_edges=list(nx.edge_dfs(nx.Graph(T2.edges), T2.nodes()))

    # Compare the level order traversal of the two trees
    if level_order_traversal(T) == level_order_traversal(T2):
        title = 'The two trees are identical'
        print("The two trees are identical.")
    else:
        title = 'The two trees are not identical'
        print("The two trees are not identical.")

    # # Plot the two graphs in one figure
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title, fontsize=10, fontweight='bold')
        # fig.suptitle("Edit operation: " + operation, fontsize=10, fontweight='bold')
    # fig.suptitle(H.nodes(), fontsize=10, fontweight='bold')
    ax1.set_title(T.nodes(),fontsize=8, fontweight='bold')
    ax2.set_title(T2.nodes(),fontsize=8, fontweight='bold')
    ax1.set_xlabel(T_edges,style='italic',color='black')
    ax2.set_xlabel(T2_edges,style='italic',color='black')
    ax1.set_ylabel("Edit distance: " + str(ED),style='italic',color='black')
    # ax.margins(0.11)

    plt.subplot(121)
    if nx.recursive_simple_cycles(T)!=None:
        # print("Graph contains a cycle. Handle it or raise an error")
        pos = nx.spring_layout(T)
        nx.draw_networkx(T, pos, **options)
    else:
        pos = topo_pos(T)
        nx.draw_networkx(T, pos, **options)
    pos1 = topo_pos(T2)
    plt.subplot(122)
    nx.draw_networkx(T2, pos1, **options1)
    plt.show() # display

def ged_operations(edit_operations):
    # Edit operations
    node_substitution = edit_operations[0]
    substitution_cost = edit_operations[1]
    edge_substitution = edit_operations[2]
    edge_substitution_cost = edit_operations[3]
    node_deletion = edit_operations[4]
    deletion_cost = edit_operations[5]
    edge_deletion = edit_operations[6] 
    edge_deletion_cost = edit_operations[7]
    node_insertion = edit_operations[8]
    insertion_cost = edit_operations[9]
    edge_insertion = edit_operations[10]
    edge_insertion_cost = edit_operations[11]

    return node_substitution, substitution_cost, edge_substitution, edge_substitution_cost, node_deletion, deletion_cost, edge_deletion, edge_deletion_cost, node_insertion, insertion_cost, edge_insertion, edge_insertion_cost

    # if len(edit_operations) > 2:
    #     node_substitution = edit_operations[2]
    # else:
    #     node_substitution = None
    # if len(edit_operations) > 3:
    #     node_substitution_cost = edit_operations[3]
    #     print('Node substitution:', node_substitution, ', Node substitution cost:', node_substitution_cost)
    # if len(edit_operations) > 4:
    #     edge_substitution = edit_operations[4]   
    # if len(edit_operations) > 5:
    #     edge_substitution_cost = edit_operations[5]
    #     print('Edge substitution:', edge_substitution, ', Edge substitution cost:', edge_substitution_cost)
    # if len(edit_operations) > 6:
    #     node_deletion = edit_operations[6]
    # if len(edit_operations) > 7:
    #     node_deletion_cost = edit_operations[7]
    #     print('Node deletion:', node_deletion, ', Node deletion cost:', node_deletion_cost)
    # if len(edit_operations) > 8:
    #     edge_deletion = edit_operations[8]
    # if len(edit_operations) > 9:
    #     edge_deletion_cost = edit_operations[9]
    #     print('Edge deletion:', edge_deletion,', Edge deletion cost:', edge_deletion_cost)
    # if len(edit_operations) > 10:
    #     node_insertion = edit_operations[10]
    # if len(edit_operations) > 11:
    #     node_insertion_cost = edit_operations[11]
    #     print('Node insertion:', node_insertion, ', Node insertion cost:', node_insertion_cost)
    # if len(edit_operations) > 12:
    #     edge_insertion = edit_operations[12]
    # if len(edit_operations) > 13:
    #     edge_insertion_cost = edit_operations[13]
    #     print('Edge insertion:', edge_insertion, ', Edge insertion cost:', edge_insertion_cost)

    # total_cost = sum(node_substitution_cost.values()) + sum(edge_substitution_cost.values()) + sum(node_deletion_cost.values()) + sum(edge_deletion_cost.values()) + sum(node_insertion_cost.values()) + sum(edge_insertion_cost.values())


def plot_edit_operations(G1, G2, edit_operations):
    """
    Plot each edit operation in the sequence of edit operations
    """

    # # nx.set_node_attributes(G1, {node: {'label': G1.nodes[node]} for node in G1.nodes()})
    # nx.set_node_attributes(G2, {node: {'label': G2.nodes[node]} for node in G2.nodes()})
    # Initialize the original graph
    T = G1.copy()
    # nx.set_node_attributes(T, {node: {'label': T.nodes[node]} for node in T.nodes()})

    # Edit operations
    if len(edit_operations) > 0:
        node_substitution = edit_operations[0]
    else:
        node_substitution = None
    if len(edit_operations) > 1:
        node_substitution_cost = edit_operations[1]
    if len(edit_operations) > 2:
        edge_substitution = edit_operations[2]
    if len(edit_operations) > 3:
        edge_substitution_cost = edit_operations[3]
    if len(edit_operations) > 4:
        node_deletion = edit_operations[4]
    if len(edit_operations) > 5:
        node_deletion_cost = edit_operations[5]
    if len(edit_operations) > 6:
        edge_deletion = edit_operations[6]
    if len(edit_operations) > 7:
        edge_deletion_cost = edit_operations[7]
    if len(edit_operations) > 8:
        node_insertion = edit_operations[8]
    if len(edit_operations) > 9:
        node_insertion_cost = edit_operations[9]
    if len(edit_operations) > 10:
        edge_insertion = edit_operations[10]
    if len(edit_operations) > 11:
        edge_insertion_cost = edit_operations[11]

    # # Initialize substitution list
    # substitution_list = []
    substitution_cost = {}
    
    # # Combine node and edge substitution lists
    # substitution_list += node_substitution
    # substitution_list += edge_substitution

    substitution_list = {}
    substitution_list.update(node_substitution)
    substitution_list.update(edge_substitution)
    
    # # Combine substitution costs
    substitution_cost.update(node_substitution_cost)
    substitution_cost.update(edge_substitution_cost)
    
    # # Perform substitutions
    # for node_or_edge, substitution in substitution_list:
    #     print(node_or_edge, substitution)
    #     # if node_or_edge is not None:
    #     if isinstance(node_or_edge, tuple) and T.has_edge(node_or_edge[0], node_or_edge[1]):
    #         if substitution is not None:
    #             if 'label' in G2[substitution[0]][substitution[1]]:
    #                 T.add_edge(substitution[0], substitution[1], label=G2[substitution[0]][substitution[1]]['label'])
    #             else:
    #                 T.add_edge(substitution[0], substitution[1])
    #             T.remove_edge(node_or_edge[0], node_or_edge[1])
    #     # elif node_or_edge in T.nodes:
    #     else: # node_or_edge is a node
    #         if substitution is not None:
    #             T.nodes[node_or_edge]['label'] = G2.nodes[substitution]['label']
    # plot_edits(T, 'Node-Edge Substitution', substitution_list, substitution_cost)


    # # Initialize deletion list
    # # deletion_list = []
    # deletion_cost = {}

    # deletion_list = node_deletion + edge_deletion
    # deletion_cost.update(node_deletion_cost)
    # deletion_cost.update(edge_deletion_cost)

    # # Perform node and edge deletion
    # for node in deletion_list:
    #     if node in T.nodes:
    #         T.remove_node(node)
    # plot_edits(T, 'Node-Edge Deletion', deletion_list, deletion_cost)

    # # Initialize insertion list 
    # # insertion_list = []
    # insertion_cost = {}

    # # Combine node and edge insertion lists, insertion costs:
    # insertion_list = node_insertion + edge_insertion
    # insertion_cost.update(node_insertion_cost)
    # insertion_cost.update(edge_insertion_cost)

    # # Perform node and edge insertion
    # for node in insertion_list:
    #     if node in G2.nodes:
    #         T.add_node(node, label=G2.nodes[node]['label'])
    # plot_edits(T, 'Node-Edge Insertion', insertion_list, insertion_cost)

    
    # Perform node substitution
    relabel_map = {node: substitution for node, substitution in node_substitution.items() if substitution is not None}
    T = nx.relabel_nodes(T, relabel_map)
    plot_edits(T, 'Node Substitution', node_substitution, node_substitution_cost)

   # Perform edge substitution
    for edge, new_edge in edge_substitution.items():
        # print(edge, new_edge)
        # if T.has_edge(edge[0], edge[1]):
        #  T.remove_edge(edge[0], edge[1])
        T.add_edge(relabel_map.get(new_edge[0], new_edge[0]), relabel_map.get(new_edge[1], new_edge[1]))
    plot_edits(T, 'Edge Substitution', edge_substitution, edge_substitution_cost)
    # plot_edits(T, 'Node-Edge Substitution', substitution_list, substitution_cost)

    T_copy = T.copy()
    # # Perform edge deletion
    for edge in edge_deletion:
        if T_copy.has_edge(edge[0], edge[1]):
            T_copy.remove_edge(edge[0], edge[1])
    plot_edits(T_copy, 'Edge Deletion', edge_deletion, edge_deletion_cost)

    # Perform node deletion
    for node in node_deletion:
        T.remove_node(node)
    plot_edits(T, 'Node Deletion', node_deletion, node_deletion_cost)

    # Perform node insertion
    for node in node_insertion:
        T.add_node(node, label=G2.nodes[node]['label'])
    plot_edits(T, 'Node Insertion', node_insertion, node_insertion_cost)
    
    
    # Perform edge insertion
    for edge in edge_insertion: 
        if not T.has_edge(edge[0], edge[1]):
            T.add_edge(edge[0], edge[1])
    plot_edits(T, 'Edge Insertion', edge_insertion, edge_insertion_cost)

    # print('Node substitution:', node_substitution, ', Node substitution cost:', node_substitution_cost)
    # print('Edge substitution:', edge_substitution, ', Edge substitution cost:', edge_substitution_cost)
    # print('Edge deletion:', edge_deletion,', Edge deletion cost:', edge_deletion_cost)
    # print('Node deletion:', node_deletion, ', Node deletion cost:', node_deletion_cost)
    # print('Node insertion:', node_insertion, ', Node insertion cost:', node_insertion_cost)
    # print('Edge insertion:', edge_insertion, ', Edge insertion cost:', edge_insertion_cost)

    total_cost = sum(node_substitution_cost.values()) + sum(edge_substitution_cost.values()) + sum(node_deletion_cost.values()) + sum(edge_deletion_cost.values()) + sum(node_insertion_cost.values()) + sum(edge_insertion_cost.values())

    compare_and_plot_graphs(T,G2,total_cost)

def graph_dfstree(G):
    # Convert the graphs to tree representations
    T = nx.dfs_tree(G)
    return T

# Compare the trees by visiting the nodes in a specific order
def level_order_traversal(T):
    nodes = []
    queue = []
    queue.append(list(T.nodes())[0])
    while len(queue) > 0:
        current = queue.pop(0)
        nodes.append(current)
        for neighbor in T.neighbors(current):
            queue.append(neighbor)
    return nodes

# def identicalBT(G1,G2):
#     # Convert the graphs to tree representations
#     T1 = nx.dfs_tree(G1)
#     T2 = nx.dfs_tree(G2)
#     # Compare the level order traversal of the two trees
#     if level_order_traversal(T1) == level_order_traversal(T2):
#         title = 'The two trees are identical'
#         print("The two trees are identical.")
#     else:
#         title = 'The two trees are not identical'
#         print("The two trees are not identical.")
#     return title

def assign_labels(G):
    labels = {}
    for i, node in enumerate(G.nodes()):
        labels[node] = 'label_' + str(i)
    nx.set_node_attributes(G, labels, 'label')
    node_labels = nx.get_node_attributes(G, 'label')
    # print('node_labels',node_labels)
    return G

def read_nodes_from_file(filename):
    # Read the json file
    with open(filename, 'r') as f:
        data = json.load(f)

    # # select two random graphs
    random_graphs = random.sample(data, 2)

    # # extract the nodes and edges for GED
    g1_nodes = random_graphs[0]["nodes"]
    g1_edges = random_graphs[0]["edge"]
    g1_adj = random_graphs[0]["adj"]

    g2_nodes = random_graphs[1]["nodes"]
    g2_edges = random_graphs[1]["edge"]
    g2_adj = random_graphs[1]["adj"]

    # create the nx DiGraph object
    g1 = nx.DiGraph()
    g2 = nx.DiGraph()

    # # add the nodes and edges to the graph
    g1.add_nodes_from(g1_nodes)
    g1.add_edges_from(g1_edges)
    g2.add_nodes_from(g2_nodes)
    g2.add_edges_from(g2_edges)

    assign_labels(g1)
    assign_labels(g2)

    # print('g1.nodes:',g1.nodes())
    # print('g1.edges:',g1.edges())
    # g1_edges=list(nx.edge_dfs(nx.Graph(g1.edges), g1.nodes()))
    # print('g1.edge_dfs:',g1_edges)
    # print('g1.adj',g1_adj,'\n')
    # print('g2.nodes:',g2.nodes())
    # print('g2.edges:',g2.edges())
    # g2_edges=list(nx.edge_dfs(nx.Graph(g2.edges), g2.nodes()))
    # print('g2.edge_dfs:',g2_edges)
    # print('g2.adj',g2_adj,'\n')

    bt_dict1, edge_dict1 = dictionaryBT(g1,g1_edges) 
    bt_dict2, edge_dict2 = dictionaryBT(g2,g2_edges) 

    options1 = {
    'node_size': 500,
    'node_color' : "pink", "edge_color" : "tab:pink",
    "width": 2,
    "with_labels": True,
    }
    plot(g1,bt_dict1,g1_edges,options1)

    options2 = {
    'node_size': 500,
    'node_color' : "#89CFF0",  "edge_color" : "tab:blue",
    "width": 2,
    "with_labels": True,
    }
    plot(g2,bt_dict2,g2_edges,options2)


    # # Extract the necessary information for GED
    # ged_nodes = case['nodes']
    # ged_edges = case['edge']
    # ged_graph = nx.DiGraph()
    # ged_graph.add_nodes_from(ged_nodes)
    # ged_graph.add_edges_from(ged_edges)

    # # Extract the necessary information for TED
    # ted_nodes = case1['nodes']
    # ted_adj = case1['adj']
    # ted_graph = nx.Graph()
    # ted_graph.add_nodes_from(ted_nodes)
    # for i, node in enumerate(ted_nodes):
    #     for neighbor in ted_adj[i]:
    #         ted_graph.add_edge(node, ted_nodes[neighbor])
    return g1,g2

def plot_tree(r_nodes, r_edge, bt_nodes, bt_edge):
    # create the nx DiGraph object
    g1 = nx.DiGraph()
    g2 = nx.DiGraph()

    # print("randomBT:", randomBT)
    # add the nodes and edges to the graph
    g1.add_nodes_from(r_nodes)
    g1.add_edges_from(r_edge)
    g2.add_nodes_from(bt_nodes)
    g2.add_edges_from(bt_edge)

    # assign_labels(g1)
    # assign_labels(g2)

    # # print('g1.nodes:',g1.nodes())
    # # print('g1.edges:',g1.edges())
    # g1_edges=list(nx.edge_dfs(nx.Graph(g1.edges), g1.nodes()))
    # # print('g1.edges_dfs:',g1_edges)
    # # # print('g1.adj',g1_adj,'\n')
    # # print('\ng2.nodes:',g2.nodes())
    # # print('g2.edges:',g2.edges())

    # g2_edges=list(nx.edge_dfs(nx.Graph(g2.edges), g2.nodes()))
    # # print('\ng2.edges_dfs:',g2_edges)
    # # print('g2.adj',g2_adj,'\n')

    # # bt_dict1, edge_dict1 = dictionaryBT(g1,g1_edges) 
    # # bt_dict2, edge_dict2 = dictionaryBT(g2,g2_edges) 

    options1 = {
    'node_size': 500,
    'node_color' : "pink", "edge_color" : "tab:pink",
    "width": 2,
    "with_labels": True,
    }
    # plot(g1,bt_dict1,g1_edges,options1)

    options2 = {
    'node_size': 500,
    'node_color' : "#89CFF0",  "edge_color" : "tab:blue",
    "width": 2,
    "with_labels": True,
    }
    # plot(g2,bt_dict2,g2_edges,options2)
    return g1, g2

def plot_mAstar(randomBT,BT):

    options1 = {
    'node_size': 500,
    'node_color' : "pink", "edge_color" : "tab:pink",
    "width": 2,
    "with_labels": True,
    }

    options2 = {
    'node_size': 500,
    'node_color' : "#89CFF0",  "edge_color" : "tab:blue",
    "width": 2,
    "with_labels": True,
    }

    g1 = nx.DiGraph()
    g2 = nx.DiGraph()

    # add the nodes and edges to the graph
    g1.add_nodes_from(randomBT["nodes"])
    g1.add_edges_from(randomBT["edge"])
    g2.add_nodes_from(BT["nodes"])
    g2.add_edges_from(BT["edge"])
    
    assign_labels(g1)
    assign_labels(g2)
    
    # print('g1.nodes:',g1.nodes())
    # print('g1.edges:',g1.edges())
    g1_edges=list(nx.edge_dfs(nx.Graph(g1.edges), g1.nodes()))
    # print('g1.edge_dfs:',g1_edges,'\n')
    # print('g1.adj',g1.adj,'\n')
    # g1_AdjLis = ELtoAL(g1_edges,g1.nodes)
    # print('g1.adj',g1_AdjLis,'\n')

    # g1_edges=edgelist_order(g1,g1_edgelist)

    # print('g2.nodes:',g2.nodes())
    g2_edges=list(nx.edge_dfs(nx.Graph(g2.edges), g2.nodes()))
    # print('g2.edge_dfs:',g1_edges,'\n')
    # print('g2.adj',g2.adj)
    # g2_AdjLis = ELtoAL(g2_edges,g2.nodes)
    # print('g2.adj',g2_AdjLis,'\n')

    g1_df=parent_child(g1_edges)    
    bt_dict, edge_dict = dictionaryBT(g1,g1_edges)
    successor(g1,g1_edges,bt_dict)
    # plot(g1,bt_dict,g1_edges,options1)
    # print('g1_data:',g1_df)
    # print('g1.edges (modified):',g1.edges())
    # print('g1_edges (modified):',g1_edges,'\n')
    g1_edge = g1_edges

    g2_data= parent_child(g2_edges)
    # print('g2.edges:',g2.edges())
    bt_dict2, edge_dict2 = dictionaryBT(g2,g2_edges)
    successor(g2,g2_edges,bt_dict2)
    # print('g2_data:',g2_data)
    # print('g2.edges (modified):',g2.edges())
    # print('g2_edges (modified):',g2_edges,'\n')
    bt_dict3, edge_dict3 = dictionaryBT(g2,g2_edges)
    g2_df=parent_child(g2_edges)
    # plot(g2,bt_dict2,g2_edges,options2)
    g2_edge = g2_edges

    g1.add_edges_from(g1_edge)
    g2.add_edges_from(g2_edge)

    return g1, g2

def set_active_sheet(workbook, sheet_name):
    workbook.active = 0
    try:
        worksheet = workbook[sheet_name]
    except KeyError:
        worksheet = workbook.create_sheet(sheet_name)
    worksheet.sheet_state = 'visible'
    
def set_active_sheet_weights(workbook, sheet_name):
    workbook.active = 0
    try:
        worksheet = workbook[sheet_name]
    except KeyError:
        worksheet = workbook.create_sheet(sheet_name)
    worksheet.sheet_state = 'visible'

def compare_dicts(dict1, dict2):
    return json.dumps(dict1, sort_keys=True) == json.dumps(dict2, sort_keys=True)

def workbook(w):
    _, workbook_name = os.path.split(w)
    return workbook_name

def ged(g1, g2, explainer_costs):
    return graph_edit_distance(g1, g2, explainer_costs)

# def modified_ged(g1, g2, explainer_costs):
#     return modified_graph_edit_distance(g1, g2, explainer_costs)

# def ted(g1, g2, explainer_costs):
#     # x_nodes, x_adj, y_nodes, y_adj, delta = preprocess_ted(g1, g2)
#     return ted.ted(x_nodes, x_adj, y_nodes, y_adj, delta)

def calculate_score(algorithm, g1, g2, explainer_costs):
    if algorithm == 'ged':
        return ged(g1, g2, explainer_costs)
    elif algorithm == 'modified_ged':
        return modified_graph_edit_distance(g1, g2, explainer_costs)
    # elif algorithm == 'ted':
    #     return ted(g1, g2, explainer_costs)
    else:
        raise ValueError('Unsupported algorithm: {}'.format(algorithm))

def main():

     # Read the costs of the edit operations from a file
    with open("costs.json") as f:
        costs = json.load(f)
   
    # G=nx.DiGraph()
    # G.add_edges_from([('A', 'B'), ('B','D'), ('D','H'), ('D','I'), ('B','E'), ('A','C'), ('C','F'),('C','G'), ('G','J'),('G','K')])
    # # G_edges=G.edges
    # G_edges=list(nx.edge_dfs(nx.Graph(G.edges), G.nodes()))
    # # G_edges=G_edge_dfs
    # # G.add_nodes_from(['A','B','C','E','D'])
    # print('G.nodes:',G.nodes())
    # print('G.edges',G.edges,'\n')
    # # print('G.edges (ordered)',G_edges,'\n')
    # print('G.edge_dfs:', G_edges)
    # bt_dict,edge_dict = dictionaryBT(G,G_edges)

    # # print('G',sorted(G.nodes()))
    # # print('G.edges:',G.edges()) # NetworkX Graph class doesn't retain the order of nodes in an edge
    # # x_nodes=G.nodes
    # adj_list = ELtoAL(G_edges,G.nodes)
    # print('G_adj:',adj_list,'\n')
    
    # # nx.ordered.Graph()
    # # list(G.adj[1])  # or list(G.neighbors(1))
    # # G.degree[1]  # the number of edges incident to 1

    # ## Get all edges linked to a given node
    # # G.in_edges(node)
    # # G.out_edges(node)  
    # # G.in_edges_iter(node) 
    # # G.out_edges_iter(node)

    
    # nodes= G.nodes 
    
    # # print('x_nodes:',x_nodes,'\n')
    # # nodes=['Priority', 'SHAP Explainer', 'LIME Explainer', 'Nearest Neighbour Explainer']
    #  # print(nodes)
    
    # # tree_dict = {"nodes":nodes, "adj":adj_list}
    # # jsonString = json.dumps(tree_dict)
    # # jsonFile = open("data.json", "w")
    # # jsonFile.write(jsonString)
    # # jsonFile.close()

    # # to_json('x.json',nodes, adj_list)
    # # edgelist=ALtoEL(nodes,adj_list)
    # # print('edgelist:',edgelist)
    # # print('******************','\n')

    # g_df=parent_child(G_edges) 

    # # edit_operation = 'delete D'
    # # plot(G,(edit_operation))
    # plot(G,bt_dict,G_edges)

    # # replace function
    # # insert
    # # delete function

    # # tree = Tree()
    # # tree.create_node("Harry", "harry")  # root node
    # # tree.create_node("Jane", "jane", parent="harry")
    # # tree.create_node("Bill", "bill", parent="harry")
    # # tree.create_node("Diane", "diane", parent="jane")
    # # tree.create_node("Mary", "mary", parent="diane")
    # # tree.create_node("Mark", "mark", parent="jane")
    # # tree.show()


    # # ## Networkx convert graph to tree preserving edge data
    # # data_G = { (u, v) : d for u, v, d in G.edges.data() }  # create a dict of { edge : data ...}
    # # print('data_G',data_G)
    # # Gt = nx.dfs_tree(G) # create tree
    # # print('Gt',Gt)
    # # nx.set_edge_attributes(Gt, data_G) # set all tree edge data at once
    # # plot(Gt,g_df)

    # successor(G,G_edges,bt_dict)
    # G_edges=list(nx.edge_dfs(nx.Graph(G.edges), G.nodes()))
    # print('G.edges (modified):',G.edges(),'\n') 
    # bt_dict1,edge_dict1 = dictionaryBT(G,G.edges())
    # # plot(G,bt_dict1,G.edges(),adj_list)
    # print('G.edge_dfs (modified):',G_edges,'\n')
    # bt_dict2,edge_dict2 = dictionaryBT(G,G_edges)
    # g_df1=parent_child(G_edges) 
   
    # #Apply DFS on graph

    # # print('AdjLis',AdjLis,'\n')
    # # stack = []
    # # stack.append(G.nodes)
    # # # path_idx = stack.index(stack)
    # # # print('path_idx',path_idx,'\n')
    # # dict = {}
    # # while stack:
    # #     print('stack:',stack)
    # #     curr = stack.pop()
    # #     print('curr:',curr)
    # #     print('AdjLis[curr]:',AdjLis[curr])
    # #     # if curr==dst:
    # #     #     return True
    # #     for v in AdjLis[curr]:
    # #         print('curr',curr)
    # #         if v in dict:      #check if vertices already visited
    # #             continue
    # #         else:
    # #             stack.append(v)
    # #             dict[v] = True     #add the vertices in dictionary as we visit it to avoid looping
    # # # return False


if __name__ == "__main__":
    main()
    
