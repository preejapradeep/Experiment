import copy
import csv
import json
import random
import time

import numpy as np

from json2graph_full import ELtoAL
from json2graph_full import ALtoEL

start_time = time.time()
explainerNames=["/Images/Anchors","/Images/Counterfactuals","/Images/GradCamTorch","/Images/IG", "/Images/LIME", "/Tabular/ALE", "/Tabular/Anchors","/Tabular/DeepSHAPGlobal", "/Tabular/DeepSHAPLocal", "/Tabular/DicePrivate","/Tabular/DicePublic","/Tabular/DisCERN","/Text/NLPClassifier","/Timeseries/CBRFox","/Tabular/IREX", "/Tabular/Importance", "/Text/LIME", "/Tabular/LIME", "/Tabular/NICE", "/Tabular/TreeSHAPGlobal", "/Tabular/TreeSHAPLocal", "/Tabular/KernelSHAPGlobal", "/Tabular/KernelSHAPLocal"]
# Assume control_nodes is a 2-dimensional array
control_nodes = np.array([["s", "p"]])

used_nodes = set()
unused_nodes = set()

def find_parent(node, edge):
    """
    Finds the parent node and its index for a given node in the tree.

    Args:
        node (int): The node whose parent is to be found.
        edge (List[Tuple[int, int]]): The edge list representing the tree.

    Returns:
        Tuple[int, int]: The parent node and its index.
    """
    for i, e in enumerate(edge):
        if node in e:
            parent = e[0] if e[1] == node else e[1]
            if i==0:
                index = i
            else:
                index = i-1
            print('parent:', parent, 'i:', i, 'index', index)
            return parent, index
    return None, None

def is_control_node(node, node_index):
    # print('node:',node,'edge:',edge,'\n')
    if len(node) < 2 and node_index != 0:
        return True

    return False

def random_node(nodes, unused_nodes):
    node = np.random.choice(unused_nodes)
    node_index = nodes.index(node)

    return node, node_index

def get_new_node(nodes, used_nodes, operation):
    discard_nodes = ['r', 'f', 't']
    unused_nodes = [node for node in (set(nodes) - set(used_nodes)) if node not in discard_nodes]
    exp_nodes = list(set(explainerNames)) # in case of repeated explainers
    # exp_nodes = list(set(explainerNames) - set(nodes)) # in case of nonrepeated explainers
    # cont_nodes = list(set(control_nodes.flatten()) - set(nodes))
    # print('cont_nodes:', cont_nodes)

    if not unused_nodes:
        operation = 'insertion'
        new_node = random.choice(exp_nodes)
        node_index = None
        node = None  
        return operation, node_index, node, new_node

    if operation == 'replacement':
        unusednodes = [node for node in unused_nodes if node not in control_nodes]
        print('unusednodes:', unusednodes)
        if not unusednodes:
            operation = 'insertion'
            new_node = random.choice(exp_nodes)
            node_index = None
            node = None
            return operation, node_index, node, new_node
        else:
            node = np.random.choice(unusednodes)
            node_index = nodes.index(node)
            # node, node_index = random_node(nodes, unused_nodes)
            new_node = ''

            if node[0] == '/':
                # node is an explainer
                if len(exp_nodes) > 0:
                    new_node = np.random.choice(exp_nodes)
                else:
                    new_node = None
            # else:
            #     # node is a control node
            #     unused_control_nodes = set(control_nodes.flatten()) - used_nodes
            #     print('\nunused_control_nodes:', unused_control_nodes)
            #     if len(unused_control_nodes) > 0:
            #         new_node = np.random.choice(control_nodes.flatten())
            #         print("c - new_node",new_node)
            #     else:
            #         print("New node")
        print('\n rel operation:', operation, 'random node:',node, 'new_node', new_node, 'node_index', node_index)

    elif operation == 'deletion':
        print('\nInside deletion')
        discard_nodes = ['r', 'f', 't']
        unused_control_nodes = [node for node in nodes if not node.startswith('/') and node not in used_nodes and node not in discard_nodes]
        print('unused_control_nodes',unused_control_nodes)
        combined_nodes = unused_nodes + unused_control_nodes
        node = np.random.choice(combined_nodes)            
        node_index = nodes.index(node)
        new_node = ''

        if not unused_nodes:
            operation = 'insertion'
            operation, node_index, node, new_node = get_new_node(nodes, used_nodes, operation)   
            print('\noperation del1:', operation, 'random node:',node, 'new_node', new_node)
        elif not unused_control_nodes:
                operation = np.random.choice(['replacement','insertion'])
                operation, node_index, node, new_node = get_new_node(nodes, used_nodes, operation) 
                print('\noperation del2:', operation, 'random node:',node, 'new_node', new_node)
        elif len(unused_control_nodes) == 1:
                    operation = 'insertion'
                    operation, node_index, node, new_node = get_new_node(nodes, used_nodes, operation) 
                    print('\noperation del3:', operation, 'random node:',node, 'node_index', node_index)  
        else: 
            operation = 'deletion'
            print('\noperation del5:', operation, 'random node:',node, 'node_index', node_index)  

    elif operation == 'insertion':
        new_node = random.choice(exp_nodes)
        node_index = None
        node = None  
        print('\n ins operation:', operation, 'random node:',node, 'new_node', new_node)

    return operation, node_index, node, new_node


def get_replacement_node(nodes, control_nodes):
    print('control_nodes:', control_nodes)
    for i, control_set in enumerate(control_nodes):
        print('i', i, 'control_set:', control_set)
        if control_set[0] in nodes:
            print('control_set[0]:', control_set[0])
            if len(control_set) == 1:
                continue
            if len(control_set) > 1 and len(control_set) >= 2 and control_set[1] not in nodes:
                print('control_set[1]:', control_set[1])
                return control_set[1]
        elif control_set[1] in nodes:
            if control_set[0] not in nodes:
                print('control_set[0]:', control_set[0])
                return control_set[0]
        else:
            return control_set[0]
        print('control_nodes:', control_nodes[-1][1])
    return control_nodes[-1][1]

def choose_random_operation(random_bt_prime, edits):
    nodes = random_bt_prime['nodes']
    adj = random_bt_prime['adj']
    edge = ALtoEL(nodes,adj)
    num_nodes = len(nodes)
    global num_operations
    num_operations = 0

    # initialize the children list for each node
    node_list = [{'id': node, 'children': []} for node in nodes]
    print('\nchoose_random_operation - edit:', edits, 'random_bt_prime:', random_bt_prime, '\n')
    print('Len:', num_nodes, len(adj))
    operation = ''
    
    if num_nodes >= 2:
        if num_nodes <= 3:
            operation = 'insertion'
        else:
            operation = np.random.choice(['replacement', 'insertion', 'deletion'])
        operation, node_index, node, new_node = get_new_node(nodes, used_nodes, operation)
        print('\n choose_random_operation - operation:', operation, 'random node:',node, 'new_node', new_node, '\n')
        
        # if new_node not in nodes and new_node not in used_nodes and node not in used_nodes:
        if operation == 'replacement':
            # Perform the replacement
            if node[0] == '/':
                nodes[node_index] = new_node
                print('\nreplacement: e - nodes:',nodes, 'adj:', adj,'\n')
            else:
                if node in control_nodes:
                    replacement_node = get_replacement_node(nodes, control_nodes)
                    print('replacement_node', replacement_node)
                    nodes[nodes.index(node)] = replacement_node
                    edge = [(e[0] if e[0] != node else replacement_node, e[1] if e[1] != node else replacement_node) for e in edge]
                    random_bt_prime['edge'] = edge
                    print('\nc - nodes:',nodes, 'adj:',adj, 'edge:',edge,'\n')
            # Add node and new_node to the set of used nodes
            # used_nodes.add(node)
            # used_nodes.add(new_node)
            # print('used nodes', used_nodes)
        
        elif operation == 'deletion':  # Perform the deletion
            node = nodes[node_index]
            print('\n del - node:', node, 'node_index:', node_index)
            # node is not the root
            if node_index != 1:
                # Check whether it is a control node or explainer
                if is_control_node(node, edge):
                    print('node', node, adj)
                    parent_index = None
                    # parent_node, parent_index  = find_parent(node, edge)
                    for i, sublist in enumerate(adj):
                        # print('adj', node, adj, sublist, i)
                        if node_index in sublist:
                            parent_node = nodes[i]
                            parent_index = i
                    # To delete a control node, move its children to the parent node of that control node 
                    if parent_index is None:
                        choose_random_operation(random_bt_prime, edits)
                    else:
                        # Append the children of the node to the parent node's adjacency list
                        adj[parent_index].extend(adj[node_index])
                        # Remove the adjacency list at node_index
                        del adj[node_index]
                        # Update the adjacency list to remove references to the deleted node and adjust indices
                        adj = [[idx-1 if idx > node_index else idx for idx in sublist if idx != node_index] for sublist in adj]
                        # Remove the node from the lists 
                        nodes.pop(node_index)
                        random_bt_prime['adj'] = adj
                        # Add node and node_index to the set of used nodes
                        # used_nodes.add(node)
                    
                elif node[0] == '/':
                    # delete the node from the nodes list
                    del nodes[node_index]
                    # Remove the adjacency list at node_index
                    del adj[node_index]
                    adj = [[idx for idx in sublist if idx != node_index] for sublist in adj]
                    for i, neighbors in enumerate(adj):
                        updated_neighbors = [n - 1 if n > node_index else n for n in neighbors]
                        adj[i] = updated_neighbors
                    random_bt_prime['adj'] = adj
                    print('\nLen:', len(nodes),len(adj))
                    print('\ndel: explainer - nodes:',nodes, 'adj:',adj, '\n')
                    # Add node and node_index to the set of used nodes
                    # used_nodes.add(node)
                else:
                    choose_random_operation(random_bt_prime, edits)
            else:
                print("Can't delete root node. Choose another node.")
                choose_random_operation(random_bt_prime, edits)

        elif operation == 'insertion':
            # new node is an explainer, insert as leaf node
            if new_node[0] == '/':
                discard_nodes = ['r', 'f', 't']
                c_nodes = list(filter(lambda node: not node.startswith('/') and node not in discard_nodes, nodes))
                parent_node = np.random.choice(list(set(c_nodes)))
                if parent_node in nodes:
                    parent_index = nodes.index(parent_node)
                    print('\nnew_node:', new_node, 'parent_node:', parent_node, 'parent_index:', parent_index)
                    # find position to insert new node in node list
                    # for i in range(parent_index + 1, len(nodes)):
                    #     if nodes[i].startswith('/'):
                    parent_adjacency_list = adj[parent_index]
                    print("parent_adjacency_list", parent_adjacency_list)
                    # if parent_adjacency_list[0] is None:
                    #     choose_random_operation(random_bt_prime, edits)
                    if len(parent_adjacency_list) > 0:
                        first_node_index = parent_adjacency_list[0]
                        # print("first_node_index", first_node_index)
                        # first_node = nodes[first_node_index]
                        node_index_new = first_node_index
                        # node_index_new = i
                        nodes.insert(node_index_new, new_node)
                        print('new_node:', new_node, "first_node_index", first_node_index, 'node_index_new:', node_index_new)
                        for i, node_adj in enumerate(adj):
                            adj[i] = [idx + 1 if idx > node_index_new - 1 else idx for idx in node_adj]
                        # Add the new_node index in front of the first node in adj[parent_index]
                        adj[parent_index].insert(0, node_index_new)
                        # Insert an empty list at the node_index_new position in the adjacency list
                        adj.insert(node_index_new, [])
                        # break
                    else:
                        print("The parent_adjacency_list is empty.")
                        choose_random_operation(random_bt_prime, edits)
                else:
                    node_index_new = len(nodes)
                    nodes.append(new_node)
                    # append new empty adjacency list to adj
                    adj.append([])
                    # update parent node's adjacency list
                    adj[parent_index].append(node_index_new)
                    adj[node_index_new].append(parent_index)
                    edge.append((parent_node, new_node))
                    adj = ELtoAL(edge,nodes)
                    random_bt_prime['adj'] = adj
                    print('insertion 2: e - nodes:',nodes, 'adj:',adj, 'edge:',edge,'\n')
                    # Add node and node_index to the set of used nodes
                    # used_nodes.add(node_index_new)
                    # used_nodes.add(new_node)
            else:
                # new node is a control node, insert as root or parent node
                nodes.insert(0, new_node)
                adj.insert(0, [])
                for i in range(num_nodes-1):
                    for j, val in enumerate(adj[i]):
                        if val >= node_index:
                            adj[i][j] += 1
                edge.append((node, new_node))
                print('insertion 3: c - nodes:',nodes, 'adj:',adj, 'edge:',edge,'\n')
                # Add node and node_index to the set of used nodes
                # used_nodes.add(node_index)
                # used_nodes.add(new_node)  
    else:
        return random_bt_prime, edits, operation      
        
    print('Final - edits:',edits, 'random_bt_prime:', random_bt_prime)   
    print('*********************************************************\n')   

    end_time = time.time()
    computation_time = end_time - start_time
    print("\nComputation time:", computation_time, "seconds", ) 

    return random_bt_prime, edits, operation
