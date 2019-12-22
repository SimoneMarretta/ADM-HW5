import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import cycle
from heapq import *
import time

def dataframe_to_adj_list(df):
    ''' Function takes dataframe and returns adjacency list and dictionary of distances.
    'Distances' is very useful in finding distance with O(1) complexity
    '''
    adj = defaultdict(set)
    distances = defaultdict(dict)
    for index, row in df.iterrows():
        n1, n2, d = row['node_1'], row['node_2'], row['distance']
        adj[n1].add((n2, d))
        adj[n2].add((n1, d))
        distances[n1][n2] = d
        distances[n2][n1] = d
    return adj, distances

class MyGraph:
    def __init__(self, options):
        self.df = options['df']
        if 'adj' in options:
            self.g, self.dist = options['adj']
        else:
            self.g, self.dist = dataframe_to_adj_list(options['df'])
        
    def edge_distance(self, node_1, node_2):
        return self.dist[node_1][node_2] 
    
    def edges_by_nodes(self, nodes):
        ''' Returns edges in the graph between specified nodes 
        '''
        edges = []
        for n in nodes:
            edges += list(zip([n] * len(self.g[n]), [x for x,y in self.g[n]]))
            
        return edges
    
    def draw(self, edges = None, coordinates = None, options = {}):
        '''  Function takes edges represented as dictionary with colors as keys.
        It draws nodes by giving coordinates. 
        '''
        G = nx.Graph()
        
        if edges:
            nodes = set()
            for color in edges:
                for node_1, node_2 in edges[color]:                
                    G.add_edge(node_1, node_2)
                    nodes |= {node_1, node_2}
            
            nodes = list(nodes)
        else:
            edges = {'b':[]}
            nodes = self.g.keys()
            for index, row in self.df.iterrows():
                edges['b'].append((row['node_1'], row['node_2']))
                G.add_edge(row['node_1'], row['node_2'])
        
        labels = dict(zip(nodes, nodes))
        
        pos = nx.spring_layout(G) if not coordinates else coordinates
        
        plt.figure(3,figsize=(16,12)) 
        plt.axis('off')
        
        nx.draw_networkx_nodes(G,pos,
                       nodelist = nodes,
                       node_color = 'b',
                       node_size = options['node_size'] if 'node_size' in options else 5,
                       alpha = 0.5)
        
        width = {'r': 4, 'b': 4, 'g': 4, 'k' : 0.1, }
        for color in edges:
            nx.draw_networkx_edges(G,pos, edgelist=edges[color],edge_color=color, width=width[color],alpha=0.5)
        
        if options and 'labels_to_show' in options:
            labels = dict(zip(options['labels_to_show'], options['labels_to_show']))
            nx.draw_networkx_labels(G,pos, labels, font_color ='m', font_size=16)
            
        if options and 'label' in options and options['label']:
            nx.draw_networkx_labels(G,pos, labels, font_color ='m', font_size=16)

    
    def adj_to_edges(self, adj):
        '''  Converts adjacency list to list of edges 
        '''
        edges = set()
        for key in adj:
            edges |= set(zip([key] * len(adj[key]), list(adj[key])))
        return edges

    def shortest_path(self, node_1, nodes_to_visit, graph = None):
        ''' Realization of modified Dijkstra algorithm. 
        Here algorithm stops when all nodes that we want to visit are visited
        '''
        heap_of_dist = []
        heapify(heap_of_dist)
        
        distances = {}
        
        edges = [(node_1,-1)]
        visited = set()
        cur_node = node_1

        cur_dist = 0
        nv_counter = 0
        
        if not graph:
            graph = self.g
            
        while True:
            neighbours =  graph[cur_node]
            for node, weight in neighbours:   
                if node not in visited:
                    if node not in distances:
                        distances[node] = cur_dist + weight
                    else:
                        distances[node] = min(cur_dist + weight, distances[node])
                    
                    heappush(heap_of_dist, (distances[node], node, cur_node))
            
            visited.add(cur_node)
            
            if cur_node in nodes_to_visit:
                print(nv_counter, cur_node)
                nv_counter += 1
            
            if nv_counter == len(nodes_to_visit):
                break
            
            while heap_of_dist:
                weight, cur_node, came_from = heappop(heap_of_dist)
                if cur_node not in visited:
                    cur_dist = weight
                    edges.append((cur_node,came_from))
                    break
                        

        node, parent = edges[-1]
        edges = dict(edges)
        path = [node]
        while parent != -1:
            parent = edges[node]
            path.append(parent)
            node = parent

        path = path[:-1][::-1]
        return distances
        
    
    def spanning_tree(self, starting_node, nodes_to_visit = set()):
        ''' Returns spanning tree starting from specific node and include specific nodes.
        If nodes to visit are not indicated it will find the whole tree.
        '''
        visited = set()
        cur_node = starting_node
        heap_of_dist = []
        heapify(heap_of_dist)
        
        #  Leaves to be removed after getting spanning tree
        leaves = set()
        
        #  Distances  from starting node to nodes that in the list of visited
        distances = {}
        
        adj = defaultdict(set)
        parent = -1
        
        while True:
            leaves -= {parent}
            counter = 0
            for node, distance in self.g[cur_node]:
                if node not in visited:
                    counter +=1
                    #  Adding to heap neighbours of current node which are not visited
                    heappush(heap_of_dist, (distance, node, cur_node))
            
            #  Presumably a leave since no neighbours left to visit
            if counter == 0:
                leaves.add(cur_node)
            
            visited.add(cur_node)
            
            #  Condition for exiting in case when required nodes are visited
            if nodes_to_visit and (visited & nodes_to_visit) == nodes_to_visit:
                break
            
            prev_node = cur_node
            while heap_of_dist and cur_node in visited:
                #  Popping from the heap a node with the smallest edge
                distance, cur_node, parent = heappop(heap_of_dist)
            
            #  Condition for exiting in case of full spanning tree
            if not heap_of_dist:
                break
            
            if prev_node != parent:
                leaves.add(prev_node)
            
            adj[cur_node].add(parent)
            adj[parent].add(cur_node)
        
        leaves -= {starting_node} | nodes_to_visit

        if nodes_to_visit:
            #  Removing redundant branches
            for l in leaves:
                #print('cur leave = ', l)
                bag = {l} 

                #  While node has one neighbour it is NOT fork node.
                while len(adj[l] - bag) == 1:
                    new_l = list(adj[l] - bag)[0]
                    #print('new L = ', new_l)
                    if l in nodes_to_visit | {starting_node}:
                        break
                    #  Removing node with one neighbour
                    del adj[l]

                    #  We need to store 
                    bag = {l, new_l}
                    l = new_l

                adj[l] -= bag
        
        return adj
    
    def detour(self, starting_node, nodes_to_visit):
        ''' Function make spanning tree, after what it traverse it.
        '''
        tm = time.time()
        #  Getting spanning tree as adjacency list
        adj = self.spanning_tree(starting_node, nodes_to_visit)
        all_nodes = set(adj.keys())
        
        print("Spannig tree is found", time.time() - tm)

        #  Adding distances to tree
        for parent in adj:
            adj[parent] = {(node, self.edge_distance(parent, node))  for node in adj[parent]}
        
        print("Edge weights are added", time.time() - tm)
        
   
        forks = defaultdict(list)
        
        visited = set()
        
        leaves = set([parent for parent, edges in list(adj.items()) if len(edges) == 1 and parent != starting_node])
        while True:
            fork_data = []
            fork_nodes = set()
            for lv in leaves:
                #  If node is a leave, we start move from a leave to fork node 
                # and store a distance to and a path of such fork node.
                fd = self.go_to_fork(adj, lv, visited, fork_nodes, forks)
                fork_data.append(fd)

           
            leaves = set()
            for fork_node, fork_distance, path in fork_data:
                #  Removing edge from fork node so we can in the future consider it as a leave (len(edges) == 1)
                leaves.add(fork_node)
                if len(path) > 1:
                    forks[fork_node].append((fork_distance, path))
    
            #  Sort fork by distance so we traverse a tree by going to bracnhes with shirter distance first
            for fork_node in forks:
                forks[fork_node] = sorted(forks[fork_node])

            #  Break when only one branch is found - the main branch
            if starting_node in leaves:
                break
        
        print("Path is found", time.time() - tm)
        path = self.get_path(forks, starting_node, start = True)
        
        #  [ Crutch ]. Here a bug in get_path but there is no time to fix. Still it return correct result in the begging
        new_path = []
        for p in path:
            new_path.append(p)
            if len(all_nodes - set(new_path)) == 0:
                return new_path
       
    
    def get_path(self, forks, node, start = False):
        ''' Returns a route by list of forks. 
        '''
        path = []
        data = forks[node]
        
        #  Remove fork so we don't go twice
        del forks[node]
        
        for _, node_list in data:
            #  Since we have list of pathes starting from leaves we need to reverse them.
            node_list = node_list[::-1]
            
            if not start:
                #  In order to exclude double presence of fork node we copy list from the second element
                node_list = node_list[1:]
            
            local_path = []
            for next_node in node_list:
                local_path.append(next_node)
                if next_node in forks:
                    #  Go to recursion if node is fork node
                    local_path += self.get_path(forks, next_node)
                
            if not start:
            #  Since we need to go in branch and go out from branch, we reverse a path to the leave and 
            # concat them (also removing the leave from reversed path so it won't count twice) 
                local_path += local_path[::-1][1:]
            
            #  We add fork node since we exclude it before, but only if it's not starting node (we don't need to return)
            path += local_path + ([node] if not start else []) 

        return path
                
    def go_to_fork(self, adj, node, visited, fork_nodes, forks):
        ''' Traverse nodes from a leave to nearest fork and return path to fork from the leave (with distance).
        Return format: fork node, distance, whole path to fork node.
        '''
        distance = 0
        if node in forks:
            #  If node is fork we assign it the maximum distance of their branches. 
            #  Since forks sorted we need the last entry
            distance = forks[node][-1][0]
        path = [node]
        next_node = node
        
        while True:
            #  Neighbours that are not visited
            not_visited = self.not_visited(adj[next_node], visited)
            
            #  We need to go only if there is only one edge available, otherwise it's fork
            if len(not_visited) != 1:
                fork_nodes.add(next_node)
                break
                
            visited.add(next_node)
            next_node, edge_dist = not_visited[0] 
            distance += edge_dist
            path.append(next_node)
  
            if next_node in fork_nodes:
                break
            
        return next_node, distance, path
        
    
    def not_visited(self, edges, visited):
        ''' Return unvisited nodes of edges 
        '''
        res = []
        for node, distance in edges:
            if node not in visited:
                res.append((node, distance))
    
        return res
        
    def assign_distances(self, adj, node, visited=set(), res = {}):
        '''  Function assings compound distance for nodes of tree (works only for tree)
        '''
        res[node] = 0
        stack = [node]
        while stack:
            cur_node = stack.pop()
            if cur_node not in visited:
                #  Mark node as visited
                visited.add(cur_node)
                
                #  For each child we add compund distance 
                for next_node, distance in adj[cur_node]:
                    if next_node not in visited:
                        #  Adding distance of parent node (cur_node) to its children
                        res[next_node] = res[cur_node] + distance
                        stack.append(next_node)
        
        return res

    def route_edges(self, path):
        ''' Function return dictiony where keys are colors and values are edges.
        Moving element wise of path we can see if previous edge is equal to reverse 
        of current edge then it means that we reached a leave so we can change color to emphasize 
        moving bacward
        '''
        colors = cycle(['r','b', 'g'])
        cur_color = next(colors)
        edges = defaultdict(list)
        for i in range(len(path)-1):
            e = (path[i], path[i+1])
            if edges[cur_color] and set(e) == set(edges[cur_color][-1]):
                cur_color = next(colors)
            edges[cur_color].append(e)
            
        return edges
        
    def traverse(self):
        '''  Graph traversal by depth
        Returns disconnected parts of graph
        '''
        parts = defaultdict(set)
        visited = set()
        stack = []
        part_counter = -1
        
        for node in self.g:
            if node not in visited:
                part_counter += 1
                stack.append(node)
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        parts[part_counter].add(node)
                        visited.add(node)
                        
                        #  Add children to stack
                        stack += list(map(lambda x: x[0], self.g[node]))
                
                
        return parts
    
    def are_nodes_connected(self, parts, nodes):
        '''  Check if nodes are in the same connected subpart of graph 
        '''
        nodes = set(nodes)
        for i in parts:
            if len(nodes - parts[i]) == 0:
                return True
            
        return False
        
    def neighbours(self, node, depth, visited = set()):
        ''' Function returns neighbours of specific node up to level = depth
        '''
        if node not in visited:
            visited.add(node)
            if depth == 0:
                return
            
            for next_node, _ in self.g[node]:
                 self.neighbours(next_node, depth - 1, visited)

        