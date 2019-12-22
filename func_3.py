# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 20:24:32 2019

@author: simo2
"""
import gzip
import pandas as pd
import math
from sys import stdin, stdout
import heapq
import networkx as nx

with gzip.open(input('Insert path of the coordinates file: ')) as f:

    features_train = pd.read_csv(f)

features_train=features_train[6:]
features_train
with gzip.open(input('Insert path of the distance nodes file: ')) as f:

    features_train2 = pd.read_csv(f)

features_train2=features_train2[6:]
features_train2['c 9th DIMACS Implementation Challenge: Shortest Paths']
with gzip.open(input('Insert path of the nodes travel times file: ')) as f:

    features_train3 = pd.read_csv(f)

features_train3=features_train3[6:]
features_train3['c 9th DIMACS Implementation Challenge: Shortest Paths']

coordinates=features_train['c 9th DIMACS Implementation Challenge: Shortest Paths'].to_list()
distance=features_train2['c 9th DIMACS Implementation Challenge: Shortest Paths'].to_list()
travel_time=features_train3['c 9th DIMACS Implementation Challenge: Shortest Paths'].to_list()
def dijkstra(adj, source, target):
    INF = ((1<<63) - 1)//2
    pred = { x:x for x in adj } 
    dist = { x:INF for x in adj }#We set all the distance to infinite
    dist[source] = 0 #Except the source's distance that is obviously zero
    PQ = []
    heapq.heappush(PQ, [dist[source], source])#we start a priority queue and we insert the tuple (dist[source], source) from the source node

    while(PQ):
        u = heapq.heappop(PQ)  # u is a tuple [u_dist, u_id] #try to fin the nearest node
        u_dist = u[0]
        u_id = u[1]
        if u_dist == dist[u_id]:
            if u_id == target:#We stop the algorithm if we have found the shortest path for the target
                break
            for v in adj[u_id]:
               v_id = v[0]
               w_uv = v[1]
               if dist[u_id] +  w_uv < dist[v_id]:#We update distances
                   dist[v_id] = dist[u_id] + w_uv
                   heapq.heappush(PQ, [dist[v_id], v_id])
                   pred[v_id] = u_id

    if dist[target]==INF:#Have we found a path?
        stdout.write("There is no path between " + source + " and " + target)
    # We go backwards from the target to the source following the shortest path
    else:
        st = []
        node = target
        while(True):
            st.append(str(node))
            if(node==pred[node]):
                break
            node = pred[node]
        path = st[::-1]
        #stdout.write("The shortest path is: " + " ".join(path))
        return path
#graph implementation node -> [ (neighbor 1, weight 1), (neighbor 2, weight 2), … , (neighbor m, weight m) ]   
node=input('insert node ')
nodes_seq=list(input('Insert node sequence ').strip().split())
distance_function=int(input('insert the distances function you want to use(1 for t(x,y),2 for d(x,y) and 3 for network distance '))
nodes_seq.insert(0, node)
final_path=[]
if distance_function==1:
    graph = {}
    for i in range(len(travel_time)):
        integer=int(travel_time[i].split()[3])
        if travel_time[i].split()[1] not in graph:

            graph[travel_time[i].split()[1]]=[]
            graph[travel_time[i].split()[1]].append((travel_time[i].split()[2],integer))
        else:
            graph[travel_time[i].split()[1]].append((travel_time[i].split()[2],integer))
if distance_function==2:
    graph = {}
    for i in range(len(distance)):
        integer=int(distance[i].split()[3])
        if distance[i].split()[1] not in graph:

            graph[distance[i].split()[1]]=[]
            graph[distance[i].split()[1]].append((distance[i].split()[2],integer))
        else:
            graph[distance[i].split()[1]].append((distance[i].split()[2],integer))
if distance_function==3:
    graph = {}
    for i in range(len(distance)):
        integer=int(distance[i].split()[3])
        if distance[i].split()[1] not in graph:

            graph[distance[i].split()[1]]=[]
            graph[distance[i].split()[1]].append((distance[i].split()[2],1))
        else:
            graph[distance[i].split()[1]].append((distance[i].split()[2],1))            

for n in range(len(nodes_seq)-1):#We calculate the final path
    path=dijkstra(graph, nodes_seq[n], nodes_seq[n+1])
    for i in range(len(path)):
        final_path.append(path[i])
g = nx.Graph()
for i in range(1,len(graph)):
    for j in range(len(graph[str(i)])):
            g.add_edge(str(i), graph[str(i)][j][0], weight=graph[str(i)][j][1])
#Creating a subgraph with only the shortest path
h = g.subgraph(final_path)
nx.draw_networkx(h, with_labels=True,node_color='r')            