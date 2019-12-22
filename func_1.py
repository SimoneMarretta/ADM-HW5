#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


#input
v=int(input('insert node '))
function=int(input('insert the distances function you want to use(1 for t(x,y),2 for d(x,y) and 3 for network distance '))
d=int(input('insert distance threshold '))
#open files according to the selected function
co=input('Insert path of the coordinates file: ')
dis=input('Insert path of the distance nodes file: ')
tim=input('Insert path of the time file: ')
with open(co,"r") as f:
    data1=f.read()
    data1=data1.split("\nv ")
    co_data=data1[1:]
for i in range(len(co_data)):
    co_data[i]=co_data[i].split(" ")

if function==1:
    with open(dis,"r") as f:
        data2=f.read()
        data2=data2.split("\n")
        distance_data=data2[7:]

    for i in range(len(distance_data)):
        distance_data[i]=distance_data[i].split(" ")[1:]
        distance_data[i]=[int(x) for x in distance_data[i]]
if function==2:
    with open(tim,"r") as f:
        data3=f.read()
        data3=data3.split("\n")
        distance_data=data3[7:]
    for i in range(len(distance_data)):
        distance_data[i]=distance_data[i].split(" ")[1:]
        distance_data[i]=[int(x) for x in distance_data[i]]

if function==3:
    with open(dis,"r") as f:
        data2=f.read()
        data2=data2.split("\n")
        distance_data=data2[7:]

    for i in range(len(distance_data)):
        distance_data[i]=distance_data[i].split(" ")[1:4]
        distance_data[i].append("1")
        distance_data[i]=[int(x) for x in distance_data[i]]
  


# In[ ]:


#converting data to dataframe and using it to creat a networkx graph
df=pd.DataFrame(distance_data, columns=['source','target', 'weight'])
graph=nx.Graph(df)


# In[ ]:


nodes=list(range(0,df.shape[0]))


# In[ ]:


def func_1(v,d):
    # it takes v and d as requried node and its threshold
    exist = [False]*(len(nodes)+1) 
    done = set() 
    distance = [0]*(len(nodes)+1) 
    q = list()
    q.append(v)
    exist[v] = True
    #finding neighbors
    while q:
        a = q.pop()
        neigh=[]
        for k in graph.neighbors(a):
            neigh.append(int(k))
        for i in neigh:
            if exist[i] == False:
                t = int(graph.edges[(a,i)]['weight']) + distance[a]
                if t <= d:
                    q.append(i)
                    exist[i]=True
                    done.add(i)
                    distance[i]=t
    return done


# In[ ]:


#visualization
l=func_1(v,d)
pos=[]
for i in l:
    pos.append((int(co_data[i][1]),int(co_data[i][2])))
pos1=dict()
for i in range(len(list(l))):
    pos1[list(l)[i]]=pos[i]
    posinitalnode=dict()
posinitalnode[v]=((int(co_data[v][1]),int(co_data[v][2])))
pos=np.array(pos).reshape(-1,2)

l=list(l)
#node color
nx.draw_networkx_nodes(graph.subgraph(v),pos=posinitalnode,
                       nodelist=[v],
                       node_color='r',
                       node_size=5,
                      )
nx.draw_networkx_nodes(graph.subgraph(l),pos=pos1,
                       nodelist=l,
                       node_color='b',
                       node_size=5,
                       )

#edge color
l.append(v)
pos1[v]=((int(co_data[v][1]),int(co_data[v][2])))

if function==1:
        nx.draw_networkx_edges(graph.subgraph(l),pos=pos1,width=1.0,alpha=0.5,edge_color='k')
if function==2:
        nx.draw_networkx_edges(graph.subgraph(l),pos=pos1,width=1.0,alpha=0.5,edge_color='y')
if function==3:
        nx.draw_networkx_edges(graph.subgraph(l),pos=pos1,width=1.0,alpha=0.5,edge_color='g')

