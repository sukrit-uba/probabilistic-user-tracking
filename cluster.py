#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 12:40:25 2017

@author: uba
"""

from sklearn import preprocessing
import numpy as np
from matplotlib import pyplot as plt 
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
import pandas as pd
import networkx as nx

data = np.loadtxt('data2.csv',delimiter=',',usecols=(1,2,3,4,5,6),skiprows=1)
#print(data)
data_scaled = preprocessing.scale(data)
#print(data_scaled)
Z = linkage(data_scaled, 'ward')
#print(data[0:10])
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        #plt.figure(figsize=(500,100))
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata
    
'''plt.figure(figsize=(50,10))
plt.title('Hierarchical Clustering Dendogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(Z,leaf_rotation=90.,leaf_font_size=8.,show_contracted=True)
plt.show()'''
    
'''fancy_dendrogram(
    Z,
    #truncate_mode='lastp',
    #p=50,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,
    max_d=16,
)
plt.show()'''

clusters = fcluster(Z, 75, criterion='distance')
#print("Number of clusters: "+str(len(clusters)))
df = pd.DataFrame(clusters)
df.columns = ["cluster"]
print("Number of clusters: ")
print(df["cluster"].unique())

#members = df[df["cluster"]==1318].index.tolist()
#print(members)

clusters_and_members = {}
for each in df['cluster'].unique():
    clusters_and_members[each] = df[df['cluster']==each].index.tolist()

def match(a,b):
    if a == b:
        return 1
    else:
        return 0
        
def similarity_score(x,y):
    score = [None]*len(x)
    for i in range(len(x)):
        score[i] = match(x[i],y[i])
    """cip_id,sip_id,uri_id,user_agent_id,site_id,device_id"""    
    weights = [2,0.5,0.5,2,1,2]
    #weights = [8,0,0,0,0,0]  
    return ((np.array(score)*np.array(weights)).sum())/8.0
  
#print(clusters_and_members)
results = {}

for key,value in clusters_and_members.items():
    #print(key,value)
    for i in range(len(value)):
        for j in range(i+1,len(value)):
            if value[i] != value[j]:
                results[value[i],value[j]]=similarity_score(data[value[i]],data[value[j]])
#            print(value[i])
#            print(value[j])
#            print(data[value[i]])
#            print(data[value[j]])
#            print(similarity_score(data[value[i]],data[value[j]]))
#            print("-------------------------------")

'''for i in range(9687):
    for j in range(i+1,9687):
        if i != j:
            results[i,j]=similarity_score(data[i],data[j])'''
            
G = nx.Graph()
G.add_nodes_from(range(9687))
vals = []    
for key,value in results.items():
#    print(value)
#    if value not in vals:
#        vals.append(value)
    if value == 1.0:
        #print(key)
        G.add_edge(key[0],key[1])

#unique similarity scores        
'''[0.0625, 0.0, 0.125, 0.3125, 0.25, 0.375, 1.0, 0.9375, 
    0.875, 0.5, 0.1875, 0.8125, 0.4375, 0.75]'''        
#print(vals)'''
     
users = list(nx.connected_components(G))
print(users)
print("Number of users found: "+str(len(users)))
