#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 15:21:16 2017

@author: uba
"""

from sklearn import preprocessing
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
import pandas as pd

data = np.loadtxt('data.csv',delimiter=',',usecols=(1,2,3,4,5),skiprows=1)

#print(data)

data_scaled = preprocessing.scale(data)

#print(data_scaled)

Z = linkage(data_scaled, 'ward')
#c, coph_dists = cophenet(Z, pdist(data_scaled))
#print(c)

#print(Z[:100])

'''plt.figure(figsize=(50,10))
plt.title('Hierarchical Clustering Dendogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(Z,leaf_rotation=90.,leaf_font_size=8.,show_contracted=True)
plt.show()'''
#plt.savefig('dendrogram.png')


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

'''last = Z[-10:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev)

acceleration = np.diff(last, 2)  # 2nd derivative of the distances
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev)
plt.show()
k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
print("clusters:", k)'''

fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=50,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,
    max_d=16,
)
plt.show()

'''clusters = fcluster(Z, 0.1, criterion='distance')
#print(type(clusters))
#for each in clusters:
#    print(each)
df = pd.DataFrame(clusters)
df.columns = ["cluster"]
print(df["cluster"].value_counts())

members = df[df['cluster'] == 2416].index.tolist()
print(df[df['cluster'] == 2416].index.tolist())'''

'''data = pd.read_csv('/home/uba/Downloads/logs/wac_1AC1_20160422_0082.log',delimiter=" ",skiprows=1,header=None)
data.columns = ["timestamp","time_taken","c_ip","filesize","s_ip","s_port","sc_status",
                "sc_bytes","cs_method","cs_uri_stem","_","rs_duration","rs_bytes",
                "c_referrer","c_user_agent","customer_id","x_ec_custom_1","None"]

for m in members:
    print(data[['c_ip','s_ip','cs_uri_stem','c_user_agent','x_ec_custom_1']].loc[m])'''