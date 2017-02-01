# probabilistic-user-tracking
## Agglomerative Hierarchical Clustering and Pairwise similarity
The approach taken here is to group the whole dataset into clusters first where the candidates of data from the same user are grouped into the same cluster. Agglomerative hierarchical clustering fits our use case because it does not require to specify the number of clusters to group the data into beforehand. Then a pairwise similarity is computed for all possible pairs inside each cluster. The data is grouped into clusters first because to compute the pairwise similarity for each possible pair from the bigger data set can be very computationally expensive. After computing the similarity score a graph is formed where the nodes are represented by each row from the dataset. Then we form edges to connect the nodes which have a high similarity score. Then all the possible unique users can be retrieved from the graph as connected subgraphs where each subgraph represents a unique user.  <br>
## Preprocessing of data
Out of all the features from the dataset, the following were used as they reveal information about a user the most:
* client ip
* server ip
* cs-uri-stem
* c-user-agent
* x-ec_custom-1 (site user had visited)

All these features were first converted into numericals values. This was done by first finding all the unique values from the above columns and then saving them in a list. Then the following fields were created:
* cip_id from client ip
* sip_id from server ip
* uri_id from cs-uri-stem
* user_agent_id from c-user-agent
* site_id from x-ec_custom-1

Values were inserted into the new columns as the index of the particular value from the respective columns from which they were created. The indexes are taken from the lists with the unique values from the respective columns.<br>
A new column named “device_id” is also created. Firstly, all the unique pairs of cip_id and user_agent_id are retrieved and stored in a list. Then values are inserted into the “device_id” column as the index of the respective cip_id and user_agent_id pairs.<br>
So the following features are used for further processing:<br>
* cip_id
* sip_id
* uri_id
* user_agent_id
* site_id
* device_id

## Clustering algorithm
The data is first scaled to a similar range to feed into the clustering algorithm. Agglomerative hierarchical clustering is used which groups the data into a hierarchy of clusters using a certain distance metric. The following link contains a tutorial for the algorithm: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/<br>
The scipy library contains the hierarchical clustering algorithm as described in the tutorial. 
The algorithm groups data into clusters which contain a single point up to a big cluster which contains all the data points.  
![alt text](https://github.com/sukrit-uba/probabilistic-user-tracking/blob/master/dendrogram2.png "Logo Title Text 1")

As seen in the above dendrogram a hierarchy of clusters is created. In the dendrogram horizontal lines are cluster merges, vertical lines tell us which clusters/labels were part of merge forming that new cluster and heights of the horizontal lines tell us about the distance that needed to be "bridged" to form the new cluster. The dendrogram can be truncated at a certain point like in the figure using the black horizontal line. The line is plotted perpendicular to the y-axis at the point approx (0,16). Using this concept we can retrieve the clusters above the line and perform pairwise similarity inside those clusters.

## Pairwise similarity
After performing clustering we get clusters which gives us a smaller search space for performing pairwise similarity. Pairwise similarity can be calculated using the following functions:<br>
```
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
    return ((np.array(score)*np.array(weights)).sum())/8.0
```
The function similarity_score accepts a pair of users, then using the function match we check if their features are the same returning 1 if they match and returning 0 otherwise. Then the list score is multiplied by a list of weights which represent the importance of each attribute in defining the similarity score.<br>
For example cip_id is quite important in determining a unique user so we assign it a higher weight. And attributes such as sip_id and uri_id contribute less in determining a unique user, so we assign it a lower weight. Then the scores are summed together and divided by the max possible score to get the normalized similarity score.<br>

## Constructing a graph and finding unique users
After finding all the pairwise similarity scores inside of all the clusters, we further construct a graph where the nodes in the graph are all the data points(rows) from our dataset. Then we connect the nodes with high similarity scores. After that we can retrieve all the connected sub graphs as unique users. 

In the image above we can see that the 3 nodes at the top are connected to each other because they have a high value of similarity score, so the connected subgraph can be considered as a single unique user. 
But the nodes at the bottom are not connected to any other nodes as they are not similar to any other nodes, so they are considered as 2 different unique users. 
