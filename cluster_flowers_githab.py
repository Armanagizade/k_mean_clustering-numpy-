#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


pd =  pd.read_csv("IRIS.csv")
m = np.shape(pd)[0]
x1 = pd["petal_length"]
x2 = pd["petal_width"]
result = pd["species"]
data = np.zeros((m,2))
data[:,0] = x1
data[:,1] = x2

max_iter = 400
number_of_centroids = 3
centroids = np.zeros((number_of_centroids , 2))
all_indexc = np.zeros((max_iter,m))
all_centroids = []
cost_function = np.zeros((max_iter,m))
all_cost_function = np.zeros((max_iter))
for i in range(max_iter): 
    for j in range(number_of_centroids):
        centroids[j,:] = data[np.random.randint(0,m),:]
    for n in range(m):
        distance = []
        for k in range(number_of_centroids):
            normij = np.linalg.norm(data[n,:] - centroids[k])
            distance.append(normij)
            
        
        all_indexc[i,n] = np.argmin(distance)
        cost_function[i,n] = np.min(distance)
    for k in range(number_of_centroids):
        centroids[k] = np.mean(data[all_indexc[i,:]==k,:],axis=0)  
    all_centroids.append(centroids)
    all_cost_function[i] = float(1/m) * np.sum(cost_function[i,:],axis=0)
best_indexc = all_indexc[np.argmin(all_cost_function),:] 
print(f"J = {np.min(all_cost_function)}")
bestsol = all_centroids[np.argmin(all_cost_function)]  
print(f"best centriods =\n {bestsol}")
fig,axes=plt.subplots(1,2,figsize=(15,5))

axes[1].scatter(data[best_indexc == np.argsort(bestsol,axis=0)[0][1],0],data[best_indexc == np.argsort(bestsol,axis=0)[0][1],1],color='r')
axes[1].scatter(data[best_indexc == np.argsort(bestsol,axis=0)[1][1],0],data[best_indexc == np.argsort(bestsol,axis=0)[1][1],1],color='g')
axes[1].scatter(data[best_indexc == np.argsort(bestsol,axis=0)[2][1],0],data[best_indexc == np.argsort(bestsol,axis=0)[2][1],1],color='b')
axes[1].set(xlabel='petal_length',ylabel='petal_width',title='k_mean')
axes[0].scatter(data[result=='Iris-setosa'][:,0],data[result=='Iris-setosa'][:,1],color='r')
axes[0].scatter(data[result=='Iris-versicolor'][:,0],data[result=='Iris-versicolor'][:,1],color='g')
axes[0].scatter(data[result=='Iris-virginica'][:,0],data[result=='Iris-virginica'][:,1],color='b')
axes[0].set(xlabel='petal_length',ylabel='petal_width',title='data set')
# %%
