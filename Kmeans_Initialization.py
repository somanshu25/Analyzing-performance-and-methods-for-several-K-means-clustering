import pandas as pd
import numpy as np
import random
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time


# In[5]:


df = pd.read_csv("./yelp.csv")
df.head()


# In[6]:


pdDate = df['yelping_since'].head(100)
pdDate = pdDate.str.split('-')
pdDate = pdDate.str.join('')
df['elite'].unique()
df['elite'].str.count(',') +1


# In[7]:


df2 = df.copy()
#df2['votes'] = df2['funny'] + df2['cool'] + df2['useful']
df2 = df2.drop(columns = ['name','user_id'])
df2['yelping_since'] = df2['yelping_since'].str[0:4]
df2= df2.replace('None',-1)
df2['elite'] = df2['elite'].str.count(',')
df2 = df2.fillna(-1)
df2['elite'] = df2['elite'] + 1


# In[8]:


df3 = df2.copy()
col_max = df3.max()
col_min = df3.min()


# In[9]:



p = col_min.size




# In[10]:




# In[11]:


col_min = np.array(col_min,dtype = float)
col_max = np.array(col_max,dtype = float)
col_min = col_min.reshape(1,p)
col_max = col_max.reshape(1,p)


# In[12]:




# In[13]:


data_original = (df3.values).astype(float)
data_original.shape
data_original = np.divide((data_original - col_min),(col_max-col_min))
data_work = data_original.copy()


# In[14]:

def getKmeanppCentroid(kpp):
    np.random.seed(100000)
    data_size = data_work.shape[0]
    cent_rand = random.randint(0,data_original.shape[0])
    centroid1 = np.random.rand(kpp,p)
    centroidDist = np.zeros((kpp-1, data_work.shape[0]))
    centroid1[0] = data_work[cent_rand]
    sum_norm = 0
    min_distance = 99999*(np.ones((data_size,1),dtype=float))
    min_clusture = -1*(np.ones((data_size,1),dtype=float))
    cumulative = np.zeros((data_size,1),dtype=float)
    for count in range(kpp-1):  
        sum_norm = 0
        #for i in range(data_size):
        #dist = LA.norm(data_work - centroid[count])
    #     smallData = data_work
    #     smallData_reshape = smallData.reshape(smallData.shape[0], 1, smallData.shape[1])
    #     centroidNew = centroid[:count+1].reshape(count+1,centroid.shape[1])
    #     print(centroidNew.shape)
    #     smallData_tile = np.tile(smallData_reshape, (1,centroidNew.shape[0],1))
    #     centroid_mat_reshape = centroidNew.reshape(1, centroidNew.shape[0], centroidNew.shape[1])
    #     compareDC = smallData_tile - centroid_mat_reshape
    #     compareDC = np.sum(np.square(compareDC), axis = 2)
    #     compareDC_index = np.argmin(compareDC, axis = 1)
    #     alloted_centroid = compareDC_index
        centroidNew = centroid1[count].reshape(1, centroid1.shape[1])
        centroidDist[count] = np.sqrt(np.sum(np.square(data_work - centroidNew),axis = 1))
        min_distance = np.min(centroidDist[:count+1], axis = 0)
        #print(min_distance)

        #min_distance = np.sqrt(np.sum(np.square((smallData - centroid_mat[alloted_centroid])), axis = 1))
    #     if min_distance[i] > dist:
    #         min_distance[i] = dist
    #         min_clusture[i] = count
    #     sum_norm = sum_norm + min_distance[i]
        sum_norm = np.sum(min_distance)
        #print('sum norm',sum_norm)
        cumulative = np.cumsum(min_distance)

        min_distance = min_distance/sum_norm
        cumulative = cumulative/sum_norm
        number = np.random.uniform(0,1)
        e
        centroid1[count+1] = data_work[next_centroid]
        #print("Itreation:",count+1)
        #print("Random number:",number)
        #print("New Centroid formed",next_centroid)
        #print('time',time2
    return centroid1


# In[34]:





# In[13]:


# In[63]:




# In[15]:
def MinMaxLoss(centroidU, Data):
    allotedCentroid = np.zeros(Data.shape[0])
    lossK = np.zeros(Data.shape[0])
    
    #print(Data.shape[0])
    for i in range(Data.shape[0]):
        DataI = Data[i].reshape(1, Data.shape[1])
        minDist = np.sqrt(np.sum(np.square(DataI - centroidU),axis=1))
        lossK[i] = np.min(minDist)
        compareDC_index = np.argmin(minDist)
        allotedCentroid[i] = compareDC_index
        #print(allotedCentroid)
    A1 = np.unique(allotedCentroid)
    minOverCentroid = np.zeros(len(A1))
    maxOverCentroid = np.zeros(len(A1))
    print('Index of the Centroids', A1)
    k = 0
    for j in A1:
        index = np.argwhere(allotedCentroid == j)
        DataPerCentroid = Data[index]
        DistK = np.sqrt(np.sum(np.square(DataPerCentroid - centroidU[int(j)]),axis=1))
        minOverCentroid[k] = np.min(DistK)
        maxOverCentroid[k] = np.max(DistK)
        k= k+1
    return minOverCentroid, maxOverCentroid, np.mean(lossK)





K = [5,25,50,100,150,250,400,500]
#K = [5,25,50,100,400]

#K = 10
B = 500
size = len(K)
p = col_min.size
T = 25
index = 0
loss_final = np.empty((size,1),dtype=float)
count_iter = 0
min_dist = np.empty((size,1),dtype=float)
max_dist = np.empty((size,1),dtype=float)
time_start = time.time()
for k in K:
    loss = np.zeros((T,1),dtype=float)
    np.random.seed(100000)
    max_distance = -9999999
    min_distance = 9999999
    time1 = time.time()
    centroid_mat = getKmeanppCentroid(k)
    distance = np.empty(k)
    alloted_centroid = np.empty(df3.shape[0])
    
    for t in range(1,T+1):
        eta = 1/(t)
        index = np.random.randint(0,data_work.shape[0],B)
        #for j in range(index,index+B):
        smallData = data_work[index]
        #print(smallData.shape)
        smallData_reshape = smallData.reshape(smallData.shape[0], 1, smallData.shape[1])
        
        smallData_tile = np.tile(smallData_reshape, (1,centroid_mat.shape[0],1))
        centroid_mat_reshape = centroid_mat.reshape(1, centroid_mat.shape[0], centroid_mat.shape[1])
        compareDC = smallData_tile - centroid_mat_reshape
        compareDC = np.sum(np.square(compareDC), axis = 2)
        compareDC_index = np.argmin(compareDC, axis = 1)
        alloted_centroid = compareDC_index
        #loss[t-1] = loss[t-1] + LA.norm(smallData - centroid_mat[alloted_centroid]) 
        A = np.unique(alloted_centroid,return_counts = True)
        loss[t-1] = np.mean(np.sqrt(np.sum(np.square(smallData - centroid_mat[alloted_centroid]),axis=1)))
        #print(A)
        for l in range(len(A[0])):
            update = A[0][l]
            count = A[1][l]
            index_temp = np.where(alloted_centroid == update)
            values_new = data_work[index_temp[0]]
            gradient_value = np.sum(values_new - centroid_mat[update.astype(int)],axis=0)/count
            centroid_mat[update.astype(int)] = centroid_mat[update.astype(int)] + eta*(gradient_value)
        val_check = np.sqrt(np.sum(np.square(smallData - centroid_mat[alloted_centroid]),axis=1))
        #index = index + B

        if min_distance > np.min(val_check):
            min_distance = np.min(val_check)
        if max_distance < np.max(val_check):
            max_distance = np.max(val_check)
        #loss[t-1] = np.mean(np.sqrt(np.sum(np.square(smallData - centroid_mat[alloted_centroid]),axis=1)))
    minVal, maxVal, lossC = MinMaxLoss(centroid_mat, data_work)


    min_dist[count_iter]= np.mean(minVal)
    max_dist[count_iter]= np.mean(maxVal)
    loss_final[count_iter] = lossC
    count_iter = count_iter+1
    print("Iteration:",count_iter)
    print("Value of K:",k)
    print("Time taken:",time.time()-time1)

iterations_count = K
print("Final Time:",time.time()-time_start)

plt.xlabel("For Each Cluster")
plt.ylabel("Mean Distance of the points from Clusture Centroids");
plt.title("K-means ++: Loss vs Iterations")
plt.plot(iterations_count,min_dist,'g--',label="Mean of Min Distance")
plt.plot(iterations_count,max_dist,'r--',label="Mean of Max Distance")
plt.plot(iterations_count,loss_final,'b--',label="Mean of Loss")
plt.legend()

plt.show()

