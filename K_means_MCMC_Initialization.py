import pandas as pd
import numpy as np
import random
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time

import math
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



def getMarkovCentroids(kM):
    np.random.seed(100000)
    q = np.zeros(kM)
    centroid = np.zeros((kM, data_work.shape[1]))

    data_size = data_work.shape[0]

    mixingTime = 1 + (math.log(4*kM/0.1)*(8/0.1))
    cent_rand = random.randint(0,data_original.shape[0])
    centroidInitial = data_work[cent_rand]
    centroid[0] = data_work[cent_rand]
    pX = np.sum(np.square(data_work - centroidInitial),axis = 1)
    pX = 1.0/(2*df3.shape[0]) + 1.0/2*(pX/np.sum(pX))
    cumulative = np.cumsum(pX)
    #print(centroid[0])

    #print(mixingTime)
    for i in range(kM-1):

        initialX_index = random.randint(0,data_original.shape[0])
        InitialX = data_work[initialX_index]

        for j in range(int(mixingTime)):
            DXqMinus1 = np.min(np.sum(np.square(InitialX.reshape((1,data_work.shape[1])) - centroid[i]),axis = 1), axis = 0)
            number = np.random.uniform(0,1)
            yjIndex = np.argmax(cumulative >= number)
            #('yjIndex', yjIndex)
            yj = data_work[yjIndex]
            qyj = pX[yjIndex]
            dyjC = np.min(np.sum(np.square(yj.reshape((1,data_work.shape[1])) - centroid[i]),axis = 1), axis = 0)
            qjXMinus1 = pX[initialX_index]
            #if qyj == 0 or DXqMinus1 == 0:
             #   pXtoY = 0
            #else:
                #pXtoY = min(1, (dyjC/DXqMinus1)*(qjXMinus1/qyj))
            dist = dyjC - DXqMinus1
            pXtoY = np.sign(dist)
            #print('dyjc',dyjC, DXqMinus1, dyjC - DXqMinus1)
            if pXtoY == 1:
                node = yj
            else:
                flip = np.random.uniform(0,1)
                if flip < np.exp(dist):
                #if flip > pXtoY:
                    node = yj
                else:
                    node = InitialX
            InitialX = node
        centroid[i+1] = node.flatten()
    return centroid

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

    centroid_mat = getMarkovCentroids(k)
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
plt.title("Markov: Loss vs Iterations")
plt.plot(iterations_count,min_dist,'g--',label="Mean of Min Distance")
plt.plot(iterations_count,max_dist,'r--',label="Mean of Max Distance")
plt.plot(iterations_count,loss_final,'b--',label="Mean of Loss")
plt.legend()

plt.show()

