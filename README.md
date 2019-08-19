# Analyzing-performance-and-methods-for-several-K-means-clustering
Compare the runtime and efficiency of Llyod’s, K-means++ and Markov chain (MCMC) methods for  K-means  seeding  and  observe  the  accuracy  relative  to  the  global  minima. The Yelp user database is considered for the comparision of these 3 algorithms.

## 1. Implement online version of the k-means clustering algorithm

Using Llyod's algorithm, I used online version of K-means clustering algorithm to cluster the datapoints with number of clusters being 10 and varying the batch sizes for online gradient descent update rule to update the centroids which could cluster the datapoints with minimum mean distance.

![Image1_Q2_Batch_Size](https://user-images.githubusercontent.com/43916672/63243154-51284080-c277-11e9-97c8-2dc2569fbe2d.png)

Here, we see that the batch size of 500 gives minimum loss with less deviation. Hence, we chose batch size of 500. The project further continues to work on the limitation of initial seeding issue with random initialization before the update rules works, hence, gives rise to local minima problem.

## 2. Implement the k-means++ initial centroid initialization

Here, we initialize the centroid using sampling from probability distribution. The probability distribution for datapoints is proportionaly based on the distance of the datapoint from already chosen centroid points. Hence, we ssample those points with high probability which are far from the already chosen centroids.

![Image_Q3_Batch_vs_Iterations](https://user-images.githubusercontent.com/43916672/63243472-4326ef80-c278-11e9-9878-c12bfbfa3775.png)

We observe that the loss in intial iterations are very less, hence, greater chance of getting the global minimal solution. The batch size of 500 gives close to ideal results.

## 3. 
