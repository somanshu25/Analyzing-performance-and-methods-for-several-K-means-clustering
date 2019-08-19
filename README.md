# Analyzing-performance-and-methods-for-several-K-means-clustering
Compare the runtime and efficiency of Llyod’s, K-means++ and Markov chain (MCMC) methods for  K-means  seeding  and  observe  the  accuracy  relative  to  the  global  minima. The Yelp user database is considered for the comparision of these 3 algorithms.

## 1. Implement online version of the k-means clustering algorithm

Using Llyod's algorithm, I used online version of K-means clustering algorithm to cluster the datapoints with number of clusters being 10 and varying the batch sizes for online gradient descent update rule to update the centroids which could cluster the datapoints with minimum mean distance.
