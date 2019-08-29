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

## 3. Implement K-means initilalization using Monte Carlo (MCMC) algorithm

Here, we work on to mitigate the issue of high runtime required for intializing the centroids using probability distribution using Markov chain approach. Hence, the probabiltiy of each state is dependent of the previous state and the current hidden state. We used one of the version of Metropolis-Hastings algorithm.

The algorithm:

● Initialize a centroid randomly which lies on the data at random
● Compute the probability distribution of all the data points with reference to the assigned centroid.
● Pick a point at random from the data set to start the markov chain. This point will act as the initial state in our sequence of points
● Sample the next state form the dataset based on the probability distribution. Compute the transitional probabilities of staying in the same state and moving to the next state using the accepting rule, which is the difference between the two distances, i.e., 
                
                dist (xj+1- x j) = D(xj+1 , C) – D(xj,C)
                
where D(x, C) is the distance square of the point x from the known centroid set C.
● Now for condition for moving to the next state:
    ○ If the difference comes out to be positive, move to the desired state as the distanceof the next state is greater than the previous state.
    ○ Or else, move to the desired state with the transitional probability:
                
                                p(xj+1/xj)= exp(dist(xj+1-xj)).
                
 We are mapping the difference of the distance to this function as our differences between the two states could be very highly negative, which implies that the probability of going to the new state is very less. We need a function which maps input of -infinity to value of y as 0 and the value of y to 1 when the value of the x is 0. 
 
Overall, we observe that the performance of initialization with MCMC gives the better results.
