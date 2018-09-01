# Recommender System for Cargo O-D cities
This is a Collaborative Filtering model that recommends Origin-Destination city routes for cargo drivers. Built for Northwestern University's Transportation Lab.

## Algorithm
Collaborative Filtering is a method for predicting preferences of a user by collecting preferences from many users. An effective way to implement Collaborative Filtering is the latent factor model approach, carried out using Matrix Factorization [1].

The algorithm for this Recommender System is a weighted, regularized Matrix Factorization, described in the paper by Hu et al. (2008) [2]. This is implemented using the [*implicit*](https://github.com/benfred/implicit) library. 
<p/>

[1] Y. Koren, R. Bell, C. Volinsky. "Matrix Factorization Techniques for Recommender Systems". 2009. 
<br/>
[2] Y. Hu, Y. Koren, C. Volinsky. ["Collaborative Filtering for Implicit Feedback Datasets"](http://yifanhu.net/PUB/cf.pdf). 2008.