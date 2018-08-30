# Recommender System for Cargo O-D cities
This is a Collaborative Filtering model to recommend Origin-Destination city preferences for cargo drivers. Built for Northwestern University's Transportation Lab.

## Background
Truck-sharing mobile apps now provide a platform for cargo drivers to search for cargo trips. Each cargo trip can be defined by an origin-city (where drivers load cargo) and a destination-city (where drivers unload cargo). Data from these apps can show us the Origin-Destination (OD) city preferences of each drivers. 

## Algorithm
Collaborative Filtering is a method for predicting preferences of a user by collecting preferences from many users. An effective way to implement Collaborative Filtering is the latent factor model approach, carried out using Matrix Factorization [1].

The algorithm for this Recommender System is a weighted, regularized Matrix Factorization, described in the paper ["Collaborative Filtering for Implicit Feedback Datasets"](http://yifanhu.net/PUB/cf.pdf) by Hu et al. (2008). This is implemented using the [*implicit*](https://github.com/benfred/implicit) library. 


[1] Y. Koren, R. Bell, C. Volinsky. "Matrix Factorization Techniques for Recommender Systems". 2009. 