# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:53:39 2019

@author: yasha_000
"""

import pandas as pd 
%matplotlib inline 
import matplotlib 
import matplotlib.pyplot as plt 
import numpy as np 

# pass in column names for each CSV as the column name is not given in the file and read them using pandas. # You can check the column names from the readme file

#read the file 
#Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code'] 
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,encoding='latin-1') 
#Reading ratings file: 
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp'] 
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1') 
#Reading items file: 
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'] 
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')


#look and feel of the data
#user 
print(users.shape) 
users.head()
#rating 
print(ratings.shape)
ratings.head()
#items
print(items.shape)
items.head()


#Taking account of test and train of rating data   
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp'] 
ratings_train = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1') 
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
ratings_train.shape, ratings_test.shape

#buliding of collaborative filterinng model 
#user-user and item-item similarity
#calculate the no.  of different user
n_users = ratings.user_id.unique().shape[0]
n_items = ratings.movie_id.unique().shape[0]

#create a user-item matrix for defining the user wit item 
data_matrix = np.zeros((n_users, n_items))
for line in ratings.itertuples():
    data_matrix[line[1]-1, line[2]-1] = line[3]

#for calculating the silimarity metrer between the user by using pairwise similarity 

from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(data_matrix, metric='cosine') 
item_similarity = pairwise_distances(data_matrix.T, metric='cosine')

#now definr the prediction base similarity '
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1) #We use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
           pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
           return pred

#making the prediction on the basis of item and user simliraity
user_prediction = predict(data_matrix, user_similarity, type='user') 
item_prediction = predict(data_matrix, item_similarity, type='item')














