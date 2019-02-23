# Import the model
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import auc_score
from lightfm.evaluation import precision_at_k
from lightfm.cross_validation import random_train_test_split

import pandas as pd
import json
import numpy as np
from numpy import linalg as LA
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix

import time

from random import randint

def file_to_df(path):
	"""
	function to load product data from .json file into a dataframe and
	preprocesses the dataframe. It creates a user-item dataframe that 
	stores user's rating of the item. 
	Args: path (str) - file path where the .json file is stored

	Returns: df_ui (pandas dataframe) - dataframe that stores user's rating of the item
										size of (num unique users, num_items)
			 df (pandas dataframe) - dataframe that stores the original preprocessed data
			 						size of (num_items, num of information about the items
			 						including reviewers' names, ratings, and text reviews)
	"""

	df = pd.DataFrame([json.loads(line) for line in open(path)])
	# remove products that do not have reviews
	df.dropna(subset = ['review times'], inplace=True)
	# remove \n in the dataframe
	df = df.replace('\n','', regex=True)
	# clean reviewer names
	for ind in df.index:
		list_reviewer_names = df.loc[ind]['reviewer names']
		df.loc[ind]['reviewer names'] = [string.replace('\n', '') for string in list_reviewer_names]
		# remove the first space in the string
		list_reviewer_names = df.loc[ind]['reviewer names']
		df.loc[ind]['reviewer names'] = [string.lstrip() for string in list_reviewer_names]
		# obtain first name
		list_reviewer_names = df.loc[ind]['reviewer names']
		df.loc[ind]['reviewer names'] = [string.split(' ')[0] for string in list_reviewer_names]

	# obtain a list of reviewer names
	list_reviewers_name = []
	for reviewer_list in df['reviewer names']:
		for reviewer_name in reviewer_list:
			if reviewer_name != 'Anonymous':
				list_reviewers_name.append(reviewer_name)
	            
	# print("Non-anonymous reviewers number: " + str(len(list_reviewers_name)))

	# print("Unique reviewers number: " + str(len(set(list_reviewers_name))))

	## create a dataframe that stores the user-item interaction info
	# first specify the columns to be list of product IDs
	list_prod_id = df['product ID'].tolist()
	df_ui = pd.DataFrame(columns=list_prod_id)

	# obtain a list of unique reviewers names
	list_unique_reviewers_name = []
	list_reviewers_name = []
	for reviewer_list in df['reviewer names']:
		for reviewer_name in reviewer_list:
			list_reviewers_name.append(reviewer_name)
	list_unique_reviewers_name = list(set(list_reviewers_name))

	# generate a dictionary with list_reviewers_names as keys 
	dict_name_product = {k:{} for k in list_unique_reviewers_name}
	# complete values in the dictionary with dictionaries of product ratings
	# loop through dataframe df
	for ind in df.index:
		list_reviewer_names = df.loc[ind]['reviewer names']
		product_id = df.loc[ind]['product ID']
		list_ratings = df.loc[ind]['rating']
		# fill missing ratings with average rating of the product
		if len(list_ratings) < len(list_reviewer_names):
			avg_rating = str(np.mean([float(rating) for rating in list_ratings]))
			for k in range(len(list_reviewer_names) - len(list_ratings)):
				list_ratings.append(avg_rating)
		for j in range(len(list_reviewer_names)):
			reviewer_name = list_reviewer_names[j]
			subdictionary = dict_name_product[reviewer_name]
			subdictionary[product_id] = list_ratings[j]
			dict_name_product[reviewer_name] = subdictionary

	# read in dict_name_product into a dataframe
	df_ui = pd.DataFrame.from_dict(dict_name_product, orient='index')
	# remove None product id
	df_ui.drop([None], axis=1, inplace=True)

	# drop anonymous ratings
	df_ui.drop(['Anonymous'], axis=0, inplace=True)

	# convert entries from string to floating number
	df_ui = df_ui.apply(pd.to_numeric)
	# fill nans with 0
	df_ui = df_ui.fillna(0)

	return df_ui, df

def recommend_to_a_user(model, user_id, item_ids, topn, df, user_name_to_ind_map, item_to_ind_map):
	"""
	function to recommend a list of products to a user using a provided pre-trained model
	Args: model (lightFM instance) - lightFM pretrained model based on a set of data in df
		  user_id (int) - user id that is used in lightFM model
		  item_ids (np array of int) - item ids that are used in lightFM model
		  topn (int) - number of top recommended items
		  df (dataframe) - the original preprocessed data that stores items information
		  user_name_to_ind_map (dictionary) - lightFM instance that is used to map users' names to lightFM indices
		  								key is users' name and value is lightFM index
		  item_to_ind_map (dictionary) - lightFM instance that is used to map item product id to lightFM indices
		  								key is item product id and value is lightFM index
	Returns: recom_list_items_names (list of strings) - list of recommended items' names for user user_id



	"""

	# compute scores that user with user_id would give for each item in item_ids
	score = model.predict(user_id, item_ids)
	# sort the scores and pick lightFM indices of top topn products with highest scores
	recom = (-score).argsort()[:topn]

	recom_list_item_prod_id = [] # store product ids of the recommended items
	recom_list_items_names = [] # store names of the recommended items

	# obtain product ids of the recommended items
	for recom_item in recom:
		# need to find the key (product id) that has value == item_ind in recom
		for prod_id, item_ind in item_to_ind_map.items():
			if item_ind == recom_item:
				recom_prod_id = prod_id
		recom_list_item_prod_id.append(recom_prod_id)
	# to find the product names using the product ids
	df = df.set_index('product ID')
	for prod_id in recom_list_item_prod_id:
		prod_name = df.loc[prod_id, 'name']
		recom_list_items_names.append(prod_name)

	return recom_list_items_names

def cosine_similarity(v1, v2):
	"""
	function to compute cosine similarity between two lists v1 and v2
	"""

	# convert list to numpy array
	v1 = np.array(v1)
	v2 = np.array(v2)
	return np.dot(v1, v2)/LA.norm(v1)/LA.norm(v2)


def recommend_based_on_product(model, item_ind, num_items, item_to_ind_map, df, topn):
	"""
	function to recommend a list of items that are most similar to the chosen item (which has lightFM id item_id)
	Args: model (lightFM instance) - a lightFM pretrained model based on interaction data in df
		  item_ind (int) - lightFM index of the item that the user chooses
		  num_items(int) - total number of items
		  item_to_ind_map (dictionary) - lightFM instance that is used to map item product id to lightFM indices
		  								key is item product id and value is lightFM index
		  df (pandas dataframe) - dataframe that stores the original preprocessed data
		  topn (int) - number of recommended products
	"""

	biases, item_features = model.get_item_representations()
	target_feature = item_features[item_ind, :]

	# compute cosine similarity
	cosine_similarity_items = []
	for i in range(num_items):
		feature = item_features[i, :]
		cosine_similarity_items.append(cosine_similarity(target_feature, feature))
	similar_item_inds = (-np.array(cosine_similarity_items)).argsort()[:topn]
	
	similar_item_prod_ids = []
	for similar_item_ind in similar_item_inds:
		for prod_id, item_id in item_to_ind_map.items():
			if similar_item_ind == item_id:
				similar_item_prod_id = prod_id
		similar_item_prod_ids.append(similar_item_prod_id)
	
	# remove itself
	similar_item_prod_ids = similar_item_prod_ids[1:]

	df = df.set_index('product ID')
	recom_list_items_names = []
	for prod_id in similar_item_prod_ids:
		prod_name = df.loc[prod_id, 'name']
		recom_list_items_names.append(prod_name)

	return recom_list_items_names


class recommender():

	def __init__(self, df, df_ui threshold):
		self.df = df
		self.df_ui = df_ui
		self.threshold = threshold

	def thresholding_dataframe(self):
		## thresholding the dataframe
		pos_thresholding = df_ui > self.threshold
		neg_thresholding = df_ui <= self.threshold
		self.df_ui[pos_thresholding] = 1
		self.df_ui[neg_thresholding] = None

	def create_dataset_lightFM(self):
		dataset, interactions, weights = create_dataset(self.df_ui)
		# num_users, num_items = dataset.interactions_shape()

		return dataset, interactions

	def lightFM_mapping(self, dataset):
		# obtain lightFM mapping for users and items
		user_name_to_ind_map = dataset.mapping()[0]
		item_to_ind_map = dataset.mapping()[2] # ind is lightFM index

		return user_name_to_ind_map, item_to_ind_map

	def train_test_split(self, test_percentage=0.2, seed=1):
		train, test = random_train_test_split(interactions, test_percentage=test_percentage, ...
			random_state=np.random.RandomState(seed))

		return train, test

	def train(self, train, epochs, learning_rate, loss, seed=1):
		model = LightFM(learning_rate=learning_rate, loss = loss,...
			random_state = seed, user_alpha=0.0001, item_alpha = 0.0001)

		model.fit(train, epochs = epochs)


		return model

	def evaluate(self, model, train, test, k=10):
		train_precision = precision_at_k(model, train, k=k).mean()
		test_precision = precision_at_k(model, test, k=k).mean()

		train_auc = auc_score(model, train).mean()
		test_auc = auc_score(model, test).mean()

		return train_precision, test_precision, train_auc, test_auc

	def prediction_for_user(self, user_id, num_items, topn):
		# items ids
		item_ids = np.array(range(num_items)) # obtain ids of all items

		# run the prediction to obtain ranks of items
		recom_list_items_names = recommend_to_a_user(model, user_id, item_ids, topn, df, ...
		user_name_to_ind_map, item_to_ind_map) # will return a list of id's for recommended items
		
		return recom_list_items_names

	def prediction_based_on_item(self, prod_id, item_to_ind_map, num_items, df, topn):
		item_ind = item_to_ind_map[prod_id]
		# product index in lightFM
		recom_list_items_names = recommend_based_on_product(model, item_ind, num_items, item_to_ind_map, df, topn)

		return recom_list_items_names

def model_training_testing_driver():

def main():


