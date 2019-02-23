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

def df_to_sparseMatrix(df):
	sp_matrix = coo_matrix(df.values)
	return sp_matrix

def create_dataset(df):
	## create a mapping between the user and item ids from our input data 
	#to indices that will be used internally by the model
	dataset = Dataset()
	list_user_names = list(df.index)
	list_items = df.columns.values
	dataset.fit((user_name for user_name in list_user_names),
            (item for item in list_items))
	
	## Build the interaction matrix
	# it encodes the interactions betwee users and items.
	# need (user, item) pair that has 1's in df
	list_pairs = list(df.stack().index)
	(interactions, weights) = dataset.build_interactions((pair for pair in list_pairs))


	return dataset, interactions, weights


def recommend_to_a_user(model, user_id, item_ids, topn, df, user_name_to_ind_map, item_to_ind_map):
	score = model.predict(user_id, item_ids)
	recom = (-score).argsort()[:topn]

	recom_list_item_prod_id = []
	recom_list_items_names = [] # store names of the recommended items

	for recom_item in recom:
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
	# convert list to numpy array
	v1 = np.array(v1)
	v2 = np.array(v2)
	return np.dot(v1, v2)/LA.norm(v1)/LA.norm(v2)

def recommend_based_on_product(model, item_ind, num_items, item_to_ind_map, df):
	biases, item_features = model.get_item_representations()
	target_feature = item_features[item_ind, :]

	# compute cosine similarity
	cosine_similarity_items = []
	for i in range(num_items):
		feature = item_features[i, :]
		cosine_similarity_items.append(cosine_similarity(target_feature, feature))
	similar_item_inds = (-np.array(cosine_similarity_items)).argsort()[:5]
	
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

	




def main():
	file_path = '../data/webscraping_whisky_exchange/sms_products.json'
	df_ui, df = file_to_df(file_path)
	
	## thresholding the dataframe
	pos_thresholding = df_ui > 3
	neg_thresholding = df_ui <= 3
	df_ui[pos_thresholding] = 1
	df_ui[neg_thresholding] = None

	# create the dataset used in lightFM
	dataset, interactions, weights = create_dataset(df_ui)
	num_users, num_items = dataset.interactions_shape()

	# user_map, item_map = dataset.mapping()
	user_name_to_ind_map = dataset.mapping()[0]
	item_to_ind_map = dataset.mapping()[2] # ind is lightFM index

	# ## split the dataframe into training and test set
	test_percentage = 0.2
	seed = 1
	train, test = random_train_test_split(interactions, test_percentage=test_percentage, random_state=np.random.RandomState(seed))

	model = LightFM(learning_rate=0.05, loss='warp',random_state = seed, user_alpha=0.0001, item_alpha = 0.0001)

	model.fit(train, epochs=5)

	train_precision = precision_at_k(model, train, k=10).mean()
	test_precision = precision_at_k(model, test, k=10).mean()

	train_auc = auc_score(model, train).mean()
	test_auc = auc_score(model, test).mean()

	# print('Precision: train %.4f, test %.4f.' % (train_precision, test_precision))
	print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

	# ## prediction for a specific user
	# # user id to be predicted
	# user_id = randint(0,num_users-1)
	# user_id = 18
	# print('Recommedations for ' + str(user_id))
	# # items ids
	# item_ids = np.array(range(num_items))

	# # run the prediction to obtain ranks of items
	# topn = 5
	# recom_list_items_names = recommend_to_a_user(model, user_id, item_ids, topn, df, user_name_to_ind_map, item_to_ind_map) # will return a list of id's for recommended items
	# print(recom_list_items_names)

	## prediction based on item
	prod_id1 = '15703'
	prod_names1 = 'Glenlivet 18 Year Old'

	item_ind = item_to_ind_map[prod_id1] # product index in lightFM
	recom_list_items_names1 = recommend_based_on_product(model, item_ind, num_items, item_to_ind_map, df)
	print(recom_list_items_names1)
	prod_id2 = '23066'
	prod_names2 = 'Bowmore Small Batch'

	item_ind = item_to_ind_map[prod_id2] # product index in lightFM
	recom_list_items_names2 = recommend_based_on_product(model, item_ind, num_items, item_to_ind_map, df)
	# print(recom_list_items_names2)

	df_recom = {prod_names1: recom_list_items_names1, prod_names2: recom_list_items_names2}
	df_recom = pd.DataFrame(data=df_recom)

	## save as csv
	df_recom.to_csv('products_rec.csv', index=True, sep=',')





if __name__ == '__main__':
	main()
