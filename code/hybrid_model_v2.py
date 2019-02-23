import re
import numpy as np
from numpy import linalg as LA
import scipy.sparse as sp
import pandas as pd
from time import time  # To time our operations
from random import randint
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix

# Import lightFM model
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import auc_score
from lightfm.evaluation import precision_at_k
from lightfm.cross_validation import random_train_test_split

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import stem_text

# import stop words using NLTK Stop words
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# Plotting tools
import matplotlib.pyplot as plt
import seaborn as sns

import json
import random
import pickle
import multiprocessing


def remove_stop_words_text(sentence):
	"""
	remove stop words in a sentence
	Args: sentence (string)
	Returns: sentence (string) - text that stop words are removed
	"""
	# break the sentence into list of tokens using gensim
	list_tokens = gensim.utils.simple_preprocess(str(sentence))
	# remove stop words in the list of tokens
	list_tokens = [token for token in list_tokens if not token in stop_words]
	# form non-stop words back to a sentence
	sentence = " ".join(list_tokens)
	return sentence


def lemmatization_text(sentence, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
	"""
	function to lemmatize text
	Args: sentence (str) - sentence to be lemmatized
	      allowed_postags (list) - allowed words postags
	Returns: sentence (str) - lemmatized sentence
	"""
	# break the sentence into list of tokens using gensim
	# list_tokens = gensim.utils.simple_preprocess(str(sentence))
	sentence = stem_text(sentence)

	return sentence


def read_corpus(df, tokens_only=False):
		"""
		function to tokenize sentences and remove punctuations using Gensim's simple_preprocess()
		tag training data as well
		"""
		for i, line in enumerate(df):
			if tokens_only:
				yield gensim.utils.simple_preprocess(line, deacc=True)
			else:
				# add tags to training data
				yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line, deacc=True), [i])



def create_df_ui(df):
	"""
	combine all the reviews for each product
	Args: df(pandas dataframe) - dataframe that stores all the information
	Returns: df_ui(pandas dataframe) - dataframe that stores user item interaction
	"""
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
	list_prod_id = df.name.tolist()
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
		product_id = df.loc[ind].name
		list_ratings = df.loc[ind]['rating']
		# fill missing ratings with average rating of the product
		if len(list_ratings) < len(list_reviewer_names):
			avg_rating = str(np.mean([float(rating) for rating in list_ratings]))
			for k in range(len(list_reviewer_names) - len(list_ratings)):
				list_ratings.append(avg_rating)
		# create a dictionary that stores product ratings that a user gives
		for j in range(len(list_reviewer_names)):
			reviewer_name = list_reviewer_names[j]
			subdictionary = dict_name_product[reviewer_name]
			subdictionary[product_id] = list_ratings[j]
			dict_name_product[reviewer_name] = subdictionary

	# read in dict_name_product into a dataframe
	df_ui = pd.DataFrame.from_dict(dict_name_product, orient='index')

	# convert entries from string to floating number
	df_ui = df_ui.apply(pd.to_numeric)
	# fill nans with 0
	df_ui = df_ui.fillna(0)
	
	return df_ui

		

def remove_stop_words_df(df, list_column_names):
	"""
	remove stop words in df[column_name] which consists of sentences (string)
	Args: df (pandas dataframe) - dataframe that stores data
		  list_column_names (list of strings) - list of column names that need to be dealt with
	Return: df - dataframe where in the specified columns, stop words are removed
	"""
	for column_name in list_column_names:
		df[column_name] = df[column_name].apply(lambda sentence: remove_stop_words_text(sentence))
	return df


def lemmatization_df(df, list_column_names):
	"""
	function to lemmatize texts in selected columns in the dataframe df
	Args: df (pandas dataframe) - dataframe that stores data
		  list_column_names (list of strings) - list of column names that need to be dealt with
	Return: df - dataframe where in the specified columns, texts are lemmatized
	"""
	for column_name in list_column_names:
		df[column_name] = df[column_name].apply(lambda sentence: lemmatization_text(sentence))
		
	return df

def nlp_preprocess(df, list_column_names):
	"""
	function to call helper functions to remove stop words and lemmatize texts in
	selected columns of the dataframe df
	Args: df (pandas dataframe) - dataframe that stores data
		  list_column_names (list of strings) - list of column names that need to be dealt with
	Return: df - dataframe where in the specified columns, texts are lemmatized
	"""


	# remove stopping words
	df = remove_stop_words_df(df, list_column_names)
	
	# lemmatize
	df = lemmatization_df(df, list_column_names)
	return df



def generate_df_interaction(df):
	# read in file that contains tasting notes
	
	df_ui = create_df_ui(df)

	return df_ui


def df_to_sparseMatrix(df):
	"""
	converts dataframe to a sparse matrix
	"""
	sp_matrix = coo_matrix(df.values)
	return sp_matrix

def create_dataset(df, item_features, list_item_features):
	"""
	function to create the dataset based on df which stores all the data including
	features (tags) of each products
	Args: df(pandas dataframe) - 
	"""
	## create a mapping between the user and item ids from our input data 
	#to indices that will be used internally by the model
	dataset = Dataset(item_identity_features=True)
	list_user_names = list(df.index)
	list_items = df.columns.values
	
	dataset.fit((user_name for user_name in list_user_names),
			(item for item in list_items), 
			item_features=(item_feature for item_feature in list_item_features))
	
	## Build the interaction matrix
	# it encodes the interactions betwee users and items.
	# need (user, item) pair that has 1's in df
	list_pairs = list(df.stack().index)
	(interactions, weights) = dataset.build_interactions((pair for pair in list_pairs))

	item_feature_matrix = dataset.build_item_features(item_features)


	return dataset, interactions, weights, item_feature_matrix


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

def recommend_based_on_product(model, prod_id, item_to_ind_map, df, item_feature_matrix, chara_list, item_feature_map):
	num_items = len(df) # number of items

	# item_features contain latent representation of each item
	biases, item_features = model.get_item_representations(item_feature_matrix)
	
	item_embedding = model.item_embeddings # contains latent representation of added item features
	
	if prod_id != None:
		item_ind = item_to_ind_map[prod_id] # product index in lightFM
		target_feature = item_features[item_ind, :] # contains the chosen products
	else:
		target_feature = 0*item_features[0, :]

	# include characteristics latent representations in target_feature
	for chara in chara_list:
		lightFM_ind_feature = item_feature_map[chara]	
		target_feature = target_feature + item_embedding[lightFM_ind_feature,:]

	
	# compute cosine similarity
	cosine_similarity_items = []
	for i in range(num_items):
		feature = item_features[i, :]
		cosine_similarity_items.append(cosine_similarity(target_feature, feature))
	print(np.array(cosine_similarity_items).argsort())
	similar_item_inds = (-np.array(cosine_similarity_items)).argsort()[:5]
	
	similar_item_prod_ids = []
	for similar_item_ind in similar_item_inds:
		for prod_id, item_id in item_to_ind_map.items():
			if similar_item_ind == item_id:
				similar_item_prod_id = prod_id
		similar_item_prod_ids.append(similar_item_prod_id)
	
	# remove itself
	similar_item_prod_ids = similar_item_prod_ids[1:]

	# df = df.set_index('product ID')
	recom_list_items_names = []
	for prod_id in similar_item_prod_ids:
		prod_name = df.loc[prod_id, 'name']
		recom_list_items_names.append(prod_name)

	return recom_list_items_names

def thresholding(threshold, df_ui):
	## thresholding the dataframe
	pos_thresholding = df_ui >= threshold
	neg_thresholding = df_ui < threshold
	df_ui[pos_thresholding] = 1
	df_ui[neg_thresholding] = None
	return df_ui

def train_model(df_withtags, df_ui, k):
	item_features = [(row.name, row['tags']) for i, row in df_withtags.iterrows()]
	# print(item_features)
	# assert False	
	# thresholding the user-item interaction dataframe
	threshold = 3.5
	df_ui = thresholding(threshold, df_ui)

	# create the dataset used in lightFM
	list_item_features = ['soft', 'sweet', 'fruit', 'smoky', 'rich', 'fresh', 'vanilla', 
	'spice', 'peaty', 'chocolate', 'oak', 'toffee','nut', 'citrus', 'creamy', 'earthy', 
	'leaf', 'nutmeg', 'cinnamon', 'banana', 'apple', 'pineapple', 'toast', 'sherry',
	'dry', 'liquoric', 'wood', 'bitter', 'coffee']

	dataset, interactions, weights, item_feature_matrix = create_dataset(df_ui, item_features, list_item_features)
	

	# ind is lightFM index
	user_name_to_ind_map, user_feature_map, item_to_ind_map, item_feature_map = dataset.mapping()
	print(len(item_to_ind_map))
	print(df_ui.shape)
	# ## split the dataframe into training and test set
	test_percentage = 0.2
	seed = 1
	train, test = random_train_test_split(interactions, test_percentage=test_percentage, random_state=np.random.RandomState(seed))


	# compute precision at k for popularity model
	# find top k popular items
	popularity_items = (-df_ui.sum(axis=0)).argsort()[:k]
	# print(popularity_items)
	num_pos = 0
	for ind in popularity_items.index:
		num_positive_item = df_ui[ind].sum(axis=0)
	precision_at_k_pop_model = num_positive_item.sum()/len(df_ui)/k
	print(precision_at_k_pop_model*100)

	model = LightFM(no_components=30, learning_rate=0.025, loss='warp',random_state = seed, user_alpha=0.0001, item_alpha = 0.0001)
	# model = LightFM(no_components=20, learning_rate=0.05, loss='warp',random_state = seed)

	# model.fit(train, epochs=10)
	model.fit(train, item_features = item_feature_matrix, epochs=10)

	return model, train, test, dataset, item_feature_matrix, item_to_ind_map, item_feature_map

def evaluate_model(model, train, test, item_feature_matrix, k):

	train_precision = precision_at_k(model, train, item_features = item_feature_matrix, k=k).mean()
	test_precision = precision_at_k(model, test, item_features = item_feature_matrix, k=k).mean()
	# train_precision = precision_at_k(model, train, k=k).mean()
	# test_precision = precision_at_k(model, test, train_interactions=train, k=k).mean()

	train_auc = auc_score(model, train, item_features = item_feature_matrix).mean()
	test_auc = auc_score(model, test, item_features = item_feature_matrix).mean()
	# train_auc = auc_score(model, train).mean()
	# test_auc = auc_score(model, test).mean()
	print('Precision: train %.4f, test %.4f.' % (train_precision*100, test_precision*100))
	print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))


def training_driver():
	df_withtags_path = 'cleanedDataFrameWithTagsV2.json'
	df_withtags = pd.read_json(df_withtags_path)
	df_ui = generate_df_interaction(df_withtags)
	print(df_withtags.columns.values)
	# print(df_withtags.head())
	# print(df_ui.head())
	# assert False
	k = 5
	model, train, test, dataset, item_feature_matrix, item_to_ind_map, item_feature_map = train_model(df_withtags, df_ui, k)
	# save the model to disk
	with open('trained_model.p', 'wb') as file:
		pickle.dump(model, file)

	with open('item_feature_map.p', 'wb') as file:
		pickle.dump(item_feature_map, file)

	with open('item_to_ind_map.p', 'wb') as file:
		pickle.dump(item_to_ind_map, file)

	with open('item_feature_matrix.p', 'wb') as file:
		pickle.dump(item_feature_matrix, file)



	num_users, num_items = dataset.interactions_shape()

	evaluate_model(model, train, test, item_feature_matrix, k)

	
def recommendation(prod_id, chara_list):
	# prod_id = 15703
	# prod_names = 'Glenlivet 18 Year Old'
	# chara_list = ['citrus', 'vanilla', 'sweet']
	## load files
	df_withtags_path = 'cleanedDataFrameWithTagsV2.json'
	df_withtags = pd.read_json(df_withtags_path)
	model = pickle.load(open('trained_model.p', 'rb'))
	item_feature_map = pickle.load(open('item_feature_map.p', 'rb'))
	item_to_ind_map = pickle.load(open('item_to_ind_map.p', 'rb'))
	item_feature_matrix = pickle.load(open('item_feature_matrix.p', 'rb'))
	
	recom_list_items_names = recommend_based_on_product(model, prod_id, item_to_ind_map, df_withtags, item_feature_matrix, chara_list, item_feature_map)
	print(recom_list_items_names)


def main():
	training_driver()
	# recommendation(15703, ['citrus', 'vanilla', 'sweet'])
	recommendation(None, ['citrus', 'vanilla', 'sweet'])
	


if __name__ == '__main__':
	main()
