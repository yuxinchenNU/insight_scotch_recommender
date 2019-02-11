import re
import numpy as np
from numpy import linalg as LA
import scipy.sparse as sp
import pandas as pd

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.summarization import keywords
from gensim.utils import lemmatize
from gensim.parsing.preprocessing import stem_text

# import stop words using NLTK Stop words
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# Plotting tools
import matplotlib.pyplot as plt

import json
import random
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

	"""
	# break the sentence into list of tokens using gensim
	
	# form lemmatized words back to a sentence
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



def preprocess_reviews(path, combine_tasting_notes):
	"""
	combine all the reviews for each product
	"""
	regular_data_path = 'sms_products.json'
	df_regular = pd.DataFrame([json.loads(line) for line in open(regular_data_path)])

	api_review_data_path = 'sms_api_review.json'
	df_api_review = pd.DataFrame(json.loads(line) for line in open(api_review_data_path))
	list_prod_id_api = list(df_api_review.columns.values)
	for prod_id in list_prod_id_api:
		# obtain list of dictionaries that contain reviews for product with prod_id
		list_review_dict = df_api_review.loc[0][prod_id]
		# index in df_regular that has product ID with prod_id
		index = df_regular.index[df_regular['product ID'] == prod_id].tolist()[0]
		reviewer_names =[]
		reviewer_rating = []
		reviews = []
		# extract each review in the list_review_dict
		for review_dict in list_review_dict:
			reviewer_names.append(review_dict['Author'])
			reviewer_rating.append(review_dict['Rating'])
			reviews.append(review_dict['Review'])
		# append these info to df_regular
		# 1) reviewer_names
		df_regular.loc[index]['reviewer names'] = df_regular.loc[index]['reviewer names']+reviewer_names
		# 2) rating
		df_regular.loc[index]['rating'] = df_regular.loc[index]['rating'] + reviewer_rating
		# 3) reviews
		df_regular.loc[index]['reviews'] = df_regular.loc[index]['reviews'] + reviews
		
	df = df_regular

	# # df = pd.DataFrame([json.loads(line) for line in open(path)])
	# df = pd.read_json(path)
	# remove products that do not have reviews
	df.dropna(subset = ['review times'], inplace=True)
	for ind in df.index:
		if len(df.loc[ind]['review times']) == 0:
			df.loc[ind]['review times'] = None

	df.dropna(subset = ['review times'], inplace=True)
	# remove products that do not have ratings
	for ind in df.index:
		rating = df.loc[ind]['rating']
		if len(rating) == 0:
			df.loc[ind]['rating'] = None
	df.dropna(subset = ['rating'], inplace=True)
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


	# remove \n in reviews and tasting notes
	# remove "Tasting Notes by " in the tasting notes
	for ind in df.index:
		list_reviews = df.loc[ind]['reviews']
		df.loc[ind]['reviews'] = [string.replace('\n', '') for string in list_reviews]
		# combine all the reviews into one text
		df.loc[ind]['reviews'] = ' '.join(df.loc[ind]['reviews'])

		tasting_note = df.loc[ind]['tasting notes']
		if tasting_note != None:
			# remove \n
			df.loc[ind]['tasting notes'] = tasting_note.replace('\n', '')
			# remove "Tasting Notes by "
			if tasting_note.find("Tasting Notes by "):
				df.loc[ind]['tasting notes'] = tasting_note.replace('Tasting Notes by ','')

		
	# combine reviews with tasting notes if combine_tasting_notes = True
	if combine_tasting_notes:
		review_tasting_notes = {}
		for ind in df.index:
			if df.loc[ind]['tasting notes'] != None:
				tasting_note = df.loc[ind]['tasting notes']

				review_tasting_notes[ind] = tasting_note + df.loc[ind]['reviews']
			else:
				review_tasting_notes[ind] = df.loc[ind]['reviews']
		df['review and tasting notes'] = df.index.map(review_tasting_notes)

	# remove empty review and tasting notes entries
	# convert "none" to NaNs
	for ind in df.index:
		if len(df.loc[ind]['review and tasting notes']) <= 1:
			df.loc[ind]['review and tasting notes'] = None

	# drop products that do not have any review and tasting notes
	df.dropna(subset=['review and tasting notes'], inplace=True)

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
	# df_ui.drop(['Anonymous'], axis=0, inplace=True)

	# convert entries from string to floating number
	df_ui = df_ui.apply(pd.to_numeric)
	# fill nans with 0
	df_ui = df_ui.fillna(0)


	# change index to product id
	df = df.set_index('product ID')
	df.drop([None], axis=0, inplace=True)
	
	return df, df_ui

		

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
	
	"""
	for column_name in list_column_names:
		df[column_name] = df[column_name].apply(lambda sentence: lemmatization_text(sentence))
		
	return df

# def bigram_generator(df):

def nlp_preprocess(df, list_column_names):
	# remove stopping words
	df = remove_stop_words_df(df, list_column_names)
	
	# lemmatize
	df = lemmatization_df(df, list_column_names)
	return df

def embed(review_tasting_notes_df, w2vmodel, feature_num):
	"""
	obtain feature vectors for each product
	"""
	# dictionary to store feature in the embeded vector space
	feature_vec_dict = {}

	# loop through all the products to obtain embedded vector for each product
	for ind in review_tasting_notes_df.index:
		text = review_tasting_notes_df.loc[ind]
		# split text into list of words
		words = text.split() 
		# print(words)
		vector = np.zeros((1, feature_num))
		print(np.sum(vector))
		# for each word, obtain its embedded vector form
		for word in words:
			try:
				vector = w2vmodel[word] + vector
			except Exception as e:
				print(e)
				print(word)
		
		feature_vec_dict[ind] = vector
	
	### NEED TO SAVE AS A MATRIX
	item_feature_matrix = np.zeros((len(review_tasting_notes_df), feature_num))
	list_keys = list(feature_vec_dict.keys())
	for i in range(len(review_tasting_notes_df)):
		# assign each row of the matrix with feature vector
		key = list_keys[i]
		item_feature_matrix[i] = feature_vec_dict[key]

	# save the matrix
	np.savetxt('item_feature_matrix.txt', item_feature_matrix, fmt='%.10f')
	return item_feature_matrix


def preprocess_text_driver():
	# read in file that contains tasting notes
	file_path = 'sms_merged_products.json'
	# file_path = '../data/webscraping_whisky_exchange/sms_merged_products.json'
	combine_tasting_notes = True
	df, df_ui = preprocess_reviews(file_path, combine_tasting_notes)
	
	# remove stopping words and lemmatize
	list_column_names = ['tasting notes', 'review and tasting notes', 'reviews']
	df = nlp_preprocess(df, list_column_names)

	print(df.columns.values)
	df.to_json('cleanedDataFrameV2.json')
	# combine all the tasting notes to extract keywords
	tasting_notes_text = ''
	for ind in df.index:
		tasting_notes_text += df.loc[ind]['tasting notes']

	kewords_list = keywords(tasting_notes_text, words=80, pos_filter=('NN','JJ'), lemmatize=True, split=True)
	return kewords_list

keywords_list = preprocess_text_driver()
print(keywords_list)
