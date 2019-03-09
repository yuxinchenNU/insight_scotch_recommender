"""
This script assigns tags to products based on the keywords extracted from
tasting notes in keywords_extraction.py
"""


import gensim
import numpy as np
import pandas as pd
import json


def word_to_keyword_ind(word, keywords_dict):
	"""
	function to one hot encode word based on keywords_dict
	Args: word (str) - word that needs to be one hot encoded
	      keywords_dict (dictionary) - a dictionary that has indices as key 
	                 and list of keywords as its value
	Returns: word_vec (numpy array) - a vector of size 1 by len(keywords_dict)
				     that represents "word" in the embeded space
	"""
	# initialize word_vec with a column vector, will convert to row vector
	word_vec = np.zeros((len(keywords_dict), 1)) 
	# loop through the dictionary keywords_dict, if word is in list value
	# then fill the key's entry in word_vec with 1
	for key, value in keywords_dict.items():
		if word in value:
			# fill the corresponding entry in word_vec with 1
			word_vec[key] = 1 
	# convert to a row vector
	word_vec = word_vec.T
	return word_vec

def reviews_to_vec(sentence, keywords_dict):
	"""
	function to encode sentence into a vector of size 1 by total number of 
	keywords in the dictionary
	Args: sentence (str) - the sentence to be encoded
	      keyword_dict (dictionary) - a dictionary that has indices as key 
	                 and list of keywords as its value
	Returns: word_vec (numpy array) - a vector that represents sentence of 
	                 size 1 by total number of keywords in keyword_dict
	                 so if a keyword in the dictionary occurs in sentence, 
	                 then the corresponding entry in word_vec is 1. 
	                 NOTE: does not consider how many times the word occurs
	"""
	# initialize the vector
	word_vec = np.zeros((1, len(keywords_dict)))
	# loop through each word in the sentence
	for word in sentence.split():
		# add 1 to the corresponding entry if the keyword occurs
		word_vec += word_to_keyword_ind(word, keywords_dict)

	# for entries that are non-zero, change to 1
	pos_indices = word_vec >= 1
	word_vec[pos_indices] = 1

	return word_vec

def word_to_firstkeyword(word, keywords_dict):
	"""
	function to standardize keywords, each value in keywords_dict is a list of
	variations of a keyword that are extracted from tasting notes
	Args: word (str) - word to be standardized if it is a keyword
	      keywords_dict (dictionary) - a dictionary that has indices as key 
	                 and list of keywords as its value
	Returns: keyword (str) - standardized keyword if word is in keywords_dict
		 			 otherwise an empty string
	"""

	keyword = ''
	for key, value in keywords_dict.items():
		if word in value:
			# set keyword to be the first element in the list of keywords
			keyword = value[0]

	return keyword

def feature_iterable_construction(prod_review, keywords_dict):
	"""
	function to generate a list of keywords for a product review
	Args: prod_review (str) - reviews of a product that needs to be assigned with keywords tags
	      keywords_dict (dictionary) - a dictionary that has indices as key 
	                 and list of keywords as its value
	Returns: feature_list (list) - list of strings 
	"""


	feature_list = []
	for word in prod_review.split():
		keyword = word_to_firstkeyword(word, keywords_dict)
		# add keywords but also need to make sure not to add duplicated ones
		if len(keyword) > 0 and keyword not in feature_list:
			feature_list.append(keyword)

	return feature_list








def generate_item_feature_matrix():
	keywords_dict = {
	0: ['soft', 'delicate', 'delic'],
	1: ['sweet', 'sweetness'],
	2: ['fruit', 'fruiti', 'fruit hint'],
	3: ['smoky', 'smokeful', 'smoki', 'smoke', 'smokey'],
	4: ['rich'],
	5: ['fresh'],
	6: ['vanilla'],
	7: ['spice', 'spici', 'spicy'],
	8: ['peaty', 'peat', 'peati'],
	9: ['chocolate', 'chocol'],
	10: ['oak', 'oaki', 'dark oak', 'oakiness', 'oaky'],
	11: ['toffee', 'toffe', 'candy', 'candi', 'honei', 'honey', 'sugar', 'raisin'],
	12: ['nut', 'nutti', 'nutty'],
	13: ['citrus', 'citru', 'lemon', 'orange', 'orang', 'lime', 'zesti', 'zest', 'zesty'],
	14: ['creamy', 'cream', 'creami'],
	15: ['earthy', 'earthi', 'earthiness', 'earth'],
	16: ['leaf', 'grass', 'grassi', 'grassy', 'grassland'],
	17: ['nutmeg'],
	18: ['cinnamon'],
	19:	['banana'], 
	20: ['apple', 'appl', 'green appl'], 
	21: ['pineapple', 'pineappl'],
	22: ['toast', 'toasti', 'toasty'],
	23: ['sherry', 'sherri'],
	24: ['dry', 'dri'],
	25: ['liquorice','liquoric'],
	26: ['wood', 'woodi'],
	27: ['bitter'],
	28: ['coffee', 'coffe']
	}

	df_path = 'cleanedDataFrameV2.json'
	df = pd.read_json(df_path)

	# get price range
	# first convert to us dollars
	df['price'] = df['price'].apply(lambda x: x.replace(u'£',''))
	df['price'] = df['price'].apply(lambda x: x.replace(',',''))
	df['price'] = df['price'].apply(lambda x: float(x)*1.3)

	tags_list = []
	# loop through each product
	item_feature_dict = {}
	for ind in df.index:
		reviews = df.loc[ind]['review and tasting notes']
		feature_list = feature_iterable_construction(reviews, keywords_dict)
		price = df.loc[ind]['price']
		if price < 100:
			feature_list.append('< $100')
		elif price < 250:
			feature_list.append('$100 - $250')
		elif price < 500:
			feature_list.append('$250 - $500')
		else:
			feature_list.append('> $500')

		tags_list.append(feature_list)



	df['tags'] = tags_list

	print(df.iloc[0]['tags'])

	df.to_json('cleanedDataFrameWithTagsV2.json')
	print(df.columns.values)

	print(df.loc[10723].index)

	# ### NEED TO SAVE AS A MATRIX
	# item_feature_matrix = np.zeros((len(df), len(keywords_dict)))
	# list_keys = list(item_feature_dict.keys())
	# for i in range(len(df)):
	# 	# assign each row of the matrix with feature vector
	# 	key = list_keys[i]
	# 	item_feature_matrix[i] = item_feature_dict[key]

	# # save the matrix
	# np.savetxt('item_feature_matrix.txt', item_feature_matrix, fmt='%.10f')
	return df


df = generate_item_feature_matrix()

