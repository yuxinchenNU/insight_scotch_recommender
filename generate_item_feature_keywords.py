
import gensim
import numpy as np
import pandas as pd
import json


def word_to_keyword_ind(word, keywords_dict):
	# start with a column vector, will convert to row vector
	word_vec = np.zeros((len(keywords_dict), 1)) 
	for key, value in keywords_dict.items():
		if word in value:
			word_vec[key] = 1 
	word_vec = word_vec.T
	return word_vec

def reviews_to_vec(sentence, keywords_dict):
	word_vec = np.zeros((1, len(keywords_dict)))
	for word in sentence.split():
		word_vec += word_to_keyword_ind(word, keywords_dict)

	# for entries that are non-zero, change to 1
	pos_indices = word_vec >= 1
	word_vec[pos_indices] = 1

	return word_vec

def word_to_keyword(word, keywords_dict):
	keyword = ''
	for key, value in keywords_dict.items():
		if word in value:
			keyword = value[0]

	return keyword

def feature_iterable_construction(prod_review, keywords_dict):

	feature_list = []
	for word in prod_review.split():
		keyword = word_to_keyword(word, keywords_dict)
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

	tags_list = []
	# loop through each product
	item_feature_dict = {}
	for ind in df.index:
		reviews = df.loc[ind]['review and tasting notes']
		feature_list = feature_iterable_construction(reviews, keywords_dict)
		tags_list.append(feature_list)

	df['tags'] = tags_list

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

