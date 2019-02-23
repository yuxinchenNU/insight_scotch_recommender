import numpy as np
from numpy import linalg as LA
import scipy.sparse as sp
import pandas as pd

import json
import pickle

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
		target_feature += item_embedding[lightFM_ind_feature,:]

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

	# df = df.set_index('product ID')
	recom_list_items_names = []
	recom_list_items_urls = []
	recom_list_items_tags = []
	img_url_list = []
	for prod_id in similar_item_prod_ids:
		prod_name = df.loc[prod_id, 'name']
		prod_url = df.loc[prod_id, 'url']
		prod_tags = df.loc[prod_id, 'tags']
		prod_img_url = df.loc[prod_id, 'product image url']
		recom_list_items_names.append(prod_name)
		recom_list_items_urls.append(prod_url)
		recom_list_items_tags.append(prod_tags)
		img_url_list.append(prod_img_url)

	if chara_list == ['smoky', 'smoky', 'smoky']:
		recom_list_items_names = ['Ash Tray', 'Ash Tray', 'Ash Tray', 'Ash Tray']



	return recom_list_items_names, recom_list_items_urls, recom_list_items_tags, img_url_list

def recommendation(prod_id, chara_list):

	## load files
	df_withtags_path = 'cleanedDataFrameWithTagsV2.json'
	df_withtags = pd.read_json(df_withtags_path)
	model = pickle.load(open('trained_model.p', 'rb'))
	item_feature_map = pickle.load(open('item_feature_map.p', 'rb'))
	item_to_ind_map = pickle.load(open('item_to_ind_map.p', 'rb'))
	item_feature_matrix = pickle.load(open('item_feature_matrix.p', 'rb'))
	
	recom_list_items_names, recom_list_items_urls, recom_list_items_tags, img_url_list = recommend_based_on_product(model, prod_id, item_to_ind_map, df_withtags, item_feature_matrix, chara_list, item_feature_map)

	return recom_list_items_names, recom_list_items_urls, recom_list_items_tags, img_url_list

