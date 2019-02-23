from flask import render_template, url_for
from flaskexample import app
from flask import request
import pandas as pd
# from sqlalchemy import create_engine
# import psycopg2
from model import *
from flask import json


# # Python code to connect to Postgres
# # You may need to modify this based on your OS,
# # as detailed in the postgres dev setup materials.
# user = 'yuxinchen'  # add your Postgres username here
# host = 'localhost'
# dbname = 'product_db'
# db = create_engine('postgres://%s%s/%s' % (user, host, dbname))
# con = None
# con = psycopg2.connect(database=dbname,
#                        user=user)

class Product:
	def __init__(self, name, url, searched_tags, othertags, img_url):
		self.name = name
		self.url = url
		self.searched_tags = searched_tags
		self.othertags = othertags
		self.img_url = img_url

@app.route('/')

@app.route('/product_input')
def product_input():
	# query_list = "SELECT index, name FROM products_info_table;"
	# query_list_results = pd.read_sql_query(query_list, con)
	# product_id_list = list(query_list_results['index'])
	# product_name_list = list(query_list_results['name'])
	df_withtags_path = 'cleanedDataFrameWithTagsV2.json'
	df_withtags = pd.read_json(df_withtags_path)
	product_list = list(df_withtags['name'])
	characteristic_list = ['soft', 'sweet', 'fruit', 'smoky', 'rich', 'fresh', 'vanilla', 'spice', 'peaty', 'chocolate', 'oak', 'toffee','nut', 'citrus', 'creamy', 'earthy', 'leaf', 'nutmeg', 'cinnamon', 'banana', 'apple', 'pineapple', 'toast', 'sherry','dry', 'wood', 'bitter', 'coffee']
	characteristic_list1 = characteristic_list[:10]
	characteristic_list2 = characteristic_list[10:20]
	characteristic_list3 = characteristic_list[20:]
	return render_template("product_search.html",
							title='Product selector',
							product_list=product_list,
							characteristic_list=characteristic_list,
							characteristic_list1=characteristic_list1,
							characteristic_list2=characteristic_list2,
							characteristic_list3=characteristic_list3)




@app.route('/output')
def recommendation_output():
	# pull 'product_name' from input field and store it
	product_name = request.args.get('yourFavScotch')
	# print('PRODUCT NAME = {}'.format(product_name))

	# chara_list = []
	# for ind in range(3):
	# 	character = request.args.get('characteristic_name' + str(ind+1))
	# 	if character != 'Choose...':
	# 		chara_list.append(character)
	if request.method == "GET":
		chara_list = request.args.getlist("check")
		print(chara_list)

	
	# # query the product ID based on product_name
	# query_info = "SELECT index FROM products_info_table WHERE name='%s';" % product_name
	# query_prod_id = list(pd.read_sql_query(query_info, con)['index'])[0]
	df_withtags_path = 'cleanedDataFrameWithTagsV2.json'
	df_withtags = pd.read_json(df_withtags_path)
	if len(product_name) > 0:
		query_prod_id = df_withtags.name[df_withtags.name == product_name].index.tolist()[0]
	else:
		query_prod_id = None

	product_name_list = list(df_withtags['name'])

	recommendation_list, recom_urls_list, recom_list_items_tags, img_url_list = recommendation(query_prod_id, chara_list)
	
	num_recom_prods = len(recommendation_list)
	list_prod_classes = []
	for ind in range(num_recom_prods):
		searched_tags_list = []
		for chara in chara_list:
			if chara in recom_list_items_tags[ind]:
				searched_tags_list.append(chara)
		
		othertags_list = [tag for tag in recom_list_items_tags[ind] if tag not in searched_tags_list]

		prod_class = Product(recommendation_list[ind], recom_urls_list[ind], searched_tags_list, othertags_list, img_url_list[ind])
		

		list_prod_classes.append(prod_class)
	


	# print(recom_urls_list)
	# still need this for the drop down menu
	# query_list = "SELECT index, name FROM products_info_table;"
	# query_list_results = pd.read_sql_query(query_list, con)
	# product_id_list = list(query_list_results['index'])
	# product_name_list = list(query_list_results['name'])

	characteristic_list = ['soft', 'sweet', 'fruit', 'smoky', 'rich', 'fresh', 'vanilla', 'spice', 'peaty', 'chocolate', 'oak', 'toffee','nut', 'citrus', 'creamy', 'earthy', 'leaf', 'nutmeg', 'cinnamon', 'banana', 'apple', 'pineapple', 'toast', 'sherry','dry', 'wood', 'bitter', 'coffee']
	characteristic_list1 = characteristic_list[:10]
	characteristic_list2 = characteristic_list[10:20]
	characteristic_list3 = characteristic_list[20:]
	return render_template("output.html", 
							list_prod_classes = list_prod_classes,
							product_name = product_name, chara_list = chara_list,
							product_list=product_name_list,
							characteristic_list=characteristic_list,
							characteristic_list1=characteristic_list1,
							characteristic_list2=characteristic_list2,
							characteristic_list3=characteristic_list3)

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")
