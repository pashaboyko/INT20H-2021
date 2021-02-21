#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask
import os
import logg
import pandas as pd
import numpy as np
from pipeline import Model_Pipeline,CleaningTextData,FillingNaN,TfIdf
from sklearn.ensemble import GradientBoostingRegressor
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import request, jsonify

log = None
app = Flask(__name__)

data=""
prediction = None
gbr_result = None


@app.route('/')
def hello_world():
	return str(data)

@app.route('/api/v1/resources/books', methods=['GET'])
def api_filter():
    query_parameters = request.args

    id = query_parameters.get('id')
    book_title = query_parameters.get('book_title')
    book_image_url = query_parameters.get('book_image_url')
    book_desc = query_parameters.get('book_desc')
    book_genre = query_parameters.get('book_genre')
    book_authors = query_parameters.get('book_authors')
    book_format = query_parameters.get('book_format')
    book_pages = query_parameters.get('book_pages')
    book_review_count = query_parameters.get('book_review_count')
    book_rating_count = query_parameters.get('book_rating_count')
    
    d = {'id':id, 'book_title':book_title,'book_image_url':book_image_url, 'book_desc':book_desc, 'book_genre':book_genre, 'book_authors':book_authors, 'book_format':book_format, 'book_pages':book_pages,'book_review_count':book_review_count, 'book_rating_count':book_rating_count }

    print(d)
    if not (id or book_title or book_image_url or book_desc or book_genre or book_authors or book_format or book_pages or book_review_count or book_rating_count):
        return page_not_found(404)

    df = pd.DataFrame(d, index=[0])
    df = df.set_index('id')

    print(df)

    submition_res = func(df)



    return jsonify(submition_res.to_json(orient="records"))

@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404

def func(X_test):

    data = pipeline.pipelineData(X_test)

    print("sdfknksdnfklsldnfnklsdnf")

    print(type(data))


    print("sdfknksdnfsdfklaslflnksalfdkklsldnfnklsdnf")
    data = pd.DataFrame(data)

    print(data)

    print(type(data))

    prediction = gbr_result.predict(data)

    submition_res = pd.DataFrame({'id': list(X_test.index), 'book_rating': prediction})
    print(submition_res)

    return submition_res


if __name__ == "__main__":
	
    log_directory = 'log'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)


    pipeline = Model_Pipeline("dft_idf_200.joblib")
    gbr_result = Model_Pipeline("GradientBoostingRegressor.joblib")
    

    #log = logg.setup_logging('Server')
    #log = logg.get_log("Web-server")

    app.run(debug=False,host='0.0.0.0')