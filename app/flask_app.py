#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask
import os
import logg
import pandas as pd
import numpy as np
from pipeline import Model_Pipeline,CleaningTextData,FillingNaN,TfIdf
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

log = None
app = Flask(__name__)

data = ""

@app.route('/')
def hello_world():
	return str(data)

if __name__ == "__main__":
	
    log_directory = 'log'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    #log = logg.setup_logging('Server')
    #log = logg.get_log("Web-server")


    X_full = pd.read_csv('test.csv', index_col='id')
    pipeline = Model_Pipeline("dft_idf_200.joblib")

    data = pipeline.pipelineData(X_full)


    print(data)
    app.run(debug=False,host='0.0.0.0')